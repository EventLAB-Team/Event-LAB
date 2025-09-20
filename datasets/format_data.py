#!/usr/bin/env python3
"""
Generalized HDF5 Formatter
Converts various sensor data formats to standardized HDF5 format based on configuration
"""

import os, h5py, subprocess, requests, json, cv2, torch, glob
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
# --- at top of module (once) ---
EVENT_T_KEYS = ("t", "timestamp", "timestamps", "time", "times")
EVENT_X_KEYS = ("x", "x_coordinate", "x_coordinates", "u", "col", "column")
EVENT_Y_KEYS = ("y", "y_coordinate", "y_coordinates", "v", "row")
EVENT_P_KEYS = ("p", "polarity", "polarities", "pol", "polarity_bit", "polarity_bits")

class DataFormatter(ABC):
    """Abstract base class for data formatters"""
    
    def __init__(self, config, data_config, dataset_name, sequence_name):
        self.config = config
        self.data_config = data_config
        self.dataset_name = dataset_name
        self.sequence_name = sequence_name
        
    @abstractmethod
    def format_to_hdf5(self, input_path, output_path):
        """Format data to HDF5"""
        pass

class BagEventFormatter(DataFormatter):
    """Formatter for ROS bag files containing event data"""
    
    def __init__(self, config, data_config, dataset_name, sequence_name):
        super().__init__(config, data_config, dataset_name, sequence_name)
        
        # Import ROS dependencies only when needed
        try:
            import rosbag
            from cv_bridge import CvBridge
            self.rosbag = rosbag
            self.bridge = CvBridge()
        except ImportError:
            raise ImportError("ROS dependencies not found. Please install rosbag and cv_bridge")
    
    def format_to_hdf5(self, input_path, output_path):
        """Convert bag file to HDF5 format (events only)"""
        print(f"Converting bag file: {input_path} → {output_path}")
        
        # Get format configuration
        format_config = self.data_config['format']['data']
        contents = format_config.get('contents', [])
        
        # Find event topics
        event_contents = [content for content in contents if 'event' in content.lower()]
        
        if not event_contents:
            raise ValueError(f"No event topics found in contents: {contents}")
        
        print(f"Event topics to process: {event_contents}")
        
        with self.rosbag.Bag(input_path, 'r') as bag:
            # Get bag info and find available event topics
            info = bag.get_type_and_topic_info()
            available_topics = set(info[1].keys())
            
            print(f"Available topics in bag: {list(available_topics)}")
            
            # Debug: Show topic info structure
            if available_topics:
                sample_topic = list(available_topics)[0]
                sample_info = info[1][sample_topic]
                print(f"Sample topic info attributes: {dir(sample_info)}")
                print(f"Sample topic info: {sample_info}")
            
            # Find which event topics actually exist in the bag
            valid_event_topics = []
            for event_topic in event_contents:
                if event_topic in available_topics:
                    valid_event_topics.append(event_topic)
                else:
                    # Try to find similar topics
                    similar_topics = [t for t in available_topics if 'event' in t.lower()]
                    if similar_topics:
                        print(f"Topic {event_topic} not found, but found similar: {similar_topics}")
                        valid_event_topics.extend(similar_topics)
            
            if not valid_event_topics:
                raise ValueError(f"No valid event topics found in bag file. Available: {list(available_topics)}")
            
            # Use the first valid event topic
            event_topic = valid_event_topics[0]
            print(f"Processing event topic: {event_topic}")
            
            with h5py.File(output_path, 'w') as h5f:
                self._create_events_structure(h5f, bag, event_topic)
                self._process_event_messages(h5f, bag, event_topic)
                
                # Add metadata
                h5f.attrs['source_file'] = os.path.basename(input_path)
                h5f.attrs['dataset_name'] = self.dataset_name
                h5f.attrs['sequence_name'] = self.sequence_name
                h5f.attrs['event_topic'] = event_topic
                h5f.attrs['available_topics'] = list(available_topics)
                h5f.attrs['formatter'] = 'BagEventFormatter'
                
                self._print_summary(h5f)
        
        print(f"✓ Conversion complete: {output_path}")
    
    def _create_events_structure(self, h5f, bag, event_topic):
        """Create HDF5 structure for events"""
        info = bag.get_type_and_topic_info()
        topic_info = info[1][event_topic]
        msg_count = topic_info.message_count
        
        # Create events group
        events_group = h5f.create_group('events')
        
        # Estimate maximum events (conservative estimate)
        max_events = msg_count * 50000  # Assume up to 50k events per message
        chunk_size = 100000  # 100k events per chunk for good I/O
        
        # Create datasets with optimal chunking
        events_group.create_dataset('x', shape=(0,), maxshape=(max_events,),
                                  dtype=np.uint16, chunks=(chunk_size,), compression='lzf')
        events_group.create_dataset('y', shape=(0,), maxshape=(max_events,),
                                  dtype=np.uint16, chunks=(chunk_size,), compression='lzf')
        events_group.create_dataset('t', shape=(0,), maxshape=(max_events,),
                                  dtype=np.uint64, chunks=(chunk_size,), compression='lzf')
        events_group.create_dataset('p', shape=(0,), maxshape=(max_events,),
                                  dtype=np.bool_, chunks=(chunk_size,), compression='lzf')
        
        # Store metadata (using correct attribute names for TopicTuple)
        events_group.attrs['topic'] = event_topic
        events_group.attrs['message_type'] = getattr(topic_info, 'msg_type', 'unknown')
        events_group.attrs['message_count'] = msg_count
        events_group.attrs['frequency'] = getattr(topic_info, 'frequency', 0.0)
        
        print(f"Created HDF5 structure for {msg_count} event messages")
        print(f"Message type: {events_group.attrs['message_type']}")
        print(f"Frequency: {events_group.attrs['frequency']:.2f} Hz")
    
    def _process_event_messages(self, h5f, bag, event_topic):
        """Process event messages from bag"""
        info = bag.get_type_and_topic_info()
        total_messages = info[1][event_topic].message_count
        total_events_processed = 0
        
        events_group = h5f['events']
        
        with tqdm(total=total_messages, desc="Processing event messages") as pbar:
            for topic, msg, timestamp in bag.read_messages(topics=[event_topic]):
                num_events = self._extract_and_store_events(events_group, msg, timestamp)
                total_events_processed += num_events
                pbar.update(1)
        
        print(f"Processed {total_events_processed:,} total events from {total_messages} messages")
    
    def _extract_and_store_events(self, events_group, msg, timestamp):
        """Extract events from message and store in HDF5"""
        events = []
        
        try:
            # Handle different event message formats
            if hasattr(msg, 'events'):
                # Standard EventArray format
                for event in msg.events:
                    # Handle different timestamp formats
                    if hasattr(event, 'ts'):
                        if hasattr(event.ts, 'to_nsec'):
                            event_time = event.ts.to_nsec()
                        else:
                            event_time = int(event.ts)
                    elif hasattr(event, 't'):
                        event_time = int(event.t)
                    else:
                        event_time = timestamp.to_nsec()
                    
                    # Handle different polarity formats
                    if hasattr(event, 'polarity'):
                        polarity = bool(event.polarity)
                    elif hasattr(event, 'pol'):
                        polarity = bool(event.pol)
                    elif hasattr(event, 'p'):
                        polarity = bool(event.p)
                    else:
                        polarity = True
                    
                    events.append([
                        int(event.x),
                        int(event.y),
                        event_time,
                        polarity
                    ])
            
            elif hasattr(msg, 'data') and hasattr(msg, 'width') and hasattr(msg, 'height'):
                # Some custom event formats store raw data
                print("Warning: Custom event format detected - may need manual adaptation")
                return 0
                
        except Exception as e:
            print(f"Warning: Could not process event message: {e}")
            return 0
        
        if not events:
            return 0
        
        # Convert to numpy and store
        events_array = np.array(events)
        num_new_events = len(events_array)
        
        # Resize datasets
        current_size = events_group['x'].shape[0]
        new_size = current_size + num_new_events
        
        for dataset_name in ['x', 'y', 't', 'p']:
            events_group[dataset_name].resize((new_size,))
        
        # Store events
        events_group['x'][current_size:new_size] = events_array[:, 0].astype(np.uint16)
        events_group['y'][current_size:new_size] = events_array[:, 1].astype(np.uint16)
        events_group['t'][current_size:new_size] = events_array[:, 2].astype(np.uint64)
        events_group['p'][current_size:new_size] = events_array[:, 3].astype(np.bool_)
        
        return num_new_events
    
    def _print_summary(self, h5f):
        """Print formatting summary"""
        if 'events' not in h5f:
            print("No events found in HDF5 file")
            return
            
        events_group = h5f['events']
        num_events = events_group['x'].shape[0]
        
        print(f"\n=== Event Formatting Summary ===")
        print(f"Total events: {num_events:,}")
        
        if num_events > 0:
            print(f"X range: {events_group['x'][:].min()} - {events_group['x'][:].max()}")
            print(f"Y range: {events_group['y'][:].min()} - {events_group['y'][:].max()}")
            print(f"Time range: {events_group['t'][:].min()} - {events_group['t'][:].max()} ns")
            
            # Calculate statistics
            time_span_ns = events_group['t'][:].max() - events_group['t'][:].min()
            duration_seconds = time_span_ns / 1e9
            event_rate = num_events / duration_seconds if duration_seconds > 0 else 0
            
            print(f"Duration: {duration_seconds:.2f} seconds")
            print(f"Average event rate: {event_rate:.0f} events/second")
            
            # File size
            total_size_mb = sum([events_group[key].nbytes for key in ['x', 'y', 't', 'p']]) / (1024*1024)
            print(f"HDF5 file size: {total_size_mb:.1f} MB")

class GeneralizedHDF5Formatter:
    """Main formatter class that delegates to specific formatters"""
    
    def __init__(self, config, data_config, dataset_name, sequence_name, query=False):
        self.config = config
        self.data_config = data_config
        self.dataset_name = dataset_name
        self.sequence_name = sequence_name
        self.query = query

        # Registry of available formatters
        self.formatters = {
            'bag': BagEventFormatter
        }
    
    def format_data(self, input_path, output_path):
        """Format data based on configuration"""
        # Get data format from config
        data_format = self.data_config['format']['data']['format'].lower()
        
        if data_format not in self.formatters:
            raise ValueError(f"Unsupported data format: {data_format}. "
                           f"Supported formats: {list(self.formatters.keys())}")
        
        # Get appropriate formatter
        formatter_class = self.formatters[data_format]
        formatter = formatter_class(self.config, self.data_config, self.dataset_name, self.sequence_name)
        
        # Perform formatting
        formatter.format_to_hdf5(input_path, output_path)
        
        print(f"✓ Data formatting complete using {formatter_class.__name__}")

def format_sequence_data(config, data_config, dataset_name, sequence_name):
    """
    Main function to format sequence data to HDF5
    
    Args:
        config: General configuration dict
        data_config: Dataset-specific configuration dict  
        dataset_name: Name of the dataset
        sequence_name: Name of the sequence
    """
    # Get file format from data config
    file_format = data_config['format']['data']['format'].lower()
    
    # Construct paths
    input_file = os.path.join(config['data_path'], dataset_name, sequence_name, 
                             f"{sequence_name}.{file_format}")
    output_file = os.path.join(config['data_path'], dataset_name, sequence_name,
                              f"{sequence_name}.{config['std_format']}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if os.path.exists(output_file):
        print(f"Formatted file already exists: {output_file}")
        return
    
    print(f"Formatting {input_file} to {output_file}")
    
    # Create formatter and format
    formatter = GeneralizedHDF5Formatter(config, data_config, dataset_name, sequence_name)
    formatter.format_data(input_file, output_file)

#!/usr/bin/env python3
"""
Generalized Event Frame Builder
Creates frame representations from event data using memory-efficient streaming
"""
class FrameAccumulator(ABC):
    """Abstract base class for different frame accumulation methods"""
    
    @abstractmethod
    def accumulate_events(self, x, y, t, p, frame_start_time, frame_end_time):
        """Accumulate events into a frame"""
        pass
    
    @abstractmethod
    def get_frame_shape(self, width, height):
        """Get the output frame shape"""
        pass

class EventCountFrameAccumulator:
    def __init__(self, width, height, max_events_per_frame, hot_pixels=None, polarity_mode='separate'):
        """
        polarity_mode:
          - 'ignore'   -> (H, W) counts (pos+neg)
          - 'separate' -> (H, W, 2) counts [pos, neg]
          - 'signed'   -> (H, W) counts (pos - neg)
          - 'pos'      -> (H, W) counts (positive only)
          - 'neg'      -> (H, W) counts (negative only)
        """
        def _to_int_scalar(v, default=None):
            if v is None:
                if default is None:
                    raise ValueError("max_events_per_frame is required")
                return int(default)
            if isinstance(v, (int, np.integer)):
                return int(v)
            a = np.asarray(v)
            if a.shape == ():          # NumPy scalar
                return int(a.item())
            if a.size == 1:            # 1-element list/array/tuple
                return int(a.reshape(()).item())
            raise ValueError(f"max_events_per_frame must be scalar; got shape {a.shape}")

        if polarity_mode not in ('ignore', 'separate', 'signed', 'pos', 'neg'):
            raise ValueError(f"Invalid polarity_mode '{polarity_mode}'")

        self.width = int(width)
        self.height = int(height)
        self.max_events = _to_int_scalar(max_events_per_frame)
        self.hot_pixels = hot_pixels
        self.polarity_mode = polarity_mode

        self.total_out_of_bounds = 0
        self.total_hot_pixels_filtered = 0
        self.warned_already = False

        # Precompute linear hot-pixel indices for O(1) isin checks
        if self.hot_pixels is not None:
            hp = np.asarray(self.hot_pixels, dtype=np.int64)
            inb = (hp[:, 0] >= 0) & (hp[:, 0] < self.width) & (hp[:, 1] >= 0) & (hp[:, 1] < self.height)
            hp = hp[inb] if hp.size else hp
            self._hot_lin = (hp[:, 1] * self.width + hp[:, 0]) if hp.size else np.empty((0,), dtype=np.int64)
            print(f"Hot pixel filtering enabled: {len(self._hot_lin)} pixels will be excluded")
        else:
            self._hot_lin = None

    def _mask_valid(self, x, y):
        return (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)

    def accumulate_by_count(self, x, y, t, p, start_idx=0):
        """
        Build one frame using at most `self.max_events` events starting at `start_idx`.

        Returns:
          frame:  (H, W) or (H, W, 2) float32 array depending on polarity_mode
          next_idx: int, index to resume from for the next frame
          t0: float, timestamp of the first used event (np.nan if none)
          t1: float, timestamp of the last  used event (np.nan if none)
        """
        N = x.size
        if start_idx >= N:
            return self._zero_frame(), start_idx, np.nan, np.nan

        end_idx = min(start_idx + self.max_events, N)
        xs = x[start_idx:end_idx]
        ys = y[start_idx:end_idx]
        ts = t[start_idx:end_idx] if (t is not None and getattr(t, "size", 0)) else None
        ps = p[start_idx:end_idx] if (p is not None and getattr(p, "size", 0)) else None

        # Bounds check
        valid = self._mask_valid(xs, ys)
        if not np.all(valid):
            self.total_out_of_bounds += int((~valid).sum())
            if not self.warned_already and (~valid).any():
                invx, invy = xs[~valid], ys[~valid]
                print("Warning: Found out-of-bounds events. Example ranges:")
                print(f"  X range: {invx.min()} - {invx.max()} (valid: 0 - {self.width-1})")
                print(f"  Y range: {invy.min()} - {invy.max()} (valid: 0 - {self.height-1})")
                print("  Will continue counting but suppress further warnings...")
                self.warned_already = True
        xs = xs[valid]; ys = ys[valid]
        if ts is not None: ts = ts[valid]
        if ps is not None: ps = ps[valid]

        # Hot-pixel removal
        if self._hot_lin is not None and xs.size:
            lin_all = (ys.astype(np.int64) * self.width + xs.astype(np.int64))
            hot = np.isin(lin_all, self._hot_lin, assume_unique=False)
            self.total_hot_pixels_filtered += int(hot.sum())
            keep = ~hot
            xs, ys = xs[keep], ys[keep]
            if ts is not None: ts = ts[keep]
            if ps is not None: ps = ps[keep]

        # If nothing useful remains
        if xs.size == 0:
            t0 = t[start_idx] if (t is not None and getattr(t, "size", 0) and start_idx < t.size) else np.nan
            t1 = t[end_idx-1] if (t is not None and getattr(t, "size", 0) and end_idx-1 < t.size) else np.nan
            return self._zero_frame(), end_idx, t0, t1

        # Vectorized accumulation (with polarity)
        H, W = self.height, self.width
        frame_size = H * W
        lin = (ys.astype(np.int64) * W + xs.astype(np.int64))

        if ps is None or self.polarity_mode == 'ignore':
            # No polarity or ignoring it
            total_counts = np.bincount(lin, minlength=frame_size).astype(np.float32, copy=False)
            out = total_counts.reshape(H, W)
        else:
            # Robust polarity masks (works for {0,1}, {-1,+1}, or bool)
            ps = np.asarray(ps)
            pos_mask = (ps > 0)
            neg_mask = ~pos_mask  # includes 0 and negative
            pos_lin = lin[pos_mask]
            neg_lin = lin[neg_mask]

            pos_counts = np.bincount(pos_lin, minlength=frame_size).astype(np.float32, copy=False) if pos_lin.size else np.zeros(frame_size, dtype=np.float32)
            neg_counts = np.bincount(neg_lin, minlength=frame_size).astype(np.float32, copy=False) if neg_lin.size else np.zeros(frame_size, dtype=np.float32)

            if self.polarity_mode == 'separate':
                out = np.stack([pos_counts.reshape(H, W), neg_counts.reshape(H, W)], axis=-1)  # (H,W,2)
            elif self.polarity_mode == 'signed':
                out = (pos_counts - neg_counts).reshape(H, W)
            elif self.polarity_mode == 'pos':
                out = pos_counts.reshape(H, W)
            elif self.polarity_mode == 'neg':
                out = neg_counts.reshape(H, W)
            else:
                # fallback: total
                out = (pos_counts + neg_counts).reshape(H, W)

        # Time metadata from actually-used events (post filtering)
        if ts is not None and ts.size:
            t0, t1 = ts[0], ts[-1]
        else:
            t0 = t[start_idx] if (t is not None and getattr(t, "size", 0) and start_idx < t.size) else np.nan
            t1 = t[end_idx-1] if (t is not None and getattr(t, "size", 0) and end_idx-1 < t.size) else np.nan

        return out, end_idx, t0, t1

    def iter_frames(self, x, y, t, p, start_idx=0):
        """
        Generator yielding frames until the stream is exhausted.
        Yields: (frame, next_idx, t0, t1) where frame is (H,W) or (H,W,2) per polarity_mode.
        """
        idx = start_idx
        N = x.size
        while idx < N:
            frame, idx, t0, t1 = self.accumulate_by_count(x, y, t, p, idx)
            yield frame, idx, t0, t1

    def get_frame_shape(self, width=None, height=None):
        w = self.width if width is None else int(width)
        h = self.height if height is None else int(height)
        if self.polarity_mode == 'separate':
            return (h, w, 2)
        else:
            return (h, w)

    def _zero_frame(self):
        if self.polarity_mode == 'separate':
            return np.zeros((self.height, self.width, 2), dtype=np.float32)
        else:
            return np.zeros((self.height, self.width), dtype=np.float32)

class CountFrameAccumulator(FrameAccumulator):
    """Simple event count accumulator (vectorized, no per-hot-pixel loop)."""
    def __init__(self, width, height, hot_pixels=None):
        self.width = width
        self.height = height
        self.hot_pixels = hot_pixels
        self.total_out_of_bounds = 0
        self.total_hot_pixels_filtered = 0
        self.warned_already = False

        # Precompute linear hot-pixel indices for O(1) isin checks
        if self.hot_pixels is not None:
            hp = np.asarray(self.hot_pixels, dtype=np.int64)
            inb = (hp[:,0] >= 0) & (hp[:,0] < self.width) & (hp[:,1] >= 0) & (hp[:,1] < self.height)
            hp = hp[inb] if hp.size else hp
            self._hot_lin = (hp[:,1] * self.width + hp[:,0]) if hp.size else np.empty((0,), dtype=np.int64)
            print(f"Hot pixel filtering enabled: {len(self._hot_lin)} pixels will be excluded")
        else:
            self._hot_lin = None

    def _mask_valid(self, x, y):
        return (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)

    def accumulate_events(self, x, y, t, p, frame_start_time, frame_end_time):
        # The builder passes only the frame’s slice; keep time check for safety.
        time_mask = (t >= frame_start_time) & (t < frame_end_time)
        if not np.any(time_mask):
            return np.zeros((self.height, self.width), dtype=np.float32)

        x = x[time_mask]; y = y[time_mask]

        valid = self._mask_valid(x, y)
        if not np.any(valid):
            return np.zeros((self.height, self.width), dtype=np.float32)

        if np.any(~valid):
            self.total_out_of_bounds += int((~valid).sum())
            if not self.warned_already:
                invx, invy = x[~valid], y[~valid]
                print(f"Warning: Found out-of-bounds events. Example ranges:")
                print(f"  X range: {invx.min()} - {invx.max()} (valid: 0 - {self.width-1})")
                print(f"  Y range: {invy.min()} - {invy.max()} (valid: 0 - {self.height-1})")
                print("  Will continue counting but suppress further warnings...")
                self.warned_already = True
        x = x[valid]; y = y[valid]

        # Hot-pixel removal
        if self._hot_lin is not None and x.size:
            lin = (y.astype(np.int64) * self.width + x.astype(np.int64))
            hot = np.isin(lin, self._hot_lin, assume_unique=False)
            self.total_hot_pixels_filtered += int(hot.sum())
            if hot.any():
                x, y = x[~hot], y[~hot]
                if x.size == 0:
                    return np.zeros((self.height, self.width), dtype=np.float32)

        # Vectorized accumulation via bincount
        frame = np.zeros((self.height * self.width,), dtype=np.float32)
        lin = (y.astype(np.int64) * self.width + x.astype(np.int64))
        counts = np.bincount(lin, minlength=frame.size)
        frame[:counts.size] = counts.astype(np.float32, copy=False)
        return frame.reshape(self.height, self.width)

    def get_frame_shape(self, width, height):
        return (height, width)

class PolarityFrameAccumulator(FrameAccumulator):
    """Separate positive/negative polarity accumulator (GPU-optional, same API)"""
    
    def __init__(self, width, height, hot_pixels=None):
        self.width = width
        self.height = height
        self.total_out_of_bounds = 0
        self.total_hot_pixels_filtered = 0
        self.warned_already = False
        self.hot_pixels = hot_pixels

        # --- NEW: optional GPU switch (off by default) and torch handle ---
        self.use_gpu = True                    # set to True externally to enable GPU
        self._torch = None
        self._device = None
        try:  # lazy import; no hard dependency
            self._torch = torch
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        except Exception:
            self._torch = None
            self._device = None
        print(f"PolarityFrameAccumulator initialized. GPU support: {'yes' if self._torch is not None else 'no'}, device: {self._device}")
        # Precompute hot-pixel linear indices for fast membership tests
        if self.hot_pixels is not None:
            hp = np.asarray(self.hot_pixels, dtype=np.int64)
            # keep only those within bounds
            inb = (hp[:,0] >= 0) & (hp[:,0] < self.width) & (hp[:,1] >= 0) & (hp[:,1] < self.height)
            hp = hp[inb] if hp.size else hp
            self._hot_lin_np = (hp[:,1] * self.width + hp[:,0]) if hp.size else np.empty((0,), dtype=np.int64)
            print(f"Hot pixel filtering enabled: {len(self._hot_lin_np)} pixels will be excluded")
        else:
            self._hot_lin_np = None

    # (unchanged signature)
    def _filter_hot_pixels(self, x, y):
        """Return boolean mask of non-hot events (CPU path)."""
        if self._hot_lin_np is None:
            return np.ones(len(x), dtype=bool)
        lin = (y.astype(np.int64) * self.width + x.astype(np.int64))
        hot_mask = np.isin(lin, self._hot_lin_np, assume_unique=False)
        self.total_hot_pixels_filtered += int(hot_mask.sum())
        return ~hot_mask

    # (unchanged signature)
    def accumulate_events(self, x, y, t, p, frame_start_time, frame_end_time):
        """Accumulate positive and negative events separately."""
        # ---- TIME WINDOW ----
        time_mask = (t >= frame_start_time) & (t < frame_end_time)
        if not np.any(time_mask):
            return np.zeros((self.height, self.width, 2), dtype=np.float32)

        x_filtered = x[time_mask]
        y_filtered = y[time_mask]
        p_filtered = p[time_mask]

        # ---- HOT PIXELS ----
        if self.hot_pixels is not None:
            hp_mask = self._filter_hot_pixels(x_filtered, y_filtered)
            x_filtered = x_filtered[hp_mask]
            y_filtered = y_filtered[hp_mask]
            p_filtered = p_filtered[hp_mask]
            if len(x_filtered) == 0:
                return np.zeros((self.height, self.width, 2), dtype=np.float32)

        # ---- BOUNDS ----
        valid_coords = (
            (x_filtered >= 0) & (x_filtered < self.width) &
            (y_filtered >= 0) & (y_filtered < self.height)
        )
        if not np.any(valid_coords):
            return np.zeros((self.height, self.width, 2), dtype=np.float32)

        x_valid = x_filtered[valid_coords]
        y_valid = y_filtered[valid_coords]
        p_valid = p_filtered[valid_coords]

        # Track OOB once
        oob = ~valid_coords
        if np.any(oob):
            self.total_out_of_bounds += int(np.sum(oob))
            if not self.warned_already:
                invalid_x = x_filtered[oob]
                invalid_y = y_filtered[oob]
                print("Warning: Found out-of-bounds events. Example ranges:")
                print(f"  X range: {invalid_x.min()} - {invalid_x.max()} (valid: 0 - {self.width-1})")
                print(f"  Y range: {invalid_y.min()} - {invalid_y.max()} (valid: 0 - {self.height-1})")
                print("  Will continue counting but suppress further warnings...")
                self.warned_already = True

        # ---- ACCUMULATE (GPU if enabled/available, else CPU) ----
        if self.use_gpu and self._torch is not None and self._device is not None:
            torch = self._torch
            with torch.no_grad():
                # to device
                x_t = torch.as_tensor(x_valid, device=self._device, dtype=torch.int64)
                y_t = torch.as_tensor(y_valid, device=self._device, dtype=torch.int64)
                p_t = torch.as_tensor(p_valid, device=self._device, dtype=torch.bool)

                # linear indices
                lin = y_t * self.width + x_t

                # frame buffer: (2, H*W)
                frame = torch.zeros((2, self.height * self.width), dtype=torch.float32, device=self._device)

                if p_t.any():
                    pos_lin = lin[p_t]
                    frame[0].index_add_(0, pos_lin, torch.ones_like(pos_lin, dtype=torch.float32))
                if (~p_t).any():
                    neg_lin = lin[~p_t]
                    frame[1].index_add_(0, neg_lin, torch.ones_like(neg_lin, dtype=torch.float32))

                # reshape to (H, W, 2) on CPU
                return frame.view(2, self.height, self.width).permute(1, 2, 0).cpu().numpy()

        # ---- CPU fallback (numpy) ----
        frame = np.zeros((self.height * self.width, 2), dtype=np.float32)
        lin = (y_valid.astype(np.int64) * self.width + x_valid.astype(np.int64))
        pos = (p_valid == True); neg = ~pos
        if np.any(pos):
            c = np.bincount(lin[pos], minlength=frame.shape[0]).astype(np.float32, copy=False)
            frame[:, 0] = c
        if np.any(neg):
            c = np.bincount(lin[neg], minlength=frame.shape[0]).astype(np.float32, copy=False)
            frame[:, 1] = c
        return frame.reshape(self.height, self.width, 2)


    # (unchanged signature)
    def get_frame_shape(self, width, height):
        return (height, width, 2)

class GeneralizedFrameBuilder:
    """Memory-efficient event frame builder using streaming"""

    def __init__(self, width, height, accumulator_type='count', max_events_per_frame=None, hot_pixels=None):
        self.width = width
        self.height = height
        self.hot_pixels = hot_pixels
        self.max_events = max_events_per_frame

        if accumulator_type == 'count':
            self.accumulator = CountFrameAccumulator(width, height, hot_pixels)
        elif accumulator_type == 'polarity':
            self.accumulator = PolarityFrameAccumulator(width, height, hot_pixels)
        elif accumulator_type == 'eventcount':
            self.accumulator = EventCountFrameAccumulator(width, height, max_events_per_frame=max_events_per_frame, hot_pixels=hot_pixels)
        else:
            raise ValueError(f"Unknown accumulator type: {accumulator_type}")
        self.accumulator_type = accumulator_type

    # ---------- helpers: time scale detection & HDF5 field access ----------
    def _parse_timestamp_val_to_scale(data_config):
        """
        Returns ticks-per-second from data_config['format']['data']['timestamp_val'].
        Accepts: 'ns'|'us'|'ms'|'s' or a positive number (ticks/s).
        """
        val = (data_config.get('format', {})
                        .get('data', {})
                        .get('timestamp_val', None))
        if val is None:
            return None  # fall back to builder detection
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ('ns', 'nanosecond', 'nanoseconds'): return 1e9
            if v in ('us', 'μs', 'microsecond', 'microseconds'): return 1e6
            if v in ('ms', 'millisecond', 'milliseconds'): return 1e3
            if v in ('s', 'sec', 'second', 'seconds'): return 1.0
            # numeric string
            try:
                num = float(v)
                if num > 0: return num
            except Exception:
                pass
            raise ValueError(f"Unsupported timestamp_val='{val}'. Use ns/us/ms/s or numeric ticks/s.")
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
        raise ValueError(f"Invalid timestamp_val={val!r} (expected ns/us/ms/s or positive number)")

    @staticmethod
    def _parse_unit_scale(attrs):
        def as_str(v):
            if isinstance(v, bytes): return v.decode("utf-8","ignore").lower()
            if isinstance(v, str):   return v.lower()
            return None
        for k, v in attrs.items():
            s = as_str(v)
            if not s: continue
            if "nano" in s or s == "ns": return 1e9
            if "micro" in s or s == "us": return 1e6
            if "milli" in s or s == "ms": return 1e3
            if s == "s" or "second" in s: return 1.0
        for v in attrs.values():
            if isinstance(v, (int, float, np.integer, np.floating)) and v > 0:
                return float(v)  # ticks per second
        return None

    def _detect_time_scale(self, h5f, ev, tds=None):
        # try attrs on group/dataset/tds
        keys = ("time_unit","time_units","unit","units","timestamp_unit",
                "t_unit","timebase","time_base","time_scale","resolution")
        attrs = {}
        for node in (ev, tds, h5f):
            if node is None or not hasattr(node, "attrs"): continue
            for k in keys:
                if k in node.attrs and k not in attrs:
                    attrs[k] = node.attrs[k]
        scale = self._parse_unit_scale(attrs)  # ticks per second
        if scale is None:
            # infer from duration vs plausible rates if needed
            scale = 1e6  # conservative default: microseconds
        return float(scale)

    @staticmethod
    def _read_first_last_t(tds, n_total):
        """Read first/last timestamp from tds (Dataset or FieldsWrapper) using scalar indexing."""
        n = int(n_total)
        if n == 0:
            return 0, 0
        first = int(tds[0])         # scalar read
        last  = int(tds[n-1])       # scalar read
        return first, last

    @staticmethod
    def _searchsorted_h5(tds, target_tick, side="left", n_total=None):
        """
        Binary search over a 1-D HDF5 dataset or FieldsWrapper.
        We cannot rely on tds.shape for FieldsWrapper, so use provided n_total.
        """
        if n_total is None:
            # best-effort fallback; works for real Datasets
            try:
                n_total = int(tds.shape[0])
            except Exception:
                raise AttributeError("n_total is required for FieldsWrapper without .shape")
        n = int(n_total)
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            tm = int(tds[mid])  # scalar read
            if tm > target_tick or (side == "right" and tm == target_tick):
                hi = mid
            else:
                lo = mid + 1
        return lo if side == "right" else lo

    def _fields_handles(self, ev):
        """
        Normalize an events container (group or dataset) to field handles:
        returns (ds, n_total) with keys ds['t'], ds['x'], ds['y'], ds['p'] (p may be None).
        Each ds[key] supports 1D indexing (cheap HDF5 reads), exposes .shape and .dtype.
        """

        # ---- helpers: lightweight 1D "views" that read slices lazily ----
        class _ColView:
            # for numeric NxC datasets (e.g., columns [x,y,t,(p)])
            def __init__(self, base, col):
                self._b, self._c = base, col
            def __getitem__(self, idx):
                return self._b[idx, self._c]
            @property
            def shape(self):
                return (self._b.shape[0],)
            @property
            def dtype(self):
                return self._b.dtype

        class _FieldView:
            # for compound dtype datasets (named fields)
            def __init__(self, base, field):
                self._b, self._f = base, field
            def __getitem__(self, idx):
                # reads only the requested slice, then selects the field
                return self._b[idx][self._f]
            @property
            def shape(self):
                return (self._b.shape[0],)
            @property
            def dtype(self):
                return self._b.dtype[self._f]

        def _first_present_key(grp, keys):
            for k in keys:
                if k in grp and isinstance(grp[k], h5py.Dataset):
                    return k
            return None

        # ---- split layout: group with x/y/t(/p) datasets (accept aliases) ----
        if isinstance(ev, h5py.Group):
            tkey = _first_present_key(ev, EVENT_T_KEYS)
            xkey = _first_present_key(ev, EVENT_X_KEYS)
            ykey = _first_present_key(ev, EVENT_Y_KEYS)
            pkey = _first_present_key(ev, EVENT_P_KEYS)

            if not (tkey or xkey or ykey):
                raise ValueError("events group has no recognizable t/x/y datasets (checked aliases).")

            # Pick any present stream to determine length
            probe = ev[tkey] if tkey else (ev[xkey] if xkey else ev[ykey])
            n_total = int(probe.shape[0])

            ds = {
                "t": ev[tkey] if tkey else None,
                "x": ev[xkey] if xkey else None,
                "y": ev[ykey] if ykey else None,
                "p": ev[pkey] if pkey else None,
            }
            return ds, n_total

        # ---- packed layout: single dataset named 'events' (compound or Nx>=3) ----
        if isinstance(ev, h5py.Dataset):
            dset = ev
            n_total = int(dset.shape[0])

            # compound dtype with named fields
            if dset.dtype.names:
                names_lut = {n.lower(): n for n in dset.dtype.names}
                def _field_name(aliases):
                    for a in aliases:
                        if a in names_lut:
                            return names_lut[a]
                    return None

                t_field = _field_name(EVENT_T_KEYS)
                x_field = _field_name(EVENT_X_KEYS)
                y_field = _field_name(EVENT_Y_KEYS)
                p_field = _field_name(EVENT_P_KEYS)

                if not (t_field or x_field or y_field):
                    raise ValueError("compound events dataset missing time/coords fields (checked aliases).")

                ds = {
                    "t": _FieldView(dset, t_field) if t_field else None,
                    "x": _FieldView(dset, x_field) if x_field else None,
                    "y": _FieldView(dset, y_field) if y_field else None,
                    "p": _FieldView(dset, p_field) if p_field else None,
                }
                return ds, n_total

            # plain numeric Nx>=3: assume columns [x,y,t,(p?)]
            if dset.ndim == 2 and dset.shape[1] >= 3:
                ds = {
                    "x": _ColView(dset, 0),
                    "y": _ColView(dset, 1),
                    "t": _ColView(dset, 2),
                    "p": _ColView(dset, 3) if dset.shape[1] > 3 else None,
                }
                return ds, n_total

            raise ValueError("Unsupported events dataset shape; expected compound or numeric Nx>=3.")

        # ---- fallback ----
        raise TypeError(f"Unsupported /events node type: {type(ev)}")

    
    def _append_metadata(self, output_dir, **updates):
        meta_path = os.path.join(output_dir, 'metadata.json')
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        meta.update({k: (float(v) if hasattr(v, "__float__") else v) for k, v in updates.items()})
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def _process_frames_by_count(self, ds, output_dir, start_idx, end_idx, chunk_size):
        """
        Stream events and write frames with at most `max_events_per_frame` valid (non-hot) events.
        Also records per-frame mid timestamps in ticks as 'event_frame_times_ticks.npy'.
        """

        max_events = self.max_events
        frame_shape = self.accumulator.get_frame_shape(self.width, self.height)

        # Infer channels from frame_shape (e.g., (H, W, 2) for polarity)
        if len(frame_shape) == 3:
            H, W, C = frame_shape
        else:
            H, W = frame_shape
            C = 1

        has_p = ("p" in ds)  # only needed if C > 1

        hot_lin = getattr(self.accumulator, "_hot_lin", None)
        warned  = bool(getattr(self.accumulator, "warned_already", False))

        def _flush(idx, frame_arr, t0_tick, t1_tick):
            np.save(os.path.join(output_dir, f"frame_{idx:06d}.npy"), frame_arr)
            if idx < 3:
                self._save_frame_preview(frame_arr, output_dir, idx)
            if t0_tick is None or t1_tick is None:
                return None
            # mid-tick (int)
            return int((int(t0_tick) + int(t1_tick)) // 2)

        idx = int(start_idx)
        frames_written = 0
        current_frame = np.zeros(frame_shape, dtype=np.float32)  # respects channels if present
        used_in_frame = 0

        # Track ticks used in current frame
        cur_t0_tick = None
        cur_t1_tick = None

        mid_ticks = []  # list of per-frame mid ticks (ints)
        flat_size = self.width * self.height

        with tqdm(desc=f"Generating frames (≤{max_events} ev/frame)") as pbar:
            while idx < end_idx:
                chunk_end = min(idx + chunk_size, end_idx)

                # Read only what we need
                x_chunk = ds["x"][idx:chunk_end]
                y_chunk = ds["y"][idx:chunk_end]
                t_chunk = ds["t"][idx:chunk_end]
                if C > 1:
                    if not has_p:
                        raise RuntimeError("Accumulator expects multi-channel frames (e.g., polarity), "
                                        "but dataset has no 'p' field.")
                    p_chunk = ds["p"][idx:chunk_end]

                # Validity check
                valid = (x_chunk >= 0) & (x_chunk < self.width) & (y_chunk >= 0) & (y_chunk < self.height)
                if not np.all(valid):
                    bad = (~valid)
                    self.accumulator.total_out_of_bounds += int(bad.sum())
                    if not warned and bad.any():
                        invx, invy = x_chunk[bad], y_chunk[bad]
                        print("Warning: Found out-of-bounds events. Example ranges:")
                        print(f"  X range: {invx.min()} - {invx.max()} (valid: 0 - {self.width-1})")
                        print(f"  Y range: {invy.min()} - {invy.max()} (valid: 0 - {self.height-1})")
                        print("  Will continue counting but suppress further warnings...")
                        warned = True
                        self.accumulator.warned_already = True

                if not np.any(valid):
                    idx = chunk_end
                    continue

                xs = x_chunk[valid]
                ys = y_chunk[valid]
                ts = t_chunk[valid]
                if C > 1:
                    ps = p_chunk[valid]

                lin = (ys.astype(np.int64) * self.width + xs.astype(np.int64))

                # Hot-pixel filter (on linear index)
                if hot_lin is not None and hot_lin.size:
                    hot = np.isin(lin, hot_lin, assume_unique=False)
                    self.accumulator.total_hot_pixels_filtered += int(hot.sum())
                    keep_mask = ~hot
                else:
                    keep_mask = np.ones(lin.shape, dtype=bool)

                if not keep_mask.any():
                    idx = chunk_end
                    continue

                # positions within the raw slice for kept events (time order preserved)
                valid_pos = np.nonzero(valid)[0]
                keep_pos_within_valid = np.nonzero(keep_mask)[0]
                kept_raw_pos = valid_pos[keep_pos_within_valid]

                need   = max_events - used_in_frame
                take_n = int(min(need, kept_raw_pos.size))

                if take_n > 0:
                    use_lin = lin[keep_mask][:take_n]
                    use_ts  = ts[keep_mask][:take_n]
                    if C > 1:
                        # Map polarity to channel index 0/1 robustly:
                        # works for {0,1} or {-1,1} or {False,True}
                        use_ps_raw = ps[keep_mask][:take_n]
                        use_pol = (use_ps_raw > 0).astype(np.int64)  # 1 for positive, 0 otherwise

                    # accumulate counts
                    if C > 1:
                        # Per-channel bincount and add into each channel plane
                        # Channel 0 (negative/off or non-positive)
                        mask0 = (use_pol == 0)
                        if mask0.any():
                            counts0 = np.bincount(use_lin[mask0], minlength=flat_size).astype(np.float32, copy=False)
                            current_frame[..., 0] += counts0.reshape(H, W)

                        # Channel 1 (positive/on)
                        mask1 = (use_pol == 1)
                        if mask1.any():
                            counts1 = np.bincount(use_lin[mask1], minlength=flat_size).astype(np.float32, copy=False)
                            current_frame[..., 1] += counts1.reshape(H, W)
                    else:
                        counts = np.bincount(use_lin, minlength=flat_size).astype(np.float32, copy=False)
                        current_frame += counts.reshape(H, W)

                    used_in_frame += take_n

                    # update frame tick range
                    first_tick = int(use_ts[0])
                    last_tick  = int(use_ts[-1])
                    if cur_t0_tick is None:
                        cur_t0_tick = first_tick
                    cur_t1_tick = last_tick

                    # Advance idx past last consumed raw row
                    last_used_pos = int(kept_raw_pos[take_n - 1])
                    idx = idx + last_used_pos + 1
                else:
                    # couldn't take more from this chunk (frame probably full)
                    pass

                # Flush if frame full, or end-of-file with partial frame
                if used_in_frame >= max_events or (idx >= end_idx and used_in_frame > 0):
                    mid = _flush(frames_written, current_frame, cur_t0_tick, cur_t1_tick)
                    if mid is not None:
                        mid_ticks.append(mid)
                    frames_written += 1
                    pbar.update(1)

                    # reset
                    current_frame.fill(0.0)
                    used_in_frame = 0
                    cur_t0_tick = None
                    cur_t1_tick = None

                # Continue within this chunk if there are still rows and we need more
                if idx < chunk_end and used_in_frame < max_events:
                    continue
                else:
                    # move to next chunk
                    idx = max(idx, chunk_end)

        # Save per-frame mid ticks
        if mid_ticks:
            np.save(os.path.join(output_dir, "event_frame_times_ticks.npy"),
                    np.asarray(mid_ticks, dtype=np.int64))

        return frames_written


    # -------------------- public API --------------------
    def build_frames(self, hdf5_path, output_dir, timewindow_ns, offset_ns=None,
                    chunk_size=100_000, max_frames=None,
                    rdcc_nbytes=64*1024*1024, rdcc_nslots=1_048_579,
                    time_scale=None, countmatch=False, max_events=None):
        """
        Build frames from HDF5 event data (streaming, unit-aware, memory-safe).
        - timewindow_ns / offset_ns are interpreted as *nanoseconds* and auto-converted.
        """

        os.makedirs(output_dir, exist_ok=True)
        print(f"Building frames: {hdf5_path} → {output_dir}")
        print(f"Accumulator: {self.accumulator_type}")
        if self.accumulator_type != "eventcount":
            print(f"Requested time window: {timewindow_ns/1e6:.1f} ms")

        def _any_present(ks, keys):
            return any(k in ks for k in keys)

        with h5py.File(hdf5_path, "r", swmr=True,
                    rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, rdcc_w0=0.0) as h5f:

            # ---- resolve events node robustly (accept alias names) ----
            ev = None
            if "events" in h5f:
                ev = h5f["events"]
            elif "columns" in h5f:  # e.g. columns/{x,y,t,p}
                ev = h5f["columns"]
            else:
                # try shallow groups first
                for _, obj in h5f.items():
                    if isinstance(obj, h5py.Group):
                        ks = obj.keys()
                        if (_any_present(ks, EVENT_X_KEYS)
                            and _any_present(ks, EVENT_Y_KEYS)
                            and _any_present(ks, EVENT_T_KEYS)):
                            ev = obj
                            break
                # deep walk if still not found
                if ev is None:
                    def _visit(name, obj):
                        nonlocal ev
                        if ev is not None:
                            return
                        if isinstance(obj, h5py.Group):
                            ks = obj.keys()
                            if (_any_present(ks, EVENT_X_KEYS)
                                and _any_present(ks, EVENT_Y_KEYS)
                                and _any_present(ks, EVENT_T_KEYS)):
                                ev = obj
                        elif isinstance(obj, h5py.Dataset):
                            # accept packed Nx>=3 "events" dataset somewhere in the tree
                            if (name.endswith("/events") or name == "events") and obj.ndim == 2 and obj.shape[1] >= 3:
                                ev = obj
                    h5f.visititems(_visit)

            if ev is None:
                raise ValueError("No events found in HDF5 file (looked for '/events', '/columns', or any group with x,y,t/timestamp/timestamps).")

            # ---- normalize fields (x,y,t[,p]) ----
            ds, n_total = self._fields_handles(ev)

            # normalize time key → ds["t"] (accept aliases)
            if ds.get("t") is None:
                for alt in ("timestamp", "timestamps", "time", "times"):
                    if ds.get(alt) is not None:
                        ds["t"] = ds[alt]
                        break
            if ds.get("t") is None:
                raise ValueError("Timestamps not found in events (expected one of t/timestamp/timestamps/time).")

            # ensure coords exist (both are required downstream)
            if ds.get("x") is None or ds.get("y") is None:
                raise ValueError("Events missing x/y coordinate fields (checked aliases).")

            # ---- Detect scale (ticks/sec) and convert offset ----
            if time_scale is None:
                scale = self._detect_time_scale(h5f, ev, ds["t"])
            else:
                scale = float(time_scale)
            self.ticks_per_second = float(scale)
            unit_str = ('ns' if scale == 1e9 else
                        'us' if scale == 1e6 else
                        'ms' if scale == 1e3 else
                        's'  if scale == 1.0 else f'{scale:g} ticks/s')
            print(f"Time unit detected: {unit_str}")

            offset_ticks = int(round((offset_ns / 1e9) * scale)) if offset_ns else None

            # ---- Get time range (ticks) via tiny reads ----
            t0_raw, tN_raw = self._read_first_last_t(ds["t"], n_total)
            start_tick = max(t0_raw, offset_ticks) if offset_ticks is not None else t0_raw
            if int(tN_raw) <= int(start_tick):
                print("Warning: No events after offset; nothing to do.")
                return

            # ---- Index bounds via HDF5-safe binary search ----
            start_idx = self._searchsorted_h5(ds["t"], int(start_tick), side="left",  n_total=n_total)
            end_idx   = self._searchsorted_h5(ds["t"], int(tN_raw),    side="right", n_total=n_total)
            end_idx   = min(end_idx, n_total)

            # =========================
            #   EVENTCOUNT MODE BRANCH
            # =========================
            if self.accumulator_type == "eventcount":
                # Coerce to a real int scalar no matter what came in
                def _to_int_scalar(v, default=50_000):
                    if v is None:
                        return int(default)
                    if isinstance(v, (int, np.integer)):
                        return int(v)
                    a = np.asarray(v)
                    if a.shape == ():
                        return int(a.item())
                    if a.size == 1:
                        return int(a.reshape(()).item())
                    raise ValueError(f"max_events_per_frame must be scalar; got shape {a.shape}")

                max_ev = max_events

                print(f"Total events (file): {n_total:,}")
                print("Processing by max events/frame (no fixed time window).")

                # Integer ceil without numpy/arrays
                num_events = max(1, int(end_idx - start_idx))
                denom      = max(1, int(max_ev))
                est_frames = (num_events + denom - 1) // denom
                print(f"Estimated frames (rough): {est_frames:,}")

                self._create_metadata_file(
                    output_dir=output_dir,
                    timewindow_ns=0,
                    start_time=int(start_tick),
                    total_frames=est_frames,
                    width=self.width, height=self.height
                )

                frames_written = self._process_frames_by_count(
                    ds=ds,
                    output_dir=output_dir,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    chunk_size=int(chunk_size)
                )

                self._append_metadata(
                    output_dir,
                    mode="eventcount",
                    max_events_per_frame=int(denom),
                    actual_frames=int(frames_written)
                )
                self._print_processing_summary()
                return


            # =========================
            #   TIME-WINDOW MODE
            # =========================
            window_ticks = int(round((timewindow_ns / 1e9) * scale))
            if window_ticks <= 0:
                raise ValueError("Effective time window is zero; increase timewindow_ns or fix countmatch inputs.")

            total_frames = int((int(tN_raw) - int(start_tick)) // window_ticks)
            if max_frames is not None:
                total_frames = min(total_frames, int(max_frames))

            duration_sec = (int(tN_raw) - int(start_tick)) / float(scale)
            print(f"Total events (file): {n_total:,}")
            print(f"Effective window: {window_ticks/scale*1e3:.3f} ms")
            print(f"Effective duration: {duration_sec:.2f} s")
            print(f"Frames to generate: {total_frames:,}")

            # Metadata
            self._create_metadata_file(
                output_dir=output_dir,
                timewindow_ns=timewindow_ns,               # keep as requested for user visibility
                start_time=int(start_tick),                # ticks (file units)
                total_frames=total_frames,
                width=self.width, height=self.height
            )

            # Stream & build
            self._process_frames_streaming(
                ds=ds,
                output_dir=output_dir,
                start_tick=int(start_tick),
                window_ticks=window_ticks,
                total_frames=total_frames,
                start_idx=start_idx, end_idx=end_idx,
                chunk_size=int(chunk_size)
            )

            # Query-side countmatch stats (for future use)
            if countmatch:
                frame_paths = sorted(glob.glob(os.path.join(output_dir, "frame_*.npy")))
                n = len(frame_paths)
                total_sum = 0.0
                pos_sum = 0.0
                neg_sum = 0.0
                saw_polarity = False

                for fp in frame_paths:
                    arr = np.load(fp, mmap_mode="r")  # cheap on memory
                    if arr.ndim == 3 and arr.shape[-1] == 2:
                        saw_polarity = True
                        ps = float(arr[..., 0].sum())
                        ns = float(arr[..., 1].sum())
                        pos_sum += ps
                        neg_sum += ns
                        total_sum += (ps + ns)
                    else:
                        total_sum += float(arr.sum())

                if n > 0:
                    updates = {
                        "countmatch": True,
                        "avg_events_per_frame": float(total_sum / n),
                        "frames_counted_for_avg": int(n),
                    }
                    if saw_polarity:
                        updates["avg_pos_events_per_frame"] = float(pos_sum / n)
                        updates["avg_neg_events_per_frame"] = float(neg_sum / n)
                    self._append_metadata(output_dir, **updates)
                else:
                    self._append_metadata(output_dir, countmatch=True,
                                        avg_events_per_frame=0.0,
                                        frames_counted_for_avg=0)

            self._print_processing_summary()


    def _process_frames_streaming(self, ds, output_dir, start_tick, window_ticks,
                                  total_frames, start_idx, end_idx, chunk_size):
        """Single-pass streaming with minimal resident memory."""
        # Only read fields required by the accumulator
        need_t = True
        need_x = True
        need_y = True
        need_p = (self.accumulator_type == "polarity")

        frame_shape = self.accumulator.get_frame_shape(self.width, self.height)
        current_frame_idx = 0
        current_frame = np.zeros(frame_shape, dtype=np.float32)

        def _flush(idx, frame_arr):
            np.save(os.path.join(output_dir, f"frame_{idx:06d}.npy"), frame_arr)
            if idx < 3:
                self._save_frame_preview(frame_arr, output_dir, idx)

        with tqdm(total=total_frames, desc="Generating frames") as pbar:
            for s in range(start_idx, end_idx, chunk_size):
                e = min(s + chunk_size, end_idx)

                x_chunk = ds["x"][s:e] if (need_x and ds["x"] is not None) else None
                y_chunk = ds["y"][s:e] if (need_y and ds["y"] is not None) else None
                t_chunk = ds["t"][s:e] if need_t else None
                p_chunk = ds["p"][s:e] if (need_p and ds["p"] is not None) else None

                # Narrow to our overall time window [start_tick, end_tick)
                # For efficiency, compute end_tick on the fly:
                end_tick_total = start_tick + total_frames * window_ticks
                in_range = (t_chunk >= start_tick) & (t_chunk < end_tick_total)
                if not np.any(in_range):
                    continue

                # Slice to in-range events
                t_chunk = t_chunk[in_range]
                x_chunk = x_chunk[in_range] if x_chunk is not None else None
                y_chunk = y_chunk[in_range] if y_chunk is not None else None
                p_chunk = p_chunk[in_range] if p_chunk is not None else None

                # Group by frame index within this chunk (non-decreasing by time)
                frame_idx_chunk = ((t_chunk - start_tick) // window_ticks).astype(np.int64)
                # Clamp (defensive)
                np.clip(frame_idx_chunk, 0, total_frames - 1, out=frame_idx_chunk)

                i = 0
                n = frame_idx_chunk.size
                while i < n:
                    fi = int(frame_idx_chunk[i])

                    # Flush any completed frames before fi
                    while current_frame_idx < fi and current_frame_idx < total_frames:
                        _flush(current_frame_idx, current_frame)
                        pbar.update(1)
                        current_frame_idx += 1
                        current_frame = np.zeros(frame_shape, dtype=np.float32)

                    # contiguous slice for this frame
                    j = i + 1
                    while j < n and frame_idx_chunk[j] == fi:
                        j += 1

                    # Accumulate into current frame
                    x_slice = x_chunk[i:j] if x_chunk is not None else np.empty((0,), dtype=np.uint16)
                    y_slice = y_chunk[i:j] if y_chunk is not None else np.empty((0,), dtype=np.uint16)
                    t_slice = t_chunk[i:j]
                    p_slice = p_chunk[i:j] if p_chunk is not None else np.empty((0,), dtype=np.bool_)

                    chunk_frame = self.accumulator.accumulate_events(
                        x_slice, y_slice, t_slice, p_slice,
                        frame_start_time=start_tick + fi * window_ticks,
                        frame_end_time=start_tick + (fi + 1) * window_ticks,
                    )
                    current_frame += chunk_frame
                    i = j

            # Flush remaining frames (including last)
            while current_frame_idx < total_frames:
                _flush(current_frame_idx, current_frame)
                pbar.update(1)
                current_frame_idx += 1
                current_frame = np.zeros(frame_shape, dtype=np.float32)
    
    def _accumulate_frame_streaming(self, events_group, frame_start_time, 
                                  frame_end_time, chunk_size, initial_frame):
        """Accumulate events for a single frame using streaming"""
        
        total_events = events_group['x'].shape[0]
        
        # Find approximate start and end indices using binary search
        timestamps = events_group['t']
        
        # Use binary search to find relevant time range
        start_idx = np.searchsorted(timestamps, frame_start_time)
        end_idx = np.searchsorted(timestamps, frame_end_time)
        
        # Add some buffer to account for chunking
        start_idx = max(0, start_idx - chunk_size)
        end_idx = min(total_events, end_idx + chunk_size)
        
        # Process in chunks
        current_frame = initial_frame.copy()
        
        for chunk_start in range(start_idx, end_idx, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_idx)
            
            # Load chunk
            x_chunk = events_group['x'][chunk_start:chunk_end]
            y_chunk = events_group['y'][chunk_start:chunk_end]
            t_chunk = events_group['t'][chunk_start:chunk_end]
            p_chunk = events_group['p'][chunk_start:chunk_end]
            
            # Accumulate events in this chunk
            chunk_frame = self.accumulator.accumulate_events(
                x_chunk, y_chunk, t_chunk, p_chunk, 
                frame_start_time, frame_end_time
            )
            
            # Add to current frame
            current_frame += chunk_frame
        
        return current_frame

    def _create_metadata_file(self, output_dir, timewindow_ns, start_time, total_frames, width, height):
        metadata = {
            'timewindow_ns': int(timewindow_ns),
            'timewindow_ms': float(timewindow_ns / 1e6),
            'start_tick': int(start_time),              # was start_time_ns
            'tick_scale': 'ticks per second',          # describe the next field
            'ticks_per_second': getattr(self, 'ticks_per_second', None),  # optional if you store it
            'total_frames': int(total_frames),
            'width': int(width),
            'height': int(height),
            'accumulator_type': self.accumulator_type,
            'frame_shape': self.accumulator.get_frame_shape(width, height),
            'dtype': 'float32',
            'hot_pixels_count': len(self.hot_pixels) if self.hot_pixels is not None else 0,
            'hot_pixels_enabled': self.hot_pixels is not None
        }
        
        # Add hot pixel coordinates to metadata if available and valid
        if self.hot_pixels is not None:
            try:
                if hasattr(self.hot_pixels, 'tolist'):
                    # It's a numpy array
                    metadata['hot_pixels'] = self.hot_pixels.tolist()
                elif isinstance(self.hot_pixels, (list, tuple)):
                    # It's already a list/tuple
                    metadata['hot_pixels'] = list(self.hot_pixels)
                else:
                    # Unknown type, don't save coordinates
                    print(f"Warning: hot_pixels has unexpected type {type(self.hot_pixels)}, not saving coordinates")
                    metadata['hot_pixels'] = []
            except Exception as e:
                print(f"Warning: Could not save hot pixels to metadata: {e}")
                metadata['hot_pixels'] = []
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved: {metadata_path}")
        if self.hot_pixels is not None:
            print(f"  Hot pixel filtering: {len(self.hot_pixels)} pixels excluded")
    
    def _save_frame_preview(self, frame, output_dir, frame_idx):
        """Save a preview image of the frame"""
        try:
            plt.figure(figsize=(8, 6))

            if frame.ndim == 2:
                # Single channel (counts) — any range is fine with a cmap,
                # but set vmin/vmax so the colorbar is meaningful.
                vmax = float(np.percentile(frame, 99)) if frame.size else 1.0
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = float(frame.max()) if frame.size else 1.0
                plt.imshow(frame, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
                plt.colorbar(label='Event Count')

            elif frame.ndim == 3 and frame.shape[2] == 2:
                # Polarity channels: normalize to [0,1] so imshow doesn't clip RGB
                pos = frame[:, :, 0].astype(np.float32, copy=False)
                neg = frame[:, :, 1].astype(np.float32, copy=False)

                flat = np.concatenate([pos.ravel(), neg.ravel()]) if frame.size else np.array([1.0], dtype=np.float32)
                vmax = float(np.percentile(flat, 99))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0

                R = np.clip(pos / vmax, 0.0, 1.0)
                G = np.clip(neg / vmax, 0.0, 1.0)
                B = G  # cyan for negative

                combined = np.dstack([R, G, B])  # float in [0,1] -> no clipping warning
                plt.imshow(combined)
                plt.title('Red: Positive, Green/Blue: Negative', fontsize=9)

            else:
                # Fallback: show nothing
                plt.imshow(np.zeros((self.height, self.width), dtype=np.float32), cmap='gray')

            # Frame title (kept last so it overrides any inner title)
            plt.title(f'Frame {frame_idx}')
            plt.axis('off')

            preview_path = os.path.join(output_dir, f'preview_frame_{frame_idx:03d}.png')
            plt.savefig(preview_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

        except ImportError:
            pass  # Skip preview if matplotlib not available
    
    def _print_processing_summary(self):
        """Print summary of frame processing including out-of-bounds and hot pixel statistics"""
        print(f"\n=== Frame Processing Summary ===")
        
        if hasattr(self.accumulator, 'total_out_of_bounds') and self.accumulator.total_out_of_bounds > 0:
            print(f"Total out-of-bounds events filtered: {self.accumulator.total_out_of_bounds:,}")
            print(f"Camera dimensions used: {self.width} x {self.height}")
            print(f"This suggests the camera dimensions might need adjustment.")
        
        if hasattr(self.accumulator, 'total_hot_pixels_filtered') and self.accumulator.total_hot_pixels_filtered > 0:
            print(f"Total hot pixel events filtered: {self.accumulator.total_hot_pixels_filtered:,}")
            print(f"Hot pixels filtered: {len(self.hot_pixels) if self.hot_pixels is not None else 0}")
        
        if (not hasattr(self.accumulator, 'total_out_of_bounds') or self.accumulator.total_out_of_bounds == 0) and \
           (not hasattr(self.accumulator, 'total_hot_pixels_filtered') or self.accumulator.total_hot_pixels_filtered == 0):
            print(f"✓ Frame processing completed successfully with no events filtered.")

class E2VIDReconstructor:
    """
    Reconstructs frames from event data using the E2VID model,
    following a similar interface to GeneralizedFrameBuilder.
    """

    def __init__(self, width, height, path_to_e2vid_model, path_to_e2vid_repo, window_type, hot_pixels=None, accumulator_type=None, max_events=None):
        """
        Initializes the E2VIDReconstructor.

        Args:
            width (int): The width of the event sensor.
            height (int): The height of the event sensor.
            path_to_e2vid_model (str): The path to the pre-trained E2VID model.
            path_to_e2vid_repo (str): The path to the cloned rpg_e2vid repository.
        """
        self.width = width
        self.height = height
        self.path_to_e2vid_model = path_to_e2vid_model
        self.path_to_e2vid_repo = path_to_e2vid_repo
        self.run_reconstruction_script = os.path.join(self.path_to_e2vid_repo, 'run_reconstruction.py')
        self.hot_pixels_file = hot_pixels
        self.window_type = window_type.lower()
        self.max_events = max_events
        self.accumulator_type = accumulator_type  # Not used in E2VID but kept for interface consistency
        if not os.path.isfile(self.run_reconstruction_script):
            raise FileNotFoundError(f"E2VID reconstruction script not found at: {self.run_reconstruction_script}")
        if not os.path.isfile(self.path_to_e2vid_model):
            raise FileNotFoundError(f"E2VID model not found at: {self.path_to_e2vid_model}")

    def build_frames(self, sequence_name, hdf5_path, output_dir, timewindow_ms, offset_ns=None, max_frames=None):
        print(f"Time window: {timewindow_ms:.1f} ms")
        """
        Builds frames from an HDF5 event data file using the E2VID model.

        Args:
            hdf5_path (str): Path to the HDF5 file with events.
            output_dir (str): Directory to save the frames.
            timewindow_ns (int): Time window in nanoseconds for each frame.
            offset_ns (int, optional): Start time offset in nanoseconds. Defaults to None.
            max_frames (int, optional): Maximum number of frames to generate. Defaults to None.
        """
        print(f"Building frames with E2VID: {hdf5_path} -> {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        # get parent dir to hdf5_path
        hdf5_parent_dir = os.path.dirname(hdf5_path)
        event_file_path_txt = os.path.join(hdf5_parent_dir, f'{sequence_name}_e2vid_events.txt')

        start_time_ns = self._prepare_event_data(hdf5_path, event_file_path_txt, offset_ns)

        if start_time_ns is None:
            start_time_ns = 0

        self._run_e2vid_reconstruction(event_file_path_txt, output_dir, timewindow_ms)

        self._process_and_save_frames(output_dir, output_dir, max_frames)

        self._create_metadata_file(output_dir, timewindow_ms, start_time_ns, max_frames)

        print("\n=== E2VID Frame Processing Summary ===")
        print(f"✓ Frame processing completed successfully.")

    def _prepare_event_data(self, hdf5_path, output_txt_path, offset_ns):
        """
        Extracts events from an HDF5 file and saves them to a text file
        in the format required by E2VID: first line "W H", then lines "t[s] x y p".
        Timestamps in HDF5 may be ns/us/ms/seconds. We detect units from the
        median positive time delta and output seconds starting at 0.0.
        """
        # ---- helpers ------------------------------------------------------------
        def _pick_key(grp, keys):
            for k in keys:
                if k in grp and isinstance(grp[k], h5py.Dataset):
                    return k
            return None

        def _any_present(ks, keys):
            return any(k in ks for k in keys)

        EVENT_T_KEYS = ("t", "timestamp", "timestamps", "time", "times")
        EVENT_X_KEYS = ("x", "x_coordinate", "x_coordinates", "u", "col", "column")
        EVENT_Y_KEYS = ("y", "y_coordinate", "y_coordinates", "v", "row")
        EVENT_P_KEYS = ("p", "polarity", "polarities", "pol", "polarity_bit", "polarity_bits")

        def _median_dt_from_dataset(t_ds, sample=200_000) -> float:
            """Return median positive tick difference from a 1-D HDF5 dataset."""
            n = int(t_ds.shape[0])
            if n <= 2:
                return 0.0
            step = max(1, n // sample)
            t = np.array(t_ds[::step])
            dt = np.diff(t)
            dt = dt[dt > 0]
            return float(np.median(dt)) if dt.size else 0.0

        def _seconds_per_tick_from_dt(dt_med: float) -> float:
            """
            Decide seconds-per-tick from typical integer Δt magnitude.
            - dt < 1        : already seconds (integer seconds)
            - 1   .. 1e3    : microseconds per tick  -> 1e-6
            - 1e3 .. 1e6    : nanoseconds per tick   -> 1e-9
            - 1e6 .. 1e9    : milliseconds per tick  -> 1e-3  (coarse clocks)
            else            : default 1.0
            """
            if dt_med <= 0:
                return 1.0
            if dt_med < 1.0:
                return 1.0
            if dt_med < 1e3:
                return 1e-6
            if dt_med < 1e6:
                return 1e-9
            if dt_med < 1e9:
                return 1e-3
            return 1.0

        # ---- event dataset/group discovery -------------------------------------
        print("Preparing event data for E2VID...")
        self.hdf5_path = hdf5_path

        def _locate_events(h5f):
            """Return either an events Group (with columns) or a Dataset (Nx>=3)."""
            events = None
            # obvious names first
            if 'events' in h5f:
                events = h5f['events']
            elif 'columns' in h5f:
                events = h5f['columns']
            elif 'data' in h5f:
                events = h5f['data']

            # shallow search if still not found
            if events is None:
                for _, obj in h5f.items():
                    if isinstance(obj, h5py.Group):
                        ks = obj.keys()
                        if (_any_present(ks, EVENT_X_KEYS)
                            and _any_present(ks, EVENT_Y_KEYS)
                            and _any_present(ks, EVENT_T_KEYS)):
                            events = obj
                            break

            # deep search
            if events is None:
                def _visit(name, obj):
                    nonlocal events
                    if events is not None:
                        return
                    if isinstance(obj, h5py.Group):
                        ks = obj.keys()
                        if (_any_present(ks, EVENT_X_KEYS)
                            and _any_present(ks, EVENT_Y_KEYS)
                            and _any_present(ks, EVENT_T_KEYS)):
                            events = obj
                    elif isinstance(obj, h5py.Dataset):
                        if (name.endswith("/events") or name == "events") and obj.ndim == 2 and obj.shape[1] >= 3:
                            events = obj
                h5f.visititems(_visit)

            if events is None:
                raise ValueError("Could not locate an events dataset/group in the HDF5.")
            return events

        # === If a text already exists, compute and return its zero-point in ns ===
        if os.path.exists(output_txt_path):
            print(f"Event text file already exists: {output_txt_path}")
            with h5py.File(hdf5_path, 'r') as h5f:
                events = _locate_events(h5f)

                if isinstance(events, h5py.Group):
                    t_key = _pick_key(events, EVENT_T_KEYS)
                    if not t_key:
                        raise ValueError("Events group has no time dataset.")
                    t_ds = events[t_key]
                    # unit detection
                    if np.issubdtype(t_ds.dtype, np.floating):
                        s_per_tick = 1.0
                    else:
                        s_per_tick = _seconds_per_tick_from_dt(_median_dt_from_dataset(t_ds))
                    t0_raw = float(t_ds[0])
                else:
                    # compound Nx>=3 dataset; assume column 2 is time if unnamed
                    if events.dtype.names:
                        names = [n.lower() for n in events.dtype.names]
                        try:
                            idx_t = next(i for i,nm in enumerate(names) if nm in EVENT_T_KEYS)
                        except StopIteration:
                            idx_t = 2
                    else:
                        idx_t = 2
                    n = int(events.shape[0])
                    if n == 0:
                        raise ValueError("No events in dataset.")
                    t0_raw = float(events[0, idx_t])
                    if np.issubdtype(events.dtype[idx_t], np.floating):
                        s_per_tick = 1.0
                    else:
                        step = max(1, n // 200_000)
                        dt_med = np.median(np.diff(np.asarray(events[::step, idx_t])))
                        s_per_tick = _seconds_per_tick_from_dt(float(dt_med))

                start_time_s = t0_raw * s_per_tick
                if offset_ns:
                    start_time_s = max(start_time_s, float(offset_ns) * 1e-9)
                start_time_ns = int(round(start_time_s * 1e9))
            return start_time_ns

        # === Otherwise, export a new text file ==================================
        with h5py.File(hdf5_path, 'r') as h5f:
            events = _locate_events(h5f)

            # --- Group of column datasets ---
            if isinstance(events, h5py.Group):
                t_key = _pick_key(events, EVENT_T_KEYS)
                x_key = _pick_key(events, EVENT_X_KEYS)
                y_key = _pick_key(events, EVENT_Y_KEYS)
                p_key = _pick_key(events, EVENT_P_KEYS)  # optional

                if not (t_key and x_key and y_key):
                    raise ValueError(f"Missing datasets in events group. "
                                    f"Found: t={t_key}, x={x_key}, y={y_key}, p={p_key}")

                t_ds = events[t_key]; x_ds = events[x_key]; y_ds = events[y_key]
                p_ds = events[p_key] if p_key else None

                n = int(t_ds.shape[0])
                if n == 0:
                    raise ValueError("No events in datasets.")

                # unit detection
                if np.issubdtype(t_ds.dtype, np.floating):
                    s_per_tick = 1.0
                else:
                    s_per_tick = _seconds_per_tick_from_dt(_median_dt_from_dataset(t_ds))
                t0_raw = float(t_ds[0])
                start_time_s = t0_raw * s_per_tick

                if offset_ns:
                    start_time_s = max(start_time_s, float(offset_ns) * 1e-9)

                with open(output_txt_path, 'w') as f:
                    f.write(f"{self.width} {self.height}\n")
                    CHUNK = 2_000_000
                    for i0 in tqdm(range(0, n, CHUNK), desc="Exporting events"):
                        i1 = min(n, i0 + CHUNK)
                        t = np.asarray(t_ds[i0:i1], dtype=np.float64) * s_per_tick
                        x = np.asarray(x_ds[i0:i1], dtype=np.int32)
                        y = np.asarray(y_ds[i0:i1], dtype=np.int32)
                        if p_ds is not None:
                            p = np.asarray(p_ds[i0:i1])
                            p = (p.astype(np.int8) > 0).astype(np.uint8) if p.dtype != np.bool_ else p.astype(np.uint8)
                        else:
                            p = np.zeros_like(x, dtype=np.uint8)

                        t -= start_time_s
                        for ti, xi, yi, pi in zip(t, x, y, p):
                            f.write(f"{ti:.9f} {int(xi)} {int(yi)} {int(pi)}\n")

                start_time_ns = int(round(start_time_s * 1e9))
                return start_time_ns

            # --- Compound Nx>=3 dataset (e.g., columns packed into one 'events') ---
            else:
                n = int(events.shape[0])
                if n == 0:
                    raise ValueError("No events in dataset.")

                # column indices (prefer names)
                if events.dtype.names:
                    names = [n.lower() for n in events.dtype.names]
                    idx_t = next((i for i,nm in enumerate(names) if nm in EVENT_T_KEYS), 2)
                    idx_x = next((i for i,nm in enumerate(names) if nm in EVENT_X_KEYS), 0)
                    idx_y = next((i for i,nm in enumerate(names) if nm in EVENT_Y_KEYS), 1)
                    idx_p = next((i for i,nm in enumerate(names) if nm in EVENT_P_KEYS), None)
                else:
                    idx_x, idx_y, idx_t = 0, 1, 2
                    idx_p = 3 if events.shape[1] > 3 else None

                # unit detection
                if np.issubdtype(events.dtype[idx_t], np.floating):
                    s_per_tick = 1.0
                else:
                    step = max(1, n // 200_000)
                    dt_med = np.median(np.diff(np.asarray(events[::step, idx_t])))
                    s_per_tick = _seconds_per_tick_from_dt(float(dt_med))

                t0_raw = float(events[0, idx_t])
                start_time_s = t0_raw * s_per_tick
                if offset_ns:
                    start_time_s = max(start_time_s, float(offset_ns) * 1e-9)

                with open(output_txt_path, 'w') as f:
                    f.write(f"{self.width} {self.height}\n")
                    CHUNK = 2_000_000
                    for i0 in tqdm(range(0, n, CHUNK), desc="Exporting events"):
                        i1 = min(n, i0 + CHUNK)
                        chunk = events[i0:i1]
                        t = np.asarray(chunk[:, idx_t], dtype=np.float64) * s_per_tick
                        x = np.asarray(chunk[:, idx_x], dtype=np.int32)
                        y = np.asarray(chunk[:, idx_y], dtype=np.int32)
                        if idx_p is not None:
                            p = np.asarray(chunk[:, idx_p])
                            p = (p.astype(np.int8) > 0).astype(np.uint8) if p.dtype != np.bool_ else p.astype(np.uint8)
                        else:
                            p = np.zeros_like(x, dtype=np.uint8)

                        t -= start_time_s
                        for ti, xi, yi, pi in zip(t, x, y, p):
                            f.write(f"{ti:.9f} {int(xi)} {int(yi)} {int(pi)}\n")

                start_time_ns = int(round(start_time_s * 1e9))
                return start_time_ns

    def _run_e2vid_reconstruction(self, event_file_path, output_dir, timewindow_ms):
        """
        Executes the E2VID reconstruction script.
        """
        if self.accumulator_type == "eventcount":
            # load the reference metadata to get the average events per frame
            command = (
                f'python '
                f'{self.run_reconstruction_script} '
                f'--i {event_file_path} '
                f'--path_to_model {self.path_to_e2vid_model} '
                f'--output_folder {output_dir} '
                f'--window_size {self.max_events} '
                f'--auto_hdr '
                f'--color '
                f'--hot_pixels_file {self.hot_pixels_file} '
            )
        elif self.window_type == "timewindow":
            command = (
                f'python '
                f'{self.run_reconstruction_script} '
                f'--i {event_file_path} '
                f'--path_to_model {self.path_to_e2vid_model} '
                f'--output_folder {output_dir} '
                f'--window_duration {timewindow_ms} '
                f'--fixed_duration '
                f'--auto_hdr '
                f'--color '
                f'--hot_pixels_file {self.hot_pixels_file}'
            )
        else:
            command = (
                f'python '
                f'{self.run_reconstruction_script} '
                f'--i {event_file_path} '
                f'--path_to_model {self.path_to_e2vid_model} '
                f'--output_folder {output_dir} '
                f'--num_events_per_pixel {timewindow_ms} '
                f'--auto_hdr '
                f'--color '
                f'--hot_pixels_file {self.hot_pixels_file}'
            )
        construct_cmd = ["pixi", "run", "bash", "-c", command]
        print("Running E2VID reconstruction...")
        subprocess.run(construct_cmd, text=True)
        self.output_dir = output_dir

    def _process_and_save_frames(self, e2vid_output_dir, final_output_dir, max_frames):
        """
        Converts the reconstructed images to .npy files.
        """
        print("Processing and saving reconstructed frames...")
        reconstruction_dir = os.path.join(e2vid_output_dir, 'reconstruction')
        image_files = sorted([f for f in os.listdir(reconstruction_dir) if f.endswith('.png')])

        if max_frames is not None:
            image_files = image_files[:max_frames]

        for i, img_name in enumerate(tqdm(image_files, desc="Saving frames")):
            img_path = os.path.join(reconstruction_dir, img_name)
            # E2VID output is typically grayscale, so we read it as such
            try:
                frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                frame = frame.astype(np.float32) / 255.0 # Normalize to [0, 1]
            except ImportError:
                frame = np.array(Image.open(img_path).convert('L')).astype(np.float32) / 255.0


            frame_filename = f"frame_{i:06d}.npy"
            frame_path = os.path.join(final_output_dir, frame_filename)
            np.save(frame_path, frame)

    def _create_metadata_file(self, output_dir, timewindow_ns, start_time_ns, total_frames):
        """
        Creates a metadata file for the reconstructed frames.
        """
        num_frames = total_frames
        if num_frames is None:
            num_frames = len([f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.npy')])

        metadata = {
            'timewindow_ns': int(timewindow_ns),
            'timewindow_ms': float(timewindow_ns / 1e6),
            'start_time_ns': int(start_time_ns),
            'total_frames': int(num_frames),
            'width': int(self.width),
            'height': int(self.height),
            'reconstructor': 'E2VID',
            'frame_shape': (self.height, self.width),
            'dtype': 'float32'
        }

        # save to common folder 
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Metadata saved to {metadata_path}")

def build_event_frames(hdf5_path, config, data_config, sequence_name, timewindow,
                      width, height, frames_dir, window_type, accumulator_type, hot_pixels=None, reference=False, max_events=None):
    """
    Main function to build event frames
    
    Args:
        hdf5_path: Path to HDF5 events file
        config: General configuration dict
        data_config: Dataset configuration dict  
        sequence_name: Name of sequence
        width, height: Camera dimensions
        frames_dir: Output directory for frames
        hot_pixels: Array of (x, y) coordinates to exclude (None = auto-load from file)
    """
    
    print(f"Debug: build_event_frames called with hot_pixels type = {type(hot_pixels)}")
    if hot_pixels is not None:
        print(f"Debug: hot_pixels length = {len(hot_pixels) if hasattr(hot_pixels, '__len__') else 'no len'}")
    
    print(f"Debug: Final hot_pixels type = {type(hot_pixels)}")
    
    # Get parameters from config
    timewindow_ms = timewindow
    timewindow_ns = int(timewindow_ms * 1e6)
    def _parse_timestamp_val_to_scale(data_config):
        """
        Returns ticks-per-second from data_config['format']['data']['timestamp_val'].
        Accepts: 'ns'|'us'|'ms'|'s' or a positive number (ticks/s).
        """
        val = (data_config.get('format', {})
                        .get('data', {})
                        .get('timestamp_val', None))
        if val is None:
            return None  # fall back to builder detection
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ('ns', 'nanosecond', 'nanoseconds'): return 1e9
            if v in ('us', 'μs', 'microsecond', 'microseconds'): return 1e6
            if v in ('ms', 'millisecond', 'milliseconds'): return 1e3
            if v in ('s', 'sec', 'second', 'seconds'): return 1.0
            # numeric string
            try:
                num = float(v)
                if num > 0: return num
            except Exception:
                pass
            raise ValueError(f"Unsupported timestamp_val='{val}'. Use ns/us/ms/s or numeric ticks/s.")
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
        raise ValueError(f"Invalid timestamp_val={val!r} (expected ns/us/ms/s or positive number)")

    
    # Get offset for this sequence
    offset_sec = None
    if ('other' in data_config and 'offset' in data_config['other'] 
        and sequence_name in data_config['other']['offset']):
        offset_sec = data_config['other']['offset'][sequence_name]
        offset_ns = int(offset_sec * 1e9)
    else:
        offset_ns = None
    
    # Get accumulator type from config
    if config['frame_generator'] == 'frames':
        accumulator_type = config.get('frame_accumulator', 'count')
        
        print(f"Building frames for sequence: {sequence_name}")
        if offset_sec:
            print(f"Using offset: {offset_sec} seconds")
        if hot_pixels is not None:
            print(f"Hot pixel filtering: {len(hot_pixels)} pixels will be excluded")

        scale_override = _parse_timestamp_val_to_scale(data_config)
        # Create frame builder with hot pixels
        if config['frame_accumulator'] == 'eventcount':
            builder = GeneralizedFrameBuilder(width, height, accumulator_type, max_events_per_frame=max_events, hot_pixels=hot_pixels)
        else:
            builder = GeneralizedFrameBuilder(width, height, accumulator_type, hot_pixels)
        
        # Build frames
        builder.build_frames(
            hdf5_path=hdf5_path,
            output_dir=frames_dir,
            timewindow_ns=timewindow_ns,
            offset_ns=offset_ns,
            chunk_size=config.get('chunk_size', 100000),
            max_frames=config.get('max_frames', None),
            time_scale=scale_override,          # <-- force the unit if provided
            max_events=max_events
        )
    elif config['frame_generator'] == 'frames' and config['frame_generator'] == 'eventcount':
        accumulator_type = config.get('frame_accumulator', 'count')
        
        print(f"Building frames for sequence: {sequence_name}")
        if offset_sec:
            print(f"Using offset: {offset_sec} seconds")
        if hot_pixels is not None:
            print(f"Hot pixel filtering: {len(hot_pixels)} pixels will be excluded")

        scale_override = _parse_timestamp_val_to_scale(data_config)
        # Create frame builder with hot pixels
        builder = GeneralizedFrameBuilder(width, height, accumulator_type, max_events_per_frame=config.get('max_events_per_frame', 25000), hot_pixels=hot_pixels)

        # Build frames
        builder.build_frames(
            hdf5_path=hdf5_path,
            output_dir=frames_dir,
            timewindow_ns=timewindow_ns,
            offset_ns=offset_ns,
            chunk_size=config.get('chunk_size', 100000),
            max_events=max_events,
            time_scale=scale_override,          # <-- force the unit if provided
        )
    else: # running the reconstructor
        # ensure the e2vid repo exists
        if not os.path.exists('./datasets/rpg_e2vid'):
            e2vid_url = "https://github.com/uzh-rpg/rpg_e2vid.git"
            print(f"Cloning e2vid repository from {e2vid_url}...")
            subprocess.run(['git', 'clone', e2vid_url, './datasets/rpg_e2vid'], check=True)

            # Get the pre-trained model
            if config.get('reconstruction_model', 'e2vid') == 'firenet':
                model_url = "https://drive.usercontent.google.com/u/0/uc?id=1Uqj8z8pDnq78JzoXdw-6radw3RPAyUPb&export=download"
            else:
                model_url = "https://drive.usercontent.google.com/u/0/uc?id=1q0rnm8OUIHk-II39qpxhp0tqBfIOK-7M&export=download"
            model_path = f"./datasets/rpg_e2vid/model/E2VID_{config['reconstruction_model']}.pth.tar"
            print(f"Downloading pre-trained model from {model_url}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            requests.get(model_url, allow_redirects=True)
            with open(model_path, 'wb') as f:
                f.write(requests.get(model_url).content)


        builder = E2VIDReconstructor(
            width=width,
            height=height,
            path_to_e2vid_model=f"./datasets/rpg_e2vid/model/E2VID_{config['reconstruction_model']}.pth.tar",
            path_to_e2vid_repo="./datasets/rpg_e2vid",
            window_type=window_type,
            hot_pixels=hot_pixels,
            accumulator_type=accumulator_type,
            max_events=max_events
        )

        builder.build_frames(
            sequence_name,
            hdf5_path=hdf5_path,
            output_dir=frames_dir,
            timewindow_ms=timewindow_ms,   # <-- ms, not mislabeled as ns
            offset_ns=offset_ns,
            max_frames=config.get('max_frames', None)
        )
