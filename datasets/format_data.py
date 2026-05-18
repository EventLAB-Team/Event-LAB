#!/usr/bin/env python3
"""Dataset formatting and event-frame generation utilities."""

import glob
import json
import os
import re
import subprocess
import time
import torch
from abc import ABC, abstractmethod

import h5py
import numpy as np
import requests
from tqdm import tqdm
from typing import Optional, Tuple
from pathlib import Path

EVENT_T_KEYS = ("t", "timestamp", "timestamps", "time", "times")
EVENT_X_KEYS = ("x", "x_coordinate", "x_coordinates", "u", "col", "column")
EVENT_Y_KEYS = ("y", "y_coordinate", "y_coordinates", "v", "row")
EVENT_P_KEYS = ("p", "polarity", "polarities", "pol", "polarity_bit", "polarity_bits")
FRAME_NPY_RE = re.compile(r"^frame_(\d+)\.npy$")

def _find_first_dataset(g: h5py.Group, candidates: tuple[str, ...], logical_name: str):
    for key in candidates:
        if key in g and isinstance(g[key], h5py.Dataset):
            return g[key]
    for _, obj in g.items():
        if isinstance(obj, h5py.Group):
            try:
                return _find_first_dataset(obj, candidates, logical_name)
            except RuntimeError:
                pass
    raise RuntimeError(
        f"Could not find {logical_name} dataset under group '{g.name}'. "
        f"Tried names: {', '.join(candidates)}"
    )

def find_event_datasets(f: h5py.File):
    g = f["events"] if ("events" in f and isinstance(f["events"], h5py.Group)) else f
    x_ds = _find_first_dataset(g, EVENT_X_KEYS, "x")
    y_ds = _find_first_dataset(g, EVENT_Y_KEYS, "y")
    t_ds = _find_first_dataset(g, EVENT_T_KEYS, "t")
    p_ds = _find_first_dataset(g, EVENT_P_KEYS, "p")
    return x_ds, y_ds, t_ds, p_ds

def sec_to_raw(t_sec: float, time_scale: float) -> int:
    return int(round(float(t_sec) / float(time_scale)))

def raw_to_sec(t_raw: int, time_scale: float) -> float:
    return float(t_raw) * float(time_scale)

def stream_event_windows_raw(
    hdf5_path: Path,
    dt_ms: float,
    chunk_size: int,
    time_scale: float,
    start_time_sec: Optional[float],
    skip: Optional[int] = None
):
    """
    Stream raw event windows from disk.

    Note: this borrowed low-level helper expects ``time_scale`` to be seconds
    per raw timestamp tick. GeneralizedFrameBuilder accepts either this value
    or the repo's usual ticks-per-second value and normalizes it before calling
    into this function.
    """
    with h5py.File(hdf5_path, "r") as f:
        x_dset, y_dset, t_dset, p_dset = find_event_datasets(f)
        N = len(t_dset)

        if N == 0:
            return

        dt_raw = int(round((dt_ms / 1000.0) / float(time_scale)))

        if dt_raw <= 0:
            raise ValueError(f"dt_ms too small for time_scale (dt_raw={dt_raw})")

        t0_raw = int(t_dset[0])
        tN_raw = int(t_dset[N - 1])

        if start_time_sec is None:
            w_start_raw = t0_raw
        else:
            w_start_raw = max(sec_to_raw(start_time_sec, time_scale), t0_raw)

        if w_start_raw >= tN_raw:
            print(f"Warning: stream start is beyond event file end (start_raw={w_start_raw}, file_end_raw={tN_raw})")
            return

        x_buf = np.empty(0, dtype=np.int64)
        y_buf = np.empty(0, dtype=np.int64)
        t_buf = np.empty(0, dtype=np.int64)
        p_buf = np.empty(0, dtype=np.int8)

        read_idx = 0
        frame_idx = 0
        t_buf_max = -1

        while w_start_raw < tN_raw:
            w_end_raw = w_start_raw + dt_raw
            t_read0 = time.perf_counter()

            accum_x, accum_y, accum_t, accum_p = [], [], [], []

            if t_buf.size > 0:
                accum_x.append(x_buf)
                accum_y.append(y_buf)
                accum_t.append(t_buf)
                accum_p.append(p_buf)
                t_buf_max = int(t_buf[-1])

            while read_idx < N and (t_buf_max < w_end_raw):
                end_idx = min(N, read_idx + chunk_size)
                t_chunk = t_dset[read_idx:end_idx].astype(np.int64, copy=False)

                if t_chunk.size > 0:
                    x_chunk = x_dset[read_idx:end_idx].astype(np.int64, copy=False)
                    y_chunk = y_dset[read_idx:end_idx].astype(np.int64, copy=False)
                    p_chunk = p_dset[read_idx:end_idx].astype(np.int8, copy=False)

                    accum_x.append(x_chunk)
                    accum_y.append(y_chunk)
                    accum_t.append(t_chunk)
                    accum_p.append(p_chunk)
                    t_buf_max = int(t_chunk[-1])

                read_idx = end_idx

            if accum_t:
                x_buf = np.concatenate(accum_x)
                y_buf = np.concatenate(accum_y)
                t_buf = np.concatenate(accum_t)
                p_buf = np.concatenate(accum_p)
            else:
                x_buf = np.empty(0, dtype=np.int64)
                y_buf = np.empty(0, dtype=np.int64)
                t_buf = np.empty(0, dtype=np.int64)
                p_buf = np.empty(0, dtype=np.int8)

            if t_buf.size:
                in_win = (t_buf >= w_start_raw) & (t_buf < w_end_raw)
                x_win = x_buf[in_win]
                y_win = y_buf[in_win]
                t_win_raw = t_buf[in_win]
                p_win = p_buf[in_win]

                leftover = t_buf >= w_end_raw
                x_buf, y_buf, t_buf, p_buf = x_buf[leftover], y_buf[leftover], t_buf[leftover], p_buf[leftover]
                t_buf_max = int(t_buf[-1]) if t_buf.size > 0 else -1
            else:
                x_win = np.empty(0, dtype=np.int64)
                y_win = np.empty(0, dtype=np.int64)
                t_win_raw = np.empty(0, dtype=np.int64)
                p_win = np.empty(0, dtype=np.int8)

            t_read1 = time.perf_counter()
            t_read_ms = (t_read1 - t_read0) * 1000.0

            if skip is None or skip <= 1 or frame_idx % skip == 0:
                yield (
                    raw_to_sec(w_start_raw, time_scale),
                    raw_to_sec(w_end_raw, time_scale),
                    w_end_raw,
                    x_win, y_win, t_win_raw, p_win,
                    frame_idx,
                    t_read_ms,
                )

            w_start_raw = w_end_raw
            frame_idx += 1

            if read_idx >= N and t_buf.size == 0 and t_buf_max < w_start_raw:
                break

@torch.no_grad()
def gpu_polarity_frame_1ch(
    x, y, p,
    H: int, W: int, device: torch.device
) -> Tuple[torch.Tensor, int]:

    # KEY FIX: keep p as-is, then promote safely before >0
    if p.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        p_cmp = p.to(torch.int16)   # prevents uint8 255 -> int8 -1 wrap
    else:
        p_cmp = p

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not torch.any(valid):
        return torch.zeros((2, H, W), device=device, dtype=torch.float32), 0

    x = x[valid]; y = y[valid]; p_cmp = p_cmp[valid]
    n_valid = int(x.numel())

    flat = H * W
    lin = y * W + x
    ch = (p_cmp > 0).to(torch.int64)

    idx = lin + ch * flat

    counts = torch.bincount(idx, minlength=2 * flat).to(torch.float32)
    frame = counts.view(2, flat).view(2, H, W)

    # flatten 1st and 2nd channelsums into single channel (polarity-agnostic)
    frame = frame.sum(dim=0, keepdim=True)

    return frame, n_valid

@torch.no_grad()
def gpu_polarity_frame_2ch(
    x, y, p,
    H: int, W: int, device: torch.device
) -> Tuple[torch.Tensor, int]:

    # KEY FIX: keep p as-is, then promote safely before >0
    if p.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        p_cmp = p.to(torch.int16)   # prevents uint8 255 -> int8 -1 wrap
    else:
        p_cmp = p

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not torch.any(valid):
        return torch.zeros((2, H, W), device=device, dtype=torch.float32), 0

    x = x[valid]; y = y[valid]; p_cmp = p_cmp[valid]
    n_valid = int(x.numel())

    flat = H * W
    lin = y * W + x
    ch = (p_cmp > 0).to(torch.int64)

    idx = lin + ch * flat

    counts = torch.bincount(idx, minlength=2 * flat).to(torch.float32)
    frame = counts.view(2, flat).view(2, H, W)

    return frame, n_valid

class _ColumnView:
    """Lazy 1-D view over a numeric NxC event dataset."""

    def __init__(self, base, col):
        self._base = base
        self._col = col

    def __getitem__(self, idx):
        return self._base[idx, self._col]

    @property
    def shape(self):
        return (self._base.shape[0],)

    @property
    def dtype(self):
        return self._base.dtype


class _FieldView:
    """Lazy 1-D view over a field in a compound event dataset."""

    def __init__(self, base, field):
        self._base = base
        self._field = field

    def __getitem__(self, idx):
        return self._base[idx][self._field]

    @property
    def shape(self):
        return (self._base.shape[0],)

    @property
    def dtype(self):
        return self._base.dtype[self._field]


def parse_timestamp_val_to_scale(data_config):
    """
    Returns ticks-per-second from data_config['format']['data']['timestamp_val'].
    Accepts: 'ns'|'us'|'ms'|'s' or a positive number (ticks/s).
    """
    val = (data_config.get('format', {})
                    .get('data', {})
                    .get('timestamp_val', None))
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ('ns', 'nanosecond', 'nanoseconds'):
            return 1e9
        if v in ('us', 'μs', 'microsecond', 'microseconds'):
            return 1e6
        if v in ('ms', 'millisecond', 'milliseconds'):
            return 1e3
        if v in ('s', 'sec', 'second', 'seconds'):
            return 1.0
        try:
            num = float(v)
            if num > 0:
                return num
        except Exception:
            pass
        raise ValueError(f"Unsupported timestamp_val='{val}'. Use ns/us/ms/s or numeric ticks/s.")
    if isinstance(val, (int, float, np.integer, np.floating)) and val > 0:
        return float(val)
    raise ValueError(f"Invalid timestamp_val={val!r} (expected ns/us/ms/s or positive number)")


def _pick_dataset_key(group, keys):
    for key in keys:
        if key in group and isinstance(group[key], h5py.Dataset):
            return key
    return None


def _has_any_key(keys, aliases):
    return any(key in keys for key in aliases)


def _locate_events_node(h5f, include_data=False):
    names = ('events', 'columns') + (('data',) if include_data else ())
    for name in names:
        if name in h5f:
            return h5f[name]

    for _, obj in h5f.items():
        if isinstance(obj, h5py.Group):
            keys = obj.keys()
            if (_has_any_key(keys, EVENT_X_KEYS)
                and _has_any_key(keys, EVENT_Y_KEYS)
                and _has_any_key(keys, EVENT_T_KEYS)):
                return obj

    event_node = None

    def _visit(name, obj):
        nonlocal event_node
        if event_node is not None:
            return
        if isinstance(obj, h5py.Group):
            keys = obj.keys()
            if (_has_any_key(keys, EVENT_X_KEYS)
                and _has_any_key(keys, EVENT_Y_KEYS)
                and _has_any_key(keys, EVENT_T_KEYS)):
                event_node = obj
        elif isinstance(obj, h5py.Dataset):
            is_event_name = name.endswith("/events") or name == "events"
            is_event_shape = obj.dtype.names or (obj.ndim == 2 and obj.shape[1] >= 3)
            if is_event_name and is_event_shape:
                event_node = obj

    h5f.visititems(_visit)
    if event_node is None:
        searched = "'/events', '/columns'" + (", '/data'" if include_data else "")
        raise ValueError(f"No events found in HDF5 file (looked for {searched}, or any group with x/y/t aliases).")
    return event_node


def _event_field_handles(events):
    """
    Normalize an events container to lazy field handles:
    ds['t'], ds['x'], ds['y'], ds['p'] where p may be None.
    """
    if isinstance(events, h5py.Group):
        t_key = _pick_dataset_key(events, EVENT_T_KEYS)
        x_key = _pick_dataset_key(events, EVENT_X_KEYS)
        y_key = _pick_dataset_key(events, EVENT_Y_KEYS)
        p_key = _pick_dataset_key(events, EVENT_P_KEYS)

        if not (t_key or x_key or y_key):
            raise ValueError("events group has no recognizable t/x/y datasets (checked aliases).")

        probe = events[t_key] if t_key else (events[x_key] if x_key else events[y_key])
        return {
            "t": events[t_key] if t_key else None,
            "x": events[x_key] if x_key else None,
            "y": events[y_key] if y_key else None,
            "p": events[p_key] if p_key else None,
        }, int(probe.shape[0])

    if isinstance(events, h5py.Dataset):
        n_total = int(events.shape[0])

        if events.dtype.names:
            names_lut = {name.lower(): name for name in events.dtype.names}

            def _field_name(aliases):
                for alias in aliases:
                    if alias in names_lut:
                        return names_lut[alias]
                return None

            t_field = _field_name(EVENT_T_KEYS)
            x_field = _field_name(EVENT_X_KEYS)
            y_field = _field_name(EVENT_Y_KEYS)
            p_field = _field_name(EVENT_P_KEYS)

            if not (t_field or x_field or y_field):
                raise ValueError("compound events dataset missing time/coords fields (checked aliases).")

            return {
                "t": _FieldView(events, t_field) if t_field else None,
                "x": _FieldView(events, x_field) if x_field else None,
                "y": _FieldView(events, y_field) if y_field else None,
                "p": _FieldView(events, p_field) if p_field else None,
            }, n_total

        if events.ndim == 2 and events.shape[1] >= 3:
            return {
                "x": _ColumnView(events, 0),
                "y": _ColumnView(events, 1),
                "t": _ColumnView(events, 2),
                "p": _ColumnView(events, 3) if events.shape[1] > 3 else None,
            }, n_total

        raise ValueError("Unsupported events dataset shape; expected compound or numeric Nx>=3.")

    raise TypeError(f"Unsupported events node type: {type(events)}")


def _linear_hot_pixels(hot_pixels, width, height):
    if hot_pixels is None:
        return None

    pixels = np.asarray(hot_pixels, dtype=np.int64)
    if pixels.size:
        pixels = pixels.reshape(-1, 2)
        in_bounds = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
            (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        )
        pixels = pixels[in_bounds]
        hot_lin = pixels[:, 1] * width + pixels[:, 0]
    else:
        hot_lin = np.empty((0,), dtype=np.int64)

    print(f"Hot pixel filtering enabled: {len(hot_lin)} pixels will be excluded")
    return hot_lin


def _valid_coord_mask(x, y, width, height):
    return (x >= 0) & (x < width) & (y >= 0) & (y < height)


def _record_out_of_bounds(owner, x, y, invalid_mask):
    count = int(np.count_nonzero(invalid_mask))
    if count == 0:
        return

    owner.total_out_of_bounds += count
    if owner.warned_already:
        return

    invalid_x = x[invalid_mask]
    invalid_y = y[invalid_mask]
    print("Warning: Found out-of-bounds events. Example ranges:")
    print(f"  X range: {invalid_x.min()} - {invalid_x.max()} (valid: 0 - {owner.width-1})")
    print(f"  Y range: {invalid_y.min()} - {invalid_y.max()} (valid: 0 - {owner.height-1})")
    print("  Will continue counting but suppress further warnings...")
    owner.warned_already = True


def _hot_pixel_keep_mask(owner, x, y, hot_lin):
    if hot_lin is None or not x.size:
        return np.ones(x.shape, dtype=bool)

    lin = y.astype(np.int64) * owner.width + x.astype(np.int64)
    hot_mask = np.isin(lin, hot_lin, assume_unique=False)
    owner.total_hot_pixels_filtered += int(hot_mask.sum())
    return ~hot_mask


def _sample_times(t_source, sample=200_000):
    n = int(t_source.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    step = max(1, n // sample)
    return np.asarray(t_source[::step], dtype=np.float64)


def _median_dt_from_times(t_source, sample=200_000):
    if int(t_source.shape[0]) <= 2:
        return 0.0
    ticks = _sample_times(t_source, sample)
    dt = np.diff(ticks)
    dt = dt[dt > 0]
    return float(np.median(dt)) if dt.size else 0.0


def _infer_seconds_per_tick(t_source, sample=200_000):
    """
    Infer seconds-per-tick from timestamp magnitude.
    Explicit dataset config is still preferred where available.
    """
    if int(t_source.shape[0]) < 3:
        return 1.0

    ticks = _sample_times(t_source, sample)
    dt = np.diff(ticks)
    dt_pos = dt[dt > 0]
    if dt_pos.size == 0:
        return 1.0

    dt_med = float(np.median(dt_pos))
    span = float(ticks[-1] - ticks[0])

    if dt_med < 1e-3:
        return 1.0
    if span > 1e11:
        return 1e-9
    if span > 1e8:
        return 1e-6
    if span > 1e5:
        return 1e-3
    if 1.0 <= dt_med <= 50.0:
        return 1e-3
    if 50.0 < dt_med <= 50_000.0:
        return 1e-6
    if dt_med > 50_000.0:
        return 1e-9
    return 1.0


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

class H5FrameWriter:
    """
    Chunked compressed event-frame writer.

    Stores all frames in one HDF5 file rather than thousands of frame_*.npy files.
    """

    def __init__(
        self,
        path,
        frame_shape,
        dtype=np.uint16,
        chunk_frames=64,
        compression="gzip",
        compression_opts=1,
        metadata=None,
    ):
        import os
        import h5py
        import numpy as np

        self.path = path
        self.frame_shape = tuple(frame_shape)
        self.dtype = np.dtype(dtype)
        self.chunk_frames = int(chunk_frames)
        self.buffer = []
        self.start_tick_buffer = []
        self.end_tick_buffer = []
        self.event_count_buffer = []
        self.n_written = 0

        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.h5 = h5py.File(path, "w")

        self.frames = self.h5.create_dataset(
            "frames",
            shape=(0, *self.frame_shape),
            maxshape=(None, *self.frame_shape),
            chunks=(self.chunk_frames, *self.frame_shape),
            dtype=self.dtype,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
        )

        self.frame_start_ticks = self.h5.create_dataset(
            "frame_start_ticks",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_frames,),
            dtype=np.int64,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
        )

        self.frame_end_ticks = self.h5.create_dataset(
            "frame_end_ticks",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_frames,),
            dtype=np.int64,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
        )

        self.event_counts = self.h5.create_dataset(
            "event_counts",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_frames,),
            dtype=np.int64,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
        )

        if metadata:
            for k, v in metadata.items():
                try:
                    self.h5.attrs[k] = v
                except TypeError:
                    self.h5.attrs[k] = str(v)

    def _coerce_frame(self, frame):
        import numpy as np

        frame = np.asarray(frame)

        if frame.shape != self.frame_shape:
            raise ValueError(
                f"Expected frame shape {self.frame_shape}, got {frame.shape}"
            )

        if np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            frame = np.clip(frame, info.min, info.max)

        return frame.astype(self.dtype, copy=False)

    def append(self, frame, frame_start_tick=-1, frame_end_tick=-1, event_count=-1):
        self.buffer.append(self._coerce_frame(frame))
        self.start_tick_buffer.append(int(frame_start_tick))
        self.end_tick_buffer.append(int(frame_end_tick))
        self.event_count_buffer.append(int(event_count))

        if len(self.buffer) >= self.chunk_frames:
            self.flush()

    def flush(self):
        import numpy as np

        if not self.buffer:
            return

        batch = np.stack(self.buffer, axis=0)
        B = batch.shape[0]

        old_n = self.n_written
        new_n = old_n + B

        self.frames.resize((new_n, *self.frame_shape))
        self.frame_start_ticks.resize((new_n,))
        self.frame_end_ticks.resize((new_n,))
        self.event_counts.resize((new_n,))

        self.frames[old_n:new_n] = batch
        self.frame_start_ticks[old_n:new_n] = np.asarray(
            self.start_tick_buffer, dtype=np.int64
        )
        self.frame_end_ticks[old_n:new_n] = np.asarray(
            self.end_tick_buffer, dtype=np.int64
        )
        self.event_counts[old_n:new_n] = np.asarray(
            self.event_count_buffer, dtype=np.int64
        )

        self.n_written = new_n

        self.buffer.clear()
        self.start_tick_buffer.clear()
        self.end_tick_buffer.clear()
        self.event_count_buffer.clear()

    def close(self):
        self.flush()
        self.h5.attrs["total_frames"] = int(self.n_written)
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

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

        self._hot_lin = _linear_hot_pixels(self.hot_pixels, self.width, self.height)

    def _mask_valid(self, x, y):
        return _valid_coord_mask(x, y, self.width, self.height)

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
            _record_out_of_bounds(self, xs, ys, ~valid)
        xs = xs[valid]; ys = ys[valid]
        if ts is not None: ts = ts[valid]
        if ps is not None: ps = ps[valid]

        # Hot-pixel removal
        keep = _hot_pixel_keep_mask(self, xs, ys, self._hot_lin)
        if not np.all(keep):
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

        self._hot_lin = _linear_hot_pixels(self.hot_pixels, self.width, self.height)

    def _mask_valid(self, x, y):
        return _valid_coord_mask(x, y, self.width, self.height)

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
            _record_out_of_bounds(self, x, y, ~valid)
        x = x[valid]; y = y[valid]

        # Hot-pixel removal
        keep = _hot_pixel_keep_mask(self, x, y, self._hot_lin)
        if not np.all(keep):
            x, y = x[keep], y[keep]
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

        self.use_gpu = True
        self._torch = None
        self._device = None
        self._warned_torch_fallback = False
        try:
            import torch

            self._torch = torch
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        except Exception:
            self._torch = None
            self._device = None
        backend = self._device.type if self._device is not None else "numpy"
        print(f"PolarityFrameAccumulator initialized. torch support: {'yes' if self._torch is not None else 'no'}, device: {backend}")
        self._hot_lin = _linear_hot_pixels(self.hot_pixels, self.width, self.height)

    def _filter_hot_pixels(self, x, y):
        """Return boolean mask of non-hot events (CPU path)."""
        return _hot_pixel_keep_mask(self, x, y, self._hot_lin)

    def accumulate_events(self, x, y, t, p, frame_start_time, frame_end_time):
        """Accumulate positive and negative events separately."""
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

        valid_coords = _valid_coord_mask(x_filtered, y_filtered, self.width, self.height)
        if not np.any(valid_coords):
            return np.zeros((self.height, self.width, 2), dtype=np.float32)

        x_valid = x_filtered[valid_coords]
        y_valid = y_filtered[valid_coords]
        p_valid = p_filtered[valid_coords]

        # Track OOB once
        oob = ~valid_coords
        if np.any(oob):
            _record_out_of_bounds(self, x_filtered, y_filtered, oob)

        if self.use_gpu and self._torch is not None and self._device is not None:
            torch = self._torch
            try:
                with torch.no_grad():
                    x_t = torch.as_tensor(x_valid, device=self._device, dtype=torch.int64)
                    y_t = torch.as_tensor(y_valid, device=self._device, dtype=torch.int64)
                    p_t = torch.as_tensor(p_valid, device=self._device, dtype=torch.bool)

                    lin = y_t * self.width + x_t

                    frame = torch.zeros((2, self.height * self.width), dtype=torch.float32, device=self._device)

                    pos_lin = lin[p_t]
                    if pos_lin.numel() > 0:
                        frame[0].index_add_(0, pos_lin, torch.ones_like(pos_lin, dtype=torch.float32))
                    neg_mask = ~p_t
                    neg_lin = lin[neg_mask]
                    if neg_lin.numel() > 0:
                        frame[1].index_add_(0, neg_lin, torch.ones_like(neg_lin, dtype=torch.float32))

                    return frame.view(2, self.height, self.width).permute(1, 2, 0).cpu().numpy()
            except Exception as exc:
                if not self._warned_torch_fallback:
                    print(f"Warning: torch accumulation failed on device '{self._device}'; falling back to NumPy. Error: {exc}")
                    self._warned_torch_fallback = True

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
    @staticmethod
    def _parse_timestamp_val_to_scale(data_config):
        return parse_timestamp_val_to_scale(data_config)

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
            if tm < target_tick or (side == "right" and tm == target_tick):
                lo = mid + 1
            else:
                hi = mid
        return lo if side == "right" else lo

    def _fields_handles(self, ev):
        return _event_field_handles(ev)

    
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
        Stream events and write event-count frames to one chunked HDF5 file.

        Output:
            output_dir/frames.h5

        Also writes:
            output_dir/event_frame_times_ticks.npy

        for compatibility with any existing downstream code that expects mid ticks.
        """

        max_events = int(getattr(self.accumulator, "max_events", self.max_events))
        frame_shape = self.accumulator.get_frame_shape(self.width, self.height)

        if len(frame_shape) == 3:
            H, W, C = frame_shape
        else:
            H, W = frame_shape
            C = 1

        has_p = ds.get("p") is not None
        hot_lin = getattr(self.accumulator, "_hot_lin", None)
        flat_size = self.width * self.height

        frame_store_path = os.path.join(output_dir, "frames.h5")

        writer = H5FrameWriter(
            frame_store_path,
            frame_shape=frame_shape,
            dtype=np.uint16,
            chunk_frames=64,
            compression="gzip",
            compression_opts=1,
            metadata={
                "width": int(self.width),
                "height": int(self.height),
                "accumulator_type": self.accumulator_type,
                "mode": "eventcount",
                "max_events_per_frame": int(max_events),
                "ticks_per_second": float(getattr(self, "ticks_per_second", -1)),
                "frame_layout": "HWC" if len(frame_shape) == 3 else "HW",
            },
        )

        def _flush(frame_idx, frame_arr, t0_tick, t1_tick):
            writer.append(
                frame_arr,
                frame_start_tick=t0_tick,
                frame_end_tick=t1_tick,
                event_count=int(np.sum(frame_arr)),
            )

            if getattr(self, "save_previews", False) and frame_idx < 3:
                self._save_frame_preview(frame_arr, output_dir, frame_idx)

            if t0_tick is None or t1_tick is None:
                return None

            try:
                if np.isnan(t0_tick) or np.isnan(t1_tick):
                    return None
            except TypeError:
                pass

            return int((int(t0_tick) + int(t1_tick)) // 2)

        idx = int(start_idx)
        frames_written = 0
        current_frame = np.zeros(frame_shape, dtype=np.float32)
        used_in_frame = 0

        cur_t0_tick = None
        cur_t1_tick = None
        mid_ticks = []

        try:
            with tqdm(desc=f"Generating frames (≤{max_events} ev/frame)") as pbar:
                while idx < end_idx:
                    chunk_end = min(idx + chunk_size, end_idx)

                    x_chunk = ds["x"][idx:chunk_end]
                    y_chunk = ds["y"][idx:chunk_end]
                    t_chunk = ds["t"][idx:chunk_end]

                    p_chunk = None
                    if C > 1:
                        if not has_p:
                            raise RuntimeError(
                                "Accumulator expects multi-channel frames, but dataset has no 'p' field."
                            )
                        p_chunk = ds["p"][idx:chunk_end]

                    valid = _valid_coord_mask(x_chunk, y_chunk, self.width, self.height)

                    if not np.all(valid):
                        _record_out_of_bounds(self.accumulator, x_chunk, y_chunk, ~valid)

                    if not np.any(valid):
                        idx = chunk_end
                        continue

                    xs = x_chunk[valid]
                    ys = y_chunk[valid]
                    ts = t_chunk[valid]
                    ps = p_chunk[valid] if p_chunk is not None else None

                    lin = ys.astype(np.int64) * self.width + xs.astype(np.int64)
                    keep_mask = _hot_pixel_keep_mask(self.accumulator, xs, ys, hot_lin)

                    if not np.any(keep_mask):
                        idx = chunk_end
                        continue

                    valid_pos = np.nonzero(valid)[0]
                    keep_pos_within_valid = np.nonzero(keep_mask)[0]
                    kept_raw_pos = valid_pos[keep_pos_within_valid]

                    need = max_events - used_in_frame
                    take_n = int(min(need, kept_raw_pos.size))

                    if take_n > 0:
                        use_lin = lin[keep_mask][:take_n]
                        use_ts = ts[keep_mask][:take_n]

                        if C > 1:
                            use_ps_raw = ps[keep_mask][:take_n]
                            use_pos = use_ps_raw > 0

                            if np.any(use_pos):
                                pos_counts = np.bincount(
                                    use_lin[use_pos],
                                    minlength=flat_size,
                                ).astype(np.float32, copy=False)
                                current_frame[..., 0] += pos_counts.reshape(H, W)

                            if np.any(~use_pos):
                                neg_counts = np.bincount(
                                    use_lin[~use_pos],
                                    minlength=flat_size,
                                ).astype(np.float32, copy=False)
                                current_frame[..., 1] += neg_counts.reshape(H, W)
                        else:
                            counts = np.bincount(
                                use_lin,
                                minlength=flat_size,
                            ).astype(np.float32, copy=False)
                            current_frame += counts.reshape(H, W)

                        used_in_frame += take_n

                        first_tick = int(use_ts[0])
                        last_tick = int(use_ts[-1])

                        if cur_t0_tick is None:
                            cur_t0_tick = first_tick
                        cur_t1_tick = last_tick

                        last_used_pos = int(kept_raw_pos[take_n - 1])
                        idx = idx + last_used_pos + 1

                    if used_in_frame >= max_events:
                        mid = _flush(frames_written, current_frame, cur_t0_tick, cur_t1_tick)
                        if mid is not None:
                            mid_ticks.append(mid)

                        frames_written += 1
                        pbar.update(1)

                        current_frame.fill(0.0)
                        used_in_frame = 0
                        cur_t0_tick = None
                        cur_t1_tick = None

                    if idx < chunk_end and used_in_frame < max_events:
                        continue

                    idx = max(idx, chunk_end)

                if used_in_frame > 0:
                    mid = _flush(frames_written, current_frame, cur_t0_tick, cur_t1_tick)
                    if mid is not None:
                        mid_ticks.append(mid)

                    frames_written += 1
                    pbar.update(1)

        finally:
            writer.close()

        if mid_ticks:
            np.save(
                os.path.join(output_dir, "event_frame_times_ticks.npy"),
                np.asarray(mid_ticks, dtype=np.int64),
            )

        return frames_written

    @staticmethod
    def _coerce_seconds_per_tick(time_scale):
        """
        Normalize timestamp scale inputs for stream_event_windows_raw.

        Existing Event-LAB config parsing returns ticks per second (for example
        1e6 for microseconds). The borrowed raw streamer expects seconds per
        tick (for example 1e-6). Accept both for convenience.
        """
        if time_scale is None:
            return None

        scale = float(time_scale)
        if scale <= 0:
            raise ValueError(f"time_scale must be positive, got {time_scale!r}")

        if scale >= 1.0:
            return 1.0 / scale
        return scale

    def _detect_seconds_per_tick(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as h5f:
            ev = _locate_events_node(h5f)
            ds, _ = self._fields_handles(ev)
            tds = ds.get("t")
            if tds is None:
                raise ValueError("Timestamps not found in events (expected one of t/timestamp/timestamps/time).")

            ticks_per_second = self._detect_time_scale(h5f, ev, tds)
            if ticks_per_second <= 0:
                raise ValueError(f"Detected invalid timestamp scale: {ticks_per_second!r}")
            return 1.0 / float(ticks_per_second)

    def _resolve_stream_seconds_per_tick(self, hdf5_path, time_scale):
        seconds_per_tick = self._coerce_seconds_per_tick(time_scale)
        if seconds_per_tick is None:
            seconds_per_tick = self._detect_seconds_per_tick(hdf5_path)

        self.ticks_per_second = 1.0 / float(seconds_per_tick)
        return float(seconds_per_tick)

    def _fast_polarity_frame_from_events(self, x, y, p):
        """
        Fast polarity accumulation using the copied torch/bincount polarity builder.

        Returns existing GeneralizedFrameBuilder polarity layout:
            [H, W, 2]

        Channel convention preserved:
            frame[..., 0] = positive events
            frame[..., 1] = negative events
        """
        import torch

        if p is None or len(p) != len(x):
            raise ValueError(
                "Polarity frame accumulation requires one polarity value per event."
            )

        frame_shape = self.accumulator.get_frame_shape(self.width, self.height)
        empty = np.zeros(frame_shape, dtype=np.float32)

        if len(x) == 0:
            return empty

        x = np.asarray(x)
        y = np.asarray(y)
        p = np.asarray(p)

        # Keep the same coordinate filtering / accounting as the existing code.
        valid = _valid_coord_mask(x, y, self.width, self.height)
        if not np.all(valid):
            _record_out_of_bounds(self.accumulator, x, y, ~valid)

        if not np.any(valid):
            return empty

        x = x[valid]
        y = y[valid]
        p = p[valid]

        # Preserve hot-pixel behaviour.
        hot_lin = getattr(self.accumulator, "_hot_lin", None)
        keep_mask = _hot_pixel_keep_mask(self.accumulator, x, y, hot_lin)

        if not np.any(keep_mask):
            return empty

        x = x[keep_mask]
        y = y[keep_mask]
        p = p[keep_mask]

        # torch.bincount is the fast bit. Prefer CUDA if available.
        # Do not default to MPS unless you've tested bincount there.
        device = getattr(self, "_fast_polarity_device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._fast_polarity_device = device

        x_t = torch.as_tensor(x, device=device, dtype=torch.long)
        y_t = torch.as_tensor(y, device=device, dtype=torch.long)
        p_t = torch.as_tensor(p, device=device)

        frame_chw, _ = gpu_polarity_frame_2ch(
            x_t,
            y_t,
            p_t,
            H=self.height,
            W=self.width,
            device=device,
        )

        # IMPORTANT:
        # gpu_polarity_frame_2ch returns [2, H, W] with:
        #   channel 0 = negative/non-positive
        #   channel 1 = positive
        #
        # Existing GeneralizedFrameBuilder polarity frames use [H, W, 2] with:
        #   channel 0 = positive
        #   channel 1 = negative
        frame_chw = frame_chw[[1, 0], :, :]  # [pos, neg]
        frame_hwc = frame_chw.permute(1, 2, 0).contiguous()

        return frame_hwc.cpu().numpy().astype(np.float32, copy=False)

    def _frame_from_raw_window(self, x, y, t, p, frame_start_tick, frame_end_tick):
        if self.accumulator_type == "eventcount":
            frame, _, _, _ = self.accumulator.accumulate_by_count(
                x, y, t, p,
                start_idx=0,
            )
            return frame.astype(np.float32, copy=False)

        if self.accumulator_type == "polarity":
            return self._fast_polarity_frame_from_events(x, y, p)

        return self.accumulator.accumulate_events(
            x, y, t, p,
            frame_start_time=frame_start_tick,
            frame_end_time=frame_end_tick,
        )

    def stream_hdf5_frames(self, hdf5_path, timewindow_ns=None, dt_ms=None,
                           start_time_sec=None, offset_ns=None, chunk_size=100_000,
                           time_scale=None, skip=None, max_frames=None,
                           return_metadata=False):
        """
        Stream accumulated frames directly from an HDF5 event file.

        This is the direct-frame counterpart to build_frames(): it uses the
        fast raw rolling-window streamer and returns frames in memory instead
        of saving frame_*.npy files. ``time_scale`` may be either seconds per
        tick or ticks per second.
        """
        if dt_ms is None:
            if timewindow_ns is None:
                raise ValueError("Provide either dt_ms or timewindow_ns.")
            dt_ms = float(timewindow_ns) / 1e6
        else:
            dt_ms = float(dt_ms)

        if dt_ms <= 0:
            raise ValueError(f"dt_ms must be positive, got {dt_ms!r}")

        if start_time_sec is None and offset_ns is not None:
            start_time_sec = float(offset_ns) * 1e-9

        seconds_per_tick = self._resolve_stream_seconds_per_tick(hdf5_path, time_scale)
        window_ticks = int(round((dt_ms / 1000.0) / seconds_per_tick))
        if window_ticks <= 0:
            raise ValueError("Effective time window is zero; increase dt_ms/timewindow_ns or fix time_scale.")

        if max_frames is not None:
            max_frames = int(max_frames)
            if max_frames <= 0:
                return

        yielded = 0
        for window in stream_event_windows_raw(
            Path(hdf5_path),
            dt_ms=dt_ms,
            chunk_size=int(chunk_size),
            time_scale=seconds_per_tick,
            start_time_sec=start_time_sec,
            skip=skip,
        ):
            (
                frame_start_sec,
                frame_end_sec,
                frame_end_tick,
                x_win,
                y_win,
                t_win_raw,
                p_win,
                frame_idx,
                read_ms,
            ) = window

            frame_start_tick = int(frame_end_tick) - window_ticks
            frame = self._frame_from_raw_window(
                x_win, y_win, t_win_raw, p_win,
                frame_start_tick=frame_start_tick,
                frame_end_tick=int(frame_end_tick),
            )

            metadata = {
                "frame_idx": int(frame_idx),
                "frame_start_sec": float(frame_start_sec),
                "frame_end_sec": float(frame_end_sec),
                "frame_start_tick": int(frame_start_tick),
                "frame_end_tick": int(frame_end_tick),
                "event_count": int(len(x_win)),
                "read_ms": float(read_ms),
                "ticks_per_second": float(self.ticks_per_second),
            }

            yielded += 1
            yield (frame, metadata) if return_metadata else frame

            if max_frames is not None and yielded >= max_frames:
                break

    def get_streamed_frame(self, hdf5_path, timewindow_ns=None, dt_ms=None,
                           start_time_sec=None, offset_ns=None, frame_index=0,
                           chunk_size=100_000, time_scale=None,
                           return_metadata=False):
        """
        Return one frame directly from an HDF5 stream.

        ``frame_index`` is relative to ``start_time_sec``/``offset_ns``. Use
        frame_index=0 to fetch the first window at the requested start time.
        """
        frame_index = int(frame_index)
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative.")

        stream = self.stream_hdf5_frames(
            hdf5_path=hdf5_path,
            timewindow_ns=timewindow_ns,
            dt_ms=dt_ms,
            start_time_sec=start_time_sec,
            offset_ns=offset_ns,
            chunk_size=chunk_size,
            time_scale=time_scale,
            max_frames=frame_index + 1,
            return_metadata=True,
        )

        for yielded_idx, (frame, metadata) in enumerate(stream):
            if yielded_idx == frame_index:
                return (frame, metadata) if return_metadata else frame

        raise IndexError(f"No streamed frame available at frame_index={frame_index}.")


    # -------------------- public API --------------------
    def _append_countmatch_stats_from_h5(self, output_dir, batch_size=256):
        """
        Compute countmatch stats from output_dir/frames.h5.

        Replaces the old frame_*.npy scan.
        """

        frame_store_path = os.path.join(output_dir, "frames.h5")

        if not os.path.exists(frame_store_path):
            self._append_metadata(
                output_dir,
                countmatch=True,
                avg_events_per_frame=0.0,
                frames_counted_for_avg=0,
                countmatch_error=f"Missing frame store: {frame_store_path}",
            )
            return

        total_sum = 0.0
        pos_sum = 0.0
        neg_sum = 0.0
        saw_polarity = False

        with h5py.File(frame_store_path, "r") as f:
            frames = f["frames"]
            n = int(frames.shape[0])

            for start in range(0, n, int(batch_size)):
                end = min(start + int(batch_size), n)
                batch = frames[start:end]

                if batch.ndim == 4 and batch.shape[-1] == 2:
                    saw_polarity = True
                    ps = float(batch[..., 0].sum())
                    ns = float(batch[..., 1].sum())
                    pos_sum += ps
                    neg_sum += ns
                    total_sum += ps + ns
                else:
                    total_sum += float(batch.sum())

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
            self._append_metadata(
                output_dir,
                countmatch=True,
                avg_events_per_frame=0.0,
                frames_counted_for_avg=0,
            )

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

        with h5py.File(hdf5_path, "r", swmr=True,
                    rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, rdcc_w0=0.0) as h5f:

            ev = _locate_events_node(h5f)
            ds, n_total = self._fields_handles(ev)

            if ds.get("t") is None:
                raise ValueError("Timestamps not found in events (expected one of t/timestamp/timestamps/time).")
            if ds.get("x") is None or ds.get("y") is None:
                raise ValueError("Events missing x/y coordinate fields (checked aliases).")
            if self.accumulator_type == "polarity" and ds.get("p") is None:
                raise ValueError("Polarity frame accumulation requires a polarity field (checked p/polarity aliases).")

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
                max_ev = max_events if max_events is not None else self.accumulator.max_events

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

            # Query-side countmatch stats from the HDF5 frame store
            if countmatch:
                self._append_countmatch_stats_from_h5(output_dir)

            self._print_processing_summary()


    def _process_frames_streaming(self, ds, output_dir, start_tick, window_ticks,
                                  total_frames, start_idx, end_idx, chunk_size):
        """
        Single-pass fixed-time-window frame generation.

        Writes one chunked HDF5 file:
            output_dir/frames.h5

        instead of thousands of frame_XXXXXX.npy files.
        """

        need_t = True
        need_x = True
        need_y = True
        need_p = (self.accumulator_type == "polarity")

        frame_shape = self.accumulator.get_frame_shape(self.width, self.height)
        current_frame_idx = 0
        current_frame = np.zeros(frame_shape, dtype=np.float32)

        frame_store_path = os.path.join(output_dir, "frames.h5")

        writer = H5FrameWriter(
            frame_store_path,
            frame_shape=frame_shape,
            dtype=np.uint16,
            chunk_frames=64,
            compression="gzip",
            compression_opts=1,
            metadata={
                "width": int(self.width),
                "height": int(self.height),
                "accumulator_type": self.accumulator_type,
                "mode": "fixed_time_window",
                "start_tick": int(start_tick),
                "window_ticks": int(window_ticks),
                "total_frames_requested": int(total_frames),
                "ticks_per_second": float(getattr(self, "ticks_per_second", -1)),
                "frame_layout": "HWC" if len(frame_shape) == 3 else "HW",
            },
        )

        def _flush(frame_idx, frame_arr):
            frame_start_tick = int(start_tick + frame_idx * window_ticks)
            frame_end_tick = int(start_tick + (frame_idx + 1) * window_ticks)

            writer.append(
                frame_arr,
                frame_start_tick=frame_start_tick,
                frame_end_tick=frame_end_tick,
                event_count=int(np.sum(frame_arr)),
            )

            if getattr(self, "save_previews", False) and frame_idx < 3:
                self._save_frame_preview(frame_arr, output_dir, frame_idx)

        end_tick_total = int(start_tick + total_frames * window_ticks)

        try:
            with tqdm(total=total_frames, desc="Generating frames") as pbar:
                for s in range(start_idx, end_idx, chunk_size):
                    e = min(s + chunk_size, end_idx)

                    x_chunk = ds["x"][s:e] if (need_x and ds["x"] is not None) else None
                    y_chunk = ds["y"][s:e] if (need_y and ds["y"] is not None) else None
                    t_chunk = ds["t"][s:e] if need_t else None
                    p_chunk = ds["p"][s:e] if (need_p and ds["p"] is not None) else None

                    in_range = (t_chunk >= start_tick) & (t_chunk < end_tick_total)
                    if not np.any(in_range):
                        continue

                    t_chunk = t_chunk[in_range]
                    x_chunk = x_chunk[in_range] if x_chunk is not None else None
                    y_chunk = y_chunk[in_range] if y_chunk is not None else None
                    p_chunk = p_chunk[in_range] if p_chunk is not None else None

                    frame_idx_chunk = ((t_chunk - start_tick) // window_ticks).astype(np.int64)
                    np.clip(frame_idx_chunk, 0, total_frames - 1, out=frame_idx_chunk)

                    i = 0
                    n = frame_idx_chunk.size

                    while i < n:
                        fi = int(frame_idx_chunk[i])

                        while current_frame_idx < fi and current_frame_idx < total_frames:
                            _flush(current_frame_idx, current_frame)
                            pbar.update(1)
                            current_frame_idx += 1
                            current_frame = np.zeros(frame_shape, dtype=np.float32)

                        j = i + 1
                        while j < n and frame_idx_chunk[j] == fi:
                            j += 1

                        x_slice = x_chunk[i:j] if x_chunk is not None else np.empty((0,), dtype=np.uint16)
                        y_slice = y_chunk[i:j] if y_chunk is not None else np.empty((0,), dtype=np.uint16)
                        t_slice = t_chunk[i:j]
                        p_slice = p_chunk[i:j] if p_chunk is not None else np.empty((0,), dtype=np.bool_)

                        if self.accumulator_type == "polarity":
                            chunk_frame = self._fast_polarity_frame_from_events(
                                x_slice,
                                y_slice,
                                p_slice,
                            )
                        else:
                            chunk_frame = self.accumulator.accumulate_events(
                                x_slice,
                                y_slice,
                                t_slice,
                                p_slice,
                                frame_start_time=start_tick + fi * window_ticks,
                                frame_end_time=start_tick + (fi + 1) * window_ticks,
                            )

                        current_frame += chunk_frame
                        i = j

                while current_frame_idx < total_frames:
                    _flush(current_frame_idx, current_frame)
                    pbar.update(1)
                    current_frame_idx += 1
                    current_frame = np.zeros(frame_shape, dtype=np.float32)

        finally:
            writer.close()

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
            'dtype': 'uint16',
            'storage_file': 'frames.h5',
            'storage_dataset': 'frames',
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
            import matplotlib.pyplot as plt

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
        print("Preparing event data for E2VID...")
        self.hdf5_path = hdf5_path

        with h5py.File(hdf5_path, 'r') as h5f:
            events = _locate_events_node(h5f, include_data=True)
            ds, n = _event_field_handles(events)

            if ds.get("t") is None or ds.get("x") is None or ds.get("y") is None:
                raise ValueError("Events missing t/x/y fields required for E2VID export.")
            if n == 0:
                raise ValueError("No events in dataset.")

            t_ds = ds["t"]
            x_ds = ds["x"]
            y_ds = ds["y"]
            p_ds = ds["p"]

            s_per_tick = _infer_seconds_per_tick(t_ds)
            t0_raw = float(t_ds[0])
            start_time_s = t0_raw * s_per_tick
            if offset_ns:
                start_time_s = max(start_time_s, float(offset_ns) * 1e-9)

            start_time_ns = int(round(start_time_s * 1e9))

            if os.path.exists(output_txt_path):
                print(f"Event text file already exists: {output_txt_path}")
                dt_med = _median_dt_from_times(t_ds)
                t1_raw = float(t_ds[1]) if n > 1 else t0_raw
                print(f"[t] dtype={t_ds.dtype} t0={t0_raw} t1={t1_raw} "
                      f"dt_med={dt_med} inferred_s_per_tick={s_per_tick}")
                return start_time_ns

            with open(output_txt_path, 'w') as f:
                f.write(f"{self.width} {self.height}\n")
                chunk_size = 2_000_000
                for i0 in tqdm(range(0, n, chunk_size), desc="Exporting events"):
                    i1 = min(n, i0 + chunk_size)
                    t = np.asarray(t_ds[i0:i1], dtype=np.float64) * s_per_tick
                    x = np.asarray(x_ds[i0:i1], dtype=np.int32)
                    y = np.asarray(y_ds[i0:i1], dtype=np.int32)
                    if p_ds is None:
                        p = np.zeros_like(x, dtype=np.uint8)
                    else:
                        p = (np.asarray(p_ds[i0:i1]) > 0).astype(np.uint8)

                    t -= start_time_s
                    for ti, xi, yi, pi in zip(t, x, y, p):
                        f.write(f"{ti:.9f} {int(xi)} {int(yi)} {int(pi)}\n")

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
        construct_cmd = ["pixi", "run", "-e", "e2vid", "bash", "-c", command]
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
            try:
                import cv2

                frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    raise ValueError(f"Could not read reconstructed image: {img_path}")
                frame = frame.astype(np.float32) / 255.0
            except ImportError:
                from PIL import Image

                frame = np.array(Image.open(img_path).convert('L')).astype(np.float32) / 255.0

            frame_filename = f"frame_{i:06d}.npy"
            frame_path = os.path.join(final_output_dir, frame_filename)
            np.save(frame_path, frame)

    def _create_metadata_file(self, output_dir, timewindow_ms, start_time_ns, total_frames):
        """
        Creates a metadata file for the reconstructed frames.
        """
        num_frames = total_frames
        if num_frames is None:
            num_frames = len([f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.npy')])

        metadata = {
            'timewindow_ns': int(timewindow_ms * 1e6),
            'timewindow_ms': float(timewindow_ms),
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
    timewindow_ms = timewindow
    timewindow_ns = int(timewindow_ms * 1e6)

    offset_sec = None
    if ('other' in data_config and 'offset' in data_config['other'] 
        and sequence_name in data_config['other']['offset']):
        offset_sec = data_config['other']['offset'][sequence_name]
        offset_ns = int(offset_sec * 1e9)
    else:
        offset_ns = None
    
    frame_generator = config.get('frame_generator', 'frames')

    if frame_generator in ('frames', 'eventcount'):
        if frame_generator == 'eventcount':
            accumulator_type = 'eventcount'
        else:
            accumulator_type = config.get('frame_accumulator', accumulator_type or 'count')

        print(f"Building frames for sequence: {sequence_name}")
        if offset_sec is not None:
            print(f"Using offset: {offset_sec} seconds")
        if hot_pixels is not None:
            print(f"Hot pixel filtering: {len(hot_pixels)} pixels will be excluded")

        max_events_per_frame = max_events
        if accumulator_type == 'eventcount' and max_events_per_frame is None:
            max_events_per_frame = config.get('max_events_per_frame', None)

        builder = GeneralizedFrameBuilder(
            width=width,
            height=height,
            accumulator_type=accumulator_type,
            max_events_per_frame=max_events_per_frame,
            hot_pixels=hot_pixels,
        )

        builder.build_frames(
            hdf5_path=hdf5_path,
            output_dir=frames_dir,
            timewindow_ns=timewindow_ns,
            offset_ns=offset_ns,
            chunk_size=config.get('chunk_size', 100000),
            max_frames=config.get('max_frames', None),
            time_scale=parse_timestamp_val_to_scale(data_config),
            max_events=max_events_per_frame,
        )
    else:
        if not os.path.exists('./datasets/rpg_e2vid'):
            e2vid_url = "https://github.com/uzh-rpg/rpg_e2vid.git"
            print(f"Cloning e2vid repository from {e2vid_url}...")
            subprocess.run(['git', 'clone', e2vid_url, './datasets/rpg_e2vid'], check=True)

            if config.get('reconstruction_model', 'e2vid') == 'firenet':
                model_url = "https://drive.usercontent.google.com/u/0/uc?id=1Uqj8z8pDnq78JzoXdw-6radw3RPAyUPb&export=download"
            else:
                model_url = "https://drive.usercontent.google.com/u/0/uc?id=1q0rnm8OUIHk-II39qpxhp0tqBfIOK-7M&export=download"
            model_path = f"./datasets/rpg_e2vid/model/E2VID_{config['reconstruction_model']}.pth.tar"
            print(f"Downloading pre-trained model from {model_url}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            response = requests.get(model_url, allow_redirects=True)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                f.write(response.content)

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
            timewindow_ms=timewindow_ms,
            offset_ns=offset_ns,
            max_frames=config.get('max_frames', None)
        )
