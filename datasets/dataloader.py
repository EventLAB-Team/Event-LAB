import os, yaml, h5py, requests

import numpy as np
from datasets.format_data import format_sequence_data, build_event_frames
from datasets.download_data import download_sequence_data

class EventDataset():
    def __init__(self, config, dataset_name, sequence_name):
        super().__init__()
        self.config = config # General configuration
        self.dataset_name = dataset_name # Name of the dataset
        self.sequence_name = sequence_name # Name of the sequence

        # Load dataset configuration
        with open(f"./datasets/{self.dataset_name}.yaml", 'r') as file:
            self.data_config = yaml.safe_load(file)

        # Generate dataset name
        self.dataset_path = os.path.join(self.config['data_path'], # data path to the dataset
                                    self.dataset_name, # name of the dataset
                                    self.sequence_name, # the sequence name
                                    )
        # Create dataset directory
        os.makedirs(self.dataset_path, exist_ok=True)

        self.dataset_raw = os.path.join(self.dataset_path,f"{self.sequence_name}.{self.data_config['format']['data']['format']}")
        self.dataset_formatted = os.path.join(self.dataset_path,f"{self.sequence_name}.{self.config['std_format']}")

        # Get the dataset sequence names
        self.dataset_sequences = []
        self.reconstruction_types = []
        self.timewindow_list = []
        self.max_events_list = []
        if self.config['frame_accumulator'] == 'eventcount':
            for num_events in self.config['num_events']:
                self.dataset_sequences.append(f"{self.sequence_name}-{self.config['frame_generator']}-{str(num_events)}")
                self.reconstruction_types.append('num_events')
                self.timewindow_list.append(num_events)
        else:
            for idx, window in enumerate(self.config['timewindows']):
                self.dataset_sequences.append(f"{self.sequence_name}-{self.config['frame_generator']}-{str(window)}")
                self.reconstruction_types.append('timewindow')
                self.timewindow_list.append(window)

        
        if self.config['frame_accumulator'] == 'eventcount':
            for i in self.config['num_events']:
                self.max_events_list.append(i)
        # Create list of dataset sequences full paths
        self.full_dataset_paths = []
        for seq in self.dataset_sequences:
            self.full_dataset_paths.append(os.path.join(self.dataset_path, seq))
        # Check for existence of a hot pixel file
        self.hot_pixel_file = os.path.join(self.config['data_path'],
                                           self.dataset_name,
                                           self.sequence_name,
                                           f"{self.sequence_name}_hot_pixels.txt")
        
        if not os.path.exists(self.hot_pixel_file):
            if self.data_config['sequences'][self.sequence_name]['hot_pixel']['available']:
                # Download the hot pixel file if available
                hot_pixel_url = self.data_config['sequences'][self.sequence_name]['hot_pixel']['url']
                print(f"Downloading hot pixel file from {hot_pixel_url}")
                response = requests.get(hot_pixel_url)
                response.raise_for_status()
                with open(self.hot_pixel_file, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Hot pixel file downloaded: {self.hot_pixel_file}")
        
        # Download the sequence data if raw data does not exist
        if not os.path.exists(self.dataset_raw):
            download_sequence_data(self.config, self.data_config, self.dataset_name, self.sequence_name)
        # Format the data
        if not os.path.exists(self.dataset_formatted):
            format_sequence_data(self.config, self.data_config, self.dataset_name, self.sequence_name)
            # Get basic dataset info without loading all data
            self._load_dataset_metadata()
    
        # Check if the reconstructed dataset exists, if not, create frames (count and/or reconstruction)
        for idx, self.full_dataset_path in enumerate(self.full_dataset_paths):
            if not os.path.exists(self.full_dataset_path):
                self._load_dataset_metadata()
                # Load hot pixels if available
                self.hot_pixels = self._load_hot_pixels()
                if self.max_events_list:
                    # Check and potentially create event frames
                    self._handle_event_frames(self.full_dataset_path, 
                                            self.timewindow_list[idx], 
                                            self.reconstruction_types[idx], 
                                            recon=self.config['frame_generator'],
                                            max_events=self.max_events_list[idx])
                else:
                    self._handle_event_frames(self.full_dataset_path, 
                                            self.timewindow_list[idx], 
                                            self.reconstruction_types[idx], 
                                            recon=self.config['frame_generator'])

        # download the ground truth, if available
        if not os.path.exists(os.path.join(self.dataset_path, f"{self.sequence_name}_ground_truth.{self.data_config['format']['ground_truth']}")):
            if self.data_config['sequences'][self.sequence_name]['ground_truth']['available']:
                gt_url = self.data_config['sequences'][self.sequence_name]['ground_truth']['url']
                print(f"Downloading ground truth from {gt_url}")
                response = requests.get(gt_url)
                response.raise_for_status()
                gt_file = os.path.join(self.dataset_path, f"{self.sequence_name}_ground_truth.{self.data_config['format']['ground_truth']}")
                with open(gt_file, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Ground truth downloaded: {gt_file}")

    def _time_scale_from_config(self):
        """
        Read format->data->timestamp_val from self.data_config.
        Returns (scale, unit_name) where scale is ticks/second.
        """
        val = None
        try:
            val = self.data_config['format']['data'].get('timestamp_val', None)
        except Exception:
            pass

        if val is None:
            return None, None

        if isinstance(val, str):
            v = val.strip().lower()
            # accept common synonyms
            if v in ('ns', 'nanosecond', 'nanoseconds'):
                return 1e9, 'ns'
            if v in ('us', 'μs', 'microsecond', 'microseconds'):
                return 1e6, 'us'
            if v in ('ms', 'millisecond', 'milliseconds'):
                return 1e3, 'ms'
            if v in ('s', 'sec', 'second', 'seconds'):
                return 1.0, 's'
            # allow "1e6", "1000000" as strings
            try:
                num = float(v)
                if num > 0:
                    return float(num), f'{num:g} ticks/s'
            except Exception:
                pass
            raise ValueError(f"Unsupported timestamp_val='{val}'. Use ns/us/ms/s or numeric ticks/s.")

        if isinstance(val, (int, float)) and val > 0:
            return float(val), f'{float(val):g} ticks/s'

        raise ValueError(f"Invalid timestamp_val={val!r} (expected ns/us/ms/s or positive number)")

    def _load_dataset_metadata(self, *, rdcc_nbytes=64*1024*1024, rdcc_nslots=1_048_579,
                            max_tail_bytes=64*1024*1024, warn_chunk_bytes=256*1024*1024):
        import os, h5py, numpy as np

        def _read_first_last_scalar_1d(dset):
            n = int(dset.shape[0])
            if n == 0: return 0, 0
            out = np.empty(1, dtype=dset.dtype)
            dset.read_direct(out, np.s_[0:1],   np.s_[:]); first = int(out[0])
            dset.read_direct(out, np.s_[n-1:n], np.s_[:]); last  = int(out[0])
            return first, last

        def _read_first_scalar_1d(dset):
            n = int(dset.shape[0])
            if n == 0: return 0
            out = np.empty(1, dtype=dset.dtype)
            dset.read_direct(out, np.s_[0:1], np.s_[:]); return int(out[0])

        def _read_last_with_budget_1d(dset, max_bytes):
            n = int(dset.shape[0])
            if n == 0: return 0
            elems = max(1, max_bytes // dset.dtype.itemsize)
            start = max(0, n - elems); count = n - start
            buf = np.empty(count, dtype=dset.dtype)
            dset.read_direct(buf, np.s_[start:n], np.s_[:count])
            return int(buf[-1])

        def _fmt(n):
            try: return f"{int(n):,}"
            except Exception: return str(n)

        def _attr_unit_scale(node):
            keys = ("time_unit","time_units","unit","units","timestamp_unit",
                    "t_unit","timebase","time_base","time_scale","resolution")
            # try to interpret attrs as ticks per second or named unit
            def as_str(v):
                if isinstance(v, bytes): return v.decode("utf-8","ignore").lower()
                if isinstance(v, str):   return v.lower()
                return None
            # named unit first
            for k in keys:
                if hasattr(node, "attrs") and k in node.attrs:
                    s = as_str(node.attrs[k])
                    if s:
                        if "nano" in s or s == "ns": return 1e9, "ns"
                        if "micro" in s or s == "us": return 1e6, "us"
                        if "milli" in s or s == "ms": return 1e3, "ms"
                        if s == "s" or "second" in s: return 1.0, "s"
            # numeric ticks/s
            for k in keys:
                if hasattr(node, "attrs") and k in node.attrs:
                    v = node.attrs[k]
                    if isinstance(v, (int, float, np.integer, np.floating)) and v > 0:
                        return float(v), f"{float(v):g} ticks/s"
            return None, None

        # ---- open file (robust to absolute/relative dataset_formatted) ----
        fpath = self.dataset_formatted
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Dataset file not found: {fpath}")

        # ---- prefer config-specified time scale ----
        cfg_scale, cfg_unit_name = self._time_scale_from_config()

        # --- add near the top of the module (once) ---
        EVENT_T_KEYS = ("t", "timestamp", "timestamps", "time", "times")
        EVENT_X_KEYS = ("x", "x_coordinate", "x_coordinates", "u", "col", "column")
        EVENT_Y_KEYS = ("y", "y_coordinate", "y_coordinates", "v", "row")
        EVENT_P_KEYS = ("p", "polarity", "polarities", "pol", "polarity_bit", "polarity_bits")

        def _first_present_key(grp, keys):
            for k in keys:
                if k in grp and isinstance(grp[k], h5py.Dataset):
                    return k
            return None

        def _any_present(ks, keys):
            # ks: group.keys(); keys: alias list
            return any(k in ks for k in keys)

        # --- replace your with-block body with this version (differences marked) ---
        with h5py.File(fpath, "r", swmr=True,
                    rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, rdcc_w0=0.0) as h5f:
            self.width, self.height = self._get_camera_dimensions(h5f)

            # ---- locate events node robustly ----
            ev = None
            if "events" in h5f:
                ev = h5f["events"]
            elif "columns" in h5f:  # layout: /columns/{x,y,t,p}
                ev = h5f["columns"]
            else:
                # fallback: discover any group containing (x,y) and a time key alias
                def _looks_like_split_group(obj):
                    if not isinstance(obj, h5py.Group):
                        return False
                    ks = obj.keys()
                    return (_any_present(ks, EVENT_T_KEYS)
                            and _any_present(ks, EVENT_X_KEYS)
                            and _any_present(ks, EVENT_Y_KEYS))

                # try shallow children first
                for _, obj in h5f.items():
                    if _looks_like_split_group(obj):
                        ev = obj
                        break

                # deep walk if still not found
                if ev is None:
                    def _visit(name, obj):
                        nonlocal ev
                        if ev is None and _looks_like_split_group(obj):
                            ev = obj
                    h5f.visititems(_visit)

            if ev is None:
                raise ValueError(
                    "No events found: expected '/events' dataset/group, or a group with x,y,t[,p] "
                    "(e.g. '/columns')."
                )

            # ---- locate datasets & count ----
            if isinstance(ev, h5py.Group):
                # split layout: group with x/y/t(/p), accept aliases
                tkey = _first_present_key(ev, EVENT_T_KEYS)
                if tkey is None:
                    raise ValueError("events-like group missing time dataset (t/timestamp/timestamps).")

                tds = ev[tkey]
                self.events_count = int(tds.shape[0])
                if self.events_count == 0:
                    print("Warning: No events found in dataset")
                    self.start_time = self.end_time = 0
                    self.duration_sec = 0.0
                    return

                try:
                    t0_raw, tN_raw = _read_first_last_scalar_1d(tds)
                except MemoryError:
                    t0_raw = _read_first_scalar_1d(tds)
                    tN_raw = _read_last_with_budget_1d(tds, max_tail_bytes)

            elif isinstance(ev, h5py.Dataset):
                # packed layout: compound with time field or numeric Nx>=3 with column 2 as time
                dset = ev
                self.events_count = int(dset.shape[0])
                if self.events_count == 0:
                    print("Warning: No events found in dataset")
                    self.start_time = self.end_time = 0
                    self.duration_sec = 0.0
                    return

                names = dset.dtype.names or ()
                if names:
                    # compound -> must have a time field (accept aliases)
                    names_lut = {n.lower(): n for n in names}
                    t_field = next((names_lut[a] for a in EVENT_T_KEYS if a in names_lut), None)
                    if t_field is None:
                        raise ValueError("compound /events dataset has no time field (t/timestamp/timestamps).")
                    try:
                        t0_raw = int(dset[0][t_field])
                        tN_raw = int(dset[-1][t_field])
                    except MemoryError:
                        n = int(dset.shape[0])
                        t0_raw = int(dset[0][t_field]) if n else 0
                        tN_raw = int(dset[n-1][t_field]) if n else 0
                else:
                    # numeric Nx>=3: assume columns [x,y,t,(p?)]
                    if dset.ndim != 2 or dset.shape[1] < 3:
                        raise ValueError("numeric /events dataset must be Nx>=3 with column 2 as time (x,y,t[,p]).")
                    try:
                        t0_raw = int(dset[0, 2])
                        tN_raw = int(dset[-1, 2])
                    except MemoryError:
                        n = int(dset.shape[0])
                        t0_raw = int(dset[0, 2]) if n else 0
                        tN_raw = int(dset[n-1, 2]) if n else 0
            else:
                raise TypeError(f"Unsupported events node type: {type(ev)}")

            dt_raw = max(0, int(tN_raw) - int(t0_raw))

            # ---- resolve time scale: config → attrs → (optional) heuristic fallback ----
            if cfg_scale:
                scale = float(cfg_scale)
                unit_name = cfg_unit_name
            else:
                attr_scale, attr_unit = _attr_unit_scale(ev)
                if isinstance(ev, h5py.Group) and attr_scale is None:
                    # use the resolved time key for attrs (was hardcoded to 't' before)
                    tkey_for_attr = tkey  # from above
                    t_attr_scale, t_attr_unit = _attr_unit_scale(ev[tkey_for_attr])
                    if t_attr_scale:
                        attr_scale, attr_unit = t_attr_scale, t_attr_unit
                if attr_scale:
                    scale = float(attr_scale); unit_name = attr_unit
                else:
                    # last resort heuristic
                    candidates = [(1e9, "ns"), (1e6, "us"), (1e3, "ms"), (1.0, "s")]
                    MAX_RATE = 80_000_000.0
                    MIN_RATE = 10.0
                    pick = None
                    for sc, nm in candidates:
                        dur = dt_raw / sc if dt_raw > 0 else 0.0
                        rate = (self.events_count / dur) if dur > 0 else float("inf")
                        if MIN_RATE <= rate <= MAX_RATE:
                            pick = (sc, nm); break
                    if pick is None: pick = (1e6, "us")
                    scale, unit_name = pick

            # ---- store + print ----
            self.time_scale = float(scale)        # ticks per second
            self.time_unit_name = unit_name or "ticks/s"
            self.start_time = int(t0_raw)
            self.end_time   = int(tN_raw)
            self.duration_sec = (dt_raw / self.time_scale) if dt_raw > 0 else 0.0

    def _to_ticks(self, value, unit="ticks"):
        """Convert value (single number) from unit -> file ticks."""
        if unit in ("tick", "ticks", None):
            return int(value)
        scale = getattr(self, "time_scale", None)
        if not scale:
            # Fallback: assume nanoseconds if unknown (legacy), but warn
            print("Warning: time_scale unknown; assuming 'ns'")
            scale = 1e9
        if unit == "s":   return int(round(value * scale))
        if unit == "ms":  return int(round(value * scale / 1e3))
        if unit == "us":  return int(round(value * scale / 1e6))
        if unit == "ns":  return int(round(value * scale / 1e9))
        raise ValueError(f"Unsupported time unit: {unit}")

    def _searchsorted_h5(self, tds, target_tick, side="left"):
        """
        Binary search over a 1-D, monotonically increasing HDF5 dataset (no full load).
        Returns index in [0, N].
        """
        n = int(tds.shape[0])
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            tm = int(tds[mid])  # 1-element read
            if tm > target_tick or (side == "right" and tm == target_tick):
                hi = mid
            else:
                lo = mid + 1
        return lo if side == "right" else lo
    
    def _get_camera_dimensions(self, h5f):
        """Get camera dimensions efficiently"""
        # First try to get from camera_info in HDF5
        if 'camera_info' in h5f:
            camera_info = h5f['camera_info']
            if 'width' in camera_info.attrs and 'height' in camera_info.attrs:
                width = int(camera_info.attrs['width'])
                height = int(camera_info.attrs['height'])
                return width, height
        
        # Try to get from dataset config
        if 'resolution' in self.data_config['dataset']:
            resolution = self.data_config['dataset']['resolution']
            width, height = resolution[0], resolution[1]
            return width, height
        
        # Default fallback
        print("Warning: Could not determine camera dimensions, using default 640x480")
        return 640, 480
    
    def _load_hot_pixels(self):
        """Load hot pixel coordinates from file"""
        # Construct hot pixels file path
        self.hot_pixels_file = os.path.join(
            self.dataset_path, 
            f"{self.sequence_name}_hot_pixels.txt"
        )
        
        if not os.path.exists(self.hot_pixels_file):
            return None
        
        try:
            hot_pixels = []
            with open(self.hot_pixels_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    
                    try:
                        # Parse "x, y" format
                        parts = line.split(',')
                        if len(parts) != 2:
                            print(f"Warning: Invalid format in hot pixels file line {line_num}: '{line}'")
                            continue
                        
                        x = int(parts[0].strip())
                        y = int(parts[1].strip())
                        
                        # Validate coordinates are within sensor bounds
                        if 0 <= x < self.width and 0 <= y < self.height:
                            hot_pixels.append((x, y))
                        else:
                            print(f"Warning: Hot pixel ({x}, {y}) is out of sensor bounds ({self.width}x{self.height})")
                            
                    except ValueError as e:
                        print(f"Warning: Could not parse line {line_num} in hot pixels file: '{line}' - {e}")
                        continue
            
            if hot_pixels:
                hot_pixels_array = np.array(hot_pixels, dtype=np.uint16)
                # Convert to numpy array for efficient filtering
                return hot_pixels_array
            else:
                print(f"No valid hot pixels found in {self.hot_pixels_file}")
                return None
                
        except Exception as e:
            print(f"Error loading hot pixels file {self.hot_pixels_file}: {e}")
            return None
    
    def _handle_event_frames(self, frames_dir, timewindow, window_type, recon=None, max_events=None):
        """Check and potentially create event frames"""
        self.max_events = max_events
        # Define frames directory
        frames_dir = frames_dir
        
        if os.path.exists(frames_dir) and not recon == 'countmatch':
            self.frames_dir = frames_dir
            return
        
        # Get offset for this sequence
        offset_sec = self._get_sequence_offset()

        
        # Calculate effective time range and estimate frames
        self._estimate_and_confirm_frame_generation(frames_dir, timewindow, offset_sec, window_type)
    
    def _parse_offset_to_seconds(self, val):
        """
        Accepts float/int (seconds), or strings with units: 'ns','us','ms','s'.
        Returns seconds (float).
        """
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)  # interpret as seconds for backward compatibility
        v = str(val).strip().lower()
        # numeric string without unit -> seconds
        try:
            return float(v)
        except Exception:
            pass
        # with unit
        import re
        m = re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(ns|us|μs|ms|s)\s*', v)
        if not m:
            raise ValueError(f"Cannot parse offset '{val}'. Use seconds or a number with unit (ns/us/ms/s).")
        num = float(m.group(1)); unit = m.group(2)
        if unit in ('s',):    return num
        if unit in ('ms',):   return num / 1e3
        if unit in ('us','μs'): return num / 1e6
        if unit == 'ns':      return num / 1e9
        raise ValueError(f"Unsupported offset unit in '{val}'")

    def _get_sequence_offset(self):
        try:
            raw = (self.data_config.get('other', {})
                                .get('offset', {})
                                .get(self.sequence_name, None))
            if raw is None:
                return None
            offset_sec = self._parse_offset_to_seconds(raw)
            # sanity check: warn if absurd compared to duration we already computed
            try:
                if hasattr(self, "duration_sec") and self.duration_sec > 0 and offset_sec > 10 * self.duration_sec:
                    print(f"Warning: offset {offset_sec} s >> duration {self.duration_sec:.2f} s")
            except Exception:
                pass
            print(f"Found offset for {self.sequence_name}: {offset_sec} seconds")
            return offset_sec
        except Exception as e:
            print(f"Could not parse offset for {self.sequence_name}: {e}")
            return None

    def _estimate_and_confirm_frame_generation(self, frames_dir, timewindow_ms, offset_sec, window_type):
        # convert window to TICKS
        if not hasattr(self, "time_scale"):
            print("Warning: time_scale unknown; assuming 'ns' for estimation")
            scale = 1e9
        else:
            scale = self.time_scale

        timewindow_ticks = int(round((timewindow_ms / 1000.0) * scale))
        if offset_sec:
            offset_ticks = int(round(offset_sec * scale))
            effective_start = max(self.start_time, offset_ticks)
        else:
            effective_start = self.start_time

        effective_duration_ticks = max(0, self.end_time - effective_start)
        if effective_duration_ticks <= 0:
            print(f"Warning: No events after offset {offset_sec}, skipping frame generation")
            return

        estimated_frames = int(effective_duration_ticks // timewindow_ticks)
        effective_duration_sec = effective_duration_ticks / float(scale)

        # Get accumulator type for storage estimation
        accumulator_type = self.config.get('frame_accumulator', 'count')
        if accumulator_type == 'polarity':
            bytes_per_pixel = 8  # 2 channels * 4 bytes (float32)
        else:
            bytes_per_pixel = 4  # 1 channel * 4 bytes (float32)
        
        # Estimate storage size
        estimated_size_mb = estimated_frames * self.width * self.height * bytes_per_pixel / (1024 * 1024)
        
        print(f"\n=== Event Frame Generation Estimation ===")
        print(f"Time window: {timewindow_ms} ms")
        print(f"Frame accumulator: {accumulator_type}")
        print(f"Sequence offset: {offset_sec if offset_sec else 'None'}")
        print(f"Effective duration: {effective_duration_sec:.2f} seconds")
        print(f"Estimated frames: {estimated_frames:,}")
        print(f"Frame resolution: {self.width} x {self.height}")
        print(f"Estimated storage: {estimated_size_mb:.1f} MB")
        
        # Ask user for confirmation
        if self.config['request_input']:
            user_response = input(f"\nGenerate {estimated_frames:,} event frames? (yes/no): ").strip().lower()
            if user_response not in ['yes', 'y']:
                print("Frame generation cancelled")
                return
        
        # Generate frames using the generalized builder
        print("\nStarting frame generation...")
        if self.config['frame_generator'] == 'reconstruction':
            hot_pxls = self.hot_pixels_file # reconstruction loads the file, not the direct array
        else:
            hot_pxls = self.hot_pixels # event count methods use the pre-loaded array
        build_event_frames(
            hdf5_path=self.dataset_formatted,
            config=self.config,
            data_config=self.data_config,
            sequence_name=self.sequence_name,
            timewindow=timewindow_ms,
            width=self.width,
            height=self.height,
            frames_dir=frames_dir,
            window_type=window_type,
            accumulator_type=self.config['frame_accumulator'],
            hot_pixels=hot_pxls,  # Pass already loaded hot pixels
            max_events=self.max_events
        )
        
        self.frames_dir = frames_dir
        print(f"✓ Frame generation complete: {frames_dir}")
    
    def get_events(self, start_time=None, end_time=None, max_events=None,
                start_idx=None, end_idx=None, chunk_size=1_000_000,
                time_unit="ticks", fields=("x","y","t","p"),
                rdcc_nbytes=64*1024*1024, rdcc_nslots=1_048_579):
        """
        Memory-efficient slice of events.

        Args:
            start_time/end_time: bounds in given time_unit ('ticks','ns','us','ms','s').
            start_idx/end_idx: index bounds (mutually exclusive with time-based).
            max_events: clamp the slice length.
            fields: subset of columns to read (default all).
            chunk_size: upper bound for materialization (you can stream if larger).
        """
        # open the FILE, not the folder
        h5_path = os.path.join(self.dataset_path, self.dataset_formatted)
        if not os.path.exists(h5_path):
            return {k: np.array([]) for k in ("x","y","t","p")}

        with h5py.File(h5_path, "r", swmr=True,
                    rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, rdcc_w0=0.0) as h5f:
            if "events" not in h5f:
                return {k: np.array([]) for k in ("x","y","t","p")}

            ev = h5f["events"]
            # Support both layouts
            if isinstance(ev, h5py.Group):
                ds = {k: ev[k] for k in ("x","y","t","p") if k in ev}
                n_total = int(next(iter(ds.values())).shape[0]) if ds else 0
                tds = ds.get("t", None)
            else:  # compound dataset
                n_total = int(ev.shape[0])
                tds = ev["t"]  # field access is cheap; element reads are scalar

            # Determine index bounds
            if start_idx is None and end_idx is None and (start_time is not None or end_time is not None):
                if tds is None:
                    raise ValueError("Timestamps dataset/field 't' not found for time-based slicing.")
                s_tick = self._to_ticks(start_time, time_unit) if start_time is not None else None
                e_tick = self._to_ticks(end_time,   time_unit) if end_time   is not None else None
                start_idx = 0 if s_tick is None else self._searchsorted_h5(tds, s_tick, side="left")
                end_idx   = n_total if e_tick is None else self._searchsorted_h5(tds, e_tick, side="right")
            else:
                if start_idx is None: start_idx = 0
                if end_idx   is None: end_idx   = n_total

            start_idx = max(0, int(start_idx))
            end_idx   = min(n_total, int(end_idx))
            if end_idx <= start_idx:
                return {k: np.array([]) for k in ("x","y","t","p")}

            if max_events is not None:
                end_idx = min(end_idx, start_idx + int(max_events))

            # Materialize only requested fields
            want = tuple(f for f in fields if f in ("x","y","t","p"))
            out = {}
            if isinstance(ev, h5py.Group):
                for k in ("x","y","t","p"):
                    if k in want and k in ev:
                        out[k] = ev[k][start_idx:end_idx]
                    elif k in ("x","y","t","p"):
                        out[k] = np.array([])
            else:
                # compound dataset: slice once per requested field
                sl = slice(start_idx, end_idx)
                for k in ("x","y","t","p"):
                    if k in want:
                        out[k] = ev[sl][k]  # h5py will read only the requested field
                    else:
                        out[k] = np.array([])

            return out

    
    def stream_events(self, chunk_size=1_000_000,
                    start_time=None, end_time=None, time_unit="ticks",
                    fields=("x","y","t","p"),
                    rdcc_nbytes=64*1024*1024, rdcc_nslots=1_048_579):
        """
        Stream events in chunks (memory efficient). Supports time-bounded streaming.

        Yields dicts with requested fields.
        """
        h5_path = os.path.join(self.dataset_path, self.dataset_formatted)
        if not os.path.exists(h5_path):
            return
        import h5py, numpy as np

        with h5py.File(h5_path, "r", swmr=True,
                    rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, rdcc_w0=0.0) as h5f:
            if "events" not in h5f:
                return
            ev = h5f["events"]

            if isinstance(ev, h5py.Group):
                ds = {k: ev[k] for k in ("x","y","t","p") if k in ev}
                n_total = int(next(iter(ds.values())).shape[0]) if ds else 0
                tds = ds.get("t", None)
            else:
                n_total = int(ev.shape[0])
                tds = ev["t"]

            # Resolve index window
            if start_time is not None or end_time is not None:
                if tds is None:
                    raise ValueError("Timestamps dataset/field 't' not found for time-bounded streaming.")
                s_tick = self._to_ticks(start_time, time_unit) if start_time is not None else None
                e_tick = self._to_ticks(end_time,   time_unit) if end_time   is not None else None
                start_idx = 0 if s_tick is None else self._searchsorted_h5(tds, s_tick, side="left")
                end_idx   = n_total if e_tick is None else self._searchsorted_h5(tds, e_tick, side="right")
            else:
                start_idx, end_idx = 0, n_total

            want = tuple(f for f in fields if f in ("x","y","t","p"))
            for s in range(start_idx, end_idx, int(chunk_size)):
                e = min(s + int(chunk_size), end_idx)
                if isinstance(ev, h5py.Group):
                    chunk = {}
                    for k in ("x","y","t","p"):
                        if k in want and k in ev:
                            chunk[k] = ev[k][s:e]
                    yield chunk
                else:
                    sl = slice(s, e)
                    chunk = {}
                    for k in ("x","y","t","p"):
                        if k in want:
                            chunk[k] = ev[sl][k]
                    yield chunk

    
    def load_frame(self, frame_idx):
        """Load a specific frame by index"""
        if not hasattr(self, 'frames_dir') or not os.path.exists(self.frames_dir):
            raise ValueError("No frames available. Generate frames first.")
        
        frame_path = os.path.join(self.frames_dir, f"frame_{frame_idx:06d}.npy")
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame {frame_idx} not found")
        
        return np.load(frame_path, mmap_mode="r")
    
    def get_frames_info(self):
        """
        Return a dict {sequence_name: metadata_dict} for any sequences that
        already have frames (detected via metadata.json). If none found, return None.
        This does not load any .npy frames into memory.
        """
        import json

        results = {}
        # Ensure we have the list of sequence dirs
        if not hasattr(self, "full_dataset_paths"):
            # Build from known fields if needed
            self.full_dataset_paths = [
                os.path.join(self.dataset_path, seq) for seq in getattr(self, "dataset_sequences", [])
            ]

        for seq, seq_path in zip(getattr(self, "dataset_sequences", []), self.full_dataset_paths):
            meta_path = os.path.join(seq_path, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    # Ensure total_frames is present; if not, compute cheaply
                    if "total_frames" not in meta:
                        try:
                            meta["total_frames"] = sum(
                                1 for n in os.listdir(seq_path)
                                if n.startswith("frame_") and n.endswith(".npy")
                            )
                        except Exception:
                            pass
                    results[seq] = meta
                except Exception as e:
                    print(f"Warning: failed to read {meta_path}: {e}")

        return results or None
    
    def get_dataset_info(self):
        self._load_dataset_metadata()
        """Get comprehensive dataset information"""
        # create list of file paths for each sequence
        file_paths = {}
        for seq in self.dataset_sequences:
            file_paths[seq] = os.path.join(self.dataset_path, seq)
        info = {
            'dataset_name': self.dataset_name,
            'sequence_name': self.sequence_name,
            'file_path': file_paths,
            'events_count': self.events_count,
            'sensor_resolution': (self.width, self.height),
            'time_range': (self.start_time, self.end_time),
            'duration_seconds': self.duration_sec,
            'event_rate': self.events_count / self.duration_sec if self.duration_sec > 0 else 0,
            'frames_info': self.get_frames_info()
        }
        return info
    
    def __repr__(self):
        return (f"Event_dataset(dataset='{self.dataset_name}', "f"sequence='{self.sequence_name}',"
                f"Saved location: {self.dataset_sequences})")