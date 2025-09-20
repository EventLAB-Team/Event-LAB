import json, os, h5py, re, warnings

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, Sequence, List, Tuple, Union, Dict

PathLike = Union[Path, str]

def create_GTtol(GT, distance=2):
    """
    Creates a ground truth matrix with vertical tolerance by manually adding 1s
    above and below the original 1s up to the specified distance.
    
    Parameters:
    - GT (numpy.ndarray): The original ground truth matrix.
    - distance (int): The maximum number of rows to add 1s above and below the detected 1s.
    
    Returns:
    - GTtol (numpy.ndarray): The modified ground truth matrix with vertical tolerance.
    """
    # Ensure GT is a binary matrix
    GT_binary = (GT > 0).astype(int)
    
    # Initialize GTtol with zeros
    GTtol = np.zeros_like(GT_binary)
    
    # Get the number of rows and columns
    num_rows, num_cols = GT_binary.shape
    
    # Iterate over each column
    for col in range(num_cols):
        # Find the indices of rows where GT has a 1 in the current column
        ones_indices = np.where(GT_binary[:, col] == 1)[0]
        
        # For each index with a 1, set 1s in GTtol within the specified vertical distance
        for row in ones_indices:
            # Determine the start and end rows, ensuring they are within bounds
            start_row = max(row - distance, 0)
            end_row = min(row + distance + 1, num_rows)  # +1 because upper bound is exclusive
            
            # Set the range in GTtol to 1
            GTtol[int(start_row):int(end_row), col] = 1
    
    return GTtol

def _detect_divisor_from_med_dt(med_dt: float) -> float:
    for divisor in (1e9, 1e6, 1e3, 1.0):  # ns, µs, ms, s
        med_dt_s = med_dt / divisor
        if 1e-4 <= med_dt_s <= 10.0:
            return divisor
    return 1e9

def _ticks_to_seconds(ticks: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, float]:
    ticks = np.asarray(ticks, dtype=np.float64)
    if ticks.size == 0:
        return np.array([], dtype=np.float64), 1.0
    if ticks.size == 1:
        return np.array([0.0], dtype=np.float64), 1.0

    diffs = np.diff(ticks)
    if np.any(diffs <= 0):
        warnings.warn("Non-monotonic or duplicate ticks detected; selection may be less regular.")
    med_dt = float(np.median(diffs))
    divisor = _detect_divisor_from_med_dt(med_dt)
    times_s = (ticks - ticks[0]) / divisor
    return times_s, divisor

def _quick_time_stats(times_s: np.ndarray) -> dict:
    t = np.asarray(times_s, dtype=float)
    diffs = np.diff(t)
    return {
        "n": int(t.size),
        "t0": float(t[0] if t.size else np.nan),
        "tN": float(t[-1] if t.size else np.nan),
        "duration_s": float((t[-1] - t[0]) if t.size else 0.0),
        "median_dt_s": float(np.median(diffs)) if diffs.size else float('nan'),
        "p1_dt_s": float(np.percentile(diffs, 1)) if diffs.size else float('nan'),
        "p99_dt_s": float(np.percentile(diffs, 99)) if diffs.size else float('nan'),
        "n_nonmono": int(np.sum(diffs <= 0)) if diffs.size else 0,
        "n_dupes": int(np.sum(np.isclose(diffs, 0.0))) if diffs.size else 0,
    }

def _make_strictly_increasing(t: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    t = t.astype(np.float64).copy()
    for i in range(1, len(t)):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + eps
    return t


def _select_indices_every_fixed_interval(times_seconds: np.ndarray, interval_sec: float) -> List[int]:
    t = np.asarray(times_seconds, dtype=np.float64)
    if t.size == 0:
        return []
    if interval_sec <= 0:
        return list(range(len(t)))
    bins = np.floor(t / interval_sec).astype(np.int64)
    keep_mask = np.empty_like(bins, dtype=bool)
    keep_mask[0] = True
    keep_mask[1:] = bins[1:] != bins[:-1]
    return np.nonzero(keep_mask)[0].tolist()

def _apply_time_filter_to_files(
    files_list: Sequence[PathLike],
    dirpath: PathLike,
    min_gap_sec: float,
    ticks_filename: str = "event_frame_times_ticks.npy",
    ticks_text_name: str = "timestamps.txt",
    debug: bool = False
) -> Dict[str, object]:
    """
    If <ticks_filename> exists and loads, interpret as tick timestamps and convert
    to seconds via `_ticks_to_seconds`. Otherwise, if <ticks_text_name> exists,
    interpret as seconds (one float per line) and use directly.
    If neither exists, skip filtering and return passthrough.

    Returns:
      {
        'files':                    filtered_files (kept only),
        'kept_idx':                 indices kept (relative to original files_list),
        'dropped_idx':              indices dropped (relative to original files_list),
        'times_s':                  relative seconds (trimmed to matched length),
        'divisor':                  tick divisor used (1.0 if seconds text; None if skipped),
        'times_s_filtered':         seconds after applying interval filter,
        'timestamps_filtered_text': newline-separated text of filtered seconds (%.18f each)
      }
    """
    files_list = list(files_list)
    orig_len = len(files_list)

    dirpath = Path(dirpath)
    npy_path = dirpath / ticks_filename
    txt_path = dirpath / ticks_text_name

    source = None
    ticks = None
    times_s = None
    divisor = None

    # ----- Try .npy ticks first -----
    if npy_path.exists():
        try:
            ticks = np.load(str(npy_path))
            ticks = np.asarray(ticks).reshape(-1)
            times_s, divisor = _ticks_to_seconds(ticks, debug=debug)
            source = "npy"
        except Exception as e:
            warnings.warn(f"Failed to load/interpret {ticks_filename}: {e}. Will try {ticks_text_name}…")

    # ----- Fallback: seconds .txt -----
    if source is None:
        if not txt_path.exists():
            warnings.warn(
                f"Neither {ticks_filename} nor {ticks_text_name} available in {dirpath} — skipping time filtering"
            )
            return {
                'files': files_list,
                'kept_idx': list(range(orig_len)),
                'dropped_idx': [],
                'times_s': None,
                'divisor': None,
                'times_s_filtered': None,
                'timestamps_filtered_text': None,
            }
        try:
            with open(txt_path, "r") as f:
                vals = []
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    s = s.rstrip(",")
                    vals.append(float(s))
            if not vals:
                raise ValueError("timestamps.txt contained no valid floats")
            times_s = np.asarray(vals, dtype=float).reshape(-1)
            divisor = 1.0  # seconds already
            source = "txt"
        except Exception as e:
            warnings.warn(f"Failed to read {ticks_text_name}: {e}. Skipping time filtering.")
            return {
                'files': files_list,
                'kept_idx': list(range(orig_len)),
                'dropped_idx': [],
                'times_s': None,
                'divisor': None,
                'times_s_filtered': None,
                'timestamps_filtered_text': None,
            }
    # ----- Align by trimming to min length (times vs files) -----
    n_times = len(times_s)
    if n_times != orig_len:
        warnings.warn(f"{'ticks/seconds' if source else 'timestamps'} length ({n_times}) != #files ({orig_len}). "
                      f"Trimming to min length and proceeding.")
        m = min(n_times, orig_len)
        times_s = times_s[:m]
        files_trimmed = files_list[:m]
        pretrim_dropped = list(range(m, orig_len))
    else:
        m = orig_len
        files_trimmed = files_list
        pretrim_dropped = []
    # ----- Select indices at fixed interval -----
    interval = float(min_gap_sec)
    kept_within_trim = _select_indices_every_fixed_interval(times_s, interval)

    # Map kept indices (within trimmed range) back to original indexing
    kept_idx = kept_within_trim
    dropped_within_trim = sorted(set(range(m)) - set(kept_within_trim))
    dropped_idx = dropped_within_trim + pretrim_dropped

    filtered_files = [files_trimmed[i] for i in kept_within_trim]
    times_s_filtered = times_s[kept_within_trim] if len(times_s) else np.array([], dtype=float)

    # Pre-formatted text content (no saving)
    timestamps_filtered_text = "".join(f"{v:.18f}\n" for v in times_s_filtered)

    if debug:
        duration = float(times_s[-1] - times_s[0]) if len(times_s) else 0.0
        expected = int(np.floor(duration / interval)) + 1 if interval > 0 and len(times_s) else len(times_s)
        print(f"[time-filter:{source}] interval={interval:.6f}s, duration≈{duration:.6f}s → "
              f"expected≈{expected}, kept={len(kept_idx)}, dropped={len(dropped_idx)} (orig={orig_len})")
        if kept_idx:
            human = f"{times_s[kept_within_trim[0]]:.6f} s (relative)"
            print(f"[time-filter:{source}] first kept index: {kept_idx[0]}  first kept: {human}")

    return {
        'files': filtered_files,
        'kept_idx': kept_idx,
        'dropped_idx': dropped_idx,
        'times_s': times_s,
        'divisor': divisor,
        'times_s_filtered': times_s_filtered,
        'timestamps_filtered_text': timestamps_filtered_text,
    }

def compute_event_rate(hdf5_path, ticks_per_sec_override):
    """
    Returns events_per_second using only t-field (constant-rate summary).
    """
    with h5py.File(hdf5_path, "r") as f:
        ev = f["events"]
        # support both group with fields and compound dset
        if isinstance(ev, h5py.Group):
            t = ev["t"]
            n = int(t.shape[0])
        else:
            t = ev.fields("t")
            n = int(ev.shape[0])
        if n == 0:
            return 0.0

        t0 = int(t[0])
        tN = int(t[n-1])
        # try to read unit/timescale from attrs, unless caller overrides
        if ticks_per_sec_override is None:
            scale = _infer_ticks_per_second(f, ev, t)
        else:
            scale = float(ticks_per_sec_override)

        dur_s = max((tN - t0) / float(scale), 1e-9)
        return n / dur_s

def _infer_ticks_per_second(h5f, ev, tds):
    def as_str(v):
        if isinstance(v, bytes): return v.decode("utf-8","ignore").lower()
        if isinstance(v, str):   return v.lower()
        return None
    keys = ("time_unit","time_units","unit","units","timestamp_unit",
            "t_unit","timebase","time_base","time_scale","resolution")
    attrs = {}
    for node in (ev, tds, h5f):
        if node is None or not hasattr(node, "attrs"): continue
        for k in keys:
            if k in node.attrs and k not in attrs:
                attrs[k] = node.attrs[k]
    for v in attrs.values():
        s = as_str(v)
        if not s: continue
        if "nano" in s or s == "ns": return 1e9
        if "micro" in s or s == "us": return 1e6
        if "milli" in s or s == "ms": return 1e3
        if s == "s" or "second" in s: return 1.0
    for v in attrs.values():
        if isinstance(v, (int, float, np.integer, np.floating)) and v > 0:
            return float(v)
    return 1e6  # safe default: microseconds

def write_countmatch_meta(out_dir, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "countmatch_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path

def read_countmatch_meta(path):
    with open(path, "r") as f:
        return json.load(f)
    
from pathlib import Path
import numpy as np
from typing import Dict, Any

def avg_dt_ms_from_txt(path: str | Path) -> Dict[str, Any]:
    """
    Load timestamps (seconds) from a text file (one timestamp per line),
    compute consecutive differences and return statistics in milliseconds.

    Returns a dict with:
      - n_timestamps: int
      - n_intervals: int
      - mean_ms, median_ms, std_ms, min_ms, max_ms
      - diffs_ms: numpy array of differences (ms)

    Non-numeric / empty lines are skipped. Raises FileNotFoundError if path missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    ts = []
    skipped = 0
    with p.open('r') as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            try:
                ts.append(float(s))
            except ValueError:
                skipped += 1
                continue

    ts = np.array(sorted(ts), dtype=float)
    if ts.size < 2:
        return {
            "n_timestamps": int(ts.size),
            "n_intervals": 0,
            "mean_ms": float('nan'),
            "median_ms": float('nan'),
            "std_ms": float('nan'),
            "min_ms": float('nan'),
            "max_ms": float('nan'),
            "diffs_ms": np.array([]),
            "skipped_lines": skipped
        }

    diffs_s = np.diff(ts)              # seconds
    diffs_ms = diffs_s * 1000.0        # milliseconds

    stats = {
        "n_timestamps": int(ts.size),
        "n_intervals": int(diffs_ms.size),
        "mean_ms": float(np.mean(diffs_ms)),
        "median_ms": float(np.median(diffs_ms)),
        "std_ms": float(np.std(diffs_ms, ddof=0)),
        "min_ms": float(np.min(diffs_ms)),
        "max_ms": float(np.max(diffs_ms)),
        "diffs_ms": diffs_ms,
        "skipped_lines": skipped
    }
    return stats

# ----------------- Public API -----------------
def compute_avg_events_per_frame_from_h5(
    h5_path: str,
    frames_dir: str,
    *,
    dset_path: Optional[str] = None,
    ts_column: int = 0,
    width: int = 240,
    height: int = 180,
    num_events_per_pixel: float = 0.35,
    n_events_per_window_override: Optional[int] = None,
    match_tolerance_ms: float = 25.0,
    out_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute per-frame event counts by mapping saved frame timestamps (frame_XXXXXXXX.png)
    to fixed-N event windows extracted from an HDF5 event file.

    Parameters
    ----------
    h5_path :
        Path to HDF5 containing event timestamp dataset (one-dim or NxM with timestamps column).
    frames_dir :
        Directory containing saved frames named like `frame_<integer>.png`.
    dset_path :
        Optional explicit HDF5 dataset path to use for timestamps (e.g. '/events/timestamps').
        If None the function heuristically picks a numeric dataset (prefers names containing 'ts'/'time').
    ts_column :
        If your chosen dataset is 2D, this selects which column contains timestamps (default 0).
    width, height, num_events_per_pixel :
        Sensor geometry and events-per-pixel used by E2VID auto-N. N = round(width * height * num_events_per_pixel)
        If you prefer to supply exact N, use `n_events_per_window_override`.
    n_events_per_window_override :
        If provided, use this N instead of computing from width*height*num_events_per_pixel.
    match_tolerance_ms :
        Tolerance when reporting matched frames (informational only). Frame-to-window matching still performed.
    out_csv :
        If provided, path to write the resulting DataFrame as CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
          frame_filename, frame_int, frame_s, matched_window, matched_err_ms, events_in_window

    Notes
    -----
    - The function heuristically converts raw event timestamps to seconds by magnitude:
        median > 1e12 -> assume nanoseconds; >1e6 -> microseconds; >1e3 -> milliseconds; else seconds.
    - Frame timestamp integers are similarly tested for units by checking which scaling places the values
      inside the event timestamp range. This removes guesswork about ms/µs/ns.
    - This implements non-overlapping fixed-N windows (FixedSizeEventReader behavior). If you used sliding
      or fixed-duration windows, adapt accordingly or ask me for that variant.
    """
    # --- helpers (local functions) ---
    def list_numeric_datasets(h5p: Path) -> Dict[str, Dict]:
        out = {}
        with h5py.File(str(h5p), "r") as f:
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                    out[name] = {"shape": obj.shape, "dtype": str(obj.dtype)}
            f.visititems(_collect)
        return out

    def load_timestamps_from_h5(h5p: Path, dset: Optional[str], col: int) -> np.ndarray:
        with h5py.File(str(h5p), "r") as f:
            if dset is None:
                # pick best candidate
                cand = []
                def _collect(name, obj):
                    if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                        cand.append(name)
                f.visititems(_collect)
                if not cand:
                    raise RuntimeError(f"No numeric dataset found in HDF5: {h5p}")
                # score candidates
                scored = []
                kws = ("timestamp","timestamps","ts","time","t","event_ts")
                for name in cand:
                    score = 0
                    lname = name.lower()
                    if any(k in lname for k in kws): score += 10
                    ds = f[name]
                    if len(ds.shape) == 1: score += 5
                    scored.append((score, name))
                scored.sort(reverse=True)
                chosen = scored[0][1]
            else:
                if dset not in f:
                    raise KeyError(f"Dataset {dset} not found in {h5p}")
                chosen = dset
            d = f[chosen][()]
            if d.ndim == 1:
                return d.astype(float)
            else:
                return d[:, col].astype(float)

    def raw_ts_to_seconds(ts_raw: np.ndarray) -> Tuple[np.ndarray, str]:
        ts = np.asarray(ts_raw, dtype=float)
        med = float(np.median(ts))
        if med > 1e12:
            return ts / 1e9, "nanoseconds"
        if med > 1e6:
            return ts / 1e6, "microseconds"
        if med > 1e3:
            return ts / 1e3, "milliseconds"
        return ts, "seconds"

    def parse_frame_filenames(frames_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        p = frames_dir_path
        fpaths = sorted([p for p in p.glob("frame_*.png")])
        if not fpaths:
            raise FileNotFoundError(f"No files matching frame_*.png in {p}")
        names = [fp.name for fp in fpaths]
        ints = []
        for n in names:
            m = re.search(r'(\d+)', n)
            if not m:
                raise ValueError(f"Filename {n} does not contain an integer timestamp")
            ints.append(int(m.group(1)))
        order = np.argsort(ints)
        sorted_names = np.array(names)[order]
        sorted_ints = np.array(ints)[order]
        return sorted_names, sorted_ints

    def detect_frame_unit_and_convert(frame_ints: np.ndarray, event_ts_s: np.ndarray) -> Tuple[np.ndarray, str, float]:
        candidates = [
            ("seconds", 1.0),
            ("milliseconds", 1e3),
            ("microseconds", 1e6),
            ("nanoseconds", 1e9),
        ]
        evmin, evmax = float(event_ts_s.min()), float(event_ts_s.max())
        best = None
        for name, div in candidates:
            fs = frame_ints.astype(float) / div
            inside_frac = float(((fs >= evmin) & (fs <= evmax)).mean())
            # tie-break with median absolute error to nearest event
            idx = np.searchsorted(event_ts_s, fs)
            idx = np.clip(idx, 0, len(event_ts_s)-1)
            med_err = float(np.median(np.abs(fs - event_ts_s[idx])))
            score = (inside_frac, -med_err)
            if best is None or score > best[0]:
                best = (score, (fs, name, div))
        return best[1]  # (frame_seconds, unit_name, divisor)

    def build_fixed_N_window_counts(event_ts_s: np.ndarray, N: int) -> np.ndarray:
        n_events = len(event_ts_s)
        n_windows = (n_events + N - 1) // N
        counts = np.empty(n_windows, dtype=int)
        for i in range(n_windows):
            start = i * N
            end = min((i+1) * N, n_events)
            counts[i] = end - start
        return counts

    def map_frames_to_window_counts(frame_seconds: np.ndarray, event_ts_s: np.ndarray,
                                    counts_per_window: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_events = len(event_ts_s)
        window_last_idx = [min((i+1)*N - 1, n_events - 1) for i in range(len(counts_per_window))]
        window_last_ts = event_ts_s[window_last_idx]
        idx = np.searchsorted(window_last_ts, frame_seconds)
        matched = np.full(len(frame_seconds), -1, dtype=int)
        matched_err_s = np.full(len(frame_seconds), np.nan, dtype=float)
        for i, ins in enumerate(idx):
            candidates = []
            for j in (ins-1, ins):
                if 0 <= j < len(window_last_ts):
                    candidates.append((abs(window_last_ts[j] - frame_seconds[i]), j))
            if candidates:
                err, jbest = min(candidates, key=lambda x: x[0])
                matched[i] = jbest
                matched_err_s[i] = err
        counts_for_frames = np.array([counts_per_window[m] if m >= 0 else np.nan for m in matched])
        return counts_for_frames, matched, matched_err_s

    # --- main flow ---
    h5p = Path(h5_path)
    frames_p = Path(frames_dir)

    # load timestamps from HDF5
    ts_raw = load_timestamps_from_h5(h5p, dset_path, ts_column)
    event_ts_s, ev_unit = raw_ts_to_seconds(ts_raw)

    # compute N
    if n_events_per_window_override is not None:
        N = int(n_events_per_window_override)
    else:
        N = int(round(width * height * num_events_per_pixel))
    if N <= 0:
        raise ValueError("Computed N <= 0; check width/height/num_events_per_pixel or pass override.")

    # build fixed-N window counts
    counts_per_window = build_fixed_N_window_counts(event_ts_s, N)

    # parse frames and detect units
    frame_names, frame_ints = parse_frame_filenames(frames_p)
    frame_seconds, frame_unit_name, frame_divisor = detect_frame_unit_and_convert(frame_ints, event_ts_s)

    # map frames to counts
    counts_for_frames, matched_windows, matched_err_s = map_frames_to_window_counts(frame_seconds, event_ts_s, counts_per_window, N)
    matched_err_ms = matched_err_s * 1000.0

    # build output DataFrame
    df = pd.DataFrame({
        "frame_filename": frame_names,
        "frame_int": frame_ints,
        "frame_s": frame_seconds,
        "matched_window": matched_windows,
        "matched_err_ms": matched_err_ms,
        "events_in_window": counts_for_frames
    })

    # optional CSV output
    if out_csv:
        try:
            df.to_csv(out_csv, index=False)
        except Exception as ex:
            raise RuntimeError(f"Failed to write CSV to {out_csv}: {ex}")

    return df