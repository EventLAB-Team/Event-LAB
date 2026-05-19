"""
Fast pseudo ground-truth builder (refactor).

Key features
- **Fast pose-only path (default)**: no frame or CSV generation required.
- Supports pose files in Parquet / HDF5 / NMEA-txt (and similar) via a universal loader.
- Matches by nearest *route-completion %* (monotonic via cumulative distance), then
  applies vertical dilation tolerance (rows around the best ref indices).
- Keeps your original CSV and qcr_event methods as fallbacks.

Typical usage
-------------
out = generate_ground_truth(
    config, dataset_config,
    dataset_name="qcr_event",
    reference_name="traverse1",
    query_name="traverse2",
    reference_data=None, query_data=None,
    timewindow=1000,
    gps_available=False,     # ignored by fast path
    fast=True                # << default True (pose-only)
)

Notes
-----
- Expects pose/GT files at:
  {data_path}/{dataset_name}/{traverse}/{traverse}_ground_truth.{ext}
  where ext = dataset_config['format']['ground_truth'] (e.g. 'parquet', 'h5', 'nmea', 'txt')
- If you’re doing count-matched regimes and already know target frame counts,
  pass n_frames_ref / n_frames_qry to ground_truth_from_pose_only().
"""

from __future__ import annotations

import os, re, glob, ast, json
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate


# =============================================================================
# Utilities: time scaling, distances, small helpers
# =============================================================================

def _to_seconds(values) -> np.ndarray:
    """
    Robustly convert a timestamp array to seconds, relative to the first sample.
    Correctly handles UNIX timestamps (in seconds) without incorrectly scaling them.
    """
    v = np.asarray(values)

    if np.issubdtype(v.dtype, np.datetime64):
        t_s = (v.astype("datetime64[ns]").astype("int64").astype(np.float64)) * 1e-6
    else:
        v = v.astype(np.float64)
        median_val = np.nanmedian(v)

        # Very large -> likely ns/us/ms; otherwise treat as seconds (epoch or relative)
        # print which one it uses
        if median_val > 1e14:      # nanoseconds
            t_s = v / 1e9
        elif median_val > 1e11:    # microseconds
            t_s = v / 1e6
        elif median_val > 1e10:    # milliseconds
            t_s = v / 1e3
        else:
            t_s = v                # seconds

    # Make relative to first finite value
    if t_s.size > 0:
        mask = np.isfinite(t_s)
        if mask.any():
            t0 = t_s[mask][0]
            t_s = t_s - t0
    return t_s


def haversine(lon1, lat1, lon2, lat2) -> float:
    """Meters between two WGS84 lon/lat points."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c * 1000.0


def euclidean(x1, y1, x2, y2) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def create_GTtol(GT: np.ndarray, distance: int = 2) -> np.ndarray:
    """
    Vertical dilation: set 1s up to +/- `distance` rows around each positive cell.
    """
    GT_binary = (GT > 0).astype(np.uint8)
    GTtol = np.zeros_like(GT_binary, dtype=np.uint8)
    num_rows, num_cols = GT_binary.shape

    for col in range(num_cols):
        ones_indices = np.where(GT_binary[:, col] == 1)[0]
        for row in ones_indices:
            start_row = max(row - distance, 0)
            end_row = min(row + distance + 1, num_rows)
            GTtol[start_row:end_row, col] = 1
    return GTtol

def create_GTtol_by_distance(
    GT: np.ndarray,
    ref_dist_m: np.ndarray,
    tol_m: float,
) -> np.ndarray:
    """
    Vertical dilation in *meters* instead of rows.

    GT:         (R, Q) binary matrix with exactly 0 or 1 '1' per column.
    ref_dist_m: (R,) cumulative distance [m] along reference route for each row.
    tol_m:      tolerance in meters (e.g., 25.0).

    For each query column j, we locate its matched reference row i (where GT[i, j] == 1),
    then mark all rows r where |ref_dist_m[r] - ref_dist_m[i]| <= tol_m as positive.
    """
    R, Q = GT.shape
    ref_dist_m = np.asarray(ref_dist_m, dtype=np.float64)
    GTtol = np.zeros_like(GT, dtype=np.uint8)

    if R == 0 or Q == 0:
        return GTtol

    for j in range(Q):
        col = GT[:, j]
        idx = np.flatnonzero(col)
        if idx.size == 0:
            continue  # no match in this column

        center_idx = idx[0]
        center_dist = ref_dist_m[center_idx]

        mask = np.abs(ref_dist_m - center_dist) <= float(tol_m)
        GTtol[mask, j] = 1

    return GTtol


# =============================================================================
# Pose loaders (Parquet / NMEA / HDF5) + dispatch
# =============================================================================

_TIME_COLS = ["timestamp", "timestamps", "time", "t", "ts", "stamp", "time_s",
              "sec", "seconds", "time_ns", "time_us", "time_ms"]
_LAT_COLS  = ["lat", "latitude", "lat_deg"]
_LON_COLS  = ["lon", "longitude", "lng", "long", "lon_deg"]
_X_COLS    = ["x", "pos_x", "x_m", "east", "easting", "utm_e"]
_Y_COLS    = ["y", "pos_y", "y_m", "north", "northing", "utm_n"]

def _pick_col(df: pd.DataFrame, cands: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in cols:
            return cols[c]
    return None


def _load_parquet_pose(path: str) -> Tuple[np.ndarray, str, float]:
    """
    Load odometry from a parquet file or a directory of parquet files.
    Returns (pose Nx3 array, kind, t0) where:
      - kind == "geo" -> [lat, lon, t_seconds]
      - kind == "xy"  -> [x, y, t_seconds]
      - t0: original start time offset applied (seconds)
    """
    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    elif os.path.isfile(path) and path.lower().endswith(".parquet"):
        files = [path]
    else:
        raise FileNotFoundError(f"Parquet path not found or not a parquet: {path}")

    if not files:
        raise FileNotFoundError(f"No parquet files found in: {path}")

    dfs = [pd.read_parquet(fp) for fp in files]
    df = pd.concat(dfs, ignore_index=True)

    tcol = _pick_col(df, _TIME_COLS)
    if tcol is None:
        raise KeyError(f"Parquet: time column not found. Tried: {_TIME_COLS}. Have: {list(df.columns)}")

    latc = _pick_col(df, _LAT_COLS)
    lonc = _pick_col(df, _LON_COLS)
    if latc and lonc:
        kind = "geo"
        xy = df[[latc, lonc]].to_numpy(dtype=np.float64, copy=False)
    else:
        xc = _pick_col(df, _X_COLS)
        yc = _pick_col(df, _Y_COLS)
        if not (xc and yc):
            raise KeyError("Parquet: missing coords (lat/lon or x/y).")
        kind = "xy"
        xy = df[[xc, yc]].to_numpy(dtype=np.float64, copy=False)

    t_s = _to_seconds(df[tcol].to_numpy())

    order = np.argsort(t_s, kind="mergesort")
    t_s = np.asarray(t_s, dtype=np.float64)[order]
    xy  = np.asarray(xy,  dtype=np.float64)[order]

    # Normalize to start at 0 and enforce strict monotonic time
    t0 = 0.0
    if np.isfinite(t_s).any():
        t0 = float(np.nanmin(t_s))
        t_s -= t0

    eps = 1e-9
    for k in range(1, len(t_s)):
        if not np.isfinite(t_s[k]) or t_s[k] <= t_s[k - 1]:
            t_s[k] = t_s[k - 1] + eps

    pose = np.column_stack([xy[:, 0], xy[:, 1], t_s])
    return pose, kind, t0


def _load_nmea_gps(nmea_file_path: str) -> Tuple[np.ndarray, str]:
    """
    Return (pose Nx3 -> lat, lon, t_seconds, 'geo').
    """
    import pynmea2
    latitudes, longitudes, timestamps = [], [], []
    first_timestamp = None
    prev_lat, prev_lon = 0.0, 0.0

    with open(nmea_file_path, encoding='utf-8') as nmea_file:
        for line in nmea_file:
            try:
                msg = pynmea2.parse(line)
                if msg.sentence_type in ['GSV', 'VTG', 'GSA']:
                    continue
                if first_timestamp is None:
                    first_timestamp = msg.timestamp
                dist_to_prev = np.linalg.norm(np.array([msg.latitude, msg.longitude]) -
                                              np.array([prev_lat, prev_lon]))
                if (msg.latitude != 0 and msg.longitude != 0 and
                    msg.latitude != prev_lat and msg.longitude != prev_lon and
                    dist_to_prev > 0.0001):
                    t, t0 = msg.timestamp, first_timestamp
                    dt = ((t.hour - t0.hour) * 3600
                          + (t.minute - t0.minute) * 60
                          + (t.second - t0.second)
                          + (getattr(t, "microsecond", 0) - getattr(t0, "microsecond", 0)) / 1e6)
                    latitudes.append(msg.latitude)
                    longitudes.append(msg.longitude)
                    timestamps.append(dt)
                    prev_lat, prev_lon = msg.latitude, msg.longitude
            except pynmea2.ParseError:
                continue

    gps = np.column_stack([latitudes, longitudes, timestamps]).astype(np.float64)
    if gps.size:
        ts = gps[:, 2]
        eps = 1e-6
        for k in range(1, len(ts)):
            if ts[k] <= ts[k - 1]:
                ts[k] = ts[k - 1] + eps
        gps[:, 2] = ts
    return gps, "geo"


def _load_hdf5_pose(path: str) -> Tuple[np.ndarray, str]:
    """
    Liberal HDF5 loader. Attempts to find coords (lat/lon or x/y) + a usable time field.
    Returns (pose Nx3, kind in {'geo','xy'}), time normalized to 0 and strictly increasing.
    """
    import h5py

    if not (os.path.isfile(path) and path.lower().endswith((".h5", ".hdf5"))):
        raise FileNotFoundError(f"Not an HDF5 file: {path}")

    structured_dfs: list[pd.DataFrame] = []
    scalar_cols: dict[str, np.ndarray] = {}

    def _canon(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
        s = re.sub(r"[^a-z0-9]+", "", s)
        s = s.replace("degrees", "deg").replace("degree", "deg")
        s = s.replace("longitude", "lon").replace("latitude", "lat")
        s = s.replace("easting", "e").replace("northing", "n")
        return s

    def _pick_col_strict(df: pd.DataFrame, cands: list[str]) -> Optional[str]:
        cols = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    def _pick_by_patterns(df: pd.DataFrame, patterns: list[str]) -> Optional[str]:
        canon_map = {_canon(c): c for c in df.columns}
        raw_cols  = list(df.columns)
        canon_cols = list(canon_map.keys())
        for pat in patterns:
            rx = re.compile(pat)
            for c in raw_cols:
                if rx.search(c.lower()):
                    return c
            for cc in canon_cols:
                if rx.search(cc):
                    return canon_map[cc]
        return None

    TIME_PATTERNS = [
        r"^timestamps?$",
        r"^(gpstime|gps_time|gpstimestamp)$",
        r"^(ros)?time(stamp)?(s|sec)?$",
        r"^(unix(time)?|utctime)$",
        r"^(tow|sow|timeofweek|secsofweek|weeksec|weekseconds)$",
        r"^time.*(ns|us|ms|s)$",
        r"^t$|^ts$|^stamp$",
    ]
    WEEK_PATTERNS = [r"^(gps)?week(num|number)?$", r"^gps_week$|^gpsweek$|^week$"]
    TOW_PATTERNS  = [r"^(gps_)?(tow|sow)$", r"^(secsofweek|weeksec|weekseconds)$", r"^timeofweek$"]
    LAT_PATTERNS  = [r"^lat$", r"^latdeg$", r"^lat.*deg$", r"^geodeticlat$", r"^latitude", r"^lat"]
    LON_PATTERNS  = [r"^lon$", r"^londeg$", r"^lon.*deg$", r"^geodeticlon$", r"^longitude", r"^lon|^lng"]
    X_PATTERNS    = [r"^x(_m)?$", r"^posx$", r"^utm(e|east|easting)", r"(^|_)east(ing)?(_m)?$", r"^enu(e|east)$", r"(^|_)e(_|$)"]
    Y_PATTERNS    = [r"^y(_m)?$", r"^posy$", r"^utm(n|north|northing)", r"(^|_)north(ing)?(_m)?$", r"^enu(n|north)$", r"(^|_)n(_|$)"]

    def _leaf_name(h5path: str) -> str:
        return h5path.split("/")[-1]

    with h5py.File(path, "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.dtype.names:
                    arr = obj[...]
                    try:
                        df = pd.DataFrame.from_records(arr)
                        df.columns = [str(c) for c in df.columns]
                        structured_dfs.append(df)
                    except Exception:
                        pass
                else:
                    try:
                        a = np.asarray(obj[...])

                        leaf_raw   = _leaf_name(name).lower()
                        leaf_canon = _canon(_leaf_name(name))

                        # 1-D vectors (time/coords candidates)
                        if a.ndim == 1 and a.shape[0] > 1:
                            scalar_cols[leaf_canon] = a
                            scalar_cols[leaf_raw]   = a
                            return

                        # 2-D arrays that look like N×2 or N×3 (or transposed)
                        if a.ndim == 2 and (2 in a.shape or 3 in a.shape):
                            if a.shape[1] in (2, 3):
                                comps = a
                            elif a.shape[0] in (2, 3):
                                comps = a.T
                            else:
                                return
                            C = comps.shape[1]
                            base = ["x", "y", "z"]
                            for i in range(min(C, 3)):
                                col = comps[:, i]
                                canon_key = base[i]
                                raw_key   = f"{leaf_raw}_{base[i]}"
                                canon_desc= f"{leaf_canon}_{base[i]}"
                                if canon_key not in scalar_cols:
                                    scalar_cols[canon_key] = col
                                scalar_cols[raw_key]    = col
                                scalar_cols[canon_desc] = col
                            return
                    except Exception:
                        pass
        h5.visititems(visit)

    def _score_df(df: pd.DataFrame) -> Tuple[int, int]:
        has_time = 1 if (_pick_col_strict(df, _TIME_COLS) or _pick_by_patterns(df, TIME_PATTERNS)) else 0
        has_geo  = 1 if (_pick_by_patterns(df, LAT_PATTERNS) and _pick_by_patterns(df, LON_PATTERNS)) else 0
        has_xy   = 1 if (_pick_by_patterns(df, X_PATTERNS) and _pick_by_patterns(df, Y_PATTERNS)) else 0
        return (has_time + (2 if has_geo else 0) + (1 if has_xy else 0), len(df))

    candidate_df: Optional[pd.DataFrame] = None
    if structured_dfs:
        structured_dfs.sort(key=_score_df, reverse=True)
        if _score_df(structured_dfs[0])[0] > 0:
            candidate_df = structured_dfs[0]

    if candidate_df is None and scalar_cols:
        df = pd.DataFrame({k: pd.Series(v) for k, v in scalar_cols.items()})
        df = df.dropna(axis=1, how="all")
        if not df.empty:
            candidate_df = df

    if candidate_df is None or candidate_df.empty:
        raise KeyError("HDF5: Could not find usable time + coordinate fields.")

    # Resolve coordinates (prefer geo)
    def _pick_by_patterns(df: pd.DataFrame, patterns: list[str]) -> Optional[str]:
        canon_map = {re.sub(r'[^a-z0-9]+', '', c.lower()): c for c in df.columns}
        raw = list(df.columns)
        for pat in patterns:
            rx = re.compile(pat)
            for c in raw:
                if rx.search(c.lower()):
                    return c
            for cc, orig in canon_map.items():
                if rx.search(cc):
                    return orig
        return None

    latc = _pick_by_patterns(candidate_df, LAT_PATTERNS)
    lonc = _pick_by_patterns(candidate_df, LON_PATTERNS)
    kind = None
    if latc and lonc:
        kind = "geo"
        xy = candidate_df[[latc, lonc]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    else:
        xc = _pick_by_patterns(candidate_df, X_PATTERNS)
        yc = _pick_by_patterns(candidate_df, Y_PATTERNS)
        if xc and yc:
            kind = "xy"
            xy = candidate_df[[xc, yc]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        else:
            raise KeyError("HDF5: Could not find coordinate columns (lat/lon or x/y).")

    tcol = _pick_col(candidate_df, _TIME_COLS) or _pick_by_patterns(candidate_df, TIME_PATTERNS)

    def _compose_week_tow(df: pd.DataFrame) -> Optional[np.ndarray]:
        wk_candidates = [r"^(gps)?week(num|number)?$", r"^gps_week$|^gpsweek$|^week$"]
        tw_candidates = [r"^(gps_)?(tow|sow)$", r"^(secsofweek|weeksec|weekseconds)$", r"^timeofweek$"]
        wk = _pick_by_patterns(df, wk_candidates)
        tw = _pick_by_patterns(df, tw_candidates)
        if wk and tw:
            weeks = pd.to_numeric(df[wk], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            sow   = pd.to_numeric(df[tw], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            if weeks.size and sow.size:
                ts = (weeks * 604800.0) + sow
                if np.isfinite(ts).any():
                    ts = ts - np.nanmin(ts)
                return ts.astype(np.float64, copy=False)
        return None

    if tcol is not None:
        t_raw = candidate_df[tcol].to_numpy()
        t_s = _to_seconds(t_raw)
    else:
        t_s = _compose_week_tow(candidate_df)
        if t_s is None:
            raise KeyError("HDF5: no usable time column or (GPS week + TOW/SOW) pair.")

    n = min(len(xy), len(t_s))
    xy, t_s = xy[:n], t_s[:n]
    mask = np.isfinite(t_s) & np.isfinite(xy).all(axis=1)
    xy, t_s = xy[mask], t_s[mask]

    order = np.argsort(t_s)
    t_s = t_s[order].astype(np.float64, copy=False)
    xy  = xy[order].astype(np.float64, copy=False)

    eps = 1e-9
    for k in range(1, len(t_s)):
        if t_s[k] <= t_s[k - 1]:
            t_s[k] = t_s[k - 1] + eps

    pose = np.column_stack([xy[:, 0], xy[:, 1], t_s])
    return pose, kind


def _load_pose_any(path: str) -> Tuple[np.ndarray, str, float]:
    """
    Dispatch to appropriate pose loader.
    Returns (pose Nx3, kind, t0_offset_seconds).
    """
    pl = path.lower()
    if os.path.isdir(path) or pl.endswith(".parquet"):
        return _load_parquet_pose(path)
    elif pl.endswith(".nmea") or pl.endswith(".txt") or pl.endswith(".log"):
        pose, kind = _load_nmea_gps(path)
        return pose, kind, 0.0
    elif pl.endswith(".h5") or pl.endswith(".hdf5"):
        pose, kind = _load_hdf5_pose(path)
        return pose, kind, 0.0
    else:
        raise ValueError(f"Unrecognized pose/odometry format: {path}")

# =============================================================================
# qcr_event route-percent GT (kept, minimal changes)
# =============================================================================

def qcr_event_ground_truth(config, dataset_config, reference_name, query_name,
                           gt_dir, reference_data, query_data, timewindow,
                           gt_tolerance=2):
    """
    Build pseudo GT via % of route completed using dataset_config['other']['gt_times'] knots.
    Uses generated frame-store metadata only; it does not create or load frame arrays.
    Rescales frame timelines to span GT durations (decouples frame count vs physical time).
    """
    def _timewindow_str(v):
        return str(v).replace(".", "_") if isinstance(v, float) else str(v)

    def _count_legacy_frames(folder):
        if not os.path.isdir(folder):
            return 0
        return len([
            f for f in os.listdir(folder)
            if f.startswith("frame_") and f.endswith(".npy")
        ])

    def _load_metadata(frames_dir):
        meta_path = os.path.join(frames_dir, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)
        return {}

    def _resolve_h5_path(frames_dir, metadata):
        storage_file = metadata.get("storage_file", "frames.h5")
        candidates = [
            os.path.join(frames_dir, storage_file),
            os.path.join(frames_dir, "frames.h5"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def _frame_store_info(frames_dir, fallback_ms=None):
        metadata = _load_metadata(frames_dir)
        n_frames = metadata.get("total_frames")
        eff_ms = metadata.get("derived_timewindow_ms", metadata.get("timewindow_ms"))

        h5_path = _resolve_h5_path(frames_dir, metadata)
        if h5_path is not None:
            try:
                import h5py

                with h5py.File(h5_path, "r") as h5:
                    if "frames" not in h5:
                        raise KeyError(f"{h5_path} does not contain dataset '/frames'.")

                    n_frames = int(h5["frames"].shape[0])

                    ticks_per_second = h5.attrs.get("ticks_per_second", None)
                    window_ticks = h5.attrs.get("window_ticks", None)
                    if ticks_per_second is not None and window_ticks is not None:
                        ticks_per_second = float(ticks_per_second)
                        window_ticks = float(window_ticks)
                        if np.isfinite(ticks_per_second) and ticks_per_second > 0:
                            eff_ms = (window_ticks / ticks_per_second) * 1000.0
            except ImportError:
                # metadata.json is enough for GT sizing, and keeps this path usable
                # in lightweight environments that do not have h5py imported.
                pass

        if n_frames is None:
            n_frames = _count_legacy_frames(frames_dir)

        n_frames = int(n_frames or 0)
        if n_frames <= 0:
            raise ValueError(
                f"Missing frame store in {frames_dir}. Expected metadata.json, "
                "frames.h5, or legacy frame_*.npy files."
            )

        if eff_ms is None:
            if fallback_ms is None:
                raise ValueError(f"No (derived_)timewindow_ms metadata in {frames_dir}")
            eff_ms = fallback_ms

        return n_frames, float(eff_ms)

    def _validate_knots(name, knots):
        if len(knots) < 2:
            raise ValueError(f"Need at least two GT time knots for {name}.")
        knots = np.asarray(knots, dtype=float)
        if not np.all(np.isfinite(knots)):
            raise ValueError(f"GT time knots for {name} contain non-finite values.")
        if np.any(np.diff(knots) < 0):
            raise ValueError(f"GT time knots for {name} must be monotonic.")
        if knots[-1] <= 0:
            raise ValueError(f"Last GT time knot for {name} must be positive.")
        return knots

    def _frame_times_rescaled(n_frames, eff_ms, gt_last_s):
        if n_frames <= 0:
            return np.zeros((0,), dtype=float)
        dt = max(eff_ms / 1000.0, 1e-9)
        t_naive = (np.arange(n_frames, dtype=float) + 0.5) * dt
        dur_naive = n_frames * dt
        scale = float(gt_last_s) / max(dur_naive, 1e-9)
        t = t_naive * scale
        return np.clip(t, 0.0, float(gt_last_s))

    timewindow_str = _timewindow_str(timewindow)
    ref_dir = os.path.join(config['data_path'], 'qcr_event', reference_name,
                           f"{reference_name}-{config['frame_generator']}-{timewindow_str}")
    query_dir = os.path.join(config['data_path'], 'qcr_event', query_name,
                             f"{query_name}-{config['frame_generator']}-{timewindow_str}")

    nominal_ms = float(timewindow) if isinstance(timewindow, (int, float)) else 1000.0

    num_ref_files, eff_ref_ms = _frame_store_info(ref_dir, fallback_ms=nominal_ms)
    num_query_files, eff_qry_ms = _frame_store_info(query_dir, fallback_ms=nominal_ms)

    try:
        ref_knots = _validate_knots(reference_name, dataset_config['other']['gt_times'][reference_name])
        qry_knots = _validate_knots(query_name, dataset_config['other']['gt_times'][query_name])
    except KeyError as e:
        raise KeyError(f"GT times missing for {e.args[0]} in dataset_config['other']['gt_times']")

    ref_end_s = float(ref_knots[-1])
    qry_end_s = float(qry_knots[-1])

    ref_times_total   = _frame_times_rescaled(num_ref_files,  eff_ref_ms, ref_end_s)
    query_times_total = _frame_times_rescaled(num_query_files, eff_qry_ms, qry_end_s)

    ref_perc_kn = [(t / ref_end_s) * 100.0 for t in ref_knots]
    qry_perc_kn = [(t / qry_end_s) * 100.0 for t in qry_knots]

    f_ref  = interpolate.interp1d(ref_knots, ref_perc_kn, bounds_error=False, fill_value=(0.0, 100.0))
    f_qry  = interpolate.interp1d(qry_knots, qry_perc_kn, bounds_error=False, fill_value=(0.0, 100.0))

    ref_percentages_total   = f_ref(ref_times_total)      # (R,)
    query_percentages_total = f_qry(query_times_total)    # (Q,)

    ref_col = ref_percentages_total[:, None]              # (R,1)
    qry_row = query_percentages_total[None, :]            # (1,Q)
    diff = np.abs(ref_col - qry_row)                      # (R,Q)
    idx = np.argmin(diff, axis=0)                         # (Q,)
    binary_gt = np.zeros_like(diff, dtype=np.uint8)       # (R,Q)
    binary_gt[idx, np.arange(diff.shape[1])] = 1

    binary_gt = binary_gt[:num_ref_files, :num_query_files]
    binary_gt = create_GTtol(binary_gt, distance=gt_tolerance)

    os.makedirs(gt_dir, exist_ok=True)
    out_path = os.path.join(gt_dir, f"{reference_name}_{query_name}_GT.npy")
    np.save(out_path, binary_gt)

    plt.figure(figsize=(10, 8))
    plt.imshow(binary_gt, cmap='gray', aspect='auto')
    plt.title('Pseudo Ground Truth Matrix (qcr_event % route)')
    plt.xlabel('Query Images'); plt.ylabel('Reference Images')
    plt.tight_layout()
    plt.savefig(out_path.replace('.npy', '.png'))
    plt.close()


# =============================================================================
# New fast pose-only GT
# =============================================================================

def _frame_centers_by_dt(T_end_s: float, dt_s: float) -> np.ndarray:
    """Midpoint-of-bin frame centers: [0.5*dt, 1.5*dt, ..., <= T_end_s]"""
    if T_end_s <= 0 or dt_s <= 0:
        return np.zeros((0,), dtype=float)
    n = int(np.floor(T_end_s / dt_s))
    if n <= 0:
        return np.array([min(0.5 * dt_s, T_end_s)], dtype=float)
    t = (np.arange(n, dtype=float) + 0.5) * dt_s
    return np.clip(t, 0.0, T_end_s)


def _frame_centers_by_n(T_end_s: float, n_frames: int) -> np.ndarray:
    """Evenly spaced centers across [0, T_end_s]."""
    n = int(max(n_frames, 1))
    return (np.arange(n, dtype=float) + 0.5) * (T_end_s / n)


def _percent_route_over_time(pose: np.ndarray, kind: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Given pose (N,3) = [x|lat, y|lon, t], return:
      times: (N,), seconds (strictly increasing, starts near 0)
      perc:  (N,), route completion percent [0..100]
      total_dist_m: total traveled distance in meters (0 if stationary)
    Uses cumulative distance (Haversine or Euclidean). If stationary, falls back to time-based %.
    """
    if pose.size == 0 or len(pose) < 2:
        times = pose[:, 2] if pose.size else np.zeros((0,), dtype=float)
        perc  = np.zeros_like(times)
        total_dist_m = 0.0
        return times, perc, total_dist_m

    xy = pose[:, :2].astype(np.float64)
    t  = pose[:, 2].astype(np.float64)

    if kind == "geo":
        d = [0.0]
        for i in range(1, len(xy)):
            d.append(haversine(xy[i-1,1], xy[i-1,0], xy[i,1], xy[i,0]))
        d = np.cumsum(np.asarray(d, dtype=np.float64))
    else:
        diffs = np.diff(xy, axis=0)
        step  = np.hypot(diffs[:,0], diffs[:,1])
        d     = np.concatenate([[0.0], np.cumsum(step)])

    total = float(d[-1])
    total_dist_m = total

    if total < 1e-6:
        # Stationary or almost; fall back to time-based percentage
        T = float(max(t[-1], 1e-9))
        perc = (t / T) * 100.0
    else:
        perc = (d / total) * 100.0

    return t, np.clip(perc, 0.0, 100.0), total_dist_m


def _percent_at_times(times_src: np.ndarray, perc_src: np.ndarray, times_query: np.ndarray) -> np.ndarray:
    """Interpolate % route at arbitrary times, clamped [0..100]."""
    if times_src.size == 0:
        return np.zeros_like(times_query)
    f = interpolate.interp1d(times_src, perc_src, bounds_error=False, fill_value=(perc_src[0], perc_src[-1]))
    return np.clip(f(times_query), 0.0, 100.0)


def _nearest_indices_monotonic(ref_perc: np.ndarray, qry_perc: np.ndarray) -> np.ndarray:
    """
    Given two monotonic increasing arrays (ref, qry), find for each qry the nearest ref index.
    O(Q log R) with small memory.
    """
    idx = np.searchsorted(ref_perc, qry_perc, side="left")
    left  = np.clip(idx - 1, 0, len(ref_perc) - 1)
    right = np.clip(idx,     0, len(ref_perc) - 1)
    choose_right = (np.abs(ref_perc[right] - qry_perc) <= np.abs(ref_perc[left] - qry_perc))
    best = np.where(choose_right, right, left).astype(np.int64)
    return best


def ground_truth_from_pose_only(config,
                                dataset_config,
                                dataset_name: str,
                                reference_name: str,
                                query_name: str,
                                timewindow_ms: float,
                                gt_tolerance: int = 2,
                                n_frames_ref: int | None = None,
                                n_frames_qry: int | None = None,
                                out_dir: str | None = None,
                                out_basename: str | None = None,
                                ref_frame_times: np.ndarray | None = None,
                                qry_frame_times: np.ndarray | None = None) -> str:
    """
    Build GT without generating frames:
      1) load ref/qry pose (lat/lon or x/y) with timestamps
      2) compute route-completion % over time from cumulative distance
      3) choose frame centres in time:
         - if ref_frame_times/qry_frame_times is given: use those directly
         - elif n_frames_ref/qry is given: use evenly spaced centres in [0, T_end]
         - else: use timewindow_ms to define centres
      4) nearest-% matching + (optional) distance-based vertical dilation
    """
    gt_ext = dataset_config['format']['ground_truth']

    ref_pose_path = os.path.join(
        config['data_path'], dataset_name, reference_name,
        f"{reference_name}_ground_truth.{gt_ext}"
    )
    qry_pose_path = os.path.join(
        config['data_path'], dataset_name, query_name,
        f"{query_name}_ground_truth.{gt_ext}"
    )

    if not os.path.exists(ref_pose_path):
        raise FileNotFoundError(f"Missing reference pose: {ref_pose_path}")
    if not os.path.exists(qry_pose_path):
        raise FileNotFoundError(f"Missing query pose: {qry_pose_path}")

    ref_pose, ref_kind, _ = _load_pose_any(ref_pose_path)
    qry_pose, qry_kind, _ = _load_pose_any(qry_pose_path)

    ref_t, ref_pct, ref_total_m = _percent_route_over_time(ref_pose, ref_kind)
    qry_t, qry_pct, qry_total_m = _percent_route_over_time(qry_pose, qry_kind)

    # -------------------------------------------------------------------------
    # Frame centres in time
    # -------------------------------------------------------------------------
    # Reference
    if ref_frame_times is not None:
        ref_centers = np.asarray(ref_frame_times, dtype=float)
    elif n_frames_ref is not None:
        ref_centers = _frame_centers_by_n(ref_t[-1], int(n_frames_ref))
    else:
        if timewindow_ms is None:
            raise ValueError("timewindow_ms must be provided if n_frames_ref and ref_frame_times are None")
        ref_centers = _frame_centers_by_dt(ref_t[-1], float(timewindow_ms) / 1000.0)

    # Query
    if qry_frame_times is not None:
        qry_centers = np.asarray(qry_frame_times, dtype=float)
    elif n_frames_qry is not None:
        qry_centers = _frame_centers_by_n(qry_t[-1], int(n_frames_qry))
    else:
        if timewindow_ms is None:
            raise ValueError("timewindow_ms must be provided if n_frames_qry and qry_frame_times are None")
        qry_centers = _frame_centers_by_dt(qry_t[-1], float(timewindow_ms) / 1000.0)

    # Route-% at those frame centres
    ref_pct_frames = _percent_at_times(ref_t, ref_pct, ref_centers)
    qry_pct_frames = _percent_at_times(qry_t,  qry_pct,  qry_centers)

    # Ensure monotonicity
    ref_pct_frames = np.maximum.accumulate(ref_pct_frames)
    qry_pct_frames = np.maximum.accumulate(qry_pct_frames)

    # Nearest-% matching
    best_ref_idx = _nearest_indices_monotonic(ref_pct_frames, qry_pct_frames)

    # Base GT
    R, Q = len(ref_pct_frames), len(qry_pct_frames)
    GT = np.zeros((R, Q), dtype=np.uint8)
    GT[best_ref_idx, np.arange(Q)] = 1

    # Distance-based tolerance in metres along reference route
    if gt_tolerance is not None and gt_tolerance > 0 and ref_total_m > 0.0:
        ref_dist_frames_m = (ref_pct_frames / 100.0) * ref_total_m
        GT = create_GTtol_by_distance(
            GT,
            ref_dist_frames_m,
            tol_m=float(gt_tolerance),
        )

    # Output
    if out_dir is None:
        out_dir = os.path.join(config['data_path'], dataset_name, "ground_truth")
    os.makedirs(out_dir, exist_ok=True)

    if out_basename is None:
        out_basename = f"{reference_name}_{query_name}_GT"

    out_npy = os.path.join(out_dir, f"{out_basename}.npy")
    np.save(out_npy, GT)

    plt.figure(figsize=(10, 8))
    plt.imshow(GT, cmap='gray', aspect='auto')
    plt.title('Pseudo Ground Truth (pose-only fast path)')
    plt.xlabel('Query (frames)')
    plt.ylabel('Reference (frames)')
    plt.tight_layout()
    plt.savefig(out_npy.replace('.npy', '.png'))
    plt.close()

    return out_npy

# =============================================================================
# Unified entry point
# =============================================================================

def generate_ground_truth(config,
                          dataset_config,
                          dataset_name: str,
                          reference_name: str,
                          query_name: str,
                          reference_data,
                          query_data,
                          timewindow: float,
                          gps_available: bool = False,
                          *,
                          fast: bool = True,
                          force_csv: bool = False,
                          n_frames_ref: int | None = None,
                          n_frames_qry: int | None = None) -> str:
    """
    Unified GT generator. Order of preference:
      1) fast=True: pose-only route-% matching (no frames/CSVs).
      2) force_csv or gps_available: CSV+Euclidean (legacy path).
      3) dataset-specific route-% (qcr_event), using frame counts & knots.

    Returns: path to the saved GT .npy
    """
    gt_dir = os.path.join(config['data_path'], dataset_name, 'ground_truth')
    os.makedirs(gt_dir, exist_ok=True)

    # If the config file frame_accumulator is "eventcount", get the number of frames for q and r
    if config.get('frame_accumulator', '') == 'eventcount':
        ref_frames_dir = os.path.join(config['data_path'], dataset_name, reference_name,
                                    f"{reference_name}-{config['frame_generator']}-{config['num_events'][0]}")
        qry_frames_dir = os.path.join(config['data_path'], dataset_name, query_name,
                                    f"{query_name}-{config['frame_generator']}-{config['num_events'][0]}")

        ref_ticks = np.load(os.path.join(ref_frames_dir, "event_frame_times_ticks.npy"))
        qry_ticks = np.load(os.path.join(qry_frames_dir, "event_frame_times_ticks.npy"))
        # Convert ticks -> seconds, relative to start
        ref_frame_times_s = (ref_ticks - ref_ticks[0]) * 1e-9
        qry_frame_times_s = (qry_ticks - qry_ticks[0]) * 1e-9
    else:
        ref_frame_times_s = None
        qry_frame_times_s = None
        
    # Pose-only fast path (default)
    if fast and gps_available:
        return ground_truth_from_pose_only(
            config=config,
            dataset_config=dataset_config,
            dataset_name=device_safe_str(dataset_name),
            reference_name=device_safe_str(reference_name),
            query_name=device_safe_str(query_name),
            timewindow_ms=float(timewindow),
            gt_tolerance=int(config.get('ground_truth_tolerance', 2)),
            n_frames_ref=None,                     # we’re using real frame times, so ignore this
            n_frames_qry=None,
            out_dir=gt_dir,
            ref_frame_times=ref_frame_times_s,
            qry_frame_times=qry_frame_times_s,
        )

    # Dataset-specific (qcr_event) fallback
    if dataset_name == "qcr_event":
        qcr_event_ground_truth(
            config,
            dataset_config,
            reference_name,
            query_name,
            gt_dir,
            reference_data,
            query_data,
            timewindow,
            gt_tolerance=int(config.get('ground_truth_tolerance', 2))
        )
        return os.path.join(gt_dir, f"{reference_name}_{query_name}_GT.npy")

    raise ValueError("No suitable GT path selected; set fast=True or provide gps_available/force_csv=True.")

# Small utility: sanitize names for filesystem safety (optional)
def device_safe_str(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]", "_", str(s))
