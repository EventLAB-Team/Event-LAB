"""Published Gardens Point GPS mapped onto the recording's camera clock."""
import csv
import json
from pathlib import Path
import warnings

import numpy as np


def camera_seconds(epoch_ns, sync, first_event_us):
    # host_epoch = anchor.delta + camera.beta_prime + alpha * camera_us * 1000
    camera = sync['camera']
    return ((np.asarray(epoch_ns, dtype=np.float64) - sync['anchor']['delta_ns']
             - camera['beta_prime_ns']) / camera['alpha'] - first_event_us * 1000) / 1e9


def load_pose(path, timewindow_ms):
    import h5py
    path = Path(path)
    sequence = path.name.removesuffix('_ground_truth.csv')
    with h5py.File(path.parent / f'{sequence}.hdf5', 'r') as recording:
        events = recording['events']
        if isinstance(events, h5py.Group):
            key = next(k for k in ('t', 'ts', 'timestamp', 'timestamps') if k in events)
            first, last = float(events[key][0]), float(events[key][-1])
        else:
            key = next(k for k in ('t', 'ts', 'timestamp', 'timestamps') if k in events.dtype.names)
            first, last = float(events[0][key]), float(events[-1][key])
    sync = json.loads((path.parent / 'sync.json').read_text())
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['in_gap'] in ('0', 'false', 'False')]
    if len(rows) < 2:
        raise ValueError('Gardens Point requires at least two valid GPS samples')
    times = camera_seconds([int(row['time_laptop_ns']) for row in rows], sync, first)
    coords = np.array([[float(row['lat']), float(row['lon'])] for row in rows])
    valid = np.isfinite(times) & np.isfinite(coords).all(axis=1)
    times, coords = times[valid], coords[valid]
    if len(times) < 2:
        raise ValueError('Gardens Point requires at least two valid GPS samples')
    order = np.argsort(times, kind='stable')
    times, coords = times[order], coords[order]
    keep = np.r_[True, np.diff(times) > 0]
    times, coords = times[keep], coords[keep]
    if len(times) < 2:
        raise ValueError('Gardens Point requires at least two valid GPS samples')
    duration = (last - first) / 1e6
    if times[-1] < 0 or times[0] > duration:
        raise ValueError('GPS and camera recording do not overlap; check sync.json')
    if times[0] > 0 or times[-1] < duration:
        warnings.warn('Gardens Point GPS does not cover the entire recording; existing endpoint-clamping semantics apply.', stacklevel=2)
    step = timewindow_ms / 1000
    centers = (np.arange(int(np.ceil(duration / step))) + .5) * step
    return np.column_stack([coords, times]), 'geo', centers
