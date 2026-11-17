"""Window manifests: sequences assembled from explicit event windows across many files.

Most datasets here are one recording per sequence, sliced into uniform windows.
Springfield-Event-VPR and NYC-Event-VPR are not: a reference "sequence" is a pooled
gallery drawn from many recordings, and a query sequence is a hand-picked (or randomly
held-out) set of windows. A manifest (``<sequence>.windows.json``) describes such a
sequence independently of the time window:

``files``
    Event recordings (paths relative to the manifest), each with an optional ``clock``
    mapping camera microseconds to the GPS track's nanosecond timebase
    (``time_ns = scale_ns_per_us * t_us + offset_ns``) and the ``track`` it uses.
``tracks``
    ``.npy`` arrays of ``[time_ns, lat, lon, valid]`` rows (paths relative to the manifest).
``tiles``
    ``{file, start_us, end_us}``: consecutive windows of the configured length from
    ``start_us``; a window is kept only when the track has a valid fix at its centre.
``anchors``
    ``{file, centre_us: [...], lat: [...], lon: [...]}``: one window per centre, of the
    configured length. ``lat``/``lon`` are optional; without them the track is used.

``resolve_windows`` turns a manifest plus a time window into concrete windows and
positions. Readers (``utils.representations.open_reader``) and ground truth both call it,
so frames and ground-truth rows always correspond one to one.
"""
import json
import os
from dataclasses import dataclass

import numpy as np

SUFFIX = '.windows.json'
FORMAT = 'eventlab-windows/1'
TRACK_COLUMNS = ('time_ns', 'lat', 'lon', 'valid')
CACHE_VERSION = 1  # bump when window resolution rules change


def is_manifest(path):
    return str(path).endswith(SUFFIX)


@dataclass
class Windows:
    files: list          # absolute file paths
    file_index: np.ndarray  # (N,) index into files
    start_us: np.ndarray    # (N,) int64, window start in the file's own timebase
    end_us: np.ndarray      # (N,) int64, exclusive
    lat: np.ndarray
    lon: np.ndarray
    sensor_size: tuple

    def __len__(self):
        return len(self.start_us)


def write_manifest(path, *, sensor_size, files, tracks=(), tiles=(), anchors=(), meta=None):
    """Write a manifest atomically. ``files``/``tracks`` entries hold absolute paths."""
    base = os.path.dirname(os.path.abspath(path))
    relative = lambda p: os.path.relpath(p, base)
    document = {
        'format': FORMAT,
        'sensor_size': list(sensor_size),
        'files': [{**entry, 'path': relative(entry['path'])} for entry in files],
        'tracks': [relative(track) for track in tracks],
        'tiles': list(tiles),
        'anchors': [{key: (np.asarray(value).tolist() if key != 'file' else value)
                     for key, value in anchor.items()} for anchor in anchors],
        'meta': meta or {},
    }
    os.makedirs(base, exist_ok=True)
    partial = path + '.part'
    with open(partial, 'w') as handle:
        json.dump(document, handle)
    os.replace(partial, path)
    return path


def save_track(path, time_ns, lat, lon, valid=None):
    """Store a GPS track as float64 ``[time_ns, lat, lon, valid]`` rows, sorted by time."""
    time_ns = np.asarray(time_ns, dtype=np.float64)
    order = np.argsort(time_ns, kind='stable')
    valid = np.ones_like(time_ns) if valid is None else np.asarray(valid, dtype=np.float64)
    table = np.stack([time_ns, np.asarray(lat, float), np.asarray(lon, float), valid], axis=1)[order]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, table)
    return path


def load_manifest(path):
    with open(path) as handle:
        document = json.load(handle)
    if document.get('format') != FORMAT:
        raise ValueError(f'{path} is not an EventLAB window manifest')
    base = os.path.dirname(os.path.abspath(path))
    for entry in document['files']:
        entry['path'] = os.path.normpath(os.path.join(base, entry['path']))
    document['tracks'] = [os.path.normpath(os.path.join(base, track)) for track in document['tracks']]
    return document


def _interpolate(track, time_ns):
    """Position at each time; invalid outside the track or next to a gap sample."""
    t, lat, lon, ok = track.T
    inside = (time_ns >= t[0]) & (time_ns <= t[-1])
    right = np.clip(np.searchsorted(t, time_ns), 1, len(t) - 1)
    valid = inside & (ok[right] > 0) & (ok[right - 1] > 0)
    return np.interp(time_ns, t, lat), np.interp(time_ns, t, lon), valid


def resolve_windows(path, timewindow_ms):
    """Concrete windows for one time window, cached beside the manifest."""
    dt_us = int(round(float(timewindow_ms) * 1000))
    if dt_us <= 0:
        raise ValueError('Time window must be positive')
    cache = f'{path[:-len(SUFFIX)]}.{dt_us}us.v{CACHE_VERSION}.npz'
    document = load_manifest(path)
    files = [entry['path'] for entry in document['files']]
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        data = np.load(cache)
        return Windows(files, data['file_index'], data['start_us'], data['end_us'],
                       data['lat'], data['lon'], tuple(document['sensor_size']))

    tracks = [np.load(track) for track in document['tracks']]
    parts = []

    def position(file, centre_us):
        entry = document['files'][file]
        clock, track = entry.get('clock'), entry.get('track')
        if clock is None or track is None:
            raise ValueError(f"{entry['path']} has no clock/track to position its windows")
        time_ns = clock['scale_ns_per_us'] * centre_us.astype(np.float64) + clock['offset_ns']
        return _interpolate(tracks[track], time_ns)

    for tile in document['tiles']:
        # EventCV's n_slices convention: ceil(span / dt), keeping the final partial window.
        # With it the Springfield gallery is exactly the published 132,569 windows at 50 ms.
        span = int(tile['end_us']) - 1 - int(tile['start_us'])
        count = -(-span // dt_us)
        if count <= 0:
            continue
        start = int(tile['start_us']) + dt_us * np.arange(count, dtype=np.int64)
        lat, lon, valid = position(tile['file'], start + dt_us / 2)
        parts.append((np.full(valid.sum(), tile['file']), start[valid], lat[valid], lon[valid]))

    for anchor in document['anchors']:
        centre = np.asarray(anchor['centre_us'], dtype=np.float64)
        start = np.round(centre - dt_us / 2).astype(np.int64)
        if 'lat' in anchor:
            lat, lon = np.asarray(anchor['lat'], float), np.asarray(anchor['lon'], float)
            valid = np.isfinite(lat) & np.isfinite(lon)
        else:
            lat, lon, valid = position(anchor['file'], centre)
        parts.append((np.full(valid.sum(), anchor['file']), start[valid], lat[valid], lon[valid]))

    if not parts:
        raise ValueError(f'{path} resolves to no windows at {timewindow_ms} ms')
    file_index, start_us, lat, lon = (np.concatenate(values) for values in zip(*parts))
    file_index = file_index.astype(np.int32)
    end_us = start_us + dt_us
    np.savez(cache + '.part.npz', file_index=file_index, start_us=start_us, end_us=end_us, lat=lat, lon=lon)
    os.replace(cache + '.part.npz', cache)
    return Windows(files, file_index, start_us, end_us, lat, lon, tuple(document['sensor_size']))


def local_metres(lat, lon, origin):
    """Equirectangular east/north metres about ``origin``; exact enough at VPR tolerances."""
    lat0, lon0 = origin
    radius = 6_371_008.8
    east = np.radians(np.asarray(lon) - lon0) * radius * np.cos(np.radians(lat0))
    north = np.radians(np.asarray(lat) - lat0) * radius
    return np.stack([east, north], axis=1)


def distance_ground_truth(reference, query, tolerance_m):
    """(R, Q) uint8 matrix: 1 where a reference window lies within ``tolerance_m`` of a query."""
    from scipy.spatial import cKDTree

    origin = (float(np.mean(reference.lat)), float(np.mean(reference.lon)))
    tree = cKDTree(local_metres(reference.lat, reference.lon, origin))
    neighbours = tree.query_ball_point(local_metres(query.lat, query.lon, origin), r=float(tolerance_m))
    gt = np.zeros((len(reference), len(query)), dtype=np.uint8)
    for column, rows in enumerate(neighbours):
        gt[rows, column] = 1
    return gt
