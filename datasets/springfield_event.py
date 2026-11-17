"""Springfield-Event-VPR: download recordings and build window manifests.

Protocol (dataset README, Hines et al. arXiv:2609.21219):

* Reference ``database``: every partition of the seven database sessions is cut into
  consecutive windows starting at its first event; windows whose centre has no valid
  synchronised GPS are dropped, and the rest are pooled into one gallery
  (132,569 windows at 50 ms).
* Queries: the windows listed in ``queries/query_slices.csv`` (5,557 = 3,169 day +
  2,388 dawn). Slice ``i`` of a session is the 50 ms window ``[t0 + 50 i, t0 + 50 (i+1))``
  from the session's first event; other time windows keep the same centre.
* Positions: camera microseconds -> laptop epoch via ``derived/sync.json``
  (``alpha``, ``beta_prime_ns``, ``anchor.delta_ns``), then ``derived/gps_interp.csv``
  interpolated at each window centre. Ground truth is GPS distance <= ``gps_tolerance``.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from utils.event_windows import save_track, write_manifest

REPO = 'AdamHines/springfield-event-vpr'
SLICE_US = 50_000  # query_slices.csv indexes 50 ms windows
SEQUENCES = {'database': ('database',), 'queries': ('day', 'dawn'),
             'query_day': ('day',), 'query_dawn': ('dawn',)}
METADATA = ('partitions.jsonl', 'derived/sync.json', 'derived/gps_interp.csv')


def _root(config):
    return Path(config['data_path']) / 'springfield_event'


def _repo(data_config):
    return data_config.get('other', {}).get('repo', REPO)


def _download(data_config, raw, patterns, request_input=False, describe=None):
    from huggingface_hub import HfApi, snapshot_download

    if request_input and describe:
        size = sum(entry.size or 0 for entry in HfApi().list_repo_tree(
            _repo(data_config), repo_type='dataset', recursive=True, path_in_repo=describe[0])
            if hasattr(entry, 'size') and describe[1](entry.path))
        answer = input(f'Download {size / 1024**3:.1f} GiB of Springfield-Event-VPR recordings? [Y/n] ')
        if answer.strip().lower() not in ('', 'y', 'yes'):
            raise RuntimeError('Download cancelled')
    snapshot_download(_repo(data_config), repo_type='dataset', local_dir=str(raw),
                      allow_patterns=list(patterns))


def _partitions(session_dir):
    """Kept partitions in index order: (file name, first event us, last event us)."""
    closed = {}
    with open(session_dir / 'partitions.jsonl') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('type') == 'close' and not record.get('discarded'):
                closed[record['idx']] = (record['file'], int(record['cam_t_first_us']), int(record['cam_t_last_us']))
    return [closed[index] for index in sorted(closed)]


def _session_clock(root, session_dir, track_index):
    """File clock entry plus the session's GPS track, saved as an .npy beside the manifests."""
    sync = json.loads((session_dir / 'derived' / 'sync.json').read_text())
    gps = pd.read_csv(session_dir / 'derived' / 'gps_interp.csv')
    track = save_track(root / 'tracks' / f'{session_dir.name}.npy',
                       gps['time_laptop_ns'], gps['lat'], gps['lon'], gps['in_gap'] == 0)
    clock = {'scale_ns_per_us': sync['camera']['alpha'] * 1000.0,
             'offset_ns': sync['camera']['beta_prime_ns'] + sync['anchor']['delta_ns']}
    return track, {'clock': clock, 'track': track_index}


def query_sessions(raw, sweeps):
    slices = pd.read_csv(raw / 'queries' / 'query_slices.csv')
    return slices[slices['sweep'].isin(sweeps)]


def build_manifest(config, data_config, sequence):
    """Write ``<sequence>.windows.json`` from downloaded metadata (no event files are read)."""
    root = _root(config)
    raw = root / 'raw'
    files, tracks, tiles, anchors = [], [], [], []
    if sequence == 'database':
        sessions = sorted(path for path in (raw / 'database').iterdir() if (path / 'partitions.jsonl').is_file())
        for session_dir in sessions:
            track, clock = _session_clock(root, session_dir, len(tracks))
            tracks.append(track)
            for name, first_us, last_us in _partitions(session_dir):
                tiles.append({'file': len(files), 'start_us': first_us, 'end_us': last_us + 1})
                files.append({'path': str(session_dir / name), **clock})
    else:
        slices = query_sessions(raw, SEQUENCES[sequence])
        for (sweep, session), rows in slices.groupby(['sweep', 'session'], sort=False):
            session_dir = raw / 'queries' / f'query_{sweep}' / session
            (name, first_us, _), = _partitions(session_dir)
            track, clock = _session_clock(root, session_dir, len(tracks))
            tracks.append(track)
            centre = first_us + (rows['slice_index'].to_numpy(np.int64) + 0.5) * SLICE_US
            anchors.append({'file': len(files), 'centre_us': centre})
            files.append({'path': str(session_dir / name), **clock})
    destination = root / sequence / f'{sequence}.windows.json'
    write_manifest(str(destination), sensor_size=data_config['dataset']['resolution'],
                   files=files, tracks=tracks, tiles=tiles, anchors=anchors,
                   meta={'dataset': 'springfield_event', 'sequence': sequence, 'repo': _repo(data_config)})
    logger.info(f'Wrote {destination} ({len(files)} recordings)')
    return destination


def prepare(config, data_config, sequence):
    """Download the metadata and recordings a sequence needs, then write its manifest."""
    if sequence not in SEQUENCES:
        raise ValueError(f'Unknown Springfield sequence: {sequence}')
    raw = _root(config) / 'raw'
    scope = 'database' if sequence == 'database' else 'queries'
    sweeps = SEQUENCES[sequence]
    session_globs = (['database/*'] if sequence == 'database'
                     else [f'queries/query_{sweep}/*' for sweep in sweeps])
    _download(data_config, raw, ['queries/query_slices.csv', 'queries/sessions.txt'] +
              [f'{glob}/{name}' for glob in session_globs for name in METADATA])

    if sequence == 'database':
        wanted = [f'database/{s.name}/{name}' for s in sorted((raw / 'database').iterdir())
                  if (s / 'partitions.jsonl').is_file() for name, _, _ in _partitions(s)]
    else:
        slices = query_sessions(raw, sweeps)
        wanted = []
        for sweep, session in slices[['sweep', 'session']].drop_duplicates().itertuples(index=False):
            session_dir = raw / 'queries' / f'query_{sweep}' / session
            wanted += [f'queries/query_{sweep}/{session}/{name}' for name, _, _ in _partitions(session_dir)]
    missing = [path for path in wanted if not (raw / path).is_file()]
    if missing:
        logger.info(f'Downloading {len(missing)} Springfield recordings for {sequence}')
        wanted_set = set(missing)
        _download(data_config, raw, missing, config.get('request_input', False),
                  describe=(scope, lambda path: path in wanted_set))
    return build_manifest(config, data_config, sequence)
