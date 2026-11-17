"""NYC-Event-VPR: raw Prophesee recordings, a random 10 % place hold-out, GPS ground truth.

Protocol (Pan et al., ICRA 2025, as used by Hines et al. arXiv:2609.21219): a random
10 % of places are queries and the remaining 90 % the reference database
(``event_dataset_generation_vg/generator.py`` in ai4ce/NYC-Event-VPR samples
``frac=0.1`` as queries and keeps the rest as the database).

Places are the GPS fixes (~1 Hz) that fall inside a recording. Each place becomes one
event window of the configured length centred on the fix time, so the split is fixed
across time windows and only the window length changes.

Timing: the recorder names each raw file after the wall-clock time logging started
(``data_<ts>.raw``, or ``event_<ts>.raw`` for 10-minute chunks) and timestamps GPS rows
with the same local clock, so a fix at wall time ``w`` sits at camera time
``t_first + (w - file_start)``. File names have one-second resolution.
"""
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from utils.event_windows import write_manifest

REPO = 'ai4ce/NYC-Event-VPR'
RAW_DIR = 'NYC-Event-VPR_raw_data'
SEQUENCES = ('database', 'queries')
NAME_TIME = re.compile(r'(?:data|event)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.raw$')


def _root(config):
    return Path(config['data_path']) / 'nyc_event'


def _options(data_config):
    other = data_config.get('other', {})
    return (other.get('repo', REPO), float(other.get('query_fraction', 0.1)),
            int(other.get('seed', 0)), bool(other.get('keep_archives', False)))


def _wall_ns(text, fmt):
    # Both clocks are the same naive local time; read them as UTC so DST never shifts one.
    stamp = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
    return np.int64(int(round(stamp.timestamp() * 1e6))) * 1000


def _remote_sessions(repo):
    """{session folder: {'gps': path, 'zip': path or None, 'zip_size': bytes}} from the Hub."""
    from huggingface_hub import HfApi

    sessions = {}
    for entry in HfApi().list_repo_tree(repo, repo_type='dataset', path_in_repo=RAW_DIR, recursive=True):
        parts = entry.path.split('/')
        if len(parts) != 3:
            continue
        info = sessions.setdefault(parts[1], {'gps': None, 'zip': None, 'zip_size': 0})
        if parts[2].startswith('GPS_data_') and parts[2].endswith('.csv'):
            info['gps'] = entry.path
        elif parts[2].startswith('data_') and parts[2].endswith('.zip'):
            info['zip'], info['zip_size'] = entry.path, getattr(entry, 'size', 0) or 0
    return {name: info for name, info in sorted(sessions.items()) if info['gps'] and info['zip']}


def _extract(archive, destination):
    """Extract only the .raw recordings, flattened into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = Path(member.filename).name
            if not name.endswith('.raw'):
                continue
            target = destination / name
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            with bundle.open(member) as source, open(str(target) + '.part', 'wb') as sink:
                shutil.copyfileobj(source, sink, length=16 * 1024 * 1024)
            Path(str(target) + '.part').replace(target)


def download(config, data_config):
    """Fetch every session's GPS log and raw recording archive, extracting the .raw files."""
    from huggingface_hub import hf_hub_download

    repo, _, _, keep_archives = _options(data_config)
    raw = _root(config) / 'raw'
    sessions = _remote_sessions(repo)
    pending = [name for name in sessions if not (raw / name / 'events' / '.complete').is_file()]
    if pending and config.get('request_input', False):
        size = sum(sessions[name]['zip_size'] for name in pending)
        answer = input(f'Download {size / 1024**3:.1f} GiB of NYC-Event-VPR raw recordings '
                       f'({len(pending)} archives, extracted locally)? [Y/n] ')
        if answer.strip().lower() not in ('', 'y', 'yes'):
            raise RuntimeError('Download cancelled')
    for name, info in sessions.items():
        hf_hub_download(repo, info['gps'], repo_type='dataset', local_dir=str(raw.parent / 'hub'))
        if name not in pending:
            continue
        logger.info(f'NYC-Event-VPR {name}: downloading {info["zip_size"] / 1024**3:.1f} GiB')
        archive = Path(hf_hub_download(repo, info['zip'], repo_type='dataset', local_dir=str(raw.parent / 'hub')))
        _extract(archive, raw / name / 'events')
        (raw / name / 'events' / '.complete').touch()
        if not keep_archives:
            archive.unlink()
    return sessions


def _recordings(events_dir, sensor_size):
    """[(path, wall_start_ns, t_first_us, t_last_us)] for each .raw file in a session."""
    import eventcv as ecv

    recordings = []
    for path in sorted(events_dir.glob('*.raw')):
        match = NAME_TIME.search(path.name)
        if not match:
            continue
        first_ms, last_ms = ecv.open(str(path), sensor_size=tuple(sensor_size)).time_span_ms
        recordings.append((path, _wall_ns(match.group(1), '%Y-%m-%d_%H-%M-%S'),
                           int(round(first_ms * 1000)), int(round(last_ms * 1000))))
    return recordings


def places(config, data_config):
    """One row per GPS fix inside a recording: file, centre_us, lat, lon, session."""
    root = _root(config)
    rows = []
    sensor_size = data_config['dataset']['resolution']
    for session_dir in sorted((root / 'raw').iterdir()):
        events = session_dir / 'events'
        gps_files = sorted((root / 'hub' / RAW_DIR / session_dir.name).glob('GPS_data_*.csv'))
        if not (events / '.complete').is_file() or not gps_files:
            continue
        gps = pd.read_csv(gps_files[0])
        gps = gps[np.isfinite(gps['Latitude']) & np.isfinite(gps['Longitude'])
                  & (gps['Latitude'].abs() > 1e-6) & (gps['Longitude'].abs() > 1e-6)]
        wall = np.array([_wall_ns(stamp, '%Y-%m-%d_%H-%M-%S_%f') for stamp in gps['Timestamp']], dtype=np.int64)
        for path, start_ns, first_us, last_us in _recordings(events, sensor_size):
            centre = first_us + (wall - start_ns) / 1000.0
            inside = (centre >= first_us) & (centre < last_us)
            for index in np.flatnonzero(inside):
                rows.append((str(path), float(centre[index]), float(gps['Latitude'].iloc[index]),
                             float(gps['Longitude'].iloc[index]), session_dir.name))
    if not rows:
        raise RuntimeError('No NYC-Event-VPR GPS fixes fall inside a downloaded recording')
    return pd.DataFrame(rows, columns=['file', 'centre_us', 'lat', 'lon', 'session'])


def _anchors(table):
    files, anchors = [], []
    for path, rows in table.groupby('file', sort=False):
        anchors.append({'file': len(files), 'centre_us': rows['centre_us'].to_numpy(),
                        'lat': rows['lat'].to_numpy(), 'lon': rows['lon'].to_numpy()})
        files.append({'path': path})
    return files, anchors


def build_manifests(config, data_config, table):
    """Seeded random hold-out of ``query_fraction`` of places; both manifests at once."""
    _, fraction, seed, _ = _options(data_config)
    order = np.random.default_rng(seed).permutation(len(table))
    is_query = np.zeros(len(table), dtype=bool)
    is_query[order[:int(round(fraction * len(table)))]] = True
    root = _root(config)
    written = {}
    for sequence, mask in (('database', ~is_query), ('queries', is_query)):
        files, anchors = _anchors(table[mask])
        destination = root / sequence / f'{sequence}.windows.json'
        write_manifest(str(destination), sensor_size=data_config['dataset']['resolution'],
                       files=files, anchors=anchors,
                       meta={'dataset': 'nyc_event', 'sequence': sequence, 'seed': seed,
                             'query_fraction': fraction, 'places': int(mask.sum()),
                             'total_places': len(table)})
        written[sequence] = destination
        logger.info(f'Wrote {destination}: {int(mask.sum())} places from {len(files)} recordings')
    return written


def prepare(config, data_config, sequence):
    if sequence not in SEQUENCES:
        raise ValueError(f'Unknown NYC-Event-VPR sequence: {sequence}')
    download(config, data_config)
    return build_manifests(config, data_config, places(config, data_config))[sequence]
