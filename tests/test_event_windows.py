"""Window manifests: resolution rules, the manifest reader, distance ground truth and the
Springfield / NYC-Event-VPR builders, all on small synthetic fixtures (no network)."""
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from utils.event_windows import (distance_ground_truth, resolve_windows, save_track,
                                 write_manifest)

METRES_PER_DEG = 111_195.0


def write_h5(path, t_us, size=(64, 48), seed=0):
    rng = np.random.default_rng(seed)
    n = len(t_us)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as handle:
        events = handle.create_group('events')
        events['t'] = np.asarray(t_us, dtype=np.int64)
        events['x'] = rng.integers(0, size[0], n).astype(np.uint16)
        events['y'] = rng.integers(0, size[1], n).astype(np.uint16)
        events['p'] = rng.integers(0, 2, n).astype(np.uint8)
        handle.attrs.update(width=size[0], height=size[1], timestamp_scale_ms=0.001)
    return path


def test_tiles_keep_final_partial_window_and_drop_windows_without_gps(tmp_path):
    clock = {'scale_ns_per_us': 1000.0, 'offset_ns': 0.0}
    # Track covers 0.0-0.30 s; the samples at 0.12 s and 0.13 s are flagged as a gap.
    times = np.arange(0, 0.31, 0.01) * 1e9
    valid = ~((times > 0.115e9) & (times < 0.135e9))
    track = save_track(tmp_path / 'track.npy', times, np.zeros_like(times), np.zeros_like(times), valid)
    manifest = write_manifest(
        str(tmp_path / 'db' / 'db.windows.json'), sensor_size=(64, 48),
        files=[{'path': str(tmp_path / 'a.h5'), 'clock': clock, 'track': 0}], tracks=[track],
        tiles=[{'file': 0, 'start_us': 0, 'end_us': 260_001}])
    windows = resolve_windows(manifest, 50)
    # Span 260 ms -> ceil(260/50) = 6 windows (the last is partial); the one centred at
    # 0.125 s touches the gap and 0.275 s lies inside the track, so 5 remain.
    assert list(windows.start_us) == [0, 50_000, 150_000, 200_000, 250_000]
    assert np.all(windows.end_us - windows.start_us == 50_000)


def test_anchors_keep_centres_across_time_windows(tmp_path):
    manifest = write_manifest(
        str(tmp_path / 'q' / 'q.windows.json'), sensor_size=(64, 48),
        files=[{'path': str(tmp_path / 'q.h5')}],
        anchors=[{'file': 0, 'centre_us': [125_000, 525_000], 'lat': [0.0, 1.0], 'lon': [2.0, 3.0]}])
    short, long = resolve_windows(manifest, 50), resolve_windows(manifest, 100)
    assert list(short.start_us) == [100_000, 500_000] and list(long.start_us) == [75_000, 475_000]
    assert list(long.lat) == [0.0, 1.0] and list(long.lon) == [2.0, 3.0]


def test_distance_ground_truth_uses_tolerance(tmp_path):
    def windows(north_m):
        north_m = np.asarray(north_m, dtype=float)
        manifest = write_manifest(
            str(tmp_path / f'{len(north_m)}' / 's.windows.json'), sensor_size=(1, 1),
            files=[{'path': 'x.h5'}],
            anchors=[{'file': 0, 'centre_us': np.arange(len(north_m)) * 1e6,
                      'lat': north_m / METRES_PER_DEG, 'lon': np.zeros_like(north_m)}])
        return resolve_windows(manifest, 50)
    reference, query = windows([0, 20, 40, 100]), windows([10, 95, 500])
    gt = distance_ground_truth(reference, query, 25.0)
    assert gt.shape == (4, 3)
    assert gt[:, 0].tolist() == [1, 1, 0, 0]
    assert gt[:, 1].tolist() == [0, 0, 0, 1]
    assert not gt[:, 2].any()


def test_manifest_reader_slices_each_window_from_its_own_file(tmp_path):
    from utils.configuration import CONFIG, read_config
    from utils.representations import open_reader

    rng = np.random.default_rng(1)
    first = write_h5(tmp_path / 'a.h5', np.sort(rng.integers(1_000, 1_001_000, 20_000)), seed=1)
    second = write_h5(tmp_path / 'b.h5', np.sort(rng.integers(5_000, 505_000, 10_000)), seed=2)
    manifest = write_manifest(
        str(tmp_path / 'm' / 'm.windows.json'), sensor_size=(64, 48),
        files=[{'path': str(first)}, {'path': str(second)}],
        anchors=[{'file': 0, 'centre_us': [301_000], 'lat': [0.0], 'lon': [0.0]},
                 {'file': 1, 'centre_us': [105_000, 405_000], 'lat': [0.0, 0.0], 'lon': [0.0, 0.0]}])
    config = {**read_config(CONFIG), 'hot_pixel_filter': False}
    reader = open_reader(manifest, 50, config=config, event_representation='count')
    assert reader.n_slices == 3 and reader.fixed_places
    with h5py.File(second) as handle:
        t = handle['events/t'][:]
    frame = np.asarray(reader.slice(2).numpy())
    assert frame.sum() == ((t >= 380_000) & (t < 430_000)).sum()
    assert reader.batch([0, 2]).shape[0] == 2
    assert np.array_equal(np.asarray(reader.slice(-1).numpy()), frame)


def test_springfield_manifests_follow_slices_and_partitions(tmp_path):
    import datasets.springfield_event as springfield

    raw = tmp_path / 'springfield_event' / 'raw'
    sync = {'camera': {'alpha': 1.0, 'beta_prime_ns': 0.0}, 'anchor': {'delta_ns': 1e18}}

    def session(directory, partitions):
        (directory / 'derived').mkdir(parents=True)
        (directory / 'derived' / 'sync.json').write_text(json.dumps(sync))
        seconds = np.arange(0, 3.01, 0.1)
        pd.DataFrame({'time_laptop_ns': (1e18 + seconds * 1e9).astype(np.int64),
                      'lat': seconds * 10 / METRES_PER_DEG, 'lon': 0.0, 'alt': 0.0,
                      'in_gap': 0}).to_csv(directory / 'derived' / 'gps_interp.csv', index=False)
        records = [json.dumps({'type': 'close', 'idx': i, 'file': name, 'cam_t_first_us': first,
                               'cam_t_last_us': last, 'discarded': discarded})
                   for i, (name, first, last, discarded) in enumerate(partitions)]
        (directory / 'partitions.jsonl').write_text('\n'.join(records))

    session(raw / 'database' / 'S1', [('events_p0000.h5', 1_000, 1_000_000, False),
                                      ('events_p0001.h5', 1_000_001, 2_000_000, False),
                                      ('events_p0002.h5', 2_000_001, 2_500_000, True)])
    session(raw / 'queries' / 'query_day' / 'Q1', [('events_p0000.h5', 52_000, 2_000_000, False)])
    session(raw / 'queries' / 'query_dawn' / 'Q2', [('events_p0000.h5', 704, 2_000_000, False)])
    pd.DataFrame({'session': ['Q1', 'Q1', 'Q2'], 'sweep': ['day', 'day', 'dawn'],
                  'slice_index': [0, 10, 3], 'psi_deg': 0, 'x_enu_m': 0, 'y_enu_m': 0}
                 ).to_csv(raw / 'queries' / 'query_slices.csv', index=False)

    config = {'data_path': str(tmp_path)}
    data_config = {'dataset': {'resolution': [1280, 720]}}
    database = resolve_windows(str(springfield.build_manifest(config, data_config, 'database')), 50)
    assert len(database.files) == 2  # the discarded partition is skipped
    assert len(database) == 20 + 20  # ceil(998.999 ms / 50) per partition
    queries = resolve_windows(str(springfield.build_manifest(config, data_config, 'queries')), 50)
    assert list(queries.start_us) == [52_000, 552_000, 150_704]
    # GPS moves 10 m/s north from t = 1e18 ns; slice 10 of Q1 is centred at 0.577 s.
    assert queries.lat[1] * METRES_PER_DEG == pytest.approx(5.77, abs=1e-3)
    day = resolve_windows(str(springfield.build_manifest(config, data_config, 'query_day')), 50)
    assert len(day) == 2


def test_nyc_places_are_split_ten_percent_and_reproducibly(tmp_path, monkeypatch):
    import datasets.nyc_event as nyc

    root = tmp_path / 'nyc_event'
    events = root / 'raw' / 'sensor_data_2023-02-14_15-06-30' / 'events'
    events.mkdir(parents=True)
    (events / '.complete').touch()
    for name in ('event_2023-02-14_15-06-33.raw', 'event_2023-02-14_15-16-33.raw'):
        (events / name).touch()
    gps_dir = root / 'hub' / nyc.RAW_DIR / 'sensor_data_2023-02-14_15-06-30'
    gps_dir.mkdir(parents=True)
    stamps = pd.date_range('2023-02-14 15:06:30', periods=1300, freq='1s')
    pd.DataFrame({'Longitude': -73.99, 'Latitude': 40.66 + np.arange(1300) * 1e-5, 'HeadMotion': 0,
                  'Timestamp': [s.strftime('%Y-%m-%d_%H-%M-%S_%f') for s in stamps]}
                 ).to_csv(gps_dir / 'GPS_data_2023-02-14_15-06-31.csv', index=False)
    # Ten-minute chunks whose camera clocks start at arbitrary values.
    monkeypatch.setattr(nyc, '_recordings', lambda directory, size: [
        (path, nyc._wall_ns(nyc.NAME_TIME.search(path.name).group(1), '%Y-%m-%d_%H-%M-%S'),
         7_000 + 1_000_000_000 * i, 7_000 + 1_000_000_000 * i + 600_000_000)
        for i, path in enumerate(sorted(directory.glob('*.raw')))])

    config = {'data_path': str(tmp_path)}
    data_config = {'dataset': {'resolution': [1280, 720]}, 'other': {'query_fraction': 0.1, 'seed': 0}}
    table = nyc.places(config, data_config)
    assert len(table) == 1200  # fixes before the first chunk and after the last are dropped
    first = table.iloc[0]
    assert first['centre_us'] == 7_000 and first['lat'] == pytest.approx(40.66 + 3e-5)
    written = nyc.build_manifests(config, data_config, table)
    database, queries = (resolve_windows(str(written[name]), 50) for name in ('database', 'queries'))
    assert (len(database), len(queries)) == (1080, 120)
    pairs = set(zip(np.round(database.lat, 7), database.start_us)) & set(zip(np.round(queries.lat, 7), queries.start_us))
    assert not pairs
    again = nyc.build_manifests(config, data_config, table)
    assert np.array_equal(resolve_windows(str(again['queries']), 100).lat, queries.lat)


def test_windows_datasets_report_status_from_manifest(tmp_path):
    from datasets.prepare import sequence_download_status
    from utils.configuration import read_config, ROOT

    config = {'data_path': str(tmp_path)}
    data_config = read_config(ROOT / 'datasets' / 'springfield_event.yaml')
    assert sequence_download_status(config, data_config, 'springfield_event', 'queries') == 'Missing'
    (tmp_path / 'springfield_event' / 'raw').mkdir(parents=True)
    assert sequence_download_status(config, data_config, 'springfield_event', 'queries') == 'Partial'
    write_manifest(str(tmp_path / 'springfield_event' / 'queries' / 'queries.windows.json'),
                   sensor_size=(1280, 720), files=[])
    assert sequence_download_status(config, data_config, 'springfield_event', 'queries') == 'Downloaded'
