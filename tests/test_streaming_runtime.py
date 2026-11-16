"""Runner/ground truth integration, using local recordings and no baseline downloads."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

pytest.importorskip('pandas')
pytest.importorskip('scipy')
from datasets import groundtruths
from utils.configuration import CONFIG, read_config, write_config


def test_fractional_metres_do_not_mean_place_count():
    truth = np.zeros((5, 1), np.uint8)
    truth[2, 0] = 1
    result = groundtruths.create_GTtol_by_distance(truth, np.array([0, 1.5, 2, 2.7, 5]), .6)
    np.testing.assert_array_equal(result[:, 0], [0, 1, 1, 0, 0])


def test_generator_forwards_fractional_gps_tolerance(tmp_path, monkeypatch):
    seen = {}
    def generate(**kwargs):
        seen.update(kwargs)
        return 'truth.npy'
    monkeypatch.setattr(groundtruths, 'ground_truth_from_pose_only', generate)
    result = groundtruths.generate_ground_truth(
        {'data_path': str(tmp_path), 'gps_tolerance': .75}, {}, 'dataset', 'ref', 'qry',
        None, None, 50, gps_available=True)
    assert result == 'truth.npy'
    assert seen['gt_tolerance'] == .75


@pytest.mark.parametrize('fail', [False, True])
def test_runner_regenerates_ground_truth_and_cleans_up(tmp_path, monkeypatch, fail):
    import eventlab_run
    events = []
    class Baseline:
        def format_data(self, *args):
            events.append(('format', args[-1]))
            if fail:
                raise RuntimeError('format failed')
        def build_execute(self, *args):
            pass
        def run(self):
            events.append(('run',))
        def parse_results(self, *args):
            pass
        def cleanup(self):
            events.append(('cleanup',))
    config = read_config(CONFIG)
    config.update(timewindows=[50, 100], gps_tolerance=2.75, data_path=str(tmp_path))
    path = tmp_path / 'config.yaml'
    write_config(path, config)
    monkeypatch.setattr(sys, 'argv', ['eventlab_run.py', 'lens', 'brisbane_event', 'sunset2', 'sunrise', '--config', str(path)])
    monkeypatch.setattr(eventlab_run.utils, 'start', lambda *a: None)
    monkeypatch.setattr(eventlab_run, 'get_dataset', lambda *a: SimpleNamespace(timewindow_list=[50, 100]))
    monkeypatch.setattr(eventlab_run, 'get_baseline', lambda *a: Baseline())
    def generate(config, *args, timewindow, **kwargs):
        events.append(('gt', timewindow, config['gps_tolerance']))
        return str(tmp_path / 'truth.npy')
    monkeypatch.setattr(eventlab_run, 'generate_ground_truth', generate)
    monkeypatch.setenv('EVENTLAB_RUNTIME_CONFIG', '')
    monkeypatch.setenv('PYTHONPATH', '')
    if fail:
        with pytest.raises(RuntimeError, match='format failed'):
            eventlab_run.main()
        assert events == [('gt', 50, 2.75), ('format', 50), ('cleanup',)]
    else:
        eventlab_run.main()
        assert events == [('gt', 50, 2.75), ('format', 50), ('run',), ('cleanup',),
                          ('gt', 100, 2.75), ('format', 100), ('run',), ('cleanup',)]


def test_reconstruction_is_blocked_before_opening_recordings(monkeypatch):
    ecv = pytest.importorskip('eventcv')
    from utils.representations import open_reader
    def fail(*args, **kwargs):
        raise AssertionError('Disabled reconstruction must not open a recording')
    monkeypatch.setattr(ecv, 'open', fail)
    with pytest.raises(ValueError, match='Reconstruction is unavailable'):
        open_reader('recording.h5', 50, config={
            'representation': 'reconstruction', 'reconstruction': {'model': 'model.onnx'}})


@pytest.mark.parametrize('name', ['lens', 'ensemble', 'vprmethods', 'eventvlad', 'spikevpr', 'megaevent', 'eventgem', 'sparse_event'])
def test_adapter_preparation_accepts_configured_voxel(tmp_path, name):
    import eventcv as ecv
    # Skip constructors (repository updates / pretrained weight downloads).
    from baselines import lens, ensemble, vprmethods, eventvlad, spikevpr, megaevent, eventgem, sparse_event
    classes = {'lens': lens.LENS_baseline, 'ensemble': ensemble.ensemble_baseline,
               'vprmethods': vprmethods.vprmethods_baseline, 'eventvlad': eventvlad.eventvlad_baseline,
               'spikevpr': spikevpr.spikevpr_baseline, 'megaevent': megaevent.megaevent_baseline,
               'eventgem': eventgem.eventgem_baseline, 'sparse_event': sparse_event.sparse_event_baseline}
    recording = tmp_path / 'events.h5'
    ecv.save(ecv.from_numpy(np.array([[1, 1, 0, 1], [2, 2, 20000, 0]], dtype=float),
                           sensor_size=(8, 8), time_unit='us'), str(recording))
    config = {'representation': 'voxel', 'representation_options': {'bins': 5},
              'hot_pixel_filter': False, 'timewindows': [10], 'data_path': str(tmp_path / 'recordings')}
    dataset_config = {'dataset': {'name': 'fixture', 'resolution': [8, 8]}}
    def data(sequence):
        return SimpleNamespace(get_dataset_info=lambda: {'sequence_name': sequence, 'dataset_name': 'fixture',
                                                         'hdf5_path': str(recording)})
    adapter = object.__new__(classes[name])
    adapter.outdir = str(tmp_path / 'output')
    adapter.baseline_config = {'use_saliency': True, 'num_target_pixels': 1}
    try:
        adapter.format_data(config, dataset_config, data('ref'), data('qry'), 10)
        assert 'voxel-' in adapter.output_dir
        assert not Path(config['data_path']).exists() # No stored frame tree in the dataset directory.
        temporary = getattr(adapter, 'temp_dir', None)
    finally:
        adapter.cleanup()
    if temporary:
        assert not Path(temporary).exists()


def test_workbook_does_not_average_different_representations(tmp_path, monkeypatch):
    from baselines.EventBaselineLab import EventBaseline
    from openpyxl import load_workbook
    monkeypatch.chdir(tmp_path)
    baseline = EventBaseline()
    baseline.matrix_type = 'similarity'
    baseline.output_dir = tmp_path / 'matrices'
    baseline.output_dir.mkdir()
    np.save(baseline.output_dir / 'similarity.npy', np.eye(2))
    row = dict(timestamp_utc='2026-11-16', run_name='fixture', ref_query='ref_qry',
               array_name='similarity', n_references=2, n_queries=2,
               **{'R@1': .5, 'R@5': 1., 'R@10': 1., 'R@15': 1., 'R@20': 1., 'R@25': 1., 'aupr': .7})
    for representation in ('polarity', 'voxel'):
        baseline.config = {'representation': representation}
        baseline.save_results([row], {('ref_qry', 'similarity'): ([1., .5], [0., 1.])}, 'fixture', 'ref_qry')
    workbook = load_workbook(tmp_path / 'output/eventlab_results.xlsx', read_only=True)
    try:
        rows = list(workbook['Summary'].values)
        assert len(rows) == 3
        assert 'polarity' in rows[1][1]
        assert 'voxel' in rows[2][1]
    finally:
        workbook.close()
