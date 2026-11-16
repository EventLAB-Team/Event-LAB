"""Exercise configured EventCV inputs without downloading models or datasets."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from utils.configuration import CONFIG, REPRESENTATIONS, main, update_config, validate_runtime
from utils.representations import (ArrayFrame, ReconstructionReader, channels, image_uint8,
                                   open_reader, representation_tag, runtime_config)


@pytest.fixture
def recording(tmp_path):
    ecv = pytest.importorskip('eventcv')
    events = np.array([[1, 1, 0, 1], [2, 2, 1000, 0], [1, 2, 20000, 1]], dtype=float)
    path = tmp_path / 'events.h5'
    ecv.save(ecv.from_numpy(events, sensor_size=(8, 8), time_unit='us'), str(path))
    return path


@pytest.mark.parametrize('name', REPRESENTATIONS[:-1])
def test_all_dense_representations_and_model_channels(recording, name):
    reader = open_reader(recording, 10, config={'representation': name, 'hot_pixel_filter': False})
    assert reader.n_slices == 3
    frame = reader.slice(0)
    for count in (1, 2, 3, 10):
        result = channels(frame, count)
        assert result.shape == (count, 8, 8)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
    assert image_uint8(frame).shape == (3, 8, 8)


def test_representation_options_change_rendered_channels(recording):
    config = {'representation': 'voxel', 'representation_options': {'bins': 5}, 'hot_pixel_filter': False}
    assert open_reader(recording, 10, config=config).slice(0).numpy().shape[0] == 5


def test_png_adapter_uses_selected_representation(recording, tmp_path):
    from PIL import Image
    from utils.eventcv_frames import render_png_sequence
    config = {'representation': 'binary', 'hot_pixel_filter': False}
    destination = tmp_path / 'frames'
    times = render_png_sequence(recording, destination, timewindow_ms=10, config=config)
    assert times == [0, .01, .02]
    image = np.asarray(Image.open(destination / 'frame_000000.png'))
    assert image.shape == (8, 8, 3)
    assert set(np.unique(image)) == {0, 255}


def test_recurrent_reader_advances_skipped_windows_and_resets():
    class Model:
        total = 0
        def reset(self):
            self.total = 0
        def __call__(self, frame):
            self.total += frame.numpy()
            return self.total.copy()[None]
    reader = SimpleNamespace(n_slices=5, slice=lambda i: ArrayFrame(np.full((1, 2, 2), i + 1)))
    reconstruction = ReconstructionReader(reader, Model())
    assert reconstruction.slice(2).numpy()[0, 0, 0] == 6
    # Peeking the same frame must not advance the recurrent state.
    assert reconstruction.slice(2).numpy()[0, 0, 0] == 6
    assert reconstruction.batch([3, 4])[:, 0, 0, 0].tolist() == [10, 15]
    assert reconstruction.slice(0).numpy()[0, 0, 0] == 1
    assert reconstruction.slice(2).numpy()[0, 0, 0] == 6


def test_configuration_cli_and_subprocess_snapshot(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'config.yaml'
    path.write_text(CONFIG.read_text())
    main(['--config', str(path), '--representation', 'voxel', '--representation_options', '{bins: 5}',
          '--gps_tolerance', '2.75'])
    monkeypatch.setenv('EVENTLAB_RUNTIME_CONFIG', str(path))
    settings = runtime_config()
    assert settings['gps_tolerance'] == 2.75
    assert settings['representation_options'] == {'bins': 5}
    validate_runtime(settings)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        update_config(path, {'reconstruction': {'state_map': []}})
    assert path.read_bytes() == before


def test_reconstruction_is_unavailable():
    with pytest.raises(ValueError, match='Reconstruction is unavailable'):
        validate_runtime({'representation': 'reconstruction'})


def test_representation_cache_changes_with_options_and_model(tmp_path):
    config = {'representation': 'voxel', 'representation_options': {'bins': 5}}
    first = representation_tag(config)
    config['representation_options']['bins'] = 9
    assert representation_tag(config) != first
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'first')
    config = {'representation': 'reconstruction', 'reconstruction': {'model': str(model)}}
    first = representation_tag(config)
    model.write_bytes(b'replacement model')
    assert representation_tag(config) != first


def test_eventgem_factory_preserves_global_eventcv(monkeypatch):
    pytest.importorskip('torch')
    import eventcv
    from utils import eventgem_run
    dataset = SimpleNamespace(ecv=eventcv)
    monkeypatch.setitem(sys.modules, 'eventgem', SimpleNamespace(dataset=dataset))
    monkeypatch.setitem(sys.modules, 'eventgem.dataset', dataset)
    seen = []
    def reader(path, dt, offset):
        seen.append((path, dt, offset))
        return SimpleNamespace(n_slices=3, slice=lambda i: ArrayFrame(np.full((2, 4, 4), i)))
    monkeypatch.setattr(eventgem_run, 'open_reader', reader)
    original = eventcv.open
    eventgem_run.patch_representation()
    result = dataset.ecv.open('test.h5', dt_ms=50, offset=12, repr='mcts').with_repr('mcts', max_window_ms=50)
    assert result.slice(1).numpy().shape == (10, 4, 4)
    assert seen == [('test.h5', 50, 12)]
    assert eventcv.open is original


def test_image_adapter_scales_raw_counts_but_preserves_rgb():
    counts = np.array([[[0, 1, 2]]], dtype=np.uint64)
    np.testing.assert_array_equal(image_uint8(counts, 1), [[[0, 127, 255]]])
    rgb = np.array([[[10]], [[60]], [[200]]], dtype=np.uint8)
    np.testing.assert_array_equal(image_uint8(rgb), rgb)


def test_cli_reconstruction_is_disabled_without_changing_file(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(CONFIG.read_text())
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(['--config', str(path), '--representation', 'reconstruction'])
    assert error.value.code == 2
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='Reconstruction is unavailable'):
        update_config(path, {'reconstruction': {'model': 'model.onnx'}})
    assert path.read_bytes() == before
