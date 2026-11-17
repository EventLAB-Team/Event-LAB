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


@pytest.mark.parametrize('name', ['count', 'polarity', 'redblue', 'mcts'])
def test_all_dense_representations_and_model_channels(recording, name):
    reader = open_reader(recording, 10, config={'representation': 'Events', 'hot_pixel_filter': False}, event_representation=name)
    assert reader.n_slices == 3
    frame = reader.slice(0)
    for count in (1, 2, 3, 10):
        result = channels(frame, count)
        assert result.shape == (count, 8, 8)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
    assert image_uint8(frame).shape == (3, 8, 8)


def test_representation_options_change_rendered_channels(recording):
    config = {'representation': 'Events', 'hot_pixel_filter': False}
    assert open_reader(recording, 10, config=config, event_representation='voxel', event_options={'bins': 5}).slice(0).numpy().shape[0] == 5


def test_png_adapter_uses_selected_representation(recording, tmp_path):
    from PIL import Image
    from utils.eventcv_frames import render_png_sequence
    config = {'representation': 'Events', 'hot_pixel_filter': False}
    destination = tmp_path / 'frames'
    times = render_png_sequence(recording, destination, timewindow_ms=10, config=config)
    assert times == [0, .01, .02]
    image = np.asarray(Image.open(destination / 'frame_000000.png'))
    assert image.shape == (8, 8, 3)
    assert set(np.unique(image)) <= {0, 255}


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
    main(['--config', str(path), '--representation', 'Events',
          '--gps_tolerance', '2.75'])
    monkeypatch.setenv('EVENTLAB_RUNTIME_CONFIG', str(path))
    settings = runtime_config()
    assert settings['gps_tolerance'] == 2.75
    assert settings['representation'] == 'Events'
    validate_runtime(settings)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        update_config(path, {'reconstruction': {'state_map': []}})
    assert path.read_bytes() == before


def test_reconstruction_mode_is_valid():
    validate_runtime({'representation': 'Reconstruction'})


def test_representation_cache_changes_with_mode_and_weights(tmp_path, monkeypatch):
    monkeypatch.setattr('utils.representations.ROOT', tmp_path)
    events = representation_tag({'representation': 'Events'})
    config = {'representation': 'Reconstruction'}
    assert representation_tag(config) != events
    model = tmp_path / 'e2vid' / 'e2vid.safetensors'
    model.parent.mkdir()
    model.write_bytes(b'first')
    first = representation_tag(config)
    model.write_bytes(b'replacement model')
    assert representation_tag(config) != first


def test_eventgem_factory_preserves_native_eventcv(monkeypatch):
    pytest.importorskip('torch')
    import eventcv
    from utils import eventgem_run
    dataset = SimpleNamespace(ecv=eventcv)
    monkeypatch.setitem(sys.modules, 'eventgem', SimpleNamespace(dataset=dataset))
    monkeypatch.setitem(sys.modules, 'eventgem.dataset', dataset)
    monkeypatch.setattr(eventgem_run, 'runtime_config', lambda: {'representation': 'Events'})
    eventgem_run.patch_representation()
    assert dataset.ecv is eventcv


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
        main(['--config', str(path), '--representation', 'unknown'])
    assert error.value.code == 2
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='Unknown runtime field'):
        update_config(path, {'reconstruction': {'model': 'model.onnx'}})
    assert path.read_bytes() == before


def test_e2vid_mode_feeds_raw_windows_and_local_weights(tmp_path, monkeypatch):
    import eventcv
    import utils.representations as representations
    weights = tmp_path / 'e2vid' / 'e2vid.safetensors'
    weights.parent.mkdir()
    weights.write_bytes(b'test weights')
    monkeypatch.setattr(representations, 'ROOT', tmp_path)
    seen = []
    raw = SimpleNamespace(n_slices=3, slice=lambda i: i)
    monkeypatch.setattr(eventcv, 'open', lambda *args, **kwargs: raw)
    class E2VID:
        def __init__(self, path):
            assert path == str(weights)
        def __call__(self, window):
            seen.append(window)
            return np.full((4, 4), window / 3, dtype=np.float32)
        def reset(self):
            seen.append('reset')
    monkeypatch.setattr(eventcv, 'E2VID', E2VID, raising=False)
    reader = open_reader('recording.h5', 10, config={'representation': 'Reconstruction'})
    assert reader.slice(2).numpy().shape == (1, 4, 4)
    assert seen == [0, 1, 2]
    reader.slice(0)
    assert seen == [0, 1, 2, 'reset', 0]


def test_global_event_format_overrides_are_rejected():
    with pytest.raises(ValueError, match='choose Events, Reconstruction'):
        validate_runtime({'representation': 'voxel'})
    with pytest.raises(ValueError, match='Retired'):
        validate_runtime({'representation': 'Events', 'representation_options': {'bins': 9}})
