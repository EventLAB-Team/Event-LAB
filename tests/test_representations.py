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


def test_recurrent_reader_renders_chunks_in_order_and_resets():
    class Model:
        total = 0
        def reset(self):
            self.total = 0
            calls.append('reset')
    model, calls = Model(), []
    def render(start, count):
        # Recurrent stand-in: frame i is the running sum of windows 1..i+1.
        calls.append((start, count))
        frames = []
        for i in range(start, start + count):
            model.total += i + 1
            frames.append(np.full((2, 2), model.total))
        return np.stack(frames)
    reconstruction = ReconstructionReader(5, model, render, chunk=2)
    assert reconstruction.slice(2).numpy().shape == (1, 2, 2)
    assert reconstruction.slice(2).numpy()[0, 0, 0] == 6
    assert calls == [(0, 2), (2, 2)]
    # Revisiting the held chunk must not advance the recurrent state.
    assert reconstruction.slice(3).numpy()[0, 0, 0] == 10
    assert reconstruction.slice(2).numpy()[0, 0, 0] == 6
    assert reconstruction.batch([4])[:, 0, 0, 0].tolist() == [15]
    assert calls == [(0, 2), (2, 2), (4, 1)]
    assert reconstruction.slice(0).numpy()[0, 0, 0] == 1
    assert calls[3:] == ['reset', (0, 2)]
    with pytest.raises(IndexError):
        reconstruction.slice(5)


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


def test_representation_cache_changes_with_mode_and_eventcv_version(monkeypatch):
    from utils import representations
    events = representation_tag({'representation': 'Events'})
    config = {'representation': 'Reconstruction'}
    monkeypatch.setattr(representations.metadata, 'version', lambda name: '1.0.13')
    first = representation_tag(config)
    assert first != events
    monkeypatch.setattr(representations.metadata, 'version', lambda name: '1.0.14')
    assert representation_tag(config) != first
    assert representation_tag({'representation': 'Events'}) == events


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


def test_e2vid_mode_reconstructs_raw_windows_in_aligned_chunks(monkeypatch):
    import eventcv
    from utils import representations
    opened, seen, devices = [], [], []
    def fake_open(path, **kwargs):
        opened.append(kwargs)
        return SimpleNamespace(n_slices=100, time_span=lambda unit: (5_000.0, 1_000_000.0))
    class E2VID:
        def __init__(self, weights=None, *, device=None):
            assert weights is None
            devices.append(device)
        def reset(self):
            seen.append('reset')
    def reconstruct(chunk, model, *, max_frames):
        assert chunk.n_slices == 100 and isinstance(model, E2VID) and 'repr' not in opened[-1]
        seen.append((opened[-1]['offset_us'], max_frames))
        return np.zeros((max_frames, 4, 4), dtype=np.float32)
    monkeypatch.setattr(eventcv, 'open', fake_open)
    monkeypatch.setattr(eventcv, 'E2VID', E2VID, raising=False)
    monkeypatch.setattr(eventcv, 'reconstruct', reconstruct, raising=False)
    monkeypatch.setattr(eventcv, 'gpu_available', lambda: True)
    reader = open_reader('recording.h5', 10, config={'representation': 'Reconstruction'})
    assert reader.n_slices == 100 and devices == ['gpu']
    assert reader.slice(70).numpy().shape == (1, 4, 4)
    # Chunks start at the recording's framing origin and advance by whole windows.
    assert seen == [(5_000, 64), (5_000 + 64 * 10_000, 36)]
    assert all(kwargs['dt_ms'] == 10 and 'offset' not in kwargs for kwargs in opened[1:])
    reader.slice(0)
    assert seen[2:] == ['reset', (5_000, 64)]
    # An explicit sequence offset becomes the framing origin of every chunk.
    seen.clear()
    open_reader('recording.h5', 10, offset_ms=12.5, config={'representation': 'Reconstruction'}).slice(0)
    assert opened[-2]['offset'] == 12.5 and seen == [(12_500, 64)]
    # Without an EventCV GPU backend, E2VID runs on the CPU.
    monkeypatch.setattr(eventcv, 'gpu_available', lambda: False)
    open_reader('recording.h5', 10, config={'representation': 'Reconstruction'})
    assert devices[-1] == 'cpu'


def test_global_event_format_overrides_are_rejected():
    with pytest.raises(ValueError, match='choose Events, Reconstruction'):
        validate_runtime({'representation': 'voxel'})
    with pytest.raises(ValueError, match='Retired'):
        validate_runtime({'representation': 'Events', 'representation_options': {'bins': 9}})
