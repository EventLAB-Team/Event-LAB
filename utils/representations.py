"""Shared EventCV representations and fixed-channel model input adapters."""
import hashlib
import json
import os
from importlib import metadata

import numpy as np

from utils.configuration import CONFIG, read_config, validate_runtime


def runtime_config():
    return read_config(os.environ.get('EVENTLAB_RUNTIME_CONFIG', CONFIG))


def representation_tag(config):
    """Keep descriptor/result caches separate when rendering settings change."""
    settings = {key: config.get(key) for key in (
        'representation', 'representation_options', 'hot_pixel_filter',
        'filter_places_sec', 'timewindows', 'gps_tolerance', 'data_path')}
    settings['input_pipeline_version'] = 2
    if config.get('representation') == 'Reconstruction':
        # EventCV ships the E2VID weights, so its version identifies the model.
        settings['model_version'] = f"eventcv-{metadata.version('eventcv')}"
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    return f"{config.get('representation', 'Events')}-{digest}"


def channels(frame, count):
    """Preserve matching channels; average groups to reduce or tile to expand."""
    array = np.asarray(frame.numpy() if hasattr(frame, 'numpy') else frame, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f'Expected a dense [C,H,W] representation, got {array.shape}')
    if array.shape[0] > count:
        array = np.stack([group.mean(axis=0) for group in np.array_split(array, count)])
    elif array.shape[0] < count:
        array = np.tile(array, (int(np.ceil(count / array.shape[0])), 1, 1))[:count]
    return np.ascontiguousarray(array)


def image_uint8(frame, count=3):
    """Convert dense representations to grayscale/RGB for image-only models."""
    source = np.asarray(frame.numpy() if hasattr(frame, 'numpy') else frame)
    array = channels(source, count)
    low, high = float(array.min()), float(array.max())
    if low < 0:
        array = array / max(abs(low), abs(high)) * 127.5 + 127.5
    elif high <= 1:
        array = array * 255
    elif source.dtype != np.uint8 or high > 255:
        array = array * (255 / high)
    return np.clip(array, 0, 255).astype(np.uint8)


class ArrayFrame:
    def __init__(self, array):
        self.array = np.asarray(array)

    def numpy(self):
        return self.array


class ReconstructionReader:
    """E2VID intensity frames rendered in sequence order, ``chunk`` windows at a time.

    ``render(start, count)`` returns frames ``start .. start + count - 1`` as ``[N, H, W]``.
    The recurrent state lives in ``model`` and carries from one chunk to the next, so the
    frames match a single ``ecv.reconstruct`` pass while only one chunk is held in memory.
    Seeking back before the held chunk resets the model and replays from the start.
    """
    def __init__(self, n_slices, model, render, chunk=64):
        self.n_slices, self.model, self.render, self.chunk = n_slices, model, render, chunk
        self.start, self.frames = 0, None

    def slice(self, index):
        if index < 0 or index >= self.n_slices:
            raise IndexError(index)
        if index < self.start:
            self.model.reset()
            self.start, self.frames = 0, None
        while self.frames is None or index >= self.start + len(self.frames):
            start = 0 if self.frames is None else self.start + len(self.frames)
            frames = np.asarray(self.render(start, min(self.chunk, self.n_slices - start)),
                                dtype=np.float32)
            if not len(frames):
                raise RuntimeError(f'E2VID returned no frames from window {start}')
            self.start, self.frames = start, frames
        return ArrayFrame(self.frames[index - self.start][None])

    def batch(self, indices):
        return np.stack([self.slice(i).numpy() for i in indices])

    def windows(self):
        return (self.slice(i) for i in range(self.n_slices))


class ManifestReader:
    """Reader over a window manifest: slice ``i`` is window ``i`` of the resolved sequence.

    Exposes the subset of EventCV's reader interface the baselines use (``n_slices``,
    ``slice``, ``batch``, ``windows``, ``sensor_size``). Windows are explicit places, so
    ``fixed_places`` tells frame thinning (``filter_places_sec``) to leave them alone.
    """
    fixed_places = True
    MAX_OPEN = 4

    def __init__(self, path, timewindow_ms, open_file):
        from utils.event_windows import resolve_windows

        self.path = str(path)
        self.windows_table = resolve_windows(self.path, timewindow_ms)
        self.n_slices = len(self.windows_table)
        self.sensor_size = self.windows_table.sensor_size
        self._open_file, self._readers = open_file, {}

    def _reader(self, file):
        if file not in self._readers:
            if len(self._readers) >= self.MAX_OPEN:
                self._readers.pop(next(iter(self._readers)))
            self._readers[file] = self._open_file(self.windows_table.files[file])
        return self._readers[file]

    def slice(self, index):
        if index < 0:
            index += self.n_slices
        if not 0 <= index < self.n_slices:
            raise IndexError(index)
        table = self.windows_table
        return self._reader(int(table.file_index[index])).slice(
            t0_us=int(table.start_us[index]), t1_us=int(table.end_us[index]))

    def __getitem__(self, index):
        return self.slice(index)

    def batch(self, indices):
        return np.stack([np.asarray(self.slice(i).numpy()) for i in indices])

    def windows(self):
        return (self.slice(i) for i in range(self.n_slices))

    def __getstate__(self):
        return {**self.__dict__, '_readers': {}}


def open_reader(path, timewindow_ms, offset_ms=None, sensor_size=None, config=None,
                *, event_representation='redblue', event_options=None):
    import eventcv as ecv
    from utils.event_windows import is_manifest

    config = runtime_config() if config is None else config
    validate_runtime(config)
    kwargs = dict(dt_ms=timewindow_ms, hot_pixel_filter=config.get('hot_pixel_filter', True))
    if offset_ms is not None:
        kwargs['offset'] = offset_ms
    if sensor_size is not None:
        kwargs['sensor_size'] = tuple(sensor_size)
    name = config.get('representation', 'Events')
    if name == 'Reconstruction' and not (hasattr(ecv, 'E2VID') and hasattr(ecv, 'reconstruct')):
        raise RuntimeError('Reconstruction requires EventCV >= 1.0.13 with E2VID support '
                           '(the `nn` build feature); run `pixi install` to update.')
    if is_manifest(path):
        # Each window names its own file and absolute time span; offsets do not apply.
        kwargs.pop('offset', None)
        def open_file(file):
            file_reader = ecv.open(str(file), **{**kwargs, 'sensor_size': tuple(manifest.sensor_size)})
            return file_reader.with_repr(event_representation, **(event_options or {})) \
                if name == 'Events' else file_reader
        manifest = ManifestReader(path, timewindow_ms, open_file)
        reader = manifest
    else:
        reader = ecv.open(str(path), **kwargs)
    if name == 'Events':
        return reader if is_manifest(path) else reader.with_repr(event_representation, **(event_options or {}))
    # E2VID owns voxelization, recurrent state and its weights (downloaded by EventCV on
    # first use). Frames keep the experiment's time windows, aligned with ground truth.
    # The device is set on the model (not ecv.reconstruct) because one model carries the
    # recurrent state across chunks; EventCV otherwise defaults E2VID to the CPU.
    model = ecv.E2VID(device='gpu' if ecv.gpu_available() else 'cpu')
    if is_manifest(path):
        # Manifest windows are explicit spans across files rather than one uniformly
        # sliced reader, so E2VID is applied window by window, as ecv.reconstruct does.
        def render(start, count):
            return np.stack([np.asarray(model(reader.slice(i))) for i in range(start, start + count)])
    else:
        origin_us = round(offset_ms * 1000) if offset_ms is not None else int(reader.time_span('us')[0])
        step_us = round(timewindow_ms * 1000)
        chunk_kwargs = {key: value for key, value in kwargs.items() if key != 'offset'}
        def render(start, count):
            # A reader framed at window `start` yields the same windows as the full reader.
            chunk = ecv.open(str(path), **chunk_kwargs, offset_us=origin_us + start * step_us)
            return ecv.reconstruct(chunk, model, max_frames=count)
    return ReconstructionReader(reader.n_slices, model, render)
