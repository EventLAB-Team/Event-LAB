"""Shared EventCV representations and fixed-channel model input adapters."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from utils.configuration import ROOT, CONFIG, read_config, validate_runtime


def runtime_config():
    return read_config(os.environ.get('EVENTLAB_RUNTIME_CONFIG', CONFIG))


def representation_tag(config):
    """Keep descriptor/result caches separate when rendering settings change."""
    settings = {key: config.get(key) for key in (
        'representation', 'representation_options', 'hot_pixel_filter',
        'filter_places_sec', 'timewindows', 'gps_tolerance', 'data_path')}
    settings['input_pipeline_version'] = 2
    if config.get('representation') == 'Reconstruction':
        model = ROOT / 'e2vid' / 'e2vid.safetensors'
        if model.is_file():
            stat = model.stat()
            settings['model_version'] = (str(model.resolve()), stat.st_size, stat.st_mtime_ns)
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
    """Keep recurrent state in sequence order without retaining rendered frames."""
    def __init__(self, reader, model):
        self.reader, self.model = reader, model
        self.n_slices = reader.n_slices
        self.next_index = 0
        self.last = None

    def slice(self, index):
        if index < 0 or index >= self.n_slices:
            raise IndexError(index)
        if index == self.next_index - 1:
            return self.last
        if index < self.next_index:
            self.model.reset()
            self.next_index = 0
        while self.next_index <= index:
            result = self.model(self.reader.slice(self.next_index))
            array = np.asarray(result.numpy() if hasattr(result, 'numpy') else result)
            if array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            if array.ndim == 2:
                array = array[None]
            self.last = ArrayFrame(array)
            self.next_index += 1
        return self.last

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
    if name == 'Reconstruction' and not hasattr(ecv, 'E2VID'):
        raise RuntimeError('Reconstruction requires an EventCV build with E2VID support. '
                           'The installed EventCV only supports ONNX reconstruction.')
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
    weights = ROOT / 'e2vid' / 'e2vid.safetensors'
    model = ecv.E2VID(str(weights)) if weights.is_file() else ecv.E2VID()
    # Feed raw event windows to E2VID, which owns voxelization and recurrent state.
    # Keep the experiment's time windows so frames remain aligned with ground truth.
    return ReconstructionReader(reader, model)
