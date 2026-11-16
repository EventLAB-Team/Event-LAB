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
    if config.get('representation') == 'reconstruction':
        settings['reconstruction'] = config.get('reconstruction', {})
        model = ROOT / Path(settings['reconstruction'].get('model', '')).expanduser()
        if model.is_file():
            stat = model.stat()
            settings['model_version'] = (str(model.resolve()), stat.st_size, stat.st_mtime_ns)
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    return f"{config.get('representation', 'polarity')}-{digest}"


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


def open_reader(path, timewindow_ms, offset_ms=None, sensor_size=None, config=None):
    import eventcv as ecv

    config = runtime_config() if config is None else config
    validate_runtime(config)
    kwargs = dict(dt_ms=timewindow_ms, hot_pixel_filter=config.get('hot_pixel_filter', True))
    if offset_ms is not None:
        kwargs['offset'] = offset_ms
    if sensor_size is not None:
        kwargs['sensor_size'] = tuple(sensor_size)
    reader = ecv.open(str(path), **kwargs)
    name = config.get('representation', 'polarity')
    if name != 'reconstruction':
        return reader.with_repr(name, **config.get('representation_options', {}))
    reconstruction = config['reconstruction']
    reader = reader.with_repr(reconstruction.get('input_representation', 'voxel'),
                              **reconstruction.get('input_options', {'bins': 5}))
    # StatefulModel also handles a stateless graph with an empty state map.
    model = ecv.StatefulModel(ecv.Model(str(ROOT / Path(reconstruction['model']).expanduser())),
                              state_map=reconstruction.get('state_map', {}))
    return ReconstructionReader(reader, model)
