"""Live place recognition: reference recordings from a camera, and a per-method retrieval worker.

The TUI's Live section records references with the camera (eventcv's own recorder, `record=`) and
registers each one as a sequence of the `live` dataset (`datasets/live.yaml`), so the batch pipeline
can use it too. It
then localises the live stream against a reference with a `LiveWorker`: a long-running process in
the method's own pixi environment (`utils/live_worker.py`) that maps the reference once -- the
descriptor bank is cached -- and then matches every window it is sent. No ground truth is involved.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from utils.configuration import ROOT, read_config, write_config
from utils.terminal_theme import EVENT_OFF, EVENT_ON

LIVE_DATASET = 'live'
# Method -> the pixi environment its worker runs in (the same split the batch baselines use).
LIVE_METHODS = {'sparse_event': 'default', 'megaevent': 'megaevent', 'spikevpr': 'spikevpr',
                'eventgem': 'eventgem'}
# Pictures are drawn at this width: real pixels in Kitty, where the terminal scales them to the view.
VIEW_WIDTH = 640


def dataset_yaml():
    return ROOT / 'datasets' / f'{LIVE_DATASET}.yaml'


def clean_name(text):
    """A recording name usable as a sequence key and a directory: letters, digits, `-` and `_`."""
    return re.sub(r'[^A-Za-z0-9_-]+', '_', text.strip()).strip('_')


def recording_path(config, name):
    # The layout every dataset uses: <data_path>/<dataset>/<sequence>/<sequence>.<format>.
    return ROOT / config['data_path'] / LIVE_DATASET / name / f'{name}.hdf5'


def recordings(config):
    """Registered references whose recording is on disk, in the order they were made."""
    if not dataset_yaml().is_file():
        return []
    return [name for name in read_config(dataset_yaml()).get('sequences') or {}
            if recording_path(config, name).is_file()]


def register(name, sensor_size, camera=''):
    """Add `name` to the live dataset, creating `datasets/live.yaml` on first use."""
    path = dataset_yaml()
    data = read_config(path) if path.is_file() else {
        'dataset': {'name': LIVE_DATASET, 'version': '1.0.0',
                    'description': 'References recorded live in the EventLAB TUI (no ground truth)',
                    'base_url': '', 'camera': camera, 'resolution': list(sensor_size)},
        'sequences': {},
        'format': {'data': {'format': 'hdf5', 'timestamp_val': 'us', 'time_scale': 1.0e-06},
                   'ground_truth': 'hdf5', 'hot_pixel': ''},
    }
    data['dataset']['resolution'] = list(sensor_size)
    if data.get('sequences') is None:
        data['sequences'] = {}
    data['sequences'][name] = {'data': {'url': ''}, 'ground_truth': {'available': False, 'url': ''},
                               'hot_pixel': {'available': False, 'url': ''}}
    write_config(path, data)


def delete_recording(config, name):
    """Remove reference `name`: its recording, its entry in the live dataset, and its cached
    descriptor banks (named `<name>-<window>ms-<digest>.npz` by utils/live_worker.py)."""
    import shutil

    shutil.rmtree(recording_path(config, name).parent, ignore_errors=True)
    if dataset_yaml().is_file():
        data = read_config(dataset_yaml())
        if name in (data.get('sequences') or {}):
            del data['sequences'][name]
            write_config(dataset_yaml(), data)
    pattern = re.compile(re.escape(name) + r'-[0-9.]+ms-[0-9a-f]{12}\.npz')
    for cache in (ROOT / 'output' / 'live').glob('*/*.npz'):
        if pattern.fullmatch(cache.name):
            cache.unlink()


def _hex(colour):
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


ON, OFF = _hex(EVENT_ON), _hex(EVENT_OFF)


def render(events, sensor_size, width=VIEW_WIDTH):
    """An (h, w, 3) uint8 picture of `events` (rows of x, y, t, p) on black: each pixel that fired
    takes the colour of its majority polarity (ON blue, OFF red), as eventcv's raw view draws it."""
    sensor_w, sensor_h = sensor_size
    height = max(1, round(sensor_h * width / sensor_w))
    image = np.zeros((height * width, 3), np.uint8)
    if len(events):
        x = np.minimum(events[:, 0].astype(np.int64) * width // sensor_w, width - 1)
        y = np.minimum(events[:, 1].astype(np.int64) * height // sensor_h, height - 1)
        cell = y * width + x
        # +1 per ON event, -1 per OFF: the sign is the majority, zero where nothing fired.
        balance = np.bincount(cell, weights=np.where(events[:, 3] > 0, 1.0, -1.0), minlength=width * height)
        fired = np.bincount(cell, minlength=width * height) > 0
        image[fired & (balance >= 0)] = ON.astype(np.uint8)
        image[fired & (balance < 0)] = OFF.astype(np.uint8)
    return image.reshape(height, width, 3)


class ReferenceView:
    """Draws reference place `i`, the reference's i-th window -- the worker's place numbering."""

    def __init__(self, path, timewindow_ms):
        import eventcv as ecv

        self.reader = ecv.open(str(path), dt_ms=timewindow_ms)
        self.places = self.reader.n_slices

    def render(self, place, width=VIEW_WIDTH):
        return render(self.reader.slice(place).numpy(), self.reader.sensor_size, width)


class LiveWorker:
    """The TUI's handle on a `utils/live_worker.py` process: send windows, receive matches.

    `on_message` is called from a reader thread with every message the worker prints (mapping
    progress, `ready`, matches, `error`). At most one window is in flight; `send` drops a window
    while the previous one is still being matched, so a slow method lowers its own update rate
    rather than queueing behind the camera.
    """

    def __init__(self, method, reference, timewindow_ms, places, on_message):
        self.method, self.on_message = method, on_message
        self.ready = False
        self.busy = threading.Event()
        self.tmp = tempfile.TemporaryDirectory(prefix='eventlab-live-')
        self.sent = 0
        environment = LIVE_METHODS[method]
        script = [str(ROOT / 'utils' / 'live_worker.py'), method, str(reference),
                  '--timewindow', str(timewindow_ms), '--places', str(places or 0)]
        env = os.environ.copy()
        if environment == 'default':
            command = [sys.executable, '-u', *script]
        else:
            # As utils.utils.pixi_run: strip the parent's pixi state so `-e` resolves afresh.
            for key in ('PIXI_ENVIRONMENT', 'PIXI_PROJECT_MANIFEST', 'PIXI_IN_SHELL'):
                env.pop(key, None)
            command = ['pixi', 'run', '-e', environment, 'python', '-u', *script]
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(ROOT), env.get('PYTHONPATH'))))
        log_dir = ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        self.log_path = log_dir / f'live-{method}.log'
        self.log = self.log_path.open('w')
        self.process = subprocess.Popen(command, cwd=ROOT, env=env, text=True, bufsize=1,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        start_new_session=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'ready' in message:
                self.ready = True
            if 'best' in message or 'error' in message:
                self.busy.clear()
            self.on_message(message)
        self.ready = False
        if self.process.wait() not in (0, -15):
            self.on_message({'error': f'{self.method} worker exited; see {self.log_path}'})

    def send(self, events, sensor_size):
        """Hand one window (rows of x, y, t_us, p) to the worker, unless one is in flight."""
        if not self.ready or self.busy.is_set() or self.process.poll() is not None:
            return False
        self.busy.set()
        # Alternating files: the worker has finished reading the previous one by now.
        path = Path(self.tmp.name) / f'window{self.sent % 2}.npy'
        self.sent += 1
        np.save(path, np.ascontiguousarray(events, dtype=np.int64))
        try:
            request = {'window': str(path), 'sensor': list(sensor_size)}
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.busy.clear()
            return False
        return True

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, 15)
        self.log.close()
        self.tmp.cleanup()
