"""Live place recognition worker: map one reference with one method, then match live windows.

Runs inside the method's own pixi environment, started by `utils.live_vpr.LiveWorker`. It speaks
one JSON object per line on stdout: `{"stage", "done", "total"}` while mapping, then `{"ready": k,
"windows": [...]}` once the reference's `k` places are described (`windows` names the reference
window each place is), then for each `{"window": path, "sensor": [w, h]}` read from stdin (an .npy
of x, y, t_us, p rows) `{"best", "window", "score", "ms"}`: the best place and its window.
`--places k` maps only `k` places, the middle windows of `k` equal stretches of the reference;
without it every window is a place. Everything else the
method prints goes to stderr, which the TUI keeps in logs/live-<method>.log.

Mapping reuses each method's own model code and the reference reader the batch pipeline uses, with
one deliberate difference: hot-pixel filtering is off. The batch readers filter with statistics
over the whole file, which a live window does not have, so the reference is described the same way
the live windows are. Descriptor banks are cached under output/live/<method>/.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# The protocol owns the real stdout; anything the methods print goes to stderr.
PROTOCOL = os.fdopen(os.dup(1), 'w', buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def say(**message):
    PROTOCOL.write(json.dumps(message) + '\n')
    PROTOCOL.flush()


def settings(name):
    from utils.configuration import baseline_config_path, read_config
    return dict(read_config(baseline_config_path(name)))


def live_config():
    from utils.representations import runtime_config
    return {**dict(runtime_config()), 'hot_pixel_filter': False, 'representation': 'Events'}


class Window:
    """One live window: the raw rows, and the same events as an EventCV stream."""

    def __init__(self, events, sensor_size):
        import eventcv as ecv

        self.events, self.sensor_size = events, tuple(sensor_size)
        self.stream = ecv.from_numpy(events, time_unit='us', sensor_size=self.sensor_size)


class SparseEvent:
    """Sparse-event VPR: L1 distance over saliency-sampled pixels of burst-clipped count frames,
    matched over a sequence of `sequence_length` windows (the batch baseline's eye() convolution).
    A place is the run of `sequence_length` windows ending at it, so only those are described."""

    def __init__(self, timewindow):
        self.cfg, self.timewindow = settings('sparse_event'), timewindow
        self.length = int(self.cfg['sequence_length'])
        self.recent = deque(maxlen=self.length)

    def frame(self, count):
        from baselines.vpr_sparse_event.src.sparse_event_vpr.utils import remove_random_bursts
        from utils.representations import channels
        return remove_random_bursts(channels(count, 1), threshold=10)[0]

    def map(self, path, windows, progress):
        from baselines.vpr_sparse_event.src.sparse_event_vpr.sparse_pixel_utils import (
            adjust_and_normalize_probabilities, get_random_pixels)
        from utils.representations import open_reader

        reader = open_reader(path, self.timewindow, config=live_config(), event_representation='count')
        # The run ending at each place; at the very start the first window stands in for earlier ones.
        runs = np.maximum(np.asarray(windows)[:, None] - np.arange(self.length)[::-1], 0)
        needed = np.unique(runs)
        m = len(needed)
        # Two passes, as the batch baseline: saliency first, then only the chosen pixels are kept.
        total = None
        for i, w in enumerate(needed):
            frame = self.frame(reader.slice(int(w)))
            total = frame.astype(np.float64) if total is None else total + frame
            progress('saliency', i + 1, 2 * m)
        saliency = np.abs(total / m)
        prob = adjust_and_normalize_probabilities(saliency) if self.cfg['use_saliency'] and saliency.any() else None
        height, width = saliency.shape
        pixels = np.array(get_random_pixels(self.cfg['num_target_pixels'], im_width=width, im_height=height,
                                            local_suppression_radius=7, prob_to_draw_from=prob))
        ys, xs = pixels[:, 0], pixels[:, 1]
        bank = np.empty((m, len(ys)), np.float32)
        for i, w in enumerate(needed):
            bank[i] = self.frame(reader.slice(int(w)))[ys, xs]
            progress('descriptors', m + i + 1, 2 * m)
        return {'bank': bank, 'runs': np.searchsorted(needed, runs), 'ys': ys, 'xs': xs}

    def load(self, state):
        self.bank, self.runs, self.ys, self.xs = state['bank'], state['runs'], state['ys'], state['xs']

    def describe(self, window):
        return self.frame(window.stream.count())[self.ys, self.xs].astype(np.float32)

    def scores(self, descriptor):
        self.recent.append(descriptor)
        queries = np.stack(self.recent)                                   # (k, P), oldest first
        k = len(queries)
        distance = np.abs(self.bank[:, None, :] - queries[None]).sum(-1)  # (described windows, k)
        # Each place's run ends at the place: its last k windows pair with the k live windows.
        return -distance[self.runs[:, self.length - k:], np.arange(k)].mean(1)


class MegaEvent:
    """MegaEvent: its checkpoint's descriptor of the red/blue frame; cosine similarity."""

    def __init__(self, timewindow):
        import torch
        from utils.megaevent_descriptors import allow_numpy2_reconstruct  # also binds np._core

        self.cfg, self.timewindow, self.torch = settings('megaevent'), timewindow, torch
        sys.path.insert(0, str(ROOT / 'baselines' / 'megaevent' / 'src'))
        from megaevent.checkpoints import load_model, resolve_model
        from megaevent.events import eval_transform
        from megaevent.runtime import device_for

        allow_numpy2_reconstruct()
        self.device = device_for(self.cfg['device'])
        checkpoint = resolve_model(self.cfg['model'], str(ROOT / self.cfg['ckpt_dir']))
        self.network, model_cfg = load_model(checkpoint, self.device)
        self.transform = eval_transform(model_cfg)

    def tensor(self, frame):
        from utils.representations import image_uint8
        return self.transform(self.torch.from_numpy(image_uint8(frame)).float().div_(255))

    def describe_batch(self, frames):
        with self.torch.no_grad():
            batch = self.torch.stack([self.tensor(f) for f in frames]).to(self.device)
            return self.network(batch).float().cpu().numpy()

    def map(self, path, windows, progress):
        from utils.representations import open_reader
        reader = open_reader(path, self.timewindow, config=live_config())
        return {'bank': batched(reader, windows, self.describe_batch, self.cfg['batch_size'], progress)}

    def load(self, state):
        self.bank = state['bank']

    def describe(self, window):
        return self.describe_batch([window.stream.redblue()])[0]

    def scores(self, descriptor):
        return self.bank @ descriptor


class SpikeVPR:
    """SpikeVPR: SEW-ResNet + MixVPR on per-polarity counts; cosine similarity."""

    def __init__(self, timewindow):
        import torch
        from utils.spikevpr_descriptors import build_model, resolve_device

        self.cfg, self.timewindow, self.torch = settings('spikevpr'), timewindow, torch
        self.device = resolve_device(self.cfg['device'])
        checkpoint = ROOT / 'baselines' / 'SpikeVPR' / 'src' / 'weights' / self.cfg['checkpoint']
        self.model = build_model(str(ROOT / 'baselines' / 'SpikeVPR'), str(checkpoint), self.cfg['encoder'],
                                 self.cfg['neuron_type'], self.cfg['out_channels'], self.cfg['out_rows'],
                                 self.device)

    def describe_batch(self, frames):
        from utils.representations import channels
        batch = np.stack([channels(f, 2) for f in frames])
        if self.cfg['channel_order'] == 'off_on':
            batch = batch[:, ::-1].copy()
        with self.torch.no_grad():
            x = self.torch.from_numpy(batch).to(self.device)
            # The released MixVPR head has a fixed spatial input (as utils/spikevpr_descriptors.py).
            x = self.torch.nn.functional.interpolate(x, size=(260, 346), mode='bilinear', align_corners=False)
            return self.model(x).float().cpu().numpy()

    def map(self, path, windows, progress):
        from utils.representations import open_reader
        reader = open_reader(path, self.timewindow, config=live_config(),
                             event_representation='polarity', event_options={'normalize': False})
        return {'bank': batched(reader, windows, self.describe_batch, self.cfg['batch_size'], progress)}

    def load(self, state):
        self.bank = state['bank']

    def describe(self, window):
        # The `polarity` representation: per-pixel counts, ON in channel 0 and OFF in channel 1.
        width, height = window.sensor_size
        x, y = window.events[:, 0].astype(np.int64), window.events[:, 1].astype(np.int64)
        on = window.events[:, 3] > 0
        counts = np.zeros((2, height * width), np.float32)
        counts[0] = np.bincount(y[on] * width + x[on], minlength=height * width)
        counts[1] = np.bincount(y[~on] * width + x[~on], minlength=height * width)
        return self.describe_batch([counts.reshape(2, height, width)])[0]

    def scores(self, descriptor):
        return self.bank @ descriptor


class EventGeM:
    """Event-GeM global retrieval: SuperEvent trunk on MCTS frames, regional GeM pooling and the
    shipped head; cosine similarity. Upstream's keypoint/homography re-ranking stays offline."""

    def __init__(self, timewindow):
        import torch

        self.cfg, self.timewindow, self.torch = settings('eventgem'), timewindow, torch
        repo = ROOT / 'baselines' / 'Event-GeM'
        # Upstream resolves its SuperEvent config and weights relative to the clone.
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        import eventgem.feature_extraction as fe
        from eventgem.dataset import EventGeMMCTS

        self.pool, self.dataset_class = fe.pool_grid_gamma, EventGeMMCTS
        requested = self.cfg['device']
        if requested == 'auto':
            requested = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(requested)
        gem = fe.EventGeM.__new__(fe.EventGeM)
        self.model, self.se_config = gem.build_superevent_model(
            Path('eventgem/external/superevent/config/super_event.yaml'),
            Path('eventgem/external/superevent/saved_models/super_event_weights.pth'), self.device)
        self.head, self.gamma = gem.load_head(self.device)

    def pooled(self, frames):
        x = np.stack([np.asarray(f.numpy() if hasattr(f, 'numpy') else f, np.float32) for f in frames])
        x = x[:, :, self.top:self.bottom, self.left:self.right]
        with self.torch.inference_mode():
            features = self.model.fpn(self.model.backbone(self.torch.from_numpy(x).to(self.device)))
            return self.pool(features.float(), self.gamma)

    def map(self, path, windows, progress):
        from utils.representations import open_reader
        # Upstream's own crop rule, read from its dataset class; frames come from the shared reader.
        crop = self.dataset_class(str(path), self.se_config, offset=0, dt_ms=self.timewindow,
                                  max_window_ms=float(self.timewindow))
        self.top, _, self.left, _ = crop.get_offsets()
        self.bottom, self.right = crop.h_end, crop.w_end
        reader = open_reader(path, self.timewindow, config=live_config(), event_representation='mcts',
                             event_options={'max_window_ms': float(self.timewindow)})
        pooled = batched(reader, windows, lambda frames: self.pooled(frames).cpu().numpy(),
                         self.cfg['keypoint_batch_size'], progress)
        with self.torch.inference_mode():
            bank = self.head(self.torch.from_numpy(pooled).to(self.device)).cpu().numpy()
        return {'bank': bank, 'crop': np.array([self.top, self.bottom, self.left, self.right])}

    def load(self, state):
        self.bank = state['bank']
        self.top, self.bottom, self.left, self.right = (int(v) for v in state['crop'])

    def describe(self, window):
        pooled = self.pooled([window.stream.mcts(max_window_ms=float(self.timewindow))])
        with self.torch.inference_mode():
            return self.head(pooled).cpu().numpy()[0]

    def scores(self, descriptor):
        return self.bank @ descriptor


METHODS = {'sparse_event': SparseEvent, 'megaevent': MegaEvent, 'spikevpr': SpikeVPR, 'eventgem': EventGeM}


def batched(reader, windows, describe, batch_size, progress):
    """Descriptors of the reader's `windows`, in that order."""
    windows = [int(w) for w in windows]
    out = []
    for start in range(0, len(windows), int(batch_size)):
        chunk = windows[start:start + int(batch_size)]
        out.append(describe([reader.slice(i) for i in chunk]))
        progress('descriptors', start + len(chunk), len(windows))
    return np.concatenate(out).astype(np.float32)


def cache_path(method, reference, timewindow, places):
    """Keyed by the recording's identity, the window, the places and the method's settings."""
    stat = Path(reference).stat()
    key = json.dumps([method, str(Path(reference).resolve()), stat.st_size, stat.st_mtime_ns, timewindow,
                      places, settings(method)], sort_keys=True, default=str)
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return ROOT / 'output' / 'live' / method / f'{Path(reference).stem}-{timewindow}ms-{digest}.npz'


def place_windows(reference, timewindow, places):
    """The reference window each place is: the middles of `places` equal stretches, or every window."""
    import eventcv as ecv
    n = ecv.open(reference, dt_ms=timewindow).n_slices
    if not places or places >= n:
        return list(range(n))
    return [(2 * k + 1) * n // (2 * places) for k in range(places)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('method', choices=sorted(METHODS))
    parser.add_argument('reference', help='The reference recording to map')
    parser.add_argument('--timewindow', type=float, required=True, help='Window length in milliseconds')
    parser.add_argument('--places', type=int, default=0, help='Places to map (0: every window)')
    args = parser.parse_args()
    timewindow = int(args.timewindow) if args.timewindow == int(args.timewindow) else args.timewindow
    reference = str(Path(args.reference).resolve())

    try:
        say(stage='loading model', done=0, total=1)
        method = METHODS[args.method](timewindow)
        windows = place_windows(reference, timewindow, args.places)
        cache = cache_path(args.method, reference, timewindow, len(windows))
        if cache.is_file():
            say(stage='cached descriptors', done=1, total=1)
            state = dict(np.load(cache))
        else:
            last = [0.0]

            def progress(stage, done, total):
                if time.monotonic() - last[0] > 0.2 or done == total:
                    last[0] = time.monotonic()
                    say(stage=stage, done=done, total=total)

            state = method.map(reference, windows, progress)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **state)
        method.load(state)
        say(ready=len(windows), windows=windows)
    except Exception as error:  # reported to the TUI rather than lost in the log
        import traceback
        traceback.print_exc()
        say(error=f'{type(error).__name__}: {error}')
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            started = time.perf_counter()
            window = Window(np.load(request['window']), request['sensor'])
            if len(window.events) == 0:
                say(error='empty window')
                continue
            score = np.asarray(method.scores(method.describe(window)), np.float64)
            best = int(np.argmax(score))
            say(best=best, window=windows[best], score=float(score[best]),
                ms=round((time.perf_counter() - started) * 1e3, 1))
        except Exception as error:
            import traceback
            traceback.print_exc()
            say(error=f'{type(error).__name__}: {error}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
