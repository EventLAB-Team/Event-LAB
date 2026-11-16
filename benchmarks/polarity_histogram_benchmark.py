#!/usr/bin/env python3
"""
Benchmark: 2-channel polarity histogram generation.

Compares EventCV running on CPU only (the representation path shipped in Event-LAB
1.1.0) against the original torch/CUDA frame builder from branch ``main``
(``datasets/format_data.py`` -> ``GeneralizedFrameBuilder(accumulator_type="polarity")``).

Three sub-commands:

    run     orchestrate a benchmark: resolves recordings, vendors main's builder,
            spawns one worker process per (variant, recording, timewindow, repeat),
            checks numerical equivalence, writes results.json / summary.md / charts
    worker  (internal) run one variant in an isolated process
    report  regenerate summary.md and the charts from an existing results.json

Every variant renders exactly the same frames 0..N-1 with N = (tN - t0) // window,
so frames/s and Mevents/s share a denominator. See benchmarks/README.md.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = "benchmarks.polarity_histogram_benchmark"

# Blob hash of datasets/format_data.py on main at the time this benchmark was written.
MAIN_BLOB_EXPECTED = "03fd28b465b1a4abd46e7a532b4810554cfd56c3"

DEFAULT_DATASETS = [
    "qcr_event/normal1",
    "brisbane_event/sunset2",
    "qut_event_walking/left",
    "nsavp/R0_FA0",
]
DEFAULT_VARIANTS = ["eventcv_cpu", "main_cuda", "main_cuda_inmem"]
ALL_VARIANTS = ["eventcv_cpu", "main_cuda", "main_cuda_inmem", "main_cpu", "main_cpu_inmem"]
KERNEL_VARIANTS = ["eventcv_kernel", "main_kernel"]

VARIANT_LABELS = {
    "eventcv_cpu": "EventCV (CPU only)",
    "main_cuda": "main: CUDA → frames.h5",
    "main_cuda_inmem": "main: CUDA, in-memory",
    "main_cpu": "main: torch CPU → frames.h5",
    "main_cpu_inmem": "main: torch CPU, in-memory",
}

# HDF5 key aliases (mirrors datasets/format_data.py on main).
EVENT_T_KEYS = ("t", "timestamp", "timestamps", "time", "times")
EVENT_X_KEYS = ("x", "x_coordinate", "x_coordinates", "u", "col", "column")
EVENT_Y_KEYS = ("y", "y_coordinate", "y_coordinates", "v", "row")
EVENT_P_KEYS = ("p", "polarity", "polarities", "pol", "polarity_bit", "polarity_bits")


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def find_event_fields(h5f):
    """Return dict of h5py datasets {x,y,t,p} using the alias tables (group layouts only)."""
    found = {}

    def _pick(group, keys):
        for k in keys:
            if k in group and isinstance(group[k], h5py.Dataset):
                return group[k]
        return None

    def _visit(name, obj):
        if found or not isinstance(obj, h5py.Group):
            return
        t = _pick(obj, EVENT_T_KEYS)
        x = _pick(obj, EVENT_X_KEYS)
        y = _pick(obj, EVENT_Y_KEYS)
        if t is not None and x is not None and y is not None:
            found.update(x=x, y=y, t=t, p=_pick(obj, EVENT_P_KEYS))

    for top in ("events", "columns"):
        if top in h5f and isinstance(h5f[top], h5py.Group):
            _visit(top, h5f[top])
            if found:
                break
    if not found:
        h5f.visititems(_visit)
    if not found:
        raise ValueError("no x/y/t event datasets found in HDF5 file")
    return found


def h5_searchsorted(tds, target, side="left", n_total=None):
    """Binary search over a sorted 1-D HDF5 dataset with scalar reads only."""
    n = int(tds.shape[0] if n_total is None else n_total)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        tm = int(tds[mid])
        if tm < target or (side == "right" and tm == target):
            lo = mid + 1
        else:
            hi = mid
    return lo


def rusage_snapshot():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {"user": ru.ru_utime, "sys": ru.ru_stime, "maxrss_mb": ru.ru_maxrss / 1024.0}


def np_polarity_hist(x, y, p, W, H):
    """Reference 2-channel histogram: [2,H,W] int64, ch0 = positive, ch1 = negative."""
    lin = y.astype(np.int64) * W + x.astype(np.int64)
    pos = np.asarray(p) > 0
    out = np.zeros((2, H * W), dtype=np.int64)
    out[0] = np.bincount(lin[pos], minlength=H * W)
    out[1] = np.bincount(lin[~pos], minlength=H * W)
    return out.reshape(2, H, W)


def load_vendor_module(path):
    spec = importlib.util.spec_from_file_location("format_data_main", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quiet_loguru():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")


# --------------------------------------------------------------------------------------
# Worker: one variant in one process
# --------------------------------------------------------------------------------------
def parse_sample_idx(s):
    return [int(v) for v in s.split(",") if v.strip() != ""] if s else []


def save_samples(path, samples):
    """samples: {frame_idx: [2,H,W] uint32}"""
    if not samples:
        return None
    idx = np.array(sorted(samples), dtype=np.int64)
    arr = np.stack([samples[i] for i in idx]).astype(np.uint32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, idx=idx, frames=arr)
    return str(path)


def worker_eventcv_cpu(a):
    os.environ.setdefault("EVENTCV_DEVICE", "cpu")
    t_import = time.perf_counter()
    import eventcv as ecv
    import importlib.metadata as md
    import_s = time.perf_counter() - t_import

    ecv.set_device("cpu")
    assert ecv.get_device() == "cpu", f"eventcv device is {ecv.get_device()!r}, expected cpu"

    t_setup = time.perf_counter()
    reader = ecv.open(
        a.hdf5,
        dt_ms=a.timewindow_ms,
        hot_pixel_filter=bool(a.hot_pixel_filter),
        sensor_size=(a.width, a.height),
    ).with_repr("polarity", normalize=False)
    n_slices = int(reader.n_slices)
    setup_s = time.perf_counter() - t_setup
    N = int(a.n_frames)
    if n_slices < N:
        raise RuntimeError(f"eventcv n_slices={n_slices} < requested N={N}")

    dur_ms = float(reader.duration_ms)
    expected_ms = (a.tn - a.t0) / a.ticks_per_second * 1e3
    if abs(dur_ms - expected_ms) > max(1.0, 1e-3 * expected_ms):
        raise RuntimeError(
            f"eventcv duration {dur_ms:.3f} ms disagrees with ticks_per_second-derived "
            f"{expected_ms:.3f} ms; check the timestamp unit"
        )

    sample_set = set(parse_sample_idx(a.sample_idx))
    samples = {}
    bs = int(a.batch_size)
    last_shape, last_dtype = None, None

    ru0 = rusage_snapshot()
    t0 = time.perf_counter()
    for start in range(0, N, bs):
        idx = list(range(start, min(start + bs, N)))
        b = reader.batch(idx)
        b = b.numpy() if hasattr(b, "numpy") else np.asarray(b)
        if sample_set and not sample_set.isdisjoint(idx):
            for j, i in enumerate(idx):
                if i in sample_set:
                    samples[i] = b[j].astype(np.uint32)
    wall = time.perf_counter() - t0
    ru1 = rusage_snapshot()
    last_shape, last_dtype = tuple(b.shape[1:]), str(b.dtype)

    return {
        "device": "cpu",
        "eventcv_device": ecv.get_device(),
        "eventcv_version": md.version("eventcv"),
        "eventcv_gpu_available": bool(ecv.gpu_available()),
        "torch_imported": "torch" in sys.modules,
        "import_s": import_s,
        "setup_s": setup_s,
        "cuda_init_s": 0.0,
        "wall_s": wall,
        "cpu_user_s": ru1["user"] - ru0["user"],
        "cpu_sys_s": ru1["sys"] - ru0["sys"],
        "ru_maxrss_mb": ru1["maxrss_mb"],
        "n_frames": N,
        "n_slices_reported": n_slices,
        "reader_n_events": int(reader.n_events),
        "reader_duration_ms": dur_ms,
        "frame_shape": last_shape,
        "frame_dtype": last_dtype,
        "batch_size": bs,
        "hot_pixel_filter": bool(a.hot_pixel_filter),
        "samples": samples,
    }


def _make_bounded_builder(fm, tail_tick):
    """Subclass whose first/last-timestamp probe clamps tN so build_frames only reads up to tail_tick."""

    class _BoundedTailBuilder(fm.GeneralizedFrameBuilder):
        _bench_tail_tick = int(tail_tick)

        def _read_first_last_t(self, tds, n_total):  # shadows the base staticmethod
            t0, tN = fm.GeneralizedFrameBuilder._read_first_last_t(tds, n_total)
            return t0, min(int(tN), self._bench_tail_tick)

    return _BoundedTailBuilder


def worker_main(a, mode):
    """mode: 'cuda' | 'cuda_inmem' | 'cpu' | 'cpu_inmem'"""
    use_cuda = mode.startswith("cuda")
    if not use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    t_import = time.perf_counter()
    import torch
    fm = load_vendor_module(a.vendor_module)
    import_s = time.perf_counter() - t_import
    quiet_loguru()

    if a.threads:
        torch.set_num_threads(int(a.threads))

    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        device = torch.device("cuda")
        t = time.perf_counter()
        torch.bincount(torch.arange(10, device=device))
        torch.cuda.synchronize()
        cuda_init_s = time.perf_counter() - t
        torch.cuda.reset_peak_memory_stats()
    else:
        device = torch.device("cpu")
        cuda_init_s = 0.0

    N = int(a.n_frames)
    sample_set = set(parse_sample_idx(a.sample_idx))
    samples = {}
    as_shipped = mode in ("cuda", "cpu")

    Builder = fm.GeneralizedFrameBuilder
    bounded = False
    if as_shipped and a.bounded_tail_tick:
        Builder = _make_bounded_builder(fm, a.bounded_tail_tick)
        bounded = True

    t_setup = time.perf_counter()
    builder = Builder(a.width, a.height, accumulator_type="polarity", hot_pixels=None)
    builder._fast_polarity_device = device
    setup_s = time.perf_counter() - t_setup

    output_bytes = 0
    n_done = 0
    event_count_total = None
    frame_shape, frame_dtype = None, None

    if as_shipped:
        scratch = Path(a.scratch_dir)
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        ru0 = rusage_snapshot()
        t0 = time.perf_counter()
        builder.build_frames(
            a.hdf5,
            str(scratch),
            timewindow_ns=int(round(a.timewindow_ms * 1e6)),
            offset_ns=None,
            chunk_size=int(a.chunk_size),
            max_frames=None,
            time_scale=float(a.ticks_per_second),
        )
        if use_cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        ru1 = rusage_snapshot()

        fpath = scratch / "frames.h5"
        output_bytes = fpath.stat().st_size
        with h5py.File(fpath, "r") as f:
            fr = f["frames"]
            n_done = int(fr.shape[0])
            frame_shape, frame_dtype = tuple(fr.shape[1:]), str(fr.dtype)
            if "event_counts" in f:
                event_count_total = int(np.asarray(f["event_counts"][:n_done]).sum())
            if "frame_start_ticks" in f and n_done:
                starts = np.asarray(f["frame_start_ticks"][:n_done], dtype=np.int64)
                w = int(a.window_ticks)
                expect = int(a.t0) + w * np.arange(n_done, dtype=np.int64)
                if not np.array_equal(starts, expect):
                    raise RuntimeError("frames.h5 frame_start_ticks do not match t0 + i*window")
            for i in sorted(sample_set):
                if i < n_done:
                    frame = np.asarray(fr[i])  # [H,W,2] uint16
                    if frame.max() >= np.iinfo(np.uint16).max:
                        raise RuntimeError(f"uint16 clipping detected in frames.h5 frame {i}")
                    samples[i] = frame.transpose(2, 0, 1).astype(np.uint32)
        if n_done != N:
            raise RuntimeError(f"build_frames wrote {n_done} frames, expected {N}")
        if not a.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    else:
        ru0 = rusage_snapshot()
        t0 = time.perf_counter()
        gen = builder.stream_hdf5_frames(
            a.hdf5,
            dt_ms=float(a.timewindow_ms),
            chunk_size=int(a.chunk_size),
            time_scale=float(a.ticks_per_second),
            max_frames=N,
        )
        for i, frame in enumerate(gen):
            n_done += 1
            if i in sample_set:
                samples[i] = np.asarray(frame).transpose(2, 0, 1).astype(np.uint32)
        if use_cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        ru1 = rusage_snapshot()
        frame_shape, frame_dtype = tuple(np.asarray(frame).shape), str(np.asarray(frame).dtype)
        if n_done != N:
            raise RuntimeError(f"stream_hdf5_frames yielded {n_done} frames, expected {N}")

    out = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if use_cuda else None,
        "gpu_name": torch.cuda.get_device_name(0) if use_cuda else None,
        "torch_num_threads": torch.get_num_threads(),
        "torch_imported": True,
        "import_s": import_s,
        "setup_s": setup_s,
        "cuda_init_s": cuda_init_s,
        "wall_s": wall,
        "cpu_user_s": ru1["user"] - ru0["user"],
        "cpu_sys_s": ru1["sys"] - ru0["sys"],
        "ru_maxrss_mb": ru1["maxrss_mb"],
        "cuda_max_alloc_mb": (torch.cuda.max_memory_allocated() / 2**20) if use_cuda else 0.0,
        "cuda_max_reserved_mb": (torch.cuda.max_memory_reserved() / 2**20) if use_cuda else 0.0,
        "n_frames": n_done,
        "frames_h5_event_count_total": event_count_total,
        "frame_shape": frame_shape,
        "frame_dtype": frame_dtype,
        "output_bytes": output_bytes,
        "bounded_tail": bounded,
        "chunk_size": int(a.chunk_size),
        "samples": samples,
    }
    return out


def _load_probe_windows(a, K):
    """Read K windows (spread over the recording) of raw events into memory."""
    with h5py.File(a.hdf5, "r") as f:
        ds = find_event_fields(f)
        n_total = int(ds["t"].shape[0])
        w = int(a.window_ticks)
        N = int(a.n_frames)
        idxs = np.unique(np.round(np.linspace(0, N - 1, K)).astype(int))
        windows = []
        for i in idxs:
            s_tick = int(a.t0) + i * w
            i0 = h5_searchsorted(ds["t"], s_tick, "left", n_total)
            i1 = h5_searchsorted(ds["t"], s_tick + w, "left", n_total)
            x = np.asarray(ds["x"][i0:i1]).astype(np.int64)
            y = np.asarray(ds["y"][i0:i1]).astype(np.int64)
            t = np.asarray(ds["t"][i0:i1]).astype(np.int64)
            p = np.asarray(ds["p"][i0:i1])
            windows.append((int(i), x, y, t, p))
    return windows


def _median_timed(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def worker_eventcv_kernel(a):
    os.environ.setdefault("EVENTCV_DEVICE", "cpu")
    import eventcv as ecv
    ecv.set_device("cpu")
    W, H = a.width, a.height
    K, reps = int(a.kernel_probe), int(a.kernel_reps)
    windows = _load_probe_windows(a, K)
    us_per_tick = 1e6 / float(a.ticks_per_second)
    rows = []
    exact = 0
    for i, x, y, t, p in windows:
        t_us = np.round((t - int(a.t0)) * us_per_tick).astype(np.int64)
        ev = np.stack([x, y, t_us, p.astype(np.int64)], axis=1)
        ref = np_polarity_hist(x, y, p, W, H)

        def _render():
            return ecv.from_numpy(ev, sensor_size=(W, H), time_unit="us").flatten(
                "polarity", normalize=False).numpy()

        out = _render()
        exact += int(np.array_equal(np.asarray(out).astype(np.int64), ref))
        rows.append({"frame": i, "n_events": int(x.size), "s": _median_timed(_render, reps)})
    return {
        "device": "cpu",
        "torch_imported": "torch" in sys.modules,
        "methods": {
            "eventcv_from_numpy_flatten_polarity": _probe_summary(rows, exact, len(windows)),
        },
    }


def _probe_summary(rows, exact, total):
    tot_s = sum(r["s"] for r in rows)
    tot_ev = sum(r["n_events"] for r in rows)
    return {
        "windows": total,
        "exact_vs_numpy": f"{exact}/{total}",
        "median_us_per_frame": statistics.median(r["s"] for r in rows) * 1e6,
        "mean_events_per_frame": tot_ev / max(1, total),
        "mevents_per_s": (tot_ev / tot_s / 1e6) if tot_s > 0 else None,
    }


def worker_main_kernel(a):
    import torch
    fm = load_vendor_module(a.vendor_module)
    quiet_loguru()
    W, H = a.width, a.height
    K, reps = int(a.kernel_probe), int(a.kernel_reps)
    windows = _load_probe_windows(a, K)
    has_cuda = torch.cuda.is_available()
    methods = {}

    b_gpu = fm.GeneralizedFrameBuilder(W, H, accumulator_type="polarity", hot_pixels=None)
    b_cpu = fm.GeneralizedFrameBuilder(W, H, accumulator_type="polarity", hot_pixels=None)
    b_cpu._fast_polarity_device = torch.device("cpu")
    if has_cuda:
        dev = torch.device("cuda")
        b_gpu._fast_polarity_device = dev
        torch.bincount(torch.arange(10, device=dev))
        torch.cuda.synchronize()

    tables = {k: [] for k in ("numpy_bincount", "cuda_kernel_only", "cuda_end_to_end", "torch_cpu_end_to_end")}
    exact = {k: 0 for k in tables}
    for i, x, y, t, p in windows:
        ref = np_polarity_hist(x, y, p, W, H)
        n = int(x.size)

        def _np():
            return np_polarity_hist(x, y, p, W, H)
        exact["numpy_bincount"] += 1
        tables["numpy_bincount"].append({"frame": i, "n_events": n, "s": _median_timed(_np, reps)})

        if has_cuda:
            x_t = torch.as_tensor(x, device=dev, dtype=torch.long)
            y_t = torch.as_tensor(y, device=dev, dtype=torch.long)
            p_t = torch.as_tensor(np.asarray(p), device=dev)

            def _k():
                fm.gpu_polarity_frame_2ch(x_t, y_t, p_t, H=H, W=W, device=dev)
                torch.cuda.synchronize()
            fr, _ = fm.gpu_polarity_frame_2ch(x_t, y_t, p_t, H=H, W=W, device=dev)
            got = fr[[1, 0]].cpu().numpy().astype(np.int64)  # kernel returns ch0=neg, ch1=pos
            exact["cuda_kernel_only"] += int(np.array_equal(got, ref))
            tables["cuda_kernel_only"].append({"frame": i, "n_events": n, "s": _median_timed(_k, reps)})

            def _e2e():
                return b_gpu._fast_polarity_frame_from_events(x, y, p)
            got = _e2e().transpose(2, 0, 1).astype(np.int64)
            exact["cuda_end_to_end"] += int(np.array_equal(got, ref))
            tables["cuda_end_to_end"].append({"frame": i, "n_events": n, "s": _median_timed(_e2e, reps)})

        def _cpu():
            return b_cpu._fast_polarity_frame_from_events(x, y, p)
        got = _cpu().transpose(2, 0, 1).astype(np.int64)
        exact["torch_cpu_end_to_end"] += int(np.array_equal(got, ref))
        tables["torch_cpu_end_to_end"].append({"frame": i, "n_events": n, "s": _median_timed(_cpu, reps)})

    for k, rows in tables.items():
        if rows:
            methods[k] = _probe_summary(rows, exact[k], len(windows))
    return {
        "device": "cuda" if has_cuda else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if has_cuda else None,
        "torch_version": torch.__version__,
        "methods": methods,
    }


def cmd_worker(a):
    t_start = time.perf_counter()
    if a.variant == "eventcv_cpu":
        res = worker_eventcv_cpu(a)
    elif a.variant in ("main_cuda", "main_cuda_inmem", "main_cpu", "main_cpu_inmem"):
        res = worker_main(a, a.variant[len("main_"):])
    elif a.variant == "eventcv_kernel":
        res = worker_eventcv_kernel(a)
    elif a.variant == "main_kernel":
        res = worker_main_kernel(a)
    else:
        raise SystemExit(f"unknown variant {a.variant}")

    samples = res.pop("samples", None)
    if samples and a.samples_out:
        res["samples_path"] = save_samples(a.samples_out, samples)
    res["variant"] = a.variant
    res["affinity_cpus"] = len(os.sched_getaffinity(0))
    res["threads_requested"] = a.threads
    res["worker_total_s"] = time.perf_counter() - t_start
    write_json(a.out, res)
    return 0


# --------------------------------------------------------------------------------------
# Runner helpers
# --------------------------------------------------------------------------------------
def git(*args):
    r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def mem_total_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return None


def block_device_info(path):
    src = sh(["findmnt", "-no", "SOURCE", "-T", str(path)])
    rota = sh(["lsblk", "-no", "ROTA", src]) if src else None
    return {"source": src, "rotational": (rota.strip() == "1") if rota else None}


def fincore_resident_frac(path):
    out = sh(["fincore", "--bytes", "--noheadings", "--output", "RES,SIZE", str(path)])
    if not out:
        return None
    try:
        res, size = (int(v) for v in out.split()[:2])
        return res / size if size else None
    except ValueError:
        return None


def warm_file(path, block=64 * 2**20):
    buf = bytearray(block)
    mv = memoryview(buf)
    n = 0
    t = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        while True:
            k = f.readinto(mv)
            if not k:
                break
            n += k
    dt = time.perf_counter() - t
    return {"seconds": dt, "mb_per_s": (n / 2**20) / dt if dt > 0 else None, "bytes": n}


def evict_file(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def resolve_ticks_per_second(data_yaml, tds):
    val = (data_yaml.get("format", {}) or {}).get("data", {}).get("timestamp_val")
    table = {"ns": 1e9, "nanosecond": 1e9, "nanoseconds": 1e9,
             "us": 1e6, "μs": 1e6, "microsecond": 1e6, "microseconds": 1e6,
             "ms": 1e3, "millisecond": 1e3, "milliseconds": 1e3,
             "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0}
    if val is not None:
        if isinstance(val, str):
            v = val.strip().lower()
            if v in table:
                return table[v], f"yaml timestamp_val={val!r}"
            return float(v), f"yaml timestamp_val={val!r}"
        return float(val), f"yaml timestamp_val={val!r}"
    for node in (tds, tds.parent):
        for key in ("units", "unit", "time_unit", "time_units", "timestamp_unit"):
            if key in node.attrs:
                s = node.attrs[key]
                s = s.decode() if isinstance(s, bytes) else str(s)
                s = s.strip().lower()
                for k, scale in table.items():
                    if s == k or (len(k) > 2 and k in s):
                        return scale, f"hdf5 attr {key}={s!r}"
    raise SystemExit(
        "cannot determine the timestamp unit: add `format.data.timestamp_val` (ns/us/ms/s) "
        "to the dataset yaml or a `units` attr to the timestamp dataset"
    )


def resolve_recording(spec, config, timewindow_ms, max_frames):
    dataset, seq = spec.split("/", 1)
    data_path = Path(config["data_path"])
    ext = config.get("std_format", "hdf5")
    hdf5 = data_path / dataset / seq / f"{seq}.{ext}"
    if not hdf5.exists():
        raise SystemExit(f"recording not found: {hdf5}")
    with open(REPO_ROOT / "datasets" / f"{dataset}.yaml") as f:
        dyaml = yaml.safe_load(f)
    W, H = (int(v) for v in dyaml["dataset"]["resolution"])
    with h5py.File(hdf5, "r") as f:
        ds = find_event_fields(f)
        tds = ds["t"]
        n_total = int(tds.shape[0])
        t0, tN = int(tds[0]), int(tds[n_total - 1])
        tps, tps_source = resolve_ticks_per_second(dyaml, tds)
        duration_s = (tN - t0) / tps
        if not (1.0 <= duration_s <= 48 * 3600):
            raise SystemExit(f"{spec}: implausible duration {duration_s:.3g} s with ticks/s={tps:g} ({tps_source})")
        w = int(round(timewindow_ms / 1000.0 * tps))
        n_full = (tN - t0) // w
        N = int(min(n_full, max_frames)) if max_frames else int(n_full)
        tail_tick = t0 + N * w
        events_in_range = h5_searchsorted(tds, tail_tick, "left", n_total)
        layout = {k: {"dtype": str(v.dtype), "chunks": v.chunks, "compression": v.compression}
                  for k, v in ds.items() if v is not None}
    real = Path(os.path.realpath(hdf5))
    return {
        "spec": spec, "dataset": dataset, "sequence": seq,
        "hdf5": str(hdf5), "realpath": str(real), "file_bytes": real.stat().st_size,
        "width": W, "height": H, "ticks_per_second": tps, "ticks_source": tps_source,
        "t0": t0, "tN": tN, "n_events_total": n_total, "duration_s": duration_s,
        "timewindow_ms": timewindow_ms, "window_ticks": w,
        "n_frames_full": int(n_full), "n_frames": N, "tail_tick": tail_tick,
        "bounded": N < n_full, "events_in_range": int(events_in_range),
        "layout": layout,
    }


def collect_provenance(args, vendor_path):
    head = git("rev-parse", "HEAD")
    main_blob = git("rev-parse", f"{args.main_ref}:datasets/format_data.py")
    wt_blob = git("hash-object", "datasets/format_data.py")
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_head": head,
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "main_ref": args.main_ref,
        "main_format_data_blob": main_blob,
        "main_blob_matches_expected": main_blob == MAIN_BLOB_EXPECTED,
        "worktree_format_data_blob": wt_blob,
        "vendor_identical_to_worktree": main_blob == wt_blob,
        "vendor_module": str(vendor_path),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "mem_total_gb": mem_total_gb(),
        "kernel": platform.release(),
        "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "cpufreq_governor": sh(["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"]),
        "argv": sys.argv[1:],
        "args": {k: v for k, v in vars(args).items()},
    }


def spawn_worker(variant, rec, run_dir, args, repeat, samples_out, kernel=False):
    tag = f"{variant}_{rec['dataset']}-{rec['sequence']}_tw{rec['timewindow_ms']:g}_r{repeat}"
    out_json = run_dir / "worker_json" / f"{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = run_dir / "scratch" / tag

    cmd = [
        sys.executable, "-m", MODULE, "worker",
        "--variant", variant,
        "--hdf5", rec["hdf5"],
        "--width", str(rec["width"]), "--height", str(rec["height"]),
        "--ticks-per-second", repr(float(rec["ticks_per_second"])),
        "--timewindow-ms", repr(float(rec["timewindow_ms"])),
        "--window-ticks", str(rec["window_ticks"]),
        "--t0", str(rec["t0"]), "--tn", str(rec["tN"]),
        "--n-frames", str(rec["n_frames"]),
        "--batch-size", str(args.batch_size),
        "--chunk-size", str(args.chunk_size),
        "--vendor-module", str(run_dir / "vendor" / "format_data_main.py"),
        "--scratch-dir", str(scratch),
        "--out", str(out_json),
    ]
    if rec["bounded"]:
        cmd += ["--bounded-tail-tick", str(rec["tail_tick"])]
    if samples_out is not None:
        cmd += ["--sample-idx", ",".join(str(i) for i in rec["sample_idx"]), "--samples-out", str(samples_out)]
    if args.hot_pixel_filter:
        cmd += ["--hot-pixel-filter"]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    if args.keep_scratch:
        cmd += ["--keep-scratch"]
    if kernel:
        cmd += ["--kernel-probe", str(args.kernel_probe), "--kernel-reps", str(args.kernel_reps)]

    env = dict(os.environ)
    env.update({"TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1", "EVENTCV_DEVICE": "cpu"})
    if variant.startswith("eventcv") or variant.startswith("main_cpu"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    if args.threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(args.threads)

    t = time.perf_counter()
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
    proc_wall = time.perf_counter() - t
    if p.returncode != 0:
        tail = log_path.read_text().splitlines()[-25:]
        raise RuntimeError(f"worker {tag} failed (exit {p.returncode}); log tail:\n" + "\n".join(tail))
    res = json.loads(out_json.read_text())
    res.update({
        "process_wall_s": proc_wall,
        "process_cpu_s": ru.ru_utime + ru.ru_stime,
        "process_maxrss_mb": ru.ru_maxrss / 1024.0,
        "log": str(log_path),
    })
    return res


# --------------------------------------------------------------------------------------
# Equivalence, aggregation, report
# --------------------------------------------------------------------------------------
def compare_samples(ref_path, other_path):
    a = np.load(ref_path)
    b = np.load(other_path)
    idx_a, idx_b = a["idx"], b["idx"]
    common = np.intersect1d(idx_a, idx_b)
    if common.size == 0:
        return {"frames": 0, "status": "NO_OVERLAP"}
    fa = a["frames"][np.searchsorted(idx_a, common)].astype(np.int64)
    fb = b["frames"][np.searchsorted(idx_b, common)].astype(np.int64)
    diff = np.abs(fa - fb)
    per_frame_exact = (diff.reshape(diff.shape[0], -1).max(axis=1) == 0)
    total_events = int(fa.sum())
    mismatched_events = int(diff.sum())
    out = {
        "frames": int(common.size),
        "exact_frames": int(per_frame_exact.sum()),
        "max_abs_diff": int(diff.max()),
        "mismatched_pixels": int((diff > 0).sum()),
        "mismatched_events_frac": (mismatched_events / total_events) if total_events else 0.0,
        "pos_sum_ref": int(fa[:, 0].sum()), "neg_sum_ref": int(fa[:, 1].sum()),
        "pos_sum_other": int(fb[:, 0].sum()), "neg_sum_other": int(fb[:, 1].sum()),
        "frames_compared": common.tolist(),
    }
    if out["exact_frames"] == out["frames"]:
        out["status"] = "PASS"
    elif out["max_abs_diff"] <= 2 and out["mismatched_events_frac"] <= 1e-4:
        out["status"] = "WARN"
    else:
        out["status"] = "FAIL"
    return out


def aggregate(runs):
    groups = {}
    for r in runs:
        key = (r["spec"], r["timewindow_ms"], r["variant"])
        groups.setdefault(key, []).append(r)
    rows = []
    for (spec, tw, variant), rs in groups.items():
        walls = [r["wall_s"] for r in rs]
        fps = [r["frames_per_s"] for r in rs]
        mev = [r["mevents_per_s"] for r in rs]
        rows.append({
            "spec": spec, "timewindow_ms": tw, "variant": variant, "runs": len(rs),
            "n_frames": rs[0]["n_frames"], "events_in_range": rs[0]["events_in_range"],
            "wall_median_s": statistics.median(walls), "wall_min_s": min(walls), "wall_max_s": max(walls),
            "fps_median": statistics.median(fps), "fps_min": min(fps), "fps_max": max(fps),
            "mevents_median": statistics.median(mev), "mevents_min": min(mev), "mevents_max": max(mev),
            "cpu_cores_median": statistics.median(r["cpu_cores"] for r in rs),
            "ru_maxrss_mb_max": max(r["ru_maxrss_mb"] for r in rs),
            "cuda_max_alloc_mb_max": max(r.get("cuda_max_alloc_mb", 0.0) or 0.0 for r in rs),
            "setup_median_s": statistics.median(r["setup_s"] for r in rs),
            "cuda_init_median_s": statistics.median(r.get("cuda_init_s", 0.0) for r in rs),
            "cache_resident_min": min((r.get("cache_resident_before") or 0.0) for r in rs),
            "output_bytes": max(r.get("output_bytes", 0) or 0 for r in rs),
            "bounded_tail": any(r.get("bounded_tail") for r in rs),
        })
    order = {v: i for i, v in enumerate(ALL_VARIANTS)}
    rows.sort(key=lambda r: (r["spec"], r["timewindow_ms"], order.get(r["variant"], 99)))
    return rows


def _fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return f"{v:,}"


def write_summary(results, run_dir):
    prov = results["provenance"]
    lines = []
    lines.append("# Polarity-histogram benchmark: EventCV (CPU) vs main's CUDA builder\n")
    lines.append(f"Run: `{run_dir.name}` — {prov['timestamp']}  ")
    if results.get("source_runs"):
        lines.append("Merged from: " + ", ".join(f"`{Path(s).name}`" for s in results["source_runs"]) + "  ")
    lines.append(f"Machine: {prov['cpu_model']} ({prov['cpu_count']} CPUs), {_fmt(prov['mem_total_gb'])} GB RAM, "
                 f"GPU: {prov.get('nvidia_smi') or 'n/a'}, kernel {prov['kernel']}  ")
    lines.append(f"Software: python {prov['python']}, numpy {prov['numpy']}, h5py {prov['h5py']}, "
                 f"eventcv {results.get('eventcv_version') or '?'}, torch {results.get('torch_version') or '?'} "
                 f"(CUDA {results.get('torch_cuda_version') or '?'})  ")
    lines.append(f"Code: Event-LAB `{prov['git_branch']}` @ `{(prov['git_head'] or '')[:10]}`"
                 f"{' (dirty)' if prov['git_dirty'] else ''}; main's `datasets/format_data.py` blob "
                 f"`{(prov['main_format_data_blob'] or '')[:10]}` "
                 f"({'expected' if prov['main_blob_matches_expected'] else 'UNEXPECTED'}, "
                 f"{'identical to' if prov['vendor_identical_to_worktree'] else 'differs from'} the working tree)  ")
    cfg = results["config"]
    lines.append(f"Protocol: cache={cfg['cache']}, repeats={cfg['repeats']}, batch_size={cfg['batch_size']}, "
                 f"chunk_size={cfg['chunk_size']}, hot_pixel_filter={cfg['hot_pixel_filter']}, "
                 f"threads={cfg['threads'] or 'default'}, max_frames={cfg['max_frames'] or 'all'}\n")

    lines.append("## Recordings\n")
    lines.append("| recording | sensor | events (M) | duration (s) | ticks/s | window (ms) | frames N | events in range (M) | file (GB) | compression |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for rec in results["recordings"]:
        comp = rec["layout"].get("t", {}).get("compression") or "none"
        lines.append(f"| {rec['spec']} | {rec['width']}×{rec['height']} | {rec['n_events_total']/1e6:,.1f} | "
                     f"{rec['duration_s']:,.1f} | {rec['ticks_per_second']:g} | {rec['timewindow_ms']:g} | "
                     f"{rec['n_frames']:,}{' (capped)' if rec['bounded'] else ''} | {rec['events_in_range']/1e6:,.1f} | "
                     f"{rec['file_bytes']/2**30:,.2f} | {comp} |")
    lines.append("")

    lines.append("## Throughput (median over repeats; min–max in brackets)\n")
    lines.append("| recording | window (ms) | variant | frames | wall (s) | frames/s | Mevents/s | CPU cores | peak RSS (MB) | GPU alloc (MB) | setup (s) | CUDA init (s) | cache resident | equivalence |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    eq_index = {(e["spec"], e["timewindow_ms"], e["other"]): e for e in results.get("equivalence", [])}
    for r in results["summary"]:
        eq = eq_index.get((r["spec"], r["timewindow_ms"], r["variant"]))
        eq_s = "reference" if r["variant"] == "eventcv_cpu" else (
            f"{eq['status']} ({eq['exact_frames']}/{eq['frames']} exact)" if eq else "-")
        if cfg["hot_pixel_filter"] and r["variant"] == "eventcv_cpu":
            eq_s = "n/a (hot-pixel filter on)"
        lines.append(
            f"| {r['spec']} | {r['timewindow_ms']:g} | {VARIANT_LABELS.get(r['variant'], r['variant'])} | "
            f"{r['n_frames']:,} | {r['wall_median_s']:,.2f} [{r['wall_min_s']:,.2f}–{r['wall_max_s']:,.2f}] | "
            f"{r['fps_median']:,.0f} [{r['fps_min']:,.0f}–{r['fps_max']:,.0f}] | "
            f"{r['mevents_median']:,.2f} [{r['mevents_min']:,.2f}–{r['mevents_max']:,.2f}] | "
            f"{r['cpu_cores_median']:.2f} | {r['ru_maxrss_mb_max']:,.0f} | {r['cuda_max_alloc_mb_max']:,.0f} | "
            f"{r['setup_median_s']:.2f} | {r['cuda_init_median_s']:.2f} | {r['cache_resident_min']*100:.0f}% | {eq_s} |")
    lines.append("")

    lines.append("## Speed-up of EventCV (CPU) over main's CUDA builder (median wall ratio, >1 = EventCV faster)\n")
    lines.append("| recording | window (ms) | vs CUDA → frames.h5 | vs CUDA in-memory | vs torch CPU in-memory |")
    lines.append("|---|---|---|---|---|")
    by = {(r["spec"], r["timewindow_ms"], r["variant"]): r for r in results["summary"]}
    for rec in results["recordings"]:
        k = (rec["spec"], rec["timewindow_ms"])
        e = by.get(k + ("eventcv_cpu",))
        if not e:
            continue
        def _ratio(v):
            o = by.get(k + (v,))
            return f"{o['wall_median_s']/e['wall_median_s']:.2f}×" if o else "-"
        lines.append(f"| {rec['spec']} | {rec['timewindow_ms']:g} | {_ratio('main_cuda')} | {_ratio('main_cuda_inmem')} | {_ratio('main_cpu_inmem')} |")
    lines.append("")

    if results.get("equivalence"):
        lines.append("## Numerical equivalence (sampled frames, [2,H,W] positive/negative counts)\n")
        lines.append("| recording | window (ms) | reference | compared | frames | exact | max |diff| | mismatched pixels | mismatched events | status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for e in results["equivalence"]:
            lines.append(f"| {e['spec']} | {e['timewindow_ms']:g} | {e['reference']} | {e['other']} | {e['frames']} | "
                         f"{e['exact_frames']} | {e['max_abs_diff']} | {e['mismatched_pixels']:,} | "
                         f"{e['mismatched_events_frac']*100:.4f}% | {e['status']} |")
        lines.append("")
        lines.append("PASS = bit-identical. WARN = differences confined to window edges (EventCV works at µs resolution; "
                     "ns recordings can shift an event within 1 µs of an edge). FAIL = anything else.\n")

    if results.get("kernel_probe"):
        kp = results["kernel_probe"]
        lines.append(f"## Kernel probe on {kp['spec']} ({kp['windows']} windows already sliced in memory, "
                     f"median of {kp['reps']} repetitions per window)\n")
        lines.append("| method | process | exact vs NumPy | median µs / frame | Mevents/s | mean events / frame |")
        lines.append("|---|---|---|---|---|---|")
        for proc, methods in kp["methods"].items():
            for name, m in methods.items():
                lines.append(f"| {name} | {proc} | {m['exact_vs_numpy']} | {m['median_us_per_frame']:,.0f} | "
                             f"{_fmt(m['mevents_per_s'], 1)} | {m['mean_events_per_frame']:,.0f} |")
        lines.append("")
        lines.append("`cuda_kernel_only` times `gpu_polarity_frame_2ch` on tensors already on the GPU; "
                     "`cuda_end_to_end` is `_fast_polarity_frame_from_events` (NumPy in, host filtering, H2D, bincount, D2H, NumPy out); "
                     "`eventcv_from_numpy_flatten_polarity` builds an EventStream from the array and renders it on the CPU.\n")

    lines.append("## Notes\n")
    lines.append("- All variants render frames 0..N-1 with N = floor((tN - t0) / window); Mevents/s uses the events inside that range.")
    lines.append("- `main: CUDA → frames.h5` is main's pipeline as shipped (HDF5 read, torch.bincount on CUDA, gzip-1 uint16 frames.h5 write). "
                 "`in-memory` uses the same kernel via `stream_hdf5_frames` without writing.")
    lines.append("- EventCV timings are the loop over `reader.batch()` with `repr=\"polarity\", normalize=False`; hot-pixel filtering is off unless stated.")
    lines.append("- CUDA context creation is excluded from `wall` and reported in the `CUDA init` column; `setup` is reader/builder construction.")
    lines.append("- Page cache: the recording is read sequentially once before timing and re-warmed whenever `fincore` reports <99% resident.")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


# ---- charts -------------------------------------------------------------------------
CHART = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "baseline": "#c3c2b7",
    "series": {  # fixed categorical slot order (dataviz reference palette, light mode)
        "eventcv_cpu": "#2a78d6", "main_cuda": "#eb6834", "main_cuda_inmem": "#1baf7a",
        "main_cpu_inmem": "#eda100", "main_cpu": "#e87ba4",
    },
}


def render_charts(results, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = results["summary"]
    if not summary:
        return []
    keys = []
    for rec in results["recordings"]:
        k = (rec["spec"], rec["timewindow_ms"])
        if k not in keys:
            keys.append(k)
    variants = [v for v in ALL_VARIANTS if any(r["variant"] == v for r in summary)]
    by = {(r["spec"], r["timewindow_ms"], r["variant"]): r for r in summary}
    multi_tw = len({k[1] for k in keys}) > 1
    xlabels = [f"{s}\n{tw:g} ms" if multi_tw else s for s, tw in keys]
    cfg = results["config"]
    paths = []
    for metric, ylabel, fname in (("fps", "frames per second", "chart_fps.png"),
                                  ("mevents", "million events per second", "chart_mevents.png")):
        n_k, n_v = len(keys), len(variants)
        fig_w = max(6.5, 1.15 * n_k * n_v + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=150)
        fig.patch.set_facecolor(CHART["surface"])
        ax.set_facecolor(CHART["surface"])
        slot = 0.8
        bw = min(slot / n_v * 0.86, 0.22)  # thin bars with a gap between neighbours; never fill the slot
        x = np.arange(n_k)
        ymax = 0.0
        for j, v in enumerate(variants):
            med = [by.get(k + (v,), {}).get(f"{metric}_median", np.nan) for k in keys]
            lo = [by.get(k + (v,), {}).get(f"{metric}_min", np.nan) for k in keys]
            hi = [by.get(k + (v,), {}).get(f"{metric}_max", np.nan) for k in keys]
            med, lo, hi = (np.array(a, dtype=float) for a in (med, lo, hi))
            xs = x - slot / 2 + bw / 2 + j * (slot / n_v)
            ax.bar(xs, np.nan_to_num(med), width=bw, color=CHART["series"][v], linewidth=0,
                   label=VARIANT_LABELS.get(v, v), zorder=3)
            ok = ~np.isnan(med)
            if ok.any():
                ax.errorbar(xs[ok], med[ok], yerr=[med[ok] - lo[ok], hi[ok] - med[ok]], fmt="none",
                            ecolor=CHART["muted"], elinewidth=1, capsize=2, capthick=1, zorder=4)
                ymax = max(ymax, float(np.nanmax(hi)))
                for xi, m, h in zip(xs[ok], med[ok], hi[ok]):
                    ax.annotate(f"{m:,.0f}" if metric == "fps" else f"{m:,.1f}", (xi, h),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                                fontsize=7, color=CHART["ink2"])
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8, color=CHART["ink2"])
        ax.set_ylabel(ylabel, fontsize=9, color=CHART["ink2"])
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.yaxis.grid(True, color=CHART["grid"], linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(CHART["baseline"])
        ax.tick_params(axis="both", colors=CHART["muted"], labelsize=8, length=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if metric == "fps" else f"{v:,.1f}"))
        title = "2-channel polarity histograms: EventCV on CPU vs main's CUDA builder"
        sub = (f"median of {cfg['repeats']} repeat(s), whiskers = min–max · page cache {cfg['cache']} · "
               f"{'hot-pixel filter on' if cfg['hot_pixel_filter'] else 'hot-pixel filter off'}")
        ax.set_title(title, loc="left", fontsize=10, color=CHART["ink"], pad=22)
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=7.5, color=CHART["muted"], va="bottom")
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8,
                        labelcolor=CHART["ink2"], handlelength=1.0, handleheight=1.0)
        for t in leg.get_texts():
            t.set_color(CHART["ink2"])
        fig.tight_layout()
        p = run_dir / fname
        fig.savefig(p, facecolor=CHART["surface"])
        plt.close(fig)
        paths.append(str(p))
    return paths


def write_report(results, run_dir):
    write_summary(results, run_dir)
    try:
        charts = render_charts(results, run_dir)
    except Exception as e:  # charts are a convenience; never lose the numbers over them
        logger.warning(f"chart rendering failed: {e}")
        charts = []
    return charts


# --------------------------------------------------------------------------------------
# run / report commands
# --------------------------------------------------------------------------------------
def cmd_run(args):
    with open(REPO_ROOT / args.config) as f:
        config = yaml.safe_load(f)
    if args.chunk_size is None:
        args.chunk_size = int(config.get("chunk_size", 100_000))

    unknown = [v for v in args.variants if v not in ALL_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; choose from {ALL_VARIANTS}")

    # Resolve recordings first (cheap; catches config errors before anything runs).
    recordings = []
    for tw in args.timewindows:
        for spec in args.datasets:
            rec = resolve_recording(spec, config, float(tw), args.max_frames)
            K = int(args.sample_frames)
            rec["sample_idx"] = np.unique(np.round(np.linspace(0, rec["n_frames"] - 1, K)).astype(int)).tolist() if K > 0 else []
            recordings.append(rec)
    if args.list:
        print(f"{'recording':28s} {'sensor':9s} {'events(M)':>10s} {'dur(s)':>8s} {'ticks/s':>8s} {'tw(ms)':>7s} {'N':>7s} {'ev in range(M)':>15s} {'file(GB)':>9s}")
        for r in recordings:
            print(f"{r['spec']:28s} {r['width']}x{r['height']:<5d} {r['n_events_total']/1e6:10.1f} {r['duration_s']:8.1f} "
                  f"{r['ticks_per_second']:8.0e} {r['timewindow_ms']:7g} {r['n_frames']:7d} {r['events_in_range']/1e6:15.1f} {r['file_bytes']/2**30:9.2f}")
        return 0

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (REPO_ROOT / args.out_root) / (f"{ts}_{args.tag}" if args.tag else ts)
    for sub in ("logs", "samples", "scratch", "vendor", "worker_json"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    logger.add(run_dir / "logs" / "runner.log", level="INFO")
    logger.info(f"run directory: {run_dir}")

    # Vendor main's frame builder so the benchmark runs main's blob regardless of the working tree.
    vendor_path = run_dir / "vendor" / "format_data_main.py"
    with open(vendor_path, "w") as f:
        r = subprocess.run(["git", "show", f"{args.main_ref}:datasets/format_data.py"], cwd=REPO_ROOT, stdout=f, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise SystemExit(f"could not read {args.main_ref}:datasets/format_data.py from git: {r.stderr.strip()}")
    prov = collect_provenance(args, vendor_path)
    if not prov["main_blob_matches_expected"]:
        logger.warning(f"main's datasets/format_data.py blob {prov['main_format_data_blob']} differs from the one this benchmark was written against ({MAIN_BLOB_EXPECTED})")
    logger.info(f"CPU {prov['cpu_model']} | GPU {prov['nvidia_smi']} | RAM {prov['mem_total_gb']:.0f} GB")

    results = {
        "provenance": prov,
        "config": {k: getattr(args, k) for k in ("datasets", "timewindows", "variants", "repeats", "max_frames", "batch_size",
                                                  "chunk_size", "sample_frames", "cache", "threads", "hot_pixel_filter",
                                                  "kernel_probe", "kernel_reps", "main_ref", "tag")},
        "data_device": block_device_info(config["data_path"]),
        "scratch_device": block_device_info(run_dir),
        "recordings": recordings,
        "runs": [],
        "equivalence": [],
        "kernel_probe": None,
    }
    write_json(run_dir / "results.json", results)

    def _prepare_cache(rec):
        real = rec["realpath"]
        info = {"cache_mode": args.cache, "cache_resident_before": fincore_resident_frac(real)}
        if args.cache == "warm":
            if info["cache_resident_before"] is None or info["cache_resident_before"] < 0.99:
                logger.info(f"warming page cache for {rec['spec']} ({rec['file_bytes']/2**30:.2f} GB)")
                info["warm"] = warm_file(real)
                info["cache_resident_before"] = fincore_resident_frac(real)
                logger.info(f"  read at {info['warm']['mb_per_s']:,.0f} MB/s; resident now {100*(info['cache_resident_before'] or 0):.0f}%")
        elif args.cache == "cold":
            evict_file(real)
            info["cache_resident_before"] = fincore_resident_frac(real)
            if info["cache_resident_before"] is not None and info["cache_resident_before"] > 0.01:
                logger.warning(f"posix_fadvise(DONTNEED) left {100*info['cache_resident_before']:.0f}% of {rec['spec']} resident; cold numbers are not trustworthy")
        return info

    sample_paths = {}  # (spec, tw, variant) -> npz
    for rec in recordings:
        key = (rec["spec"], rec["timewindow_ms"])
        logger.info(f"=== {rec['spec']} @ {rec['timewindow_ms']:g} ms: N={rec['n_frames']:,} frames, "
                    f"{rec['events_in_range']/1e6:,.1f} M events{' (capped by --max-frames)' if rec['bounded'] else ''}")
        for rep in range(args.repeats):
            k = rep % len(args.variants)
            order = args.variants[k:] + args.variants[:k]
            for variant in order:
                cache_info = _prepare_cache(rec)
                samples_out = None
                if rep == 0 and rec["sample_idx"]:
                    samples_out = run_dir / "samples" / f"{variant}_{rec['dataset']}-{rec['sequence']}_tw{rec['timewindow_ms']:g}.npz"
                try:
                    res = spawn_worker(variant, rec, run_dir, args, rep, samples_out)
                except RuntimeError as e:
                    logger.error(str(e))
                    if args.fail_fast:
                        raise
                    continue
                res.update({
                    "spec": rec["spec"], "dataset": rec["dataset"], "sequence": rec["sequence"],
                    "timewindow_ms": rec["timewindow_ms"], "repeat": rep,
                    "events_in_range": rec["events_in_range"],
                    "frames_per_s": res["n_frames"] / res["wall_s"],
                    "mevents_per_s": rec["events_in_range"] / res["wall_s"] / 1e6,
                    "cpu_cores": (res["cpu_user_s"] + res["cpu_sys_s"]) / res["wall_s"],
                })
                res.update(cache_info)
                if res.get("samples_path"):
                    sample_paths[key + (variant,)] = res["samples_path"]
                if variant == "eventcv_cpu":
                    results["eventcv_version"] = res.get("eventcv_version")
                elif "torch_version" in res:
                    results["torch_version"] = res["torch_version"]
                    results["torch_cuda_version"] = res.get("torch_cuda_version")
                results["runs"].append(res)
                logger.info(f"  r{rep} {variant:16s} wall {res['wall_s']:8.2f} s  {res['frames_per_s']:7.0f} fps  "
                            f"{res['mevents_per_s']:6.2f} Mev/s  cpu {res['cpu_cores']:.2f} cores  rss {res['ru_maxrss_mb']:,.0f} MB"
                            f"{'  gpu ' + format(res['cuda_max_alloc_mb'], ',.0f') + ' MB' if res.get('cuda_max_alloc_mb') else ''}")
                write_json(run_dir / "results.json", results)

        # equivalence for this recording
        ref = sample_paths.get(key + ("eventcv_cpu",))
        pairs = []
        if ref:
            for v in args.variants:
                if v != "eventcv_cpu" and key + (v,) in sample_paths:
                    pairs.append(("eventcv_cpu", v))
        if key + ("main_cuda",) in sample_paths and key + ("main_cuda_inmem",) in sample_paths:
            pairs.append(("main_cuda", "main_cuda_inmem"))
        for a_v, b_v in pairs:
            cmp = compare_samples(sample_paths[key + (a_v,)], sample_paths[key + (b_v,)])
            cmp.update({"spec": rec["spec"], "timewindow_ms": rec["timewindow_ms"], "reference": a_v, "other": b_v,
                        "informational": bool(args.hot_pixel_filter and "eventcv_cpu" in (a_v, b_v))})
            results["equivalence"].append(cmp)
            logger.info(f"  equivalence {a_v} vs {b_v}: {cmp['status']} ({cmp.get('exact_frames')}/{cmp.get('frames')} exact, max|diff|={cmp.get('max_abs_diff')})")

        # kernel probe on the first recording only
        if args.kernel_probe and results["kernel_probe"] is None:
            logger.info(f"  kernel probe: {args.kernel_probe} windows x {args.kernel_reps} reps")
            kp = {"spec": rec["spec"], "timewindow_ms": rec["timewindow_ms"], "windows": args.kernel_probe,
                  "reps": args.kernel_reps, "methods": {}}
            for kv in KERNEL_VARIANTS:
                try:
                    res = spawn_worker(kv, rec, run_dir, args, 0, None, kernel=True)
                    kp["methods"][kv] = res["methods"]
                except RuntimeError as e:
                    logger.error(str(e))
                    if args.fail_fast:
                        raise
            results["kernel_probe"] = kp
        write_json(run_dir / "results.json", results)

    results["summary"] = aggregate(results["runs"])
    write_json(run_dir / "results.json", results)
    write_json(run_dir / "equivalence.json", results["equivalence"])
    charts = write_report(results, run_dir)
    if not args.keep_scratch:
        shutil.rmtree(run_dir / "scratch", ignore_errors=True)
    logger.info(f"done: {run_dir / 'summary.md'}" + (f" + {', '.join(Path(c).name for c in charts)}" if charts else ""))
    print(run_dir)
    return 0


def merge_results(run_dirs):
    """Merge several runs' results.json into one (recordings/runs/equivalence concatenated)."""
    merged = None
    seen = set()
    repeats = set()
    for rd in run_dirs:
        r = json.loads((Path(rd) / "results.json").read_text())
        repeats.add(r["config"].get("repeats"))
        if merged is None:
            merged = {k: (list(v) if isinstance(v, list) else v) for k, v in r.items()}
            merged["recordings"], merged["runs"], merged["equivalence"] = [], [], []
            merged["source_runs"] = []
        merged["source_runs"].append(str(rd))
        for rec in r["recordings"]:
            key = (rec["spec"], rec["timewindow_ms"])
            if key not in seen:
                seen.add(key)
                merged["recordings"].append(rec)
        merged["runs"].extend(r["runs"])
        merged["equivalence"].extend(r.get("equivalence") or [])
        if not merged.get("kernel_probe") and r.get("kernel_probe"):
            merged["kernel_probe"] = r["kernel_probe"]
        for k in ("eventcv_version", "torch_version", "torch_cuda_version"):
            merged.setdefault(k, r.get(k))
    merged["config"] = dict(merged["config"])
    merged["config"]["repeats"] = "/".join(str(x) for x in sorted(repeats, reverse=True)) if len(repeats) > 1 else repeats.pop()
    merged["config"]["datasets"] = [rec["spec"] for rec in merged["recordings"]]
    merged["summary"] = aggregate(merged["runs"])
    return merged


def cmd_report(args):
    run_dirs = [Path(d).resolve() for d in args.run_dir]
    if len(run_dirs) == 1 and not args.out:
        run_dir = run_dirs[0]
        results = json.loads((run_dir / "results.json").read_text())
        if not results.get("summary"):
            results["summary"] = aggregate(results["runs"])
    else:
        results = merge_results(run_dirs)
        run_dir = Path(args.out).resolve() if args.out else run_dirs[0].parent / (run_dirs[0].name + "_merged")
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "results.json", results)
        write_json(run_dir / "equivalence.json", results["equivalence"])
    charts = write_report(results, run_dir)
    print(run_dir / "summary.md")
    for c in charts:
        print(c)
    return 0


# --------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(prog=f"python -m {MODULE}", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("run", help="run the benchmark")
    r.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, metavar="DATASET/SEQ")
    r.add_argument("--timewindows", nargs="+", type=float, default=[50.0], metavar="MS")
    r.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, metavar="VARIANT", help=f"choose from {ALL_VARIANTS}")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--max-frames", type=int, default=None, help="cap N (smoke runs); main_cuda then uses a bounded-tail reader")
    r.add_argument("--batch-size", type=int, default=64, help="EventCV reader.batch() size")
    r.add_argument("--chunk-size", type=int, default=None, help="events per HDF5 read for main's builder (default: config.yaml chunk_size)")
    r.add_argument("--sample-frames", type=int, default=16, help="frames sampled for the equivalence check (0 = skip)")
    r.add_argument("--cache", choices=["warm", "cold", "none"], default="warm")
    r.add_argument("--threads", type=int, default=None, help="pin OMP/MKL/torch threads in workers")
    r.add_argument("--hot-pixel-filter", action="store_true", help="EventCV hot_pixel_filter=True (as the 1.1.0 baselines use); disables equivalence")
    r.add_argument("--kernel-probe", type=int, default=0, metavar="K", help="also time the histogram kernels on K pre-sliced windows of the first recording")
    r.add_argument("--kernel-reps", type=int, default=20)
    r.add_argument("--main-ref", default="main", help="git ref whose datasets/format_data.py is benchmarked")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--out-root", default="output/benchmarks/polarity_histogram")
    r.add_argument("--tag", default=None)
    r.add_argument("--keep-scratch", action="store_true")
    r.add_argument("--fail-fast", action="store_true")
    r.add_argument("--list", action="store_true", help="resolve recordings and exit")
    r.set_defaults(func=cmd_run)

    w = sp.add_parser("worker", help="internal: run one variant in this process")
    w.add_argument("--variant", required=True)
    w.add_argument("--hdf5", required=True)
    w.add_argument("--width", type=int, required=True)
    w.add_argument("--height", type=int, required=True)
    w.add_argument("--ticks-per-second", type=float, required=True)
    w.add_argument("--timewindow-ms", type=float, required=True)
    w.add_argument("--window-ticks", type=int, required=True)
    w.add_argument("--t0", type=int, required=True)
    w.add_argument("--tn", type=int, required=True)
    w.add_argument("--n-frames", type=int, required=True)
    w.add_argument("--bounded-tail-tick", type=int, default=None)
    w.add_argument("--sample-idx", default="")
    w.add_argument("--samples-out", default=None)
    w.add_argument("--batch-size", type=int, default=64)
    w.add_argument("--chunk-size", type=int, default=100_000)
    w.add_argument("--hot-pixel-filter", action="store_true")
    w.add_argument("--threads", type=int, default=None)
    w.add_argument("--vendor-module", required=True)
    w.add_argument("--scratch-dir", required=True)
    w.add_argument("--keep-scratch", action="store_true")
    w.add_argument("--kernel-probe", type=int, default=64)
    w.add_argument("--kernel-reps", type=int, default=20)
    w.add_argument("--out", required=True)
    w.set_defaults(func=cmd_worker)

    rp = sp.add_parser("report", help="regenerate summary.md and charts from results.json; several run dirs are merged")
    rp.add_argument("run_dir", nargs="+")
    rp.add_argument("--out", default=None, help="output dir for a merged report (default: <first run>_merged)")
    rp.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
