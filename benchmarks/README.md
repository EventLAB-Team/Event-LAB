# Polarity-histogram benchmark: EventCV (CPU only) vs. main's CUDA frame builder

`polarity_histogram_benchmark.py` measures how fast **2-channel polarity histogram**
frames (positive-polarity counts in channel 0, negative in channel 1, fixed time
windows) are generated from an Event-LAB HDF5 recording by:

| variant | what it times | code path |
|---|---|---|
| `eventcv_cpu` (primary) | EventCV on the CPU only, as used by Event-LAB 1.1.0 | `ecv.open(hdf5, dt_ms=…).with_repr("polarity", normalize=False)` then `reader.batch()` in batches of `--batch-size` frames (the pattern in `baselines/lens.py`) |
| `main_cuda` (primary) | the original pipeline from branch `main`, as shipped: HDF5 read → `torch.bincount` on CUDA → `frames.h5` (uint16, gzip-1) | `GeneralizedFrameBuilder("polarity").build_frames()` from `datasets/format_data.py` |
| `main_cuda_inmem` | same CUDA kernel without the `frames.h5` write | `GeneralizedFrameBuilder.stream_hdf5_frames()` |
| `main_cpu_inmem`, `main_cpu` (opt-in) | main's torch-CPU fallback | same as above with `_fast_polarity_device = cpu` |

The question it answers: *is EventCV competitive on CPU alone against the GPU builder?*
EventCV never touches the GPU here — its worker runs with `EVENTCV_DEVICE=cpu`,
`CUDA_VISIBLE_DEVICES=""`, asserts `eventcv.get_device() == "cpu"` and records that
torch was never imported.

## Running

Always go through pixi (a bare `python` on this machine resolves `eventcv` to a
different checkout):

```bash
pixi run python -m benchmarks.polarity_histogram_benchmark run --list        # what would run
pixi run bench-polarity --datasets qcr_event/normal1 --max-frames 1000 --repeats 1 --kernel-probe 64   # ~1 min smoke
pixi run bench-polarity                                                       # default: 4 recordings, 3 repeats
pixi run bench-polarity --datasets nsavp/R0_FA0 --repeats 2 --tag nsavp       # the big one, separately
pixi run python -m benchmarks.polarity_histogram_benchmark report <run_dir> [<run_dir> …] [--out DIR]   # regenerate (or merge) summary/charts
```

Useful options (`run --help` lists all): `--datasets DATASET/SEQ …` (files are
`{data_path}/{dataset}/{seq}/{seq}.hdf5`, resolution and timestamp unit come from
`datasets/<dataset>.yaml`), `--timewindows MS …`, `--variants …`, `--repeats`,
`--max-frames` (smoke runs), `--batch-size`, `--cache warm|cold|none`,
`--hot-pixel-filter` (EventCV's pre-scan, as the 1.1.0 baselines use it — turns the
equivalence check off), `--kernel-probe K` (histogram kernels on K pre-sliced windows
of the first recording), `--threads N`, `--tag`, `--keep-scratch`, `--fail-fast`.

Results go to `output/benchmarks/polarity_histogram/<timestamp>[_tag]/`:
`summary.md` (the tables), `chart_fps.png`, `chart_mevents.png`, `results.json`
(everything, incl. provenance), `equivalence.json`, `logs/` (one per worker),
`samples/` (the frames used for the equivalence check), `vendor/format_data_main.py`
(main's builder, extracted with `git show main:datasets/format_data.py`).

## Protocol

- **One process per run.** Each (variant, recording, window, repeat) is a separate
  subprocess, so peak RSS, CPU time and GPU memory are per variant and the EventCV
  process cannot initialise CUDA.
- **Equal work.** For every recording the runner computes the window in ticks
  `w = round(dt_ms/1000 · ticks_per_second)` and `N = floor((t_last − t_first)/w)`
  (capped by `--max-frames`); every variant renders exactly frames `0..N-1`.
  frames/s = N / wall, Mevents/s = events with `t < t_first + N·w` / wall.
  (EventCV's `n_slices` and `stream_hdf5_frames` would otherwise also emit the
  trailing partial window that `build_frames` drops.)
- **Page cache.** Recordings live on a rotational disk, so a cold read dominates
  everything. Default `--cache warm`: the file is read sequentially once, and before
  every timed run `fincore` must report ≥99 % resident (it is re-warmed otherwise —
  `main_cuda`'s multi-GB `frames.h5` writes can evict it). `--cache cold` evicts the
  file with `posix_fadvise(DONTNEED)` before each run (no sudo needed).
- **Interleaving.** Within a recording the variant order is rotated on every repeat.
  Reported numbers are the median over repeats with min–max whiskers.
- **What is inside `wall`.** EventCV: the `reader.batch()` loop (reader construction is
  `setup`). main: the `build_frames()` / generator call; CUDA context creation is done
  before the timer and reported separately (`CUDA init`). `main_cuda` includes the
  gzip-1 `frames.h5` write to the run directory (NVMe).
- **Same inputs.** Hot-pixel filtering is off for all variants (main: `hot_pixels=None`,
  EventCV: `hot_pixel_filter=False`); no sequence offsets; `time_scale` is always
  passed explicitly to main's builder (its auto-detect assumes µs for files without a
  unit attribute, which is wrong for the ns recordings).
- **Equivalence.** `--sample-frames K` frames spread over the sequence are dumped by
  every variant as `[2,H,W]` counts (main's `[H,W,2]` is transposed; `frames.h5` is
  read back and checked for uint16 clipping) and compared to EventCV's. PASS =
  bit-identical; WARN = only sub-µs window-edge differences (EventCV keeps timestamps
  in µs; several recordings are in ns); FAIL = anything else.
- **Provenance.** `results.json` records the git HEAD, the blob hash of
  `main:datasets/format_data.py` (the benchmark warns if it is not the one it was
  written against), eventcv / torch / CUDA versions, CPU, GPU, RAM, disk type and
  the full command line.

## Reading the results

`summary.md` has, per recording and variant: frames, wall time, frames/s, Mevents/s,
CPU cores used (CPU time / wall — EventCV's CPU path is single-threaded, so ≈1.0),
peak RSS, GPU allocation, setup and CUDA-init times, cache residency and the
equivalence status; then a speed-up table (median-wall ratio, >1 = EventCV faster)
and, with `--kernel-probe`, a table of the histogram kernels alone on pre-sliced
events (EventCV `from_numpy(...).flatten("polarity")`, NumPy `bincount`, the CUDA
kernel with and without host↔device copies, torch on CPU).

## Pitfalls baked into the script

- `repr="polarity"` alone returns a uint8 0–255 image; the raw counts need
  `normalize=False`.
- `build_frames(max_frames=N)` still reads the whole file, so `--max-frames` runs use
  a bounded-tail subclass (flagged `bounded_tail` in the results) that clamps the last
  timestamp; full runs use main's class unmodified.
- `main_cuda_inmem` uses a different HDF5 streamer than `build_frames` (rolling buffer,
  default 1 MB h5 chunk cache vs 64 MB), so it differs from `main_cuda` in I/O pattern
  as well as in the missing write.
- The nsavp recording writes ≈26 GB of uint16 frames before gzip in `main_cuda`; keep
  enough free space in the run directory. Scratch is deleted after the sample frames
  are read back unless `--keep-scratch`.
