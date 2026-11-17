"""
Render EventCV frames to the on-disk layouts external baselines expect.

Several upstream methods (ensemble-event-vpr, VPR-methods-evaluation) do not read
events at all -- they consume folders of images. This module is the single place
that turns an event recording into such a folder via EventCV,
replacing the pre-rendered frame directories the v1.1.0 migration retired.

Event data is read exclusively through EventCV: one reader per (recording, time
window), rendered lazily so a multi-gigabyte recording never lands in memory.
"""
import json
import os

from PIL import Image
from utils.representations import open_reader, image_uint8
from loguru import logger

# Both upstream consumers order frames with a plain lexicographic sort, so the
# index must be zero padded. EventCV's export_png defaults to 5 digits; Event-LAB
# has always used 6.
FRAME_DIGITS = 6
FRAME_PREFIX = "frame_"


def kept_place_indices(n_slices, timewindow_ms, min_gap_sec):
    """
    Greedy temporal thinning over fixed-duration frames.

    EventCV renders uniform dt_ms frames, so frame i sits at a deterministic
    time i * dt_sec (the absolute offset is constant and cancels in the
    pairwise gap). This reproduces the greedy "keep frames >= min_gap_sec
    apart" filter that used to live in make_frame_source, without needing any
    per-frame tick metadata.
    """
    if not min_gap_sec or min_gap_sec <= 0:
        return list(range(n_slices))
    dt_sec = float(timewindow_ms) / 1000.0
    if dt_sec <= 0:
        return list(range(n_slices))

    kept = []
    last_kept_t = None
    for i in range(n_slices):
        t = i * dt_sec
        if last_kept_t is None or (t - last_kept_t) >= min_gap_sec:
            kept.append(i)
            last_kept_t = t
    return kept


def render_png_sequence(hdf5_path, out_dir, *, timewindow_ms, offset_ms=None,
                        sensor_size=None, config=None,
                        min_gap_sec=0.0):
    """
    Render one recording to `out_dir/frame_%06d.png` and return the frame times.

    Returns the time of each written frame in seconds relative to the start of
    the framing (i.e. `i * timewindow_ms / 1000` for the kept indices), which is
    exactly what ensemble-event-vpr's `timestamps.txt` expects.

    The selected EventCV representation is adapted to RGB for image-only models.
    """
    os.makedirs(out_dir, exist_ok=True)
    reader = open_reader(hdf5_path, timewindow_ms, offset_ms, sensor_size, config)
    n_slices = int(reader.n_slices)
    # Window manifests are already one window per place; thinning would desync GT.
    if getattr(reader, 'fixed_places', False):
        min_gap_sec = 0
    kept = kept_place_indices(n_slices, timewindow_ms, min_gap_sec)
    if not kept:
        raise ValueError(
            f"No frames to render from {hdf5_path} "
            f"(n_slices={n_slices}, min_gap_sec={min_gap_sec}).")

    dt_sec = float(timewindow_ms) / 1000.0
    if len(kept) == n_slices:
        # Fast path: stream every window lazily, never materialising the recording.
        frames = reader.windows()
    else:
        frames = (reader.slice(i) for i in kept)

    for index, frame in enumerate(frames):
        rgb = image_uint8(frame).transpose(1, 2, 0)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"{FRAME_PREFIX}{index:0{FRAME_DIGITS}d}.png"))
    logger.info(f"Rendered {len(kept)} frames ({timewindow_ms} ms) to {out_dir}")
    return [i * dt_sec for i in kept]


def write_frame_sidecars(frames_dir, times_s, *, start_time_ns, timewindow_ms,
                         width, height, metadata_dir=None):
    """
    Write the `timestamps.txt` + `metadata.json` pair ensemble-event-vpr requires.

    Upstream's `load_e2vid_abs_times` reads relative seconds from
    `<frames_dir>/timestamps.txt` and adds `start_time_ns` from the metadata file
    in the PARENT directory, raising KeyError if that key is absent. The
    metadata.json written by the retired frame pipeline carried `start_tick`
    instead, so this writes the file rather than copying one.
    """
    with open(os.path.join(frames_dir, "timestamps.txt"), "w") as handle:
        handle.writelines(f"{t:.18f}\n" for t in times_s)

    target = metadata_dir or os.path.dirname(os.path.normpath(frames_dir))
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "metadata.json"), "w") as handle:
        json.dump({
            "start_time_ns": int(start_time_ns),
            "timewindow_ms": float(timewindow_ms),
            "timewindow_ns": int(float(timewindow_ms) * 1e6),
            "total_frames": len(times_s),
            "width": int(width),
            "height": int(height),
            "source": "eventcv",
        }, handle, indent=2)


def offsets_from_dataset_config(dataset_config, ref_name, query_name):
    """
    Resolve the per-sequence stream offsets in milliseconds, or (None, None).

    EventCV's bare `offset` is an absolute timestamp in milliseconds, which is
    exactly what utils.utils.convert_offset produces from the seconds-scale
    values in the dataset YAML.
    """
    from utils.utils import convert_offset

    if "other" in dataset_config and "offset" in dataset_config["other"]:
        return convert_offset(
            dataset_config["other"]["offset"][ref_name],
            dataset_config["other"]["offset"][query_name],
            dataset_config["other"]["offset_time_scale"])
    return None, None
