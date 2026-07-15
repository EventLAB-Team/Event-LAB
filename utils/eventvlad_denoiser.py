import os
# Let unsupported ops fall back to CPU when running on Apple MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import time
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from loguru import logger
from tqdm import tqdm
sys.path.append('./baselines/EventVLAD')
from networks import EventDenoiser  # add other models in build_model if needed
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import eventcv as ecv


# ----------------- Device -----------------
def pick_device(use_gpu: bool, device_override: str = None) -> torch.device:
    """Select a compute device: explicit override, else cuda -> mps -> cpu when use_gpu."""
    if device_override:
        return torch.device(device_override)
    if not use_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device):
    """Block until queued device work finishes (for accurate timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

# ----------------- Model factory -----------------
def build_model(model_type: str, dep_u: int, dep_s: int, slope: float) -> torch.nn.Module:
    mt = model_type.lower()
    if mt in ["event_denoiser", "denoiser", "default"]:
        return EventDenoiser(3, slope=slope, dep_U=dep_u, dep_S=dep_s)
    raise ValueError(f"Unknown model_type: {model_type}")


def clean_state_dict(sd):
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def load_checkpoint_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = clean_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
    # DataParallel only makes sense for (multi-)CUDA; MPS/CPU use a plain module.
    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    model.eval()
    return model


# ----------------- EventCV stream -----------------
def open_event_reader(hdf5_path, dt_ms, offset, repr_name, hot_pixel_filter):
    """Open an EventCV reader with fixed-duration framing (dt_ms) and a chosen representation."""
    kwargs = {"dt_ms": dt_ms, "repr": repr_name, "hot_pixel_filter": hot_pixel_filter}
    if offset is not None:
        kwargs["offset"] = offset
    return ecv.open(hdf5_path, **kwargs)


def _to_numpy(x):
    """Coerce an EventCV/torch batch result into a NumPy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def iter_reader_batches(reader, batch_size):
    """
    Iterate over the reader's fixed frames in dense batches using reader.batch().

    Yields
    ------
    np.ndarray of shape [B, C, H, W]
        A rendered batch of consecutive frame slices.
    """
    n = int(reader.n_slices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = reader.batch(list(range(start, end)))  # [B, C, H, W]
        yield _to_numpy(batch)


# ----------------- Frame utils -----------------
def make_divisible(img, mult):
    if mult <= 1:
        return img
    H, W = img.shape[:2]
    return img[: H - (H % mult) if H % mult else H,
               : W - (W % mult) if W % mult else W]


def collapse_frame(frame: np.ndarray, npy_mode: str) -> np.ndarray:
    """Collapse a single [C,H,W] (or [H,W]) rendered slice to a 2-D grayscale frame."""
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3:
        c = frame.shape[0]
        if c == 1:
            return frame[0]
        if c == 2:
            if npy_mode == "sum":
                return frame[0] + frame[1]
            if npy_mode == "pos":
                return frame[0]
            if npy_mode == "neg":
                return frame[1]
            if npy_mode == "diff":
                return frame[0] - frame[1]
            raise ValueError(f"Unsupported npy_mode: {npy_mode}")
        # Unknown channel count: average across channels.
        return frame.mean(axis=0)
    raise ValueError(f"Unsupported frame shape from reader: {frame.shape}")


def normalize_frame(frame: np.ndarray, percentile: float) -> np.ndarray:
    """Percentile-normalize a grayscale frame to [0,1] (matches the old NumPy behaviour)."""
    frame = frame.astype(np.float32, copy=False)
    if frame.size == 0:
        return frame
    vmax = float(np.percentile(np.abs(frame), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(np.abs(frame))) if frame.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return np.clip(frame / vmax, 0.0, 1.0).astype(np.float32, copy=False)


def prep_input_triplet(i0: np.ndarray, i1: np.ndarray, i2: np.ndarray,
                       size: int, dep_u: int, rotate180: bool) -> torch.Tensor:
    """Prepare a 1x3xHxW tensor from three grayscale [0,1] images."""
    m = 2 ** dep_u if dep_u > 0 else 1
    i0 = make_divisible(i0, m)
    i1 = make_divisible(i1, m)
    i2 = make_divisible(i2, m)

    if size and size > 0:
        i0 = cv2.resize(i0, (size, size), interpolation=cv2.INTER_AREA)
        i1 = cv2.resize(i1, (size, size), interpolation=cv2.INTER_AREA)
        i2 = cv2.resize(i2, (size, size), interpolation=cv2.INTER_AREA)

    if rotate180:
        i0 = cv2.rotate(i0, cv2.ROTATE_180)
        i1 = cv2.rotate(i1, cv2.ROTATE_180)
        i2 = cv2.rotate(i2, cv2.ROTATE_180)

    t0 = torch.from_numpy(i0[None, ...])  # (1,H,W)
    t1 = torch.from_numpy(i1[None, ...])
    t2 = torch.from_numpy(i2[None, ...])
    x = torch.cat([t0, t1, t2], dim=0)[None, ...].contiguous().float()  # (1,3,H,W)
    return x


def tensor_to_uint8(img_t):
    """Accepts (1,1,H,W) or (1,C,H,W) or (H,W), returns uint8 HxW image."""
    if torch.is_tensor(img_t):
        img = img_t.detach().cpu().numpy()
    else:
        img = np.asarray(img_t)
    if img.ndim == 4:
        img = img[:, 0, ...]
    if img.ndim == 3:
        img = img[0]
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Denoise triplets streamed from an EventCV recording.")
    ap.add_argument("--hdf5_path", required=True, help="Path to the formatted event HDF5 recording")
    ap.add_argument("--dt_ms", type=float, required=True, help="Fixed frame duration (timewindow) in ms")
    ap.add_argument("--offset", type=float, default=None, help="Framing origin offset in ms")
    ap.add_argument("--repr", default="count", help="EventCV representation to render (e.g. count)")
    ap.add_argument("--hot_pixel_filter", action=argparse.BooleanOptionalAction, default=True,
                    help="Apply EventCV hot-pixel filtering")
    ap.add_argument("--batch_size", type=int, default=512, help="Number of slices rendered per reader.batch() call")
    ap.add_argument("--model_path", required=True, help="Path to model checkpoint")
    ap.add_argument("--model_type", default="event_denoiser", help="Model type (e.g., event_denoiser)")
    ap.add_argument("--dep_u", type=int, default=5, help="Model dep_U (divisibility power)")
    ap.add_argument("--dep_s", type=int, default=5, help="Model dep_S")
    ap.add_argument("--slope", type=float, default=0.2, help="Model slope param")
    ap.add_argument("--size", type=int, default=256, help="Resize to NxN (0 to disable)")
    ap.add_argument("--rotate180", action="store_true", help="Rotate inputs 180 degrees")
    ap.add_argument("--use_gpu", action="store_true", help="Use an accelerator if available (cuda, then Apple mps)")
    ap.add_argument("--device", default=None,
                    help="Explicit device override (e.g. cuda, mps, cpu); takes precedence over --use_gpu")
    ap.add_argument("--stride", type=int, default=1, help="Sliding window stride over triplets")
    ap.add_argument("--show", type=int, default=3, help="Show first N results (0 = headless)")
    ap.add_argument("--save_dir", default=None, help="Optional: save denoised PNGs here")

    # Kept for compatibility with existing commands/configs.
    ap.add_argument(
        "--npy_mode",
        default="sum",
        choices=["sum", "pos", "neg", "diff"],
        help="How to collapse a 2-channel polarity representation to grayscale",
    )
    ap.add_argument(
        "--npy_percentile",
        type=float,
        default=99.0,
        help="Percentile for normalization to [0,1]",
    )

    args = ap.parse_args()

    # Select the compute device (cuda -> mps -> cpu, or an explicit override).
    device = pick_device(args.use_gpu, args.device)

    # Build + load model
    logger.info(f"Building model: {args.model_type}")
    net = build_model(args.model_type, dep_u=args.dep_u, dep_s=args.dep_s, slope=args.slope)
    net = load_checkpoint_into_model(net, args.model_path, device=device)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Open the EventCV stream directly instead of loading frames from disk.
    logger.info(f"Opening EventCV stream: {args.hdf5_path} (dt_ms={args.dt_ms}, offset={args.offset})")
    reader = open_event_reader(
        args.hdf5_path,
        dt_ms=args.dt_ms,
        offset=args.offset,
        repr_name=args.repr,
        hot_pixel_filter=args.hot_pixel_filter,
    )
    n_slices = int(reader.n_slices)

    shown = 0
    timings = []

    logger.info(f"Begin denoising {n_slices} slices on {device.type}")

    window = []
    start_idx = 0
    processed = 0

    pbar = tqdm(total=n_slices, desc="Denoising", unit="slice")
    for batch in iter_reader_batches(reader, args.batch_size):
        # batch: [B, C, H, W]
        for i in range(batch.shape[0]):
            frame = collapse_frame(batch[i], args.npy_mode)
            frame = normalize_frame(frame, args.npy_percentile)

            window.append(frame)
            pbar.update(1)

            if len(window) < 3:
                continue

            if start_idx % args.stride == 0:
                i0, i1, i2 = window[0], window[1], window[2]

                x = prep_input_triplet(
                    i0,
                    i1,
                    i2,
                    size=args.size,
                    dep_u=args.dep_u,
                    rotate180=args.rotate180,
                )

                x = x.to(device, non_blocking=True)

                with torch.no_grad():
                    sync_device(device)

                    t0 = time.perf_counter()
                    y = net(x)

                    sync_device(device)

                    t1 = time.perf_counter()
                    timings.append(t1 - t0)

                den = tensor_to_uint8(y)
                noisy_mean = tensor_to_uint8(x.mean(dim=1, keepdim=True))

                # Equivalent to old center-frame naming.
                base = f"frame_{start_idx + 1:06d}"

                if args.save_dir:
                    out_path = os.path.join(args.save_dir, f"{base}_denoised.png")
                    cv2.imwrite(out_path, den)

                if args.show and shown < args.show:
                    plt.figure(figsize=(8, 4))

                    plt.subplot(1, 2, 1)
                    plt.imshow(noisy_mean, cmap="gray")
                    plt.title(f"Noisy (mean)\n{base}", fontsize=9)
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(den, cmap="gray")
                    plt.title("Denoised", fontsize=9)
                    plt.axis("off")

                    plt.tight_layout()
                    plt.show()
                    shown += 1

                processed += 1

            window.pop(0)
            start_idx += 1
    pbar.close()

    if processed == 0:
        raise RuntimeError(
            f"No triplets processed from {args.hdf5_path}. "
            f"Frame count={n_slices}, stride={args.stride}"
        )

    if timings:
        avg = sum(timings) / len(timings)
        logger.info(f"Processed {len(timings)} triplets. Avg time: {avg:.4f}s (per inference)")

if __name__ == "__main__":
    main()
