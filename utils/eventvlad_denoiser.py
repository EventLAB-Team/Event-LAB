import os
import glob
import time
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# ---- import your models here ----
import sys
sys.path.append('./baselines/EventVLAD')
from networks import EventDenoiser  # add other models in build_model if needed


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


def load_checkpoint_into_model(model, ckpt_path, use_gpu):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = clean_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
    if use_gpu and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    model.eval()
    return model


# ----------------- I/O utils -----------------
def make_divisible(img, mult):
    if mult <= 1:
        return img
    H, W = img.shape[:2]
    return img[: H - (H % mult) if H % mult else H,
               : W - (W % mult) if W % mult else W]


def to_float01(arr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Normalize to [0,1] sensibly (handles ints, large floats, and already-normalized floats)."""
    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr
    maxv = float(arr.max())
    # already normalized?
    if maxv <= 1.0 + 1e-6 and arr.min() >= -1e-6:
        return np.clip(arr, 0.0, 1.0)
    # 8-bit style?
    if np.issubdtype(arr.dtype, np.integer) and maxv <= 255:
        return np.clip(arr / 255.0, 0.0, 1.0)
    vmax = float(np.percentile(arr, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = max(maxv, 1.0)
    return np.clip(arr / vmax, 0.0, 1.0)


def load_gray_float_from_image(path: str) -> np.ndarray:
    """Load a standard image file as grayscale float32 in [0,1]."""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return (im.astype(np.float32) / 255.0)


def load_gray_float_from_npy(path: str, npy_mode: str, percentile: float) -> np.ndarray:
    """Load .npy or .npz. Accepts (H,W) or (H,W,2) [pos,neg]. Returns float32 in [0,1]."""
    if path.lower().endswith(".npz"):
        data = np.load(path)
        if "pos" in data and "neg" in data:
            arr = np.stack([data["pos"], data["neg"]], axis=-1)
        else:
            # fall back: first array in file
            key = list(data.keys())[0]
            arr = data[key]
    else:
        arr = np.load(path)

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 2:
        if npy_mode == "sum":
            gray = arr[..., 0] + arr[..., 1]
        elif npy_mode == "pos":
            gray = arr[..., 0]
        elif npy_mode == "neg":
            gray = arr[..., 1]
        elif npy_mode == "diff":
            # map to non-negative for denoiser input
            diff = arr[..., 0].astype(np.float32) - arr[..., 1].astype(np.float32)
            gray = np.abs(diff)
        else:
            raise ValueError(f"Unsupported npy_mode: {npy_mode}")
    else:
        # Unknown shape: take mean across channels
        gray = arr.mean(axis=-1) if arr.ndim == 3 else arr.squeeze()

    return to_float01(gray, percentile=percentile)


def load_gray_float_any(path: str, npy_mode: str, percentile: float) -> np.ndarray:
    if path.lower().endswith((".npy", ".npz")):
        return load_gray_float_from_npy(path, npy_mode=npy_mode, percentile=percentile)
    return load_gray_float_from_image(path)


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
    ap = argparse.ArgumentParser(description="Denoise triplets from a folder (supports images and NumPy arrays).")
    ap.add_argument("--input_dir", help="Folder with inputs (.png/.jpg/.bmp/.tif/.npy/.npz)")
    ap.add_argument("--model_path", required=True, help="Path to model checkpoint")
    ap.add_argument("--model_type", default="event_denoiser", help="Model type (e.g., event_denoiser)")
    ap.add_argument("--dep_u", type=int, default=5, help="Model dep_U (divisibility power)")
    ap.add_argument("--dep_s", type=int, default=5, help="Model dep_S")
    ap.add_argument("--slope", type=float, default=0.2, help="Model slope param")
    ap.add_argument("--size", type=int, default=256, help="Resize to NxN (0 to disable)")
    ap.add_argument("--rotate180", action="store_true", help="Rotate inputs 180 degrees")
    ap.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    ap.add_argument("--stride", type=int, default=1, help="Sliding window stride over triplets")
    ap.add_argument("--show", type=int, default=3, help="Show first N results (0 = headless)")
    ap.add_argument("--save_dir", default=None, help="Optional: save denoised PNGs here")
    # NumPy handling
    ap.add_argument("--npy_mode", default="sum", choices=["sum", "pos", "neg", "diff"],
                    help="How to convert (H,W,2) npy/npz to grayscale")
    ap.add_argument("--npy_percentile", type=float, default=99.0,
                    help="Percentile for normalization to [0,1] when scaling NumPy arrays")
    args = ap.parse_args()

    # Collect files (images + numpy arrays)
    numpy_exts = ("*.npy", "*.npz")
    files = []
    for e in numpy_exts:
        files.extend(glob.glob(os.path.join(args.input_dir, e)))
    files = sorted(files)
    if len(files) < 3:
        raise RuntimeError(f"Need at least 3 inputs in {args.input_dir}; found {len(files)}")

    # Build + load model
    print("Building model:", args.model_type)
    net = build_model(args.model_type, dep_u=args.dep_u, dep_s=args.dep_s, slope=args.slope)
    net = load_checkpoint_into_model(net, args.model_path, use_gpu=args.use_gpu)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    shown = 0
    timings = []

    print("Begin testing on", "GPU" if (args.use_gpu and torch.cuda.is_available()) else "CPU")
    for i in range(0, len(files) - 2, args.stride):
        f0, f1, f2 = files[i], files[i + 1], files[i + 2]

        # Load as grayscale float in [0,1], handling .npy/.npz or standard images
        i0 = load_gray_float_any(f0, npy_mode=args.npy_mode, percentile=args.npy_percentile)
        i1 = load_gray_float_any(f1, npy_mode=args.npy_mode, percentile=args.npy_percentile)
        i2 = load_gray_float_any(f2, npy_mode=args.npy_mode, percentile=args.npy_percentile)

        x = prep_input_triplet(i0, i1, i2, size=args.size, dep_u=args.dep_u, rotate180=args.rotate180)
        if args.use_gpu and torch.cuda.is_available():
            x = x.cuda(non_blocking=True)

        with torch.no_grad():
            if args.use_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = net(x)
            if args.use_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append(t1 - t0)

        den = tensor_to_uint8(y)
        noisy_mean = tensor_to_uint8(x.mean(dim=1, keepdim=True))  # mean of 3 inputs

        base = os.path.splitext(os.path.basename(f1))[0]  # center frame name
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

    if timings:
        avg = sum(timings) / len(timings)
        print(f"Processed {len(timings)} triplets. Avg time: {avg:.4f}s (per inference)")


if __name__ == "__main__":
    main()