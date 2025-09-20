# baselines/eventvlad_featureextraction_matconv.py
import os, sys, glob
from os.path import isfile
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---- keep your path + imports exactly as provided ----
sys.path.append('./baselines/EventVLAD')
from baselines.EventVLAD.networks.EventVLAD import Imagenet_vgg
from baselines.EventVLAD.networks.netvlad import NetVLAD, EmbedNet, TripletNet

# -------------------- loader helpers --------------------
def _device(dev=None):
    if isinstance(dev, torch.device): return dev
    if isinstance(dev, str): return torch.device(dev)
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _safe_torch_load(path):
    # Prefer safe loading when available; fall back quietly on older torch.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _strip_prefix(k, prefix):
    return k[len(prefix):] if k.startswith(prefix) else k

def _remap_state_for_triplet(state):
    """
    Map various checkpoint layouts to TripletNet(embed_net(base_model, net_vlad)) EXACT names:
      - VGG:   embed_net.base_model.<layer>
      - NetVLAD: embed_net.net_vlad.<param>
    """
    if "state_dict" in state: state = state["state_dict"]
    if "model_state_dict" in state: state = state["model_state_dict"]

    fixed = {}
    for k, v in state.items():
        # strip common wrappers
        for pref in ("module.", "model.", "net.", "triplet.", "tripletnet.", "triplet_model.", "embed.", "embed_net."):
            k = _strip_prefix(k, pref)

        # normalize obvious roots
        if k.startswith("base_model."):
            k = "embed_net.base_model." + k[len("base_model."):]
        elif k.startswith("net_vlad."):
            k = "embed_net.net_vlad." + k[len("net_vlad."):]
        elif k.startswith("encoder."):
            # many checkpoints store VGG as "encoder.*"
            k = "embed_net.base_model." + k[len("encoder."):]
        elif k.startswith("vlad.") or k.startswith("netvlad."):
            k = "embed_net.net_vlad." + k.split(".", 1)[1]
        elif k.startswith("pool."):
            # some store NetVLAD as "pool.*"
            k = "embed_net.net_vlad." + k[len("pool."):]
        elif k.startswith("embed_net.base_model.pool."):
            # rare mis-save of vlad under base_model.pool.*
            k = "embed_net.net_vlad." + k[len("embed_net.base_model.pool."):]
        else:
            # bare VGG layer names (convX_Y, reluX_Y, poolX, fc6/fc7/fc8)
            if k.startswith(("conv1_","conv2_","conv3_","conv4_","conv5_","relu","pool","fc6","fc7","fc8")):
                k = "embed_net.base_model." + k
            # bare NetVLAD param names
            elif k.startswith(("centroids", "conv.weight", "conv.bias", "lastfc.weight", "lastfc.bias")):
                k = "embed_net.net_vlad." + k

        fixed[k] = v
    return fixed

def _infer_vlad_dims(mapped_state):
    """
    Infer (K, D) for NetVLAD from the mapped state.
    """
    if "embed_net.net_vlad.centroids" in mapped_state:
        w = mapped_state["embed_net.net_vlad.centroids"]  # [K, D]
        return int(w.shape[0]), int(w.shape[1])
    if "embed_net.net_vlad.conv.weight" in mapped_state:
        w = mapped_state["embed_net.net_vlad.conv.weight"]  # [K, D, 1, 1]
        return int(w.shape[0]), int(w.shape[1])
    raise RuntimeError("Could not infer NetVLAD (K, D): no centroids or conv.weight in checkpoint.")

def _ensure_required_vlad_params(mapped):
    """
    Some checkpoints omit NetVLAD conv.bias. If the model expects it,
    synthesize a zero bias so strict=True can succeed.
    """
    need_bias_key = "embed_net.net_vlad.conv.bias"
    if need_bias_key not in mapped:
        # infer K (num_clusters)
        if "embed_net.net_vlad.conv.weight" in mapped:
            K = int(mapped["embed_net.net_vlad.conv.weight"].shape[0])
            dtype = mapped["embed_net.net_vlad.conv.weight"].dtype
        elif "embed_net.net_vlad.centroids" in mapped:
            K = int(mapped["embed_net.net_vlad.centroids"].shape[0])
            dtype = mapped["embed_net.net_vlad.centroids"].dtype
        else:
            raise RuntimeError("Cannot infer K to synthesize conv.bias.")

        mapped[need_bias_key] = torch.zeros(K, dtype=dtype)  # CPU is fine
    return mapped

# -------------------- dataset with the MATCONV meta --------------------
class _EventVGGPreprocess(Dataset):
    """
    - Resizes to 224x224
    - Converts to RGB by default (see note), uint8-like scale (no /255)
    - Subtracts channel means from EventVLAD Imagenet VGG meta (std=1)
    """
    def __init__(self, items, rgb_input=True):
        self.items = items
        self.rgb_input = rgb_input
        # From Imagenet_matconvnet_vgg_verydeep_16_dag.meta (RGB order)
        self.mean = np.array([122.7449417, 114.9440994, 101.6417770], dtype=np.float32)

    def __len__(self): return len(self.items)

    def _load(self, it):
        if isinstance(it, (np.ndarray, np.generic)):
            img = it
        else:
            img = cv2.imread(it, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {it}")

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if self.rgb_input:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            if self.rgb_input:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # matconv-style: keep [0..255]-scale floats, subtract means
        img = img.astype(np.float32)
        img -= self.mean  # RGB means
        img = np.transpose(img, (2, 0, 1))  # CHW
        return torch.from_numpy(img)

    def __getitem__(self, idx):
        return self._load(self.items[idx])

# -------------------- public API --------------------
def build_eventvlad_model_from_tar(weights_path: str, num_clusters: int = 64, device=None):
    """
    Build TripletNet(EmbedNet(Imagenet_vgg, NetVLAD)) and load weights from .tar strictly.
    NOTE: num_clusters is ignored if the checkpoint indicates a different K; we match the checkpoint.
    """
    assert isfile(weights_path), f"Missing weights file: {weights_path}"
    dev = _device(device)

    # Base (their MatConvNet-style VGG16 -> fc8:1000)
    base = Imagenet_vgg(weights_path=None)  # we load from the .tar, not a separate .pth

    # Load and map checkpoint FIRST so we can infer dims
    raw = _safe_torch_load(weights_path)
    mapped = _remap_state_for_triplet(raw)
    K, D = _infer_vlad_dims(mapped)  # e.g., K=64, D=1000
    mapped = _ensure_required_vlad_params(mapped)

    # Build NetVLAD to EXACT dims from checkpoint
    vlad = NetVLAD(num_clusters=K, dim=D, normalize_input=True)

    embed = EmbedNet(base, vlad)
    model = TripletNet(embed).to(dev).eval()

    # Pre-validate to ensure strict=True will pass (no surprises)
    model_keys = set(model.state_dict().keys())
    mapped_keys = set(mapped.keys())
    missing = sorted(model_keys - mapped_keys)
    unexpected = sorted(mapped_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Key mismatch after mapping.\n"
            f"Missing ({len(missing)}): {missing[:12]}{' ...' if len(missing)>12 else ''}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:12]}{' ...' if len(unexpected)>12 else ''}"
        )

    model.load_state_dict(mapped, strict=True)
    return model

def _coerce_items(images):
    """
    Accept: list/tuple/ndarray of paths, a single path, a directory, a glob, or a .txt/.lst file of paths.
    Returns a sorted list of file paths.
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm")
    # list/tuple already
    if isinstance(images, (list, tuple, np.ndarray)):
        return list(images)

    # single string cases
    if isinstance(images, str):
        # text list of paths
        if images.lower().endswith((".txt", ".lst")) and isfile(images):
            with open(images, "r") as f:
                paths = [ln.strip() for ln in f if ln.strip()]
            return paths
        # directory
        if os.path.isdir(images):
            files = [os.path.join(images, f) for f in os.listdir(images)]
            files = [p for p in files if isfile(p) and p.lower().endswith(exts)]
            files.sort()
            return files
        # glob
        g = sorted(glob.glob(images))
        if len(g):
            return [p for p in g if isfile(p)]
        # single file path
        return [images]

    # fallback
    return [images]

@torch.inference_mode()
def extract_eventvlad_features(
    model, images, batch_size=16, num_workers=4, device=None, rgb_input=True,
):
    dev = _device(device)
    model = model.to(dev).eval()

    image_list = _coerce_items(images)
    if len(image_list) == 0:
        raise ValueError("No images found to process.")
    ds = _EventVGGPreprocess(image_list, rgb_input=rgb_input)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(num_workers > 0),
        # drop_last=False  # keep default; we’ll fix the 1-sample case below
    )

    feats = []
    for batch in tqdm(dl, desc="EventVLAD", leave=False):
        batch = batch.to(dev, non_blocking=True)
        out = model.feature_extract(batch)           # may be [B, D] or [D] when B==1
        if out.dim() == 1:                           # <— ensure 2-D
            out = out.unsqueeze(0)                   # -> [1, D]
        out_cpu = out.detach().cpu().to(torch.float32).numpy()
        feats.append(out_cpu)

    return np.concatenate(feats, axis=0)             # now all chunks are [b_i, D]