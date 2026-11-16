"""
Extract MegaEvent descriptors for a single Event-LAB sequence.

Runs inside the ``megaevent`` pixi environment, invoked as a subprocess by
``baselines/megaevent.py``.

MegaEvent's own entrypoints are bypassed on purpose:
  * ``megaevent.cli`` is dead code (it passes keyword arguments to a function
    that takes a single Namespace).
  * ``main.py --mode eval`` builds StreamOptions without an offset, so its
    windows start at the recording's t_min instead of the dataset epoch, which
    desynchronises it from Event-LAB's ground truth and every other baseline.
  * ``retrieval.topk`` only ever returns top-k, never the full matrix Event-LAB
    scores.
So the internals are called directly and the raw descriptor bank is written out.
"""
# MUST precede any megaevent import. checkpoints.load_model references
# np._core.multiarray._reconstruct, but on numpy 1.x (Event-LAB pins numpy<2)
# np._core is a lazy forward-compat shim whose `multiarray` attribute is not
# bound until something imports it explicitly -- importing torch is not enough.
# MegaEvent's own environment is numpy 2.x, which is why upstream never hits it.
import numpy._core.multiarray  # noqa: F401

import argparse
import os
import sys
from pathlib import Path

# Direct script execution puts utils/ first; resolve the project package instead.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

import torch
from utils.representations import open_reader, image_uint8, runtime_config


def allow_numpy2_reconstruct():
    """
    Let torch's weights_only unpickler accept a checkpoint pickled under numpy 2.

    checkpoints.load_model allowlists `np._core.multiarray._reconstruct`, but
    torch matches allowlist entries by the *name* recorded in the pickle. The
    published checkpoints were saved with numpy 2.x, so they record
    "numpy._core.multiarray._reconstruct", whereas on Event-LAB's numpy 1.26 the
    very same object reports __module__ == "numpy.core.multiarray". The entry
    therefore never matches and the load is rejected. Registering the object
    under the numpy-2 path fixes the lookup without weakening weights_only.
    """
    import torch

    torch.serialization.add_safe_globals(
        [(np._core.multiarray._reconstruct, "numpy._core.multiarray._reconstruct")]
    )


class RepresentationDataset:
    def __init__(self, path, window_ms, offset, config, transform):
        self.path, self.window_ms, self.offset = path, window_ms, offset
        self.config, self.transform = config, transform
        self.reader = None
        self.length = self.open().n_slices

    def open(self):
        if self.reader is None:
            self.reader = open_reader(self.path, self.window_ms, self.offset, config=self.config)
        return self.reader

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        frame = image_uint8(self.open().slice(index))
        return self.transform(torch.from_numpy(frame).float().div_(255))

    def __getstate__(self):
        return {**self.__dict__, 'reader': None}


def main():
    ap = argparse.ArgumentParser(description="Extract MegaEvent descriptors for one sequence")
    ap.add_argument("--hdf5_path", required=True)
    ap.add_argument("--out", required=True, help="Destination .npy for the (N, D) descriptors")
    ap.add_argument("--repo_path", default="./baselines/megaevent")
    ap.add_argument("--model", default="megaevent_vits14",
                    choices=["megaevent_vits14", "megaevent_vitb14"])
    ap.add_argument("--ckpt_dir", required=True, help="Absolute checkpoint directory")
    ap.add_argument("--window_ms", type=float, required=True)
    ap.add_argument("--offset", type=float, default=None,
                    help="Absolute timestamp in milliseconds")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()

    src = os.path.abspath(os.path.join(args.repo_path, "src"))
    if src not in sys.path:
        sys.path.insert(0, src)

    from megaevent.checkpoints import load_model, resolve_model
    from megaevent.eval import extract
    from megaevent.events import eval_transform
    from megaevent.runtime import device_for

    allow_numpy2_reconstruct()

    device = device_for(args.device)
    checkpoint = resolve_model(args.model, args.ckpt_dir)
    logger.info(f"MegaEvent {args.model} on {device}: {checkpoint}")
    network, cfg = load_model(checkpoint, device)
    logger.info(f"checkpoint representation={cfg.representation} desc_dim={cfg.desc_dim} "
                f"eval_size={getattr(cfg, 'eval_img_size', None) or cfg.H}")

    config = runtime_config()
    dataset = RepresentationDataset(args.hdf5_path, args.window_ms, args.offset, config, eval_transform(cfg))
    # Recurrent state must follow the whole sequence, not interleaved worker batches.
    if config.get('representation') == 'reconstruction':
        args.workers = 0
    logger.info(f"{len(dataset)} windows from {args.hdf5_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # extract() writes the bank atomically at exactly this path. descriptor_bank()
    # is avoided: its cache key omits the offset, hot-pixel flag and
    # representation, so it silently reuses stale banks across configurations.
    bank = extract(network, dataset, args.out, device, args.batch_size, args.workers)
    logger.info(f"Wrote {bank.shape} descriptors to {args.out}")


if __name__ == "__main__":
    main()
