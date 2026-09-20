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

import numpy as np
from loguru import logger

import eventcv as ecv


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


def use_eventcv_redblue(events_module):
    """
    Replace MegaEvent's hand-rolled NumPy accumulator with EventCV's `redblue`.

    `representations.accumulate_numpy` and `EventStream.redblue()` produce
    bit-identical (3, H, W) uint8 frames -- verified elementwise on Brisbane
    sunset2 -- but redblue renders in Rust and is 17-45x faster over a 1 s window.
    Patched here rather than in the clone: Event-LAB never modifies a checked-out
    upstream repo.
    """
    original = events_module.render_stream

    def render_stream(stream, representation, window_ms=50):
        if representation == "accumulate":
            return np.asarray(stream.redblue().numpy())
        return original(stream, representation, window_ms=window_ms)

    events_module.render_stream = render_stream
    logger.info("Using eventcv redblue for the 'accumulate' representation")


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
    ap.add_argument("--hot_pixel_filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eventcv_redblue", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    src = os.path.abspath(os.path.join(args.repo_path, "src"))
    if src not in sys.path:
        sys.path.insert(0, src)

    from megaevent import events as events_module
    from megaevent.checkpoints import load_model, resolve_model
    from megaevent.eval import extract
    from megaevent.events import EventDataset, StreamOptions
    from megaevent.runtime import device_for

    if args.eventcv_redblue:
        use_eventcv_redblue(events_module)
    allow_numpy2_reconstruct()

    device = device_for(args.device)
    checkpoint = resolve_model(args.model, args.ckpt_dir)
    logger.info(f"MegaEvent {args.model} on {device}: {checkpoint}")
    network, cfg = load_model(checkpoint, device)
    logger.info(f"representation={cfg.representation} desc_dim={cfg.desc_dim} "
                f"eval_size={getattr(cfg, 'eval_img_size', None) or cfg.H}")

    options = StreamOptions(
        window_ms=args.window_ms,
        hot_pixel_filter=args.hot_pixel_filter,
        offset_ms=args.offset,
    )
    dataset = EventDataset(args.hdf5_path, cfg, options)
    logger.info(f"{len(dataset)} windows from {args.hdf5_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # extract() writes the bank atomically at exactly this path. descriptor_bank()
    # is avoided: its cache key omits the offset, hot-pixel flag and
    # representation, so it silently reuses stale banks across configurations.
    bank = extract(network, dataset, args.out, device, args.batch_size, args.workers)
    logger.info(f"Wrote {bank.shape} descriptors to {args.out}")


if __name__ == "__main__":
    main()
