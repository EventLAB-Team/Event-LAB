"""
Launch Event-GeM's own CLI with the two macOS fixes it needs.

Runs inside the ``eventgem`` pixi environment with cwd set to the Event-GeM
clone, invoked as a subprocess by ``baselines/eventgem.py``. Event-GeM's
``main.py`` is executed unmodified -- Event-LAB never edits a checked-out
upstream repo -- so the two incompatibilities are monkeypatched here instead:

1. ``feature_extraction.py`` hardcodes ``num_workers=4`` on a DataLoader whose
   Dataset holds a live ``eventcv.EventReader``. Under macOS's spawn start
   method that must be pickled, and fails with
   ``TypeError: cannot pickle 'builtins.EventReader' object``. It only works on
   Linux because fork does not pickle. ``baselines/eventvlad.py`` guards the
   equivalent case the same way.
2. ``EventGeM.__init__`` picks ``cuda`` or ``cpu`` with no MPS branch. The
   device is assigned at the very end of ``__init__`` and the model is not built
   until ``extract_superevent_features``, so overriding it afterwards is safe.
   Off by default: MaxViT on MPS is unverified, and CPU is correct if slower.

Everything else -- argument parsing, feature extraction, re-ranking, and writing
original_sim_mat.npy / reranked_sim_mat.npy -- is upstream's own code.
"""
import argparse
import multiprocessing as mp
import os
import sys

import torch
from loguru import logger


def patch_dataloader():
    """Force single-process loading when the start method cannot fork."""
    if mp.get_start_method(allow_none=True) == "fork":
        return
    original = torch.utils.data.DataLoader

    class SingleProcessDataLoader(original):
        def __init__(self, *args, **kwargs):
            if kwargs.get("num_workers", 0):
                logger.info(
                    f"Forcing num_workers={kwargs['num_workers']} -> 0 "
                    "(spawn cannot pickle eventcv.EventReader)")
                kwargs["num_workers"] = 0
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = SingleProcessDataLoader


def patch_device(requested):
    """Override Event-GeM's cuda-or-cpu choice once __init__ has run."""
    import eventgem.feature_extraction as fe

    if requested == "auto":
        if torch.cuda.is_available() or not torch.backends.mps.is_available():
            return
        device = torch.device("mps")
    else:
        device = torch.device(requested)

    original_init = fe.EventGeM.__init__

    def patched_init(self, args):
        original_init(self, args)
        self.device = device
        logger.info(f"Overriding Event-GeM device -> {device}")

    fe.EventGeM.__init__ = patched_init


def main():
    ap = argparse.ArgumentParser(
        description="Run Event-GeM's main.py with Event-LAB's compatibility patches",
        add_help=False)
    ap.add_argument("--eventlab-device", default="cpu",
                    choices=["auto", "cpu", "cuda", "mps"],
                    help="cpu (default) keeps upstream behaviour on macOS; "
                         "auto promotes to MPS when CUDA is absent")
    known, passthrough = ap.parse_known_args()

    patch_dataloader()
    if known.eventlab_device != "cpu":
        patch_device(known.eventlab_device)

    # main.py lives at the Event-GeM repo root, which is also the required cwd:
    # the submodule check and the --se-config/--se-weights defaults are relative.
    sys.path.insert(0, os.getcwd())
    import main as eventgem_main

    sys.argv = ["main.py"] + passthrough
    logger.info(f"Running Event-GeM: {' '.join(sys.argv)}")
    eventgem_main.main()


if __name__ == "__main__":
    main()
