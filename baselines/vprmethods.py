import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from utils.eventcv_frames import offsets_from_dataset_config, render_png_sequence

# Methods whose model is fetched with torch.hub.load (vpr_models/__init__.py).
# torch.hub prompts on stdin the first time it sees an untrusted repo, which is an
# EOFError under subprocess, so the one the run needs is pre-approved below.
_TORCH_HUB_REPOS = {
    "cosplace": "gmberton/cosplace",
    "eigenplaces": "gmberton/eigenplaces",
    "eigenplaces-indoor": "Enrico-Chiavassa/Indoor-VPR",
    "salad": "serizba/salad",
    "salad-indoor": "Enrico-Chiavassa/Indoor-VPR",
    "cricavpr": "Lu-Feng/CricaVPR",
    "megaloc": "gmberton/MegaLoc",
    "edtformer": "Tong-Jin01/EDTformer",
}


def _hub_repo_for(method):
    if method.startswith("anyloc"):
        return "AnyLoc/DINO"
    return _TORCH_HUB_REPOS.get(method)


class vprmethods_baseline(EventBaseline):
    def __init__(self):
        super().__init__()
        self.name = "vprmethods"
        self.repo_path = "./baselines/vpr_methods"
        self.url = "https://github.com/gmberton/VPR-methods-evaluation.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)

        self.baseline_config_path = './baselines/vprmethods.yaml'
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)

        self.outdir = './output/vprmethods'
        os.makedirs(self.outdir, exist_ok=True)
        self.matrix_type = 'distance'

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Render the reference and query streams to two flat image folders.

        VPR-methods-evaluation reads `--database_folder`/`--queries_folder` with a
        recursive glob over .jpg/.jpeg/.png and a plain `sorted()`, so the folders
        must contain nothing but zero-padded frames. `--no_labels` is passed, so no
        filename convention is imposed beyond ordering.
        """
        self.config = config
        self.timewindow = timewindow

        if config['frame_generator'] == 'reconstruction':
            raise NotImplementedError(
                "vprmethods no longer builds E2VID reconstructions: the frame-generation "
                "pipeline was retired in the EventCV migration (commit 22e0fd2). Set "
                "frame_generator: frames in config.yaml to render event frames with "
                "EventCV instead.")

        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()
        ref_sequence = ref_info['sequence_name']
        query_sequence = query_info['sequence_name']
        # Timewindow-tagged, matching the label this baseline has always written.
        self.ref_name = f'{ref_sequence}-{config["frame_generator"]}-{timewindow}'
        self.query_name = f'{query_sequence}-{config["frame_generator"]}-{timewindow}'

        resolution = tuple(dataset_config['dataset']['resolution'])  # (W, H)
        min_gap_sec = float(config.get("filter_places_sec", 0))
        representation = self.baseline_config.get('representation', 'redblue')
        ref_offset, qry_offset = offsets_from_dataset_config(
            dataset_config, ref_sequence, query_sequence)

        self.temp_dir = tempfile.mkdtemp(prefix="vprmethods_data_")
        self.ref_dir = os.path.join(self.temp_dir, self.ref_name)
        self.query_dir = os.path.join(self.temp_dir, self.query_name)
        for hdf5_path, out_dir, offset in (
            (ref_info['hdf5_path'], self.ref_dir, ref_offset),
            (query_info['hdf5_path'], self.query_dir, qry_offset),
        ):
            render_png_sequence(
                hdf5_path, out_dir,
                timewindow_ms=timewindow, offset_ms=offset,
                sensor_size=resolution, representation=representation,
                min_gap_sec=min_gap_sec)

        self.output_dir = os.path.join(
            self.outdir,
            f"{ref_info['dataset_name']}",
            f"{ref_sequence}_{query_sequence}",
            f"{config['frame_generator']}_{timewindow}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _trust_torch_hub_repo(self, method):
        """
        Add the method's torch.hub repo to torch's trusted list, if configured.

        Upstream calls torch.hub.load() without trust_repo=, so on a fresh machine
        it blocks on an interactive y/N prompt and dies with EOFError under
        subprocess. This writes the same `trusted_list` entry that trust_repo=True
        would, for the one repo the configured method needs -- which is third-party
        code that will be downloaded and executed, hence the explicit yaml gate.
        """
        repo = _hub_repo_for(method)
        if repo is None:
            return
        if not self.baseline_config.get('trust_torch_hub', True):
            logger.warning(
                f"trust_torch_hub is false and {method} loads {repo} via torch.hub; "
                "the run will stop at an interactive trust prompt.")
            return

        import torch.hub

        hub_dir = torch.hub.get_dir()
        os.makedirs(hub_dir, exist_ok=True)
        listing = os.path.join(hub_dir, "trusted_list")
        entry = "_".join(repo.split("/"))
        existing = set()
        if os.path.exists(listing):
            with open(listing) as handle:
                existing = {line.strip() for line in handle}
        if entry in existing:
            return
        with open(listing, "a") as handle:
            handle.write(entry + "\n")
        logger.info(f"Trusting torch.hub repo {repo} for method '{method}' ({listing})")

    def build_execute(self, config, data_config, ground_truth):
        """Compose the upstream main.py invocation."""
        self.ground_truth = ground_truth
        cfg = self.baseline_config
        self._trust_torch_hub_repo(cfg["method"])
        ref_dir = Path(self.ref_dir).resolve()
        query_dir = Path(self.query_dir).resolve()
        # Upstream offers cuda|cpu only -- there is no mps option -- and defaults
        # to cuda, so it must be set explicitly or it fails on macOS.
        device = cfg.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        eval_cmd = (
            f'python -u main.py '
            f'--method {cfg["method"]} '
            f'--backbone {cfg["backbone"]} '
            f'--descriptors_dimension {cfg["descriptors_dimension"]} '
            f'--no_labels '
            f'--database_folder {ref_dir} '
            f'--queries_folder {query_dir} '
            f'--device {device} '
            f'--save_descriptors'
        )
        self.full_cmd = ["pixi", "run", "bash", "-c", eval_cmd]

    def run(self):
        """Run upstream in its own tree, then build the matrix from its descriptors."""
        logger.info(f"Running vprmethods: {' '.join(self.full_cmd)}")
        subprocess.run(self.full_cmd, check=True, cwd=self.repo_path)

        log_dir = sorted(Path(self.repo_path).glob("logs/default/*"), key=os.path.getmtime)[-1]
        database_descriptors = np.load(log_dir / "database_descriptors.npy")
        query_descriptors = np.load(log_dir / "queries_descriptors.npy")
        logger.info(f"Descriptors: database {database_descriptors.shape}, "
                    f"queries {query_descriptors.shape}")

        # Event-LAB scores (references, queries); upstream descriptors are L2-normalised.
        D = (1 - (query_descriptors @ database_descriptors.T)).T
        np.save(f"{self.output_dir}/distance_matrix.npy", D)
        logger.info(f"Saved distance matrix {D.shape} to {self.output_dir}")

    def parse_results(self, GT):
        all_files = sorted(list(Path(self.output_dir).glob("*.npy")))
        all_names = [os.path.basename(f).replace(".npy", "") for f in all_files]
        all_arrays = [np.load(f) for f in all_files]
        if not all_arrays:
            logger.warning(f"No .npy result files found in {self.output_dir}")
            return
        GThard = np.load(GT)
        for name, array in zip(all_names, all_arrays):
            if array.shape != GThard.shape:
                logger.warning(
                    f"{name} is {array.shape} but the ground truth is {GThard.shape}; "
                    "run_metrics will nearest-neighbour resize the GT to match.")

        timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        rows, pr_curves = self.run_metrics(
            all_names,
            all_arrays,
            GThard,
            timestamp,
            self.name,
            f'{self.ref_name}_{self.query_name}',
            matrix_type=self.matrix_type,
            outdir=self.output_dir,
            tolerance=self.config.get('ground_truth_tolerance', 0.0)
        )
        self.save_results(rows, pr_curves, self.name, f'{self.ref_name}_{self.query_name}')

    def cleanup(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
