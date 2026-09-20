"""
Starting point for a new Event-LAB baseline.

Copy this file to `baselines/<name>.py`, add `baselines/<name>.yaml`, register the
class in `baselines/get_baseline.py`, and add the command name to `VPR-Baselines`
in `config.yaml`. See docs/contributing_baselines.rst.

Event data is read through EventCV, never from pre-rendered frame directories --
`ecv.open(hdf5_path, dt_ms=..., offset=...)` streams a multi-gigabyte recording
without materialising it. Compare `baselines/sparse_event.py` for an in-process
baseline, `baselines/vprmethods.py` for one that shells out to an upstream CLI.
"""
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

import eventcv as ecv
from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from utils.eventcv_frames import offsets_from_dataset_config


class name_baseline(EventBaseline):
    def __init__(self):
        super().__init__()
        self.name = "<name>"
        self.repo_path = "./baselines/<name>"
        self.url = "https://github.com/<name>"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)

        self.baseline_config_path = './baselines/<name>.yaml'
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)

        self.outdir = './output/<name>'
        os.makedirs(self.outdir, exist_ok=True)
        # 'distance' (lower is better) or 'similarity' (higher is better).
        # Required: it drives the max()-array flip in EventBaseline.run_metrics.
        self.matrix_type = 'distance'

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """Locate the two recordings and prepare whatever the method consumes."""
        self.config = config
        self.timewindow = timewindow

        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()
        self.ref_name = ref_info['sequence_name']
        self.query_name = query_info['sequence_name']

        # EventCV's bare `offset` is an absolute timestamp in milliseconds. Datasets
        # without an `other.offset` block yield (None, None), which ecv.open ignores.
        ref_offset, qry_offset = offsets_from_dataset_config(
            dataset_config, self.ref_name, self.query_name)

        # One reader per recording. `repr` picks the representation: "count",
        # "redblue" (3-channel RGB), "mcts", "voxel", "tsurf", ... With a repr set,
        # reader[i] is a dense [C, H, W] array and reader.batch(idx) a [B, C, H, W]
        # batch, so a torch DataLoader can collate the reader directly.
        self.ref_reader = ecv.open(
            ref_info['hdf5_path'], dt_ms=timewindow, offset=ref_offset,
            hot_pixel_filter=True,
            sensor_size=tuple(dataset_config['dataset']['resolution']))
        self.query_reader = ecv.open(
            query_info['hdf5_path'], dt_ms=timewindow, offset=qry_offset,
            hot_pixel_filter=True,
            sensor_size=tuple(dataset_config['dataset']['resolution']))

        self.output_dir = os.path.join(
            self.outdir,
            f"{ref_info['dataset_name']}",
            f"{self.ref_name}_{self.query_name}",
            f"{config['frame_generator']}_{timewindow}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def build_execute(self, config, data_config, ground_truth):
        """Prepare any command, model or temporary file the method needs. May be a no-op."""
        self.ground_truth = ground_truth

    def run(self):
        """
        Execute the method and save the result matrices.

        Every matrix must be shaped (n_references, n_queries) -- recallAtK argsorts
        along axis 0 and normalises over columns, so a transposed matrix scores
        silently wrong. For cosine descriptors the idiom is:
            D = (1 - (query_feats @ ref_feats.T)).T
        Write intermediates somewhere other than self.output_dir: parse_results
        globs every *.npy in there and would score them as results.
        """
        raise NotImplementedError("Compute the matrix and np.save it into self.output_dir")

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
        """Remove temporary files created by the wrapper."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
