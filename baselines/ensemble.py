import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from utils.eventcv_frames import (offsets_from_dataset_config, render_png_sequence,
                                  write_frame_sidecars)


class ensemble_baseline(EventBaseline):
    def __init__(self):
        super().__init__()
        self.name = "ensemble"
        self.repo_path = "./baselines/ensemble"
        self.url = "https://github.com/AdamDHines/ensemble-event-vpr.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)

        self.baseline_config_path = './baselines/ensemble.yaml'
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)

        # run_eevpr.py does `sys.path.append(netvlad_folder); from netvlad import NetVLAD`
        if not os.path.exists(self.baseline_config['netvlad_path']):
            clone_repo("https://github.com/Nanne/pytorch-NetVlad.git",
                       destination=self.baseline_config['netvlad_path'])

        self.outdir = './output/ensemble'
        os.makedirs(self.outdir, exist_ok=True)
        self.matrix_type = 'distance'

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Render every configured time window from the event stream with EventCV.

        Ensemble-Event-VPR compares a set of temporal windows in one invocation,
        so all of `config['timewindows']` are staged here and the orchestrator
        breaks after the first call (eventlab_run.py:90-92).

        Upstream reads only images. For each window and side it requires
        `<root>/<dataset>/<seq>/<seq>-<subfolder>-<w>/<subfolder>/frame_*.png`,
        a sibling `timestamps.txt` of relative seconds, and a `metadata.json`
        one level up carrying `start_time_ns`.
        """
        self.config = config
        self.dataset = dataset_config['dataset']['name']
        self.frames_subfolder = config['frame_generator']

        if config['frame_generator'] == 'reconstruction':
            raise NotImplementedError(
                "ensemble no longer builds E2VID reconstructions: the frame-generation "
                "pipeline was retired in the EventCV migration (commit 22e0fd2). Set "
                "frame_generator: frames in config.yaml to render event frames with "
                "EventCV instead.")
        if config['frame_accumulator'] == 'eventcount':
            raise NotImplementedError(
                "ensemble requires fixed-duration windows: run_eevpr.py aligns traverses "
                "against --window_duration and raises without one. Set "
                "frame_accumulator: polarity in config.yaml.")

        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()
        self.ref_name = ref_info['sequence_name']
        self.query_name = query_info['sequence_name']

        self.windows = list(config['timewindows'])
        if not self.windows:
            raise ValueError("ensemble needs at least one entry in config['timewindows'].")

        resolution = tuple(dataset_config['dataset']['resolution'])  # (W, H)
        min_gap_sec = float(config.get("filter_places_sec", 0))
        representation = self.baseline_config.get('representation', 'redblue')
        ref_offset, qry_offset = offsets_from_dataset_config(
            dataset_config, self.ref_name, self.query_name)

        self.temp_dir = tempfile.mkdtemp(prefix="ensemble_data_")
        for window in self.windows:
            for sequence, hdf5_path, offset in (
                (self.ref_name, ref_info['hdf5_path'], ref_offset),
                (self.query_name, query_info['hdf5_path'], qry_offset),
            ):
                frames_dir = os.path.join(
                    self.temp_dir, self.dataset, sequence,
                    f"{sequence}-{self.frames_subfolder}-{window}",
                    self.frames_subfolder)
                times_s = render_png_sequence(
                    hdf5_path, frames_dir,
                    timewindow_ms=window, offset_ms=offset,
                    sensor_size=resolution, representation=representation,
                    min_gap_sec=min_gap_sec)
                write_frame_sidecars(
                    frames_dir, times_s,
                    start_time_ns=int(offset or 0) * 1_000_000,  # ms -> ns, integer to keep precision
                    timewindow_ms=window,
                    width=resolution[0], height=resolution[1])

        self.output_dir = os.path.join(
            self.outdir,
            f"{ref_info['dataset_name']}",
            f"{self.ref_name}_{self.query_name}",
            f"{config['frame_generator']}_{timewindow}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def build_execute(self, config, data_config, ground_truth):
        """Compose the run_eevpr.py invocation."""
        self.ground_truth = ground_truth
        eval_cmd = (
            f'python -u {os.path.join(self.repo_path, "run_eevpr.py")} '
            f'-ds "{self.dataset}" '
            f'-r {self.ref_name} '
            f'-q {self.query_name} '
            f'-w {self.windows} '
            f'-n [] '
            f'-d "{self.temp_dir}" '
            f'-nv {self.baseline_config["netvlad_path"]} '
            f'-o {self.output_dir} '
            # Upstream defaults this to "reconstruction"; the staged tree is named
            # for the frame generator, so it has to be passed explicitly.
            f'--frames_subfolder {self.frames_subfolder}'
        )
        self.eval_cmd = ["pixi", "run", "bash", "-c", eval_cmd]

    def run(self):
        logger.info(f"Running ensemble: {' '.join(self.eval_cmd)}")
        result = subprocess.run(self.eval_cmd, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Ensemble evaluation failed with return code {result.returncode}")

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
