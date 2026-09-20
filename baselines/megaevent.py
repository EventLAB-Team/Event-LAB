import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from utils.utils import convert_offset, pixi_run


class megaevent_baseline(EventBaseline):
    def __init__(self):
        super().__init__()
        self.name = "megaevent"
        self.repo_path = "./baselines/megaevent"
        self.url = "https://github.com/AdamDHines/megaevent.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)

        self.baseline_config_path = './baselines/megaevent.yaml'
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)

        # Weights download themselves on first use via huggingface_hub, at the
        # revision pinned in src/megaevent/models.json. ckpt_dir must be absolute:
        # megaevent's own default is cwd-relative and would land anywhere.
        self.ckpt_dir = os.path.abspath(self.baseline_config['ckpt_dir'])
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.outdir = './output/megaevent'
        os.makedirs(self.outdir, exist_ok=True)
        # Descriptors are L2-normalised, so the dot product is cosine similarity.
        self.matrix_type = 'similarity'

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """Resolve the two recordings, their offsets, and the output directory."""
        self.config = config
        self.timewindow = timewindow

        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()
        self.ref_name = ref_info['sequence_name']
        self.query_name = query_info['sequence_name']
        self.ref_hdf5 = ref_info['hdf5_path']
        self.query_hdf5 = query_info['hdf5_path']

        self.ref_offset, self.qry_offset = None, None
        if "other" in dataset_config and "offset" in dataset_config["other"]:
            self.ref_offset, self.qry_offset = convert_offset(
                dataset_config['other']['offset'][self.ref_name],
                dataset_config['other']['offset'][self.query_name],
                dataset_config['other']['offset_time_scale'])

        self.output_dir = os.path.join(
            self.outdir,
            f"{ref_info['dataset_name']}",
            f"{self.ref_name}_{self.query_name}",
            f"{config['frame_generator']}_{timewindow}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Kept outside output_dir, which parse_results globs for result matrices.
        self.descriptor_dir = os.path.join(
            self.outdir, "descriptors", ref_info['dataset_name'])
        os.makedirs(self.descriptor_dir, exist_ok=True)

    def _descriptor_path(self, sequence, offset):
        cfg = self.baseline_config
        tag = f"{sequence}-{cfg['model']}-{self.timewindow}ms"
        if offset is not None:
            tag += f"-off{int(offset)}"
        return os.path.join(self.descriptor_dir, f"{tag}.npy")

    def _extract_cmd(self, hdf5_path, destination, offset):
        cfg = self.baseline_config
        cmd = (
            f"python -u utils/megaevent_descriptors.py "
            f"--hdf5_path {hdf5_path} --out {destination} "
            f"--repo_path {self.repo_path} --model {cfg['model']} "
            f"--ckpt_dir {self.ckpt_dir} --window_ms {self.timewindow} "
            f"--batch_size {cfg['batch_size']} --workers {cfg['workers']} "
            f"--device {cfg['device']}"
        )
        if offset is not None:
            cmd += f" --offset {offset}"
        if not cfg.get('eventcv_redblue', True):
            cmd += " --no-eventcv-redblue"
        return cmd

    def build_execute(self, config, data_config, ground_truth):
        self.ground_truth = ground_truth
        self.ref_descriptors = self._descriptor_path(self.ref_name, self.ref_offset)
        self.query_descriptors = self._descriptor_path(self.query_name, self.qry_offset)
        self.ref_cmd = self._extract_cmd(self.ref_hdf5, self.ref_descriptors, self.ref_offset)
        self.query_cmd = self._extract_cmd(self.query_hdf5, self.query_descriptors, self.qry_offset)

    def run(self):
        """Extract both descriptor banks, then build the reference x query matrix."""
        for label, destination, command in (
            ("reference", self.ref_descriptors, self.ref_cmd),
            ("query", self.query_descriptors, self.query_cmd),
        ):
            if os.path.exists(destination):
                logger.info(f"Reusing cached {label} descriptors: {destination}")
                continue
            result = pixi_run("megaevent", command)
            if result.returncode != 0:
                raise RuntimeError(
                    f"MegaEvent {label} descriptor extraction failed "
                    f"with return code {result.returncode}")

        ref = np.load(self.ref_descriptors, mmap_mode="r")
        qry = np.load(self.query_descriptors, mmap_mode="r")
        logger.info(f"Descriptors: reference {ref.shape}, query {qry.shape}")

        # Chunked over queries so an 8448-D bank never materialises a huge temporary.
        similarity = np.empty((ref.shape[0], qry.shape[0]), dtype=np.float32)
        chunk = 2048
        reference_block = np.asarray(ref, dtype=np.float32)
        for start in range(0, qry.shape[0], chunk):
            block = np.asarray(qry[start:start + chunk], dtype=np.float32)
            similarity[:, start:start + block.shape[0]] = reference_block @ block.T

        np.save(os.path.join(self.output_dir, "similarity_matrix.npy"), similarity)
        logger.info(f"Saved similarity matrix {similarity.shape} to {self.output_dir}")

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
            shutil.rmtree(self.temp_dir)
