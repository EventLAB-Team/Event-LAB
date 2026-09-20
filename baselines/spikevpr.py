import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import gdown
import numpy as np
import yaml
from loguru import logger

from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from utils.utils import convert_offset, pixi_run

# SpikeVPR's MixVPR head is built with in_h * in_w == 99, which is the layer4
# feature map of a 260x346 input and nothing else (factory.py:17-19, 41-43).
_SUPPORTED_RESOLUTION = (346, 260)  # (W, H), matching the dataset YAML convention


class spikevpr_baseline(EventBaseline):
    def __init__(self):
        super().__init__()
        self.name = "spikevpr"
        self.repo_path = "./baselines/SpikeVPR"
        self.url = "https://github.com/GeoffroyK/SpikeVPR.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)

        self.baseline_config_path = './baselines/spikevpr.yaml'
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)

        # Upstream's download_weights.sh is an unusable placeholder (BASE_URL is
        # literally REPLACE_ME, and it shells out to sha256sum, which macOS lacks),
        # so fetch the checkpoint from the public Drive folder the README links to.
        # Guarded independently of the clone: a failed download must be retried.
        self.checkpoint = os.path.join(
            self.repo_path, "src", "weights", self.baseline_config['checkpoint'])
        if not os.path.exists(self.checkpoint):
            os.makedirs(os.path.dirname(self.checkpoint), exist_ok=True)
            gdown.download(id=self.baseline_config['gdrive_id'],
                           output=self.checkpoint, quiet=False)
        self._verify_checkpoint()

        self.outdir = './output/spikevpr'
        os.makedirs(self.outdir, exist_ok=True)
        # Cosine similarity between L2-normalised descriptors: higher is better.
        self.matrix_type = 'similarity'

    def _verify_checkpoint(self):
        """
        Checksum the Drive download to catch truncation or an HTML error page.

        Deliberately a warning, not a hard failure. The file currently served
        from the README's Drive folder does not match the digest in the repo's
        src/weights/SHA256SUMS.txt -- that manifest is stale, and the served file
        loads cleanly as a 271-key SEW-ResNet34 + MixVPR state dict. Raising here
        would make the baseline permanently unrunnable on an upstream bookkeeping
        slip, so `sha256` below pins what the folder actually serves.
        """
        expected = self.baseline_config.get('sha256')
        if not expected:
            return
        digest = hashlib.sha256()
        with open(self.checkpoint, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            logger.warning(
                f"Checksum mismatch for {self.checkpoint}: expected {expected}, got {actual}. "
                "The upstream checkpoint may have been re-uploaded. Delete the file to "
                "re-download it, or update `sha256` in baselines/spikevpr.yaml if this is "
                "the intended weights file.")
            return
        logger.info(f"Verified checksum of {os.path.basename(self.checkpoint)}")

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """Resolve the two recordings, their offsets, and the output directory."""
        self.config = config
        self.timewindow = timewindow

        resolution = tuple(dataset_config['dataset']['resolution'])
        if resolution != _SUPPORTED_RESOLUTION:
            raise ValueError(
                f"SpikeVPR only supports {_SUPPORTED_RESOLUTION} (W, H) sensors; "
                f"{dataset_config['dataset']['name']} is {resolution}. Its MixVPR head is "
                "built for a 99-element feature map and will fail on any other resolution.")

        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()
        self.ref_name = ref_info['sequence_name']
        self.query_name = query_info['sequence_name']
        self.ref_hdf5 = ref_info['hdf5_path']
        self.query_hdf5 = query_info['hdf5_path']
        self.resolution = resolution

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

        # Descriptors are cached OUTSIDE output_dir: parse_results globs every
        # *.npy in there and would otherwise score the descriptor banks too.
        self.descriptor_dir = os.path.join(
            self.outdir, "descriptors", ref_info['dataset_name'])
        os.makedirs(self.descriptor_dir, exist_ok=True)

    def _descriptor_path(self, sequence, offset):
        cfg = self.baseline_config
        tag = f"{sequence}-{self.timewindow}ms-ec{cfg['event_count']}-{cfg['channel_order']}"
        if offset is not None:
            tag += f"-off{int(offset)}"
        return os.path.join(self.descriptor_dir, f"{tag}.npy")

    def _extract_cmd(self, hdf5_path, destination, offset):
        cfg = self.baseline_config
        cmd = (
            f"python -u utils/spikevpr_descriptors.py "
            f"--hdf5_path {hdf5_path} --out {destination} "
            f"--repo_path {self.repo_path} --checkpoint {self.checkpoint} "
            f"--dt_ms {self.timewindow} "
            f"--sensor_width {self.resolution[0]} --sensor_height {self.resolution[1]} "
            f"--encoder {cfg['encoder']} --neuron_type {cfg['neuron_type']} "
            f"--out_channels {cfg['out_channels']} --out_rows {cfg['out_rows']} "
            f"--event_count {cfg['event_count']} --channel_order {cfg['channel_order']} "
            f"--batch_size {cfg['batch_size']} --device {cfg['device']}"
        )
        if offset is not None:
            cmd += f" --offset {offset}"
        return cmd

    def build_execute(self, config, data_config, ground_truth):
        """Compose the two descriptor-extraction commands."""
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
            result = pixi_run("spikevpr", command)
            if result.returncode != 0:
                raise RuntimeError(
                    f"SpikeVPR {label} descriptor extraction failed "
                    f"with return code {result.returncode}")

        ref = np.load(self.ref_descriptors)
        qry = np.load(self.query_descriptors)
        logger.info(f"Descriptors: reference {ref.shape}, query {qry.shape}")

        # Event-LAB scores (references, queries); SpikeVPR descriptors are already
        # L2-normalised, so the dot product is the cosine similarity matrix.
        similarity = (ref @ qry.T).astype(np.float32)
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
