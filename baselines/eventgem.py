import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from utils.utils import convert_offset, pixi_run

# superevent carries the trunk weights; vprtutorial supplies recallAtK, which
# eventgem/analysis.py imports at module load. eventlab is deliberately omitted:
# it pins an ancient Event-LAB revision and would clone Event-LAB inside itself.
_SUBMODULES = ["eventgem/external/superevent", "eventgem/external/vprtutorial"]


class eventgem_baseline(EventBaseline):
    def __init__(self):
        super().__init__()
        self.name = "eventgem"
        self.repo_path = "./baselines/Event-GeM"
        self.url = "https://github.com/AdamDHines/Event-GeM.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)
        self._init_submodules()

        self.baseline_config_path = './baselines/eventgem.yaml'
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)

        self.outdir = './output/eventgem'
        os.makedirs(self.outdir, exist_ok=True)
        # Despite the "sim_mat" filenames, both matrices hold distances:
        # feature_extraction.py does sim.neg_().add_(1.0), and re-ranked values
        # go negative as inliers are subtracted. Lower is better.
        self.matrix_type = 'distance'

    def _init_submodules(self):
        """
        Check out only the submodules the default path needs.

        clone_repo() runs a plain `git clone`, and --recursive is not an option
        here: SuperEvent nests a SuperGlue submodule behind an ssh:// URL that
        fails without keys. The SuperEvent trunk weights arrive with this step --
        there is no download code for them anywhere.
        """
        missing = [m for m in _SUBMODULES
                   if not os.path.isdir(os.path.join(self.repo_path, m))
                   or not os.listdir(os.path.join(self.repo_path, m))]
        if not missing:
            return
        logger.info(f"Initialising Event-GeM submodules: {missing}")
        subprocess.run(["git", "-C", self.repo_path, "submodule", "update", "--init"] + missing,
                       check=True)

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Resolve names, offsets and output paths.

        There is no data preparation: Event-GeM reads
        <data-root>/<dataset>/<seq>/<seq>.hdf5 and
        <data-root>/<dataset>/ground_truth/<ref>_<qry>_GT.npy, which is exactly
        the layout Event-LAB already produces, so data_path is passed straight in.
        """
        self.config = config
        self.timewindow = timewindow
        self.dataset_name = dataset_config['dataset']['name']

        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()
        self.ref_name = ref_info['sequence_name']
        self.query_name = query_info['sequence_name']

        self.ref_offset, self.qry_offset = 0, 0
        if "other" in dataset_config and "offset" in dataset_config["other"]:
            self.ref_offset, self.qry_offset = convert_offset(
                dataset_config['other']['offset'][self.ref_name],
                dataset_config['other']['offset'][self.query_name],
                dataset_config['other']['offset_time_scale'])

        self.data_root = os.path.abspath(config['data_path'])
        self.output_dir = os.path.join(
            self.outdir,
            f"{ref_info['dataset_name']}",
            f"{self.ref_name}_{self.query_name}",
            f"{config['frame_generator']}_{timewindow}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Feature and keypoint caches are large; keep them out of output_dir,
        # which parse_results globs for result matrices.
        #
        # The timewindow is part of the path on purpose. Event-GeM caches under
        # <feature-out>/<dataset>/<ref>-<query>/ with no dt_ms in the name
        # (feature_extraction.py:296-299), so without this the second timewindow
        # of an Event-LAB sweep would silently reuse the first one's features.
        self.feature_out = os.path.join(
            os.path.abspath(self.outdir), "features", f"{config['frame_generator']}_{timewindow}")
        self.keypoint_out = os.path.join(
            os.path.abspath(self.outdir), "keypoints", f"{config['frame_generator']}_{timewindow}")

        # Where upstream writes its two matrices (main.py:116-118).
        self.similarity_dir = os.path.join(
            self.data_root, self.dataset_name,
            f"{self.ref_name}-{self.query_name}-similarity")

    def build_execute(self, config, data_config, ground_truth):
        """Compose the Event-GeM invocation."""
        self.ground_truth = ground_truth
        cfg = self.baseline_config
        launcher = os.path.abspath("utils/eventgem_run.py")
        self.eventgem_cmd = (
            f"python -u {launcher} --eventlab-device {cfg['device']} "
            f"-d {self.dataset_name} -r {self.ref_name} -q {self.query_name} "
            f"--data-root {self.data_root} "
            f"--feature-out {self.feature_out} --keypoint-out {self.keypoint_out} "
            f"--dt-ms {int(self.timewindow)} --max-window-ms {float(self.timewindow)} "
            f"--ref-offset {int(self.ref_offset)} --query-offset {int(self.qry_offset)} "
            f"--top-k {cfg['top_k']} --match-filter {cfg['match_filter']} "
            f"--match-ratio {cfg['match_ratio']} --ransac-thresh {cfg['ransac_thresh']} "
            f"--inlier-weight {cfg['inlier_weight']} "
            f"--keypoint-batch-size {cfg['keypoint_batch_size']}"
        )

    def run(self):
        """Run Event-GeM in its own tree, then copy its matrices into output_dir."""
        if not os.path.exists(self.ground_truth):
            raise FileNotFoundError(
                f"Event-GeM loads the ground truth only after feature extraction, so a "
                f"missing file wastes the whole run. Expected: {self.ground_truth}")

        # cwd must be the clone: the submodule check and the --se-config/--se-weights
        # defaults are all relative. PYTHONPATH needs the Event-LAB root because
        # eventgem/utils/eventlab_config.py imports `datasets` at module load, and
        # the eventlab submodule is intentionally left uninitialised.
        repo = os.path.abspath(self.repo_path)
        result = pixi_run(
            "eventgem", self.eventgem_cmd, cwd=repo,
            extra_env={"PYTHONPATH": os.pathsep.join([repo, os.path.abspath(".")])})
        if result.returncode != 0:
            raise RuntimeError(f"Event-GeM failed with return code {result.returncode}")

        harvested = 0
        for source_name, target_name in (("original_sim_mat.npy", "eventgem_base.npy"),
                                         ("reranked_sim_mat.npy", "eventgem_reranked.npy")):
            source = os.path.join(self.similarity_dir, source_name)
            if not os.path.exists(source):
                logger.warning(f"Event-GeM did not produce {source}")
                continue
            target = os.path.join(self.output_dir, target_name)
            shutil.copyfile(source, target)
            logger.info(f"Harvested {source_name} -> {target} {np.load(target, mmap_mode='r').shape}")
            harvested += 1
        if harvested == 0:
            raise RuntimeError(f"No Event-GeM matrices found in {self.similarity_dir}")

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
