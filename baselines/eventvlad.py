import os, tempfile, subprocess, yaml
import numpy as np
from pathlib import Path
from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from baselines.VPR_Tutorial.evaluation.metrics import recallAtK, createPR
import prettytable
import openpyxl
from datetime import datetime, timezone
import re, gdown, time
import utils.functional as FUNC
import shutil 
from datasets.dataloader import make_frame_source
from loguru import logger
from utils.utils import convert_offset
import eventcv as ecv

class eventvlad_baseline(EventBaseline):
    def __init__(self, config, dataset_config, reference, query):
        super().__init__()

        # Set the experimental details as instance variables
        self.config = config
        self.dataset_config = dataset_config
        self.reference = reference
        self.query = query

        # Set the baseline name
        self.name = "eventvlad"

        # Check if the baseline repository is already cloned
        self.repo_path = "./baselines/EventVLAD"

        # Baseline URL
        self.url = "https://github.com/alexjunholee/EventVLAD.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)
            file_id = "1xdoGI7vmNelaR_D9-FUk5SbB3webqa5c" # Denoiser weights
            out = "./baselines/EventVLAD/denoiser_brisbane"
            gdown.download(id=file_id, output=out, quiet=False)

        # Set the path to the baseline configuration file
        self.baseline_config_path = './baselines/eventvlad.yaml'
        
        # Set the type of matrix generated for evaluation (distance or similarity)
        self.matrix_type = 'distance'

        # Load the baseline configuration
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)

        # Check if the pytorch-NetVlad path exists
        if not os.path.exists(self.baseline_config['netvlad_path']):
            netvlad_url = "https://github.com/Nanne/pytorch-NetVlad.git"
            clone_repo(netvlad_url, destination=self.baseline_config['netvlad_path'])
        # Download the NetVLAD weights if not already present    
        if not os.path.exists('./baselines/EventVLAD/vgg16_eventvlad.tar'):
            logger.info("Downloading the eventvlad weights...")
            # Get the eventvlad weights
            file_id = "1rSIhH1pk8ADxfqYQXoos_hTuWyfiWSu3"
            out = './baselines/EventVLAD/vgg16_eventvlad.tar'
            gdown.download(id=file_id, output=out, quiet=False)

        # Create the data output folder
        self.outdir = './output/eventvlad'
        os.makedirs(self.outdir, exist_ok=True)

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Format the reference and query data for the EventVLAD baseline.

        Frame loading is now storage-agnostic through:

            from datasets.dataloader import make_frame_source

        This supports the new frames.h5 storage system without manually listing
        frame_*.npy files.
        """

        self.config = config
        self.dataset_config = dataset_config
        self.reference = reference
        self.query = query

        # Get experimental details
        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()

        ref_name = ref_info["sequence_name"]
        query_name = query_info["sequence_name"]
        self.ref_name = ref_name
        self.query_name = query_name

        # Load data with or without an offset
        if "other" in dataset_config and "offset" in dataset_config["other"]:
            ref_offset, qry_offset = convert_offset(
                dataset_config['other']['offset'][ref_name],
                dataset_config['other']['offset'][query_name],
                dataset_config['other']['offset_time_scale'])
        else:
            ref_offset, qry_offset = None, None

        # Stream parameters passed to the denoiser subprocess, which opens the
        # EventCV stream itself (a live reader object cannot cross the process
        # boundary as a CLI argument).
        self.timewindow = timewindow
        self.ref_offset = ref_offset
        self.qry_offset = qry_offset
        self.ref_hdf5_path = ref_info['hdf5_path']
        self.query_hdf5_path = query_info['hdf5_path']

        # Denoised output directories used by build_execute/run.
        self.ref_dir_out = os.path.join(
            config["data_path"],
            dataset_config["dataset"]["name"],
            ref_name,
            f"{ref_name}-frames-{timewindow}-denoised",
        )

        self.query_dir_out = os.path.join(
            config["data_path"],
            dataset_config["dataset"]["name"],
            query_name,
            f"{query_name}-frames-{timewindow}-denoised",
        )

        # Main baseline output directory
        self.output_dir = os.path.join(
            self.outdir,
            f"{ref_info['dataset_name']}",
            f"{ref_info['sequence_name']}_{query_info['sequence_name']}",
            f"{config['frame_generator']}_{timewindow}",
        )

        os.makedirs(self.output_dir, exist_ok=True)


    def build_execute(self, config, data_config, ground_truth):
        """
        Build a commandline execute for the baseline with the provided reference, query, and ground truth data.
        """
        # Denoise the images and output them to the temporary directory
        # Build the command as a single string
        if not os.path.exists(self.ref_dir_out) or len(os.listdir(self.ref_dir_out)) == 0:
            os.makedirs(self.ref_dir_out, exist_ok=True)
            ref_convert = (
                # Include command line arugments specific to the baseline
                f'python utils/eventvlad_denoiser.py '
                f'--hdf5_path {self.ref_hdf5_path} '
                f'--dt_ms {self.timewindow} '
                + (f'--offset {self.ref_offset} ' if self.ref_offset is not None else '')
                + f'--model_path baselines/EventVLAD/denoiser_brisbane '
                f'--save_dir {self.ref_dir_out} '
                f'--use_gpu '
                f'--show {0}'
            )
            # Convert all data to denoised images
            self.ref_convert_cmd_str = ["pixi", "run", "bash", "-c", ref_convert]
            result = subprocess.run(self.ref_convert_cmd_str, check=True)
            logger.info("STDOUT:", result.stdout)
            if result.stderr:
                logger.error("STDERR:", result.stderr)
            if result.returncode != 0:
                raise RuntimeError(f"Baseline evaluation failed with return code {result.returncode}")
        if not os.path.exists(self.query_dir_out) or len(os.listdir(self.query_dir_out)) == 0:
            os.makedirs(self.query_dir_out, exist_ok=True)
            query_convert = (
                # Include command line arugments specific to the baseline
                f"python utils/eventvlad_denoiser.py "
                f"--hdf5_path {self.query_hdf5_path} "
                f"--dt_ms {self.timewindow} "
                + (f"--offset {self.qry_offset} " if self.qry_offset is not None else "")
                + f"--model_path baselines/EventVLAD/denoiser_brisbane "
                f"--save_dir {self.query_dir_out} "
                f"--use_gpu "
                f"--show {0}"
            )
            self.query_convert_cmd_str = ["pixi", "run", "bash", "-c", query_convert]
            result = subprocess.run(self.query_convert_cmd_str, check=True)
            logger.info("STDOUT:", result.stdout)
            if result.stderr:
                logger.error("STDERR:", result.stderr)
            if result.returncode != 0:
                raise RuntimeError(f"Baseline evaluation failed with return code {result.returncode}")

    def run(self):
        """
        Run the baseline.
        """
        import multiprocessing as mp
        from baselines.eventvlad_featureextraction import build_eventvlad_model_from_tar, extract_eventvlad_features, _device
        # cuda -> mps -> cpu (Apple GPU support for the NetVLAD forward pass)
        device = _device()
        logger.info(f"Extracting EventVLAD features on {device.type}")
        model = build_eventvlad_model_from_tar(
            weights_path="./baselines/EventVLAD/vgg16_eventvlad.tar",
            num_clusters=64,
            device=device,
        )
        # Only use DataLoader worker processes under 'fork' (Linux). Under 'spawn'
        # (macOS/Windows) each worker re-imports eventlab_run.py, and the
        # './baselines/EventVLAD' entry on sys.path shadows the project 'utils'
        # package (baselines/EventVLAD/utils.py), crashing the workers. Loading
        # inline (num_workers=0) avoids the re-import entirely.
        num_workers = 4 if mp.get_start_method(allow_none=True) == "fork" else 0
        ref_feats = extract_eventvlad_features(model, self.ref_dir_out, batch_size=8, num_workers=num_workers, device=device)
        query_feats = extract_eventvlad_features(model, self.query_dir_out, batch_size=8, num_workers=num_workers, device=device)
        D = (1 - (query_feats @ ref_feats.T)).T
        # Save the distance matrix
        np.save(os.path.join(self.output_dir, "distance_matrix.npy"), D)

    def parse_results(self, GT):
        """
        Summary sheet: upsert by (run_name, ref_query, array_name)
        Per-run sheet (self.name): upsert summary by (ref_query, array_name),
        and upsert each PR block keyed by "PR curve for {ref_query} :: {array_name}".
        """
        # gather files
        all_files = sorted(list(Path(self.output_dir).glob("*.npy")))
        all_names = [os.path.basename(f).replace(".npy", "") for f in all_files]
        all_arrays = [np.load(f) for f in all_files]
        GThard = np.load(GT)
        if not all_arrays:
            logger.warning("No .npy result files found in", self.output_dir)
            return
    
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

        # Run evaluation metrics
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

        # Save results to excel spreadsheet
        self.save_results(rows, pr_curves, self.name, f'{self.ref_name}_{self.query_name}')

    def cleanup(self):
        """
        Clean up temporary files.
        """
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)