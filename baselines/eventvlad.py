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

class eventvlad_baseline(EventBaseline):
    def __init__(self):
        super().__init__()

        self.name = "eventvlad"
        # Check if the baseline repository is already cloned
        self.repo_path = "./baselines/EventVLAD"
        # Baseline URL
        self.url = "https://github.com/alexjunholee/EventVLAD.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)
            file_id = "1xdoGI7vmNelaR_D9-FUk5SbB3webqa5c"  # from your URL
            out = "./baselines/EventVLAD/denoiser_brisbane"

            gdown.download(id=file_id, output=out, quiet=False)  # no need for the usercontent URL

        self.baseline_config_path = './baselines/eventvlad.yaml'
        self.matrix_type = 'distance' # options are 'similarity' or 'distance'

        # Load the baseline configuration
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)
        # Check if the pytorch-NetVlad path exists
        if not os.path.exists(self.baseline_config['netvlad_path']):
            netvlad_url = "https://github.com/Nanne/pytorch-NetVlad.git"
            clone_repo(netvlad_url, destination=self.baseline_config['netvlad_path'])
        if not os.path.exists(os.path.join(self.baseline_config['netvlad_path'], 'vgg16_eventvlad.tar')):
            print("Downloading the eventvlad weights...")
            # Get the eventvlad weights
            file_id = "1rSIhH1pk8ADxfqYQXoos_hTuWyfiWSu3"
            out = './baselines/EventVLAD/vgg16_eventvlad.tar'
            gdown.download(id=file_id, output=out, quiet=False)  # no need for the usercontent URL

        # Create the data output folder
        self.outdir = './output/eventvlad'
        os.makedirs(self.outdir, exist_ok=True)

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Format the reference and query data for the baseline.
        """
        self.config=config
        # Get experimental details
        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()

        ref_name = ref_info['sequence_name']
        query_name = query_info['sequence_name']

        # from ref_info['file_path'] dict, find the directory that matches ref/query name and timewindow
        self.ref_key = [d for d in ref_info['file_path'] if ref_name in d and str(timewindow) in d]
        self.query_key = [d for d in query_info['file_path'] if query_name in d and str(timewindow) in d]
        self.ref_directory = ref_info['file_path'][self.ref_key[0]]
        self.query_directory = query_info['file_path'][self.query_key[0]]
        self.ref_name = self.ref_key[0]
        self.query_name = self.query_key[0]

        from pathlib import Path
        import re

        _RX_FRAME = re.compile(r"^frame_(\d+)\.npy$")

        def list_frame_files(dirpath: str):
            paths = []
            for p in Path(dirpath).iterdir():
                m = _RX_FRAME.fullmatch(p.name)
                if m:
                    paths.append((int(m.group(1)), p))
            paths.sort(key=lambda t: t[0])  # numeric sort
            return [p for _, p in paths]

        # usage
        ref_files   = list_frame_files(self.ref_directory)
        query_files = list_frame_files(self.query_directory)
        # after you have ref_files, query_files and min_gap_sec
        min_gap_sec = float(config.get("filter_places_sec", 60))

        ref_res   = FUNC._apply_time_filter_to_files(ref_files,   self.ref_directory,  min_gap_sec, debug=False)
        query_res = FUNC._apply_time_filter_to_files(query_files, self.query_directory, min_gap_sec, debug=False)

        # Replace file lists with filtered ones
        ref_files   = ref_res['files']
        query_files = query_res['files']

        # proceed to load arrays
        self.reference_data = np.array([np.load(p) for p in ref_files])
        self.query_data     = np.array([np.load(p) for p in query_files])

        # OPTIONAL: Create temporary directory to store converted data, if not using numpy arrays
        self.temp_dir = tempfile.mkdtemp(prefix="baseline_data_")
        self.ref_dir = os.path.join(self.temp_dir, ref_name)
        self.query_dir = os.path.join(self.temp_dir, query_name)
        self.ref_dir_out = os.path.join(self.temp_dir, f"{ref_name}_denoised")
        self.query_dir_out = os.path.join(self.temp_dir, f"{query_name}_denoised")
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.query_dir, exist_ok=True)
        os.makedirs(self.ref_dir_out, exist_ok=True)
        os.makedirs(self.query_dir_out, exist_ok=True)

        # store the reference and query data to the temporary directory
        for i, arr in enumerate(self.reference_data):
            np.save(os.path.join(self.ref_dir, f"frame_{i:06d}.npy"), arr)
        for i, arr in enumerate(self.query_data):
            np.save(os.path.join(self.query_dir, f"frame_{i:06d}.npy"), arr)

        self.output_dir = os.path.join(self.outdir, f"{ref_info['dataset_name']}", f"{ref_info['sequence_name']}_{query_info['sequence_name']}",
                                       f"{config['frame_generator']}_{timewindow}")
        os.makedirs(self.output_dir, exist_ok=True)


    def build_execute(self, config, data_config, ground_truth):
        """
        Build a commandline execute for the baseline with the provided reference, query, and ground truth data.
        """
        # Denoise the images and output them to the temporary directory
        # Build the command as a single string
        ref_convert = (
            # Include command line arugments specific to the baseline
            f'python utils/eventvlad_denoiser.py '
            f'--input_dir {self.ref_dir} '
            f'--model_path baselines/EventVLAD/denoiser_brisbane '
            f'--save_dir {self.ref_dir_out} '
            f'--use_gpu '
            f'--show {0}'
        )
        query_convert = (
            # Include command line arugments specific to the baseline
            f"python utils/eventvlad_denoiser.py "
            f"--input_dir {self.query_dir} "
            f"--model_path baselines/EventVLAD/denoiser_brisbane "
            f"--save_dir {self.query_dir_out} "
            f"--use_gpu "
            f"--show {0}"
        )
        # Convert all data to denoised images
        self.ref_convert_cmd_str = ["pixi", "run", "bash", "-c", ref_convert]
        result = subprocess.run(self.ref_convert_cmd_str, check=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"Baseline evaluation failed with return code {result.returncode}")
        self.query_convert_cmd_str = ["pixi", "run", "bash", "-c", query_convert]
        result = subprocess.run(self.query_convert_cmd_str, check=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"Baseline evaluation failed with return code {result.returncode}")

    def run(self):
        """
        Run the baseline.
        """
        import torch
        from baselines.eventvlad_featureextraction import build_eventvlad_model_from_tar, extract_eventvlad_features
        model = build_eventvlad_model_from_tar(
            weights_path="./baselines/EventVLAD/vgg16_eventvlad.tar",
            num_clusters=64,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        ref_feats = extract_eventvlad_features(model, self.ref_dir_out, batch_size=8, num_workers=4)
        query_feats = extract_eventvlad_features(model, self.query_dir_out, batch_size=8, num_workers=4)
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
            print("No .npy result files found in", self.output_dir)
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