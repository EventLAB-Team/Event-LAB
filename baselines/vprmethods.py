import os, yaml
import numpy as np
from pathlib import Path
from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from datetime import datetime, timezone
import re, subprocess
import utils.functional as FUNC
import tempfile

class vprmethods_baseline(EventBaseline):
    def __init__(self):
        super().__init__()

        self.name = "vprmethods"
        # Check if the baseline repository is already cloned
        self.repo_path = "./baselines/vpr_methods"
        # Baseline URL
        self.url = "https://github.com/gmberton/VPR-methods-evaluation.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)
        self.baseline_config_path = './baselines/vprmethods.yaml'
        # Load the baseline configuration
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)
        # Create the data output folder
        self.outdir = './output/vprmethods'
        os.makedirs(self.outdir, exist_ok=True)
        self.matrix_type = 'distance' # options are 'similarity' or 'distance'

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Format the reference and query data for the baseline.
        """
        self.config = config
        # Get experimental details
        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()

        ref_name = ref_info['sequence_name']
        query_name = query_info['sequence_name']

        # from ref_info['file_path'] dict, find the directory that matches ref/query name and timewindow
        self.ref_name = f'{ref_name}-{config["frame_generator"]}-{timewindow}'
        self.query_name = f'{query_name}-{config["frame_generator"]}-{timewindow}'
        self.ref_directory = ref_info['file_path'][self.ref_name]
        self.query_directory = query_info['file_path'][self.query_name]

        _RX_FRAME = re.compile(r"^frame_(\d+)\.png$")

        def list_frame_files(dirpath: str):
            paths = []
            for p in Path(dirpath).iterdir():
                m = _RX_FRAME.fullmatch(p.name)
                if m:
                    paths.append((int(m.group(1)), p))
            paths.sort(key=lambda t: t[0])  # numeric sort
            return [p for _, p in paths]

        # usage
        if config['frame_generator'] == 'reconstruction':
            self.ref_directory = os.path.join(self.ref_directory, 'reconstruction')
            self.query_directory = os.path.join(self.query_directory, 'reconstruction')

        ref_files   = list_frame_files(self.ref_directory)
        query_files = list_frame_files(self.query_directory)
        # after you have ref_files, query_files and min_gap_sec
        min_gap_sec = float(config.get("filter_places_sec", 60))

        ref_res   = FUNC._apply_time_filter_to_files(ref_files,   self.ref_directory,  min_gap_sec, debug=False)
        query_res = FUNC._apply_time_filter_to_files(query_files, self.query_directory, min_gap_sec, debug=False)

        # Replace file lists with filtered ones
        ref_files   = ref_res['files']
        query_files = query_res['files']
        print(len(ref_files), "reference frames after filtering")

        # OPTIONAL: Create temporary directory to store converted data, if not using numpy arrays
        self.temp_dir = tempfile.mkdtemp(prefix="baseline_data_")
        self.ref_dir = self.ref_directory
        self.query_dir = self.query_directory
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.query_dir, exist_ok=True)
        # import shutil
        # # Copy files to temporary directory using shutil
        # for idx, ref_file in enumerate(ref_files):
        #     shutil.copy(ref_file, os.path.join(self.ref_dir, f"frame_{idx:06d}.png"))

        # for idx, query_file in enumerate(query_files):
        #     shutil.copy(query_file, os.path.join(self.query_dir, f"frame_{idx:06d}.png"))


        # Set the output folder
        self.output_dir = os.path.join(self.outdir, f"{ref_info['dataset_name']}", f"{ref_info['sequence_name']}_{query_info['sequence_name']}",
                                       f"{config['frame_generator']}_{timewindow}")
        os.makedirs(self.output_dir, exist_ok=True)

    def build_execute(self, config, data_config, ground_truth):
        """
        Build a commandline execute for the baseline with the provided reference, query, and ground truth data.
        """
        if config['frame_generator'] == 'reconstruction':
            eval_cmd = (
                f'python main.py '
                f'--method {self.baseline_config["method"]} '
                f'--backbone {self.baseline_config["backbone"]} '
                f'--descriptors_dimension {self.baseline_config["descriptors_dimension"]} '
                f'--no_labels '
                f'--database_folder {self.ref_dir} '
                f'--queries_folder {self.query_dir} '
                f'--save_descriptors'
            )
        self.full_cmd = ["pixi", "run", "bash", "-c", eval_cmd]

    def run(self):
        """
        Run the baseline.
        """
        '''
        Implement run logic here to retrieve distance matrix and save it for analysis.
        '''
        subprocess.run(self.full_cmd, check=True, cwd='baselines/vpr_methods')
        # Retrieve the features from the latest log directory
        log_dir = sorted(Path(self.repo_path).glob("logs/default/*"), key=os.path.getmtime)[-1]
        # Load the `databse_descriptors.npy` and `query_descriptors.npy` files
        database_descriptors = np.load(log_dir / "database_descriptors.npy")
        query_descriptors = np.load(log_dir / "queries_descriptors.npy")
        # Compute the distance matrix
        D = (1 - (query_descriptors @ database_descriptors.T)).T
        # Save the distance matrices
        np.save(f"{self.output_dir}/distance_matrix.npy", D)
        
    
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
                tolerance=self.config['ground_truth_tolerance']
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