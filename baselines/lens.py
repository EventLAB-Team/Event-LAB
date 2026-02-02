import os, tempfile, subprocess, csv, yaml
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from datetime import datetime, timezone
import re
import utils.functional as FUNC

class LENS_baseline(EventBaseline):
    def __init__(self):
        super().__init__()

        self.name = "LENS"
        # Check if the baseline repository is already cloned
        self.repo_path = "./baselines/LENS"
        # Baseline URL
        self.url = "https://github.com/AdamDHines/LENS.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)
        self.lens_config_path = './baselines/lens.yaml'
        # Load the LENS configuration
        with open(self.lens_config_path, 'r') as file:
            self.lens_config = yaml.safe_load(file)
        # Create the data output folder
        self.outdir = './output/lens'
        os.makedirs(self.outdir, exist_ok=True)
        self.matrix_type = 'similarity' # options are 'similarity' or 'distance'

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Format the reference and query data for LENS baseline.
        """
        self.config = config
        # Get experimental details
        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()
        self.ref_name = ref_info['sequence_name']
        self.query_name = query_info['sequence_name']
        # from ref_info['file_path'] dict, find the directory that matches ref/query name and timewindow
        self.ref_key = [d for d in ref_info['file_path'] if self.ref_name in d and str(timewindow) in d]
        self.query_key = [d for d in query_info['file_path'] if self.query_name in d and str(timewindow) in d]
        self.ref_directory = ref_info['file_path'][self.ref_key[0]]
        self.query_directory = query_info['file_path'][self.query_key[0]]
        self.ref_name = self.ref_key[0]
        self.query_name = self.query_key[0]

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
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="lens_data_")
        self.ref_dir = os.path.join(self.temp_dir, self.ref_name)
        self.query_dir = os.path.join(self.temp_dir, self.query_name)
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.query_dir, exist_ok=True)
        
        # Convert reference arrays to PNGs and create CSV
        self.reference_places = len(ref_files)
        
        # Create reference CSV file
        ref_csv_path = os.path.join(self.temp_dir, f"{self.ref_name}.csv")
        with open(ref_csv_path, 'w', newline='') as ref_csv:
            writer = csv.writer(ref_csv)
            writer.writerow(['Image_name', 'index'])
            
            for idx, ref_file in enumerate(tqdm(ref_files, desc="Formatting reference data to LENS requirements")):
                ref_data = np.load(ref_file)
                # if the config reconstruction is polarity, sum the two polarities
                if not config['frame_generator'] == 'reconstruction' and (config['frame_accumulator'] == 'polarity' or config['frame_accumulator'] == 'eventcount'):
                    ref_data = np.sum(ref_data, axis=2)
                filename = f"{idx:06d}.png"
                
                # Create and save image
                ref_clipped = np.clip(ref_data, 0, 255).astype(np.uint8)
                ref_img = Image.fromarray(ref_clipped)
                ref_img.save(os.path.join(self.ref_dir, filename))
                
                # Write to CSV
                writer.writerow([filename, idx])
        
        # Convert query arrays to PNGs and create CSV
        self.query_places = len(query_files)
        
        # Create query CSV file
        query_csv_path = os.path.join(self.temp_dir, f"{self.query_name}.csv")
        with open(query_csv_path, 'w', newline='') as query_csv:
            writer = csv.writer(query_csv)
            writer.writerow(['Image_name', 'index'])
            
            for idx, query_file in enumerate(tqdm(query_files, desc="Formatting query data to LENS requirements")):
                query_data = np.load(query_file)
                if not config['frame_generator'] == 'reconstruction' and (config['frame_accumulator'] == 'polarity' or config['frame_accumulator'] == 'eventcount'):
                    query_data = np.sum(query_data, axis=2)
                filename = f"{idx:06d}.png"
                
                # Create and save image
                query_clipped = np.clip(query_data, 0, 255).astype(np.uint8)
                query_img = Image.fromarray(query_clipped)
                query_img.save(os.path.join(self.query_dir, filename))
                
                # Write to CSV
                writer.writerow([filename, idx])

        self.output_dir = os.path.join(self.outdir, f"{ref_info['dataset_name']}", f"{ref_info['sequence_name']}_{query_info['sequence_name']}",
                                       f"{config['frame_generator']}_{timewindow}")
        os.makedirs(self.output_dir, exist_ok=True)

    def build_execute(self, config, data_config, ground_truth):
        """
        Build a commandline execute for LENS baseline with the provided reference, query, and ground truth data.
        """
        # Store ground truth path for later use
        self.ground_truth = ground_truth
        
            # Build the command as a single string
        train_cmd_str = (
            f"python {os.path.join(self.repo_path, 'main.py')} "
            f"--data_dir {self.temp_dir} "
            f"--camera . "
            f"--dataset . "
            f"--reference {self.ref_name} "
            f"--reference_places {self.reference_places} "
            f"--dims {self.lens_config['dims'][0]} {self.lens_config['dims'][1]} "
            f"--roi_dim {data_config['dataset']['resolution'][1]} {data_config['dataset']['resolution'][0]} "
            f"--feature_multiplier {self.lens_config['feature_multiplier']} "
            f"--train_model "
            f"--models_dir {self.repo_path}/lens/models/ "
            f"--output_dir {self.output_dir} "
            f"--output_subfolder "
            f"--nocuda"
        )
    
        # Wrap it with pixi run
        self.train_cmd = ["pixi", "run", "bash", "-c", train_cmd_str]

        # Build the command as a single string
        eval_cmd_str = (
            f"python {os.path.join(self.repo_path, 'main.py')} "
            f"--data_dir {self.temp_dir} "
            f"--camera . "
            f"--dataset . "
            f"--reference {self.ref_name} "
            f"--reference_places {self.reference_places} "
            f"--query {self.query_name} "
            f"--query_places {self.query_places} "
            f"--dims {self.lens_config['dims'][0]} {self.lens_config['dims'][1]} "
            f"--roi_dim {data_config['dataset']['resolution'][1]} {data_config['dataset']['resolution'][0]} "
            f"--feature_multiplier {self.lens_config['feature_multiplier']} "
            f"--models_dir {self.repo_path}/lens/models/ "
            f"--output_dir {self.output_dir} "
            f"--sequence_length {self.lens_config['sequence_length']} "
            f"--timebin {self.lens_config['timebin']} "
            f"--GT_tolerance {0} "
            f"--output_subfolder "
            f"--gt_dir {config['data_path']}/{data_config['dataset']['name']}/ground_truth "
            f"--nocuda"
        )
        
        # Wrap it with pixi run
        self.eval_cmd = ["pixi", "run", "bash", "-c", eval_cmd_str]

    def run(self):
        """
        Run the LENS baseline.
        """
        print(f"Running LENS baseline with command: {' '.join(self.train_cmd)}")
        
        # Run from the current directory, not the repo directory
        # since we're providing full paths in the command
        # Check if the model already exists, skip training if it does
        model_path = os.path.join(self.repo_path, 
                                  'lens', 'models', 
        f"{self.ref_name}_LENS_IN{self.lens_config['dims'][0] * self.lens_config['dims'][1]}_FN{self.lens_config['dims'][0] * self.lens_config['dims'][1]*self.lens_config['feature_multiplier']}_DB{self.reference_places}.pth")

        if not os.path.exists(model_path):
            result = subprocess.run(self.train_cmd, capture_output=True, text=True)
        
            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            if result.returncode != 0:
                raise RuntimeError(f"LENS baseline failed with return code {result.returncode}")
        
            print(f"Running LENS baseline with command: {' '.join(self.eval_cmd)}")
        # Run evaluation command
        result = subprocess.run(self.eval_cmd, capture_output=True, text=True)  
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"LENS evaluation failed with return code {result.returncode}")
    
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