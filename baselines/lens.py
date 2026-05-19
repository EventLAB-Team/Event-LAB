import os, tempfile, subprocess, csv, yaml
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from datetime import datetime, timezone
from datasets.dataloader import make_frame_source
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

        LENS expects image folders plus CSV manifests. Event-LAB now stores
        frames in frames.h5, so this wrapper streams frames from either the new
        HDF5 store or legacy frame_*.npy files and materializes only the PNGs
        needed by the external LENS dataloader.
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

        min_gap_sec = float(config.get("filter_places_sec", 60))

        ref_source = make_frame_source(
            self.ref_directory,
            min_gap_sec=min_gap_sec,
            legacy_time_filter_fn=FUNC._apply_time_filter_to_files,
        )
        query_source = make_frame_source(
            self.query_directory,
            min_gap_sec=min_gap_sec,
            legacy_time_filter_fn=FUNC._apply_time_filter_to_files,
        )
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="lens_data_")
        self.ref_dir = os.path.join(self.temp_dir, self.ref_name)
        self.query_dir = os.path.join(self.temp_dir, self.query_name)
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.query_dir, exist_ok=True)

        def frame_to_uint8(frame):
            frame = np.asarray(frame)
            if frame.ndim == 3:
                frame = np.sum(frame, axis=-1)
            return np.clip(frame, 0, 255).astype(np.uint8)

        def write_lens_sequence(frame_source, out_dir, csv_path, desc):
            if len(frame_source) <= 0:
                raise ValueError(f"No frames available for LENS input: {frame_source.path}")

            places = 0
            with open(csv_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Image_name', 'index', 'gps_coordinate'])

                with tqdm(total=len(frame_source), desc=desc) as pbar:
                    for batch in frame_source.iter_batches(64):
                        for frame in batch:
                            filename = f"{places:06d}.png"
                            Image.fromarray(frame_to_uint8(frame)).save(os.path.join(out_dir, filename))
                            writer.writerow([filename, places, 0])
                            places += 1
                            pbar.update(1)

            return places
        
        ref_csv_path = os.path.join(self.temp_dir, f"{self.ref_name}.csv")
        query_csv_path = os.path.join(self.temp_dir, f"{self.query_name}.csv")

        self.reference_places = write_lens_sequence(
            ref_source,
            self.ref_dir,
            ref_csv_path,
            "Formatting reference data to LENS requirements",
        )
        self.query_places = write_lens_sequence(
            query_source,
            self.query_dir,
            query_csv_path,
            "Formatting query data to LENS requirements",
        )

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
            f"python -u {os.path.join(self.repo_path, 'main.py')} "
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
            f"python -u {os.path.join(self.repo_path, 'main.py')} "
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
            result = subprocess.run(self.train_cmd, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"LENS baseline failed with return code {result.returncode}")
        
            print(f"Running LENS baseline with command: {' '.join(self.eval_cmd)}")
        # Run evaluation command
        result = subprocess.run(self.eval_cmd, text=True)
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
