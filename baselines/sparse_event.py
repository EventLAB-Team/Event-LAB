import os, yaml, torch
import numpy as np
from pathlib import Path
from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from datetime import datetime, timezone
import utils.functional as FUNC

class sparse_event_baseline(EventBaseline):
    def __init__(self):
        super().__init__()
        self.name = "sparse_event_vpr"
        # Check if the baseline repository is already cloned
        self.repo_path = "./baselines/vpr_sparse_event"
        # Baseline URL
        self.url = "https://github.com/Tobias-Fischer/sparse-event-vpr.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)
        self.baseline_config_path = './baselines/sparse_event.yaml'
        # Load the baseline configuration
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)
        # Create the data output folder
        self.outdir = './output/sparse_event'
        os.makedirs(self.outdir, exist_ok=True)
        # Matrix type
        self.matrix_type = 'distance' # options are 'similarity' or 'distance'

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Format the reference and query data for the baseline.
        """
        self.config=config
        from baselines.vpr_sparse_event.src.sparse_event_vpr.sparse_pixel_utils import adjust_and_normalize_probabilities, get_random_pixels
        from baselines.vpr_sparse_event.src.sparse_event_vpr.utils import remove_random_bursts
        # Get experimental details
        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()

        ref_name = ref_info['sequence_name']
        query_name = query_info['sequence_name']

        # from ref_info['file_path'] dict, find the directory that matches ref/query name and timewindow
        self.ref_key = [d for d in ref_info['file_path'] if ref_name in d and str(timewindow) in d]
        self.query_key = [d for d in query_info['file_path'] if query_name in d and str(timewindow) in d]
        self.ref_name = self.ref_key[0]
        self.query_name = self.query_key[0]
        self.ref_directory = ref_info['file_path'][self.ref_key[0]]
        self.query_directory = query_info['file_path'][self.query_key[0]]

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

        # Save dropped indices (relative to original unfiltered order)
        self.ref_dropped_idx   = ref_res['dropped_idx']
        self.query_dropped_idx = query_res['dropped_idx']

        # (optional) also keep kept_idx if you need it elsewhere
        self.ref_kept_idx   = ref_res['kept_idx']
        self.query_kept_idx = query_res['kept_idx']

        # proceed to load arrays
        self.reference_data = np.array([np.load(p) for p in ref_files])
        self.query_data     = np.array([np.load(p) for p in query_files])
        
        # if config uses frames with polarity, sum the two polarities over the list of arrays
        if config['frame_generator'] == 'frames' and (config['frame_accumulator'] == 'eventcount' or config['frame_accumulator'] == 'polarity'):
            self.reference_data = [arr.sum(axis=2) for arr in self.reference_data]
            self.query_data     = [arr.sum(axis=2) for arr in self.query_data]
            self.reference_data = np.array(self.reference_data)
            self.query_data     = np.array(self.query_data)

        print(self.reference_data.shape, self.query_data.shape)
        # Remove random bursts from the data
        self.reference_data_noburst = remove_random_bursts(self.reference_data, threshold=10)
        self.query_data_noburst = remove_random_bursts(self.query_data, threshold=10)

        # Get the mean and variance of reference and query data
        self.reference_event_means = self.reference_data_noburst.mean(axis=0)

        # Get the proababilities for the reference data
        if self.baseline_config['use_saliency']:
            prob_to_draw_from = adjust_and_normalize_probabilities(self.reference_event_means)
        else:
            prob_to_draw_from = None
        random_pixels = np.array(get_random_pixels(self.baseline_config['num_target_pixels'], 
                                                   im_width=dataset_config["dataset"]["resolution"][0], 
                                                   im_height=dataset_config["dataset"]["resolution"][1], 
                                                   local_suppression_radius=7, 
                                                   prob_to_draw_from=prob_to_draw_from))

        # Apply sparse pixel sampling
        x_coords = random_pixels[:, 1]
        y_coords = random_pixels[:, 0]
        self.sparse_reference_data = self.reference_data[:, y_coords, x_coords]
        self.sparse_query_data = self.query_data[:, y_coords, x_coords]

        # Create sparse_event dict
        self.frames_sets = {
            "all_pixels": (self.reference_data_noburst, self.query_data_noburst),
            "subset": ( self.sparse_reference_data,  self.sparse_query_data)
        }

        self.output_dir = os.path.join(self.outdir, f"{ref_info['dataset_name']}", f"{ref_info['sequence_name']}_{query_info['sequence_name']}",
                                       f"{config['frame_generator']}_{timewindow}")
        os.makedirs(self.output_dir, exist_ok=True)

    def build_execute(self, config, data_config, ground_truth):
        """
        Build a commandline execute for the baseline with the provided reference, query, and ground truth data.
        """
        pass

    def run(self):
        """
        Run the baseline.
        """
        print(f"Running baseline")
        from baselines.vpr_sparse_event.src.sparse_event_vpr.sparse_pixel_utils import compute_distance_matrices
        # Get the current device
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # Parse results and return them in a standardized format
        distance_matrices = compute_distance_matrices(self.frames_sets, device, self.baseline_config['sequence_length'])
        # Save a png of the distance matrices
        import matplotlib.pyplot as plt
        for key, matrix in distance_matrices.items():
            plt.figure(figsize=(10, 8))
            plt.imshow(matrix, cmap='viridis')
            plt.colorbar()
            plt.title(f'Distance Matrix - {key}')
            plt.xlabel('Query Frames')
            plt.ylabel('Reference Frames')
            plt.savefig(f"{self.output_dir}/distance_matrix_{key}.png")
            plt.close()
        # Save the distance matrices
        np.save(f"{self.output_dir}/all_pixels_seq.npy", distance_matrices['all_pixels_seq'])
        np.save(f"{self.output_dir}/subset_seq.npy", distance_matrices['subset_seq'])
    
    def parse_results(self, GT):
        # gather files
        all_files  = sorted(list(Path(self.output_dir).glob("*.npy")))
        all_names  = [os.path.basename(f).replace(".npy", "") for f in all_files]
        all_arrays = [np.load(f) for f in all_files]
        GThard     = np.load(GT)

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
        """
        Clean up temporary files.
        """
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)