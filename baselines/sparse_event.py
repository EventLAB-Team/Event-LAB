import os, yaml, torch
import numpy as np
from pathlib import Path
from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from datetime import datetime, timezone
import utils.functional as FUNC
from utils.utils import convert_offset
from tqdm import tqdm
import eventcv as ecv

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
        # set the device
        self.device = torch.device("cuda" if torch.cuda.is_available()
                            else "cpu")

    def format_data(self, config, dataset_config, reference, query, timewindow, ref_offset=None, qry_offset=None):
        """
        Format the reference and query data for the baseline, using chunked
        loading so we never load all frames at once just to compute saliency.
        """
        self.config = config
        from baselines.vpr_sparse_event.src.sparse_event_vpr.sparse_pixel_utils import (
            adjust_and_normalize_probabilities,
            get_random_pixels,
        )
        from baselines.vpr_sparse_event.src.sparse_event_vpr.utils import remove_random_bursts

        # Get experimental details
        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()

        self.ref_name = ref_info['sequence_name']
        self.query_name = query_info['sequence_name']

        # Load data with or without an offset
        if "other" in dataset_config and "offset" in dataset_config["other"]:
            ref_offset, qry_offset = convert_offset(
                dataset_config['other']['offset'][self.ref_name],
                dataset_config['other']['offset'][self.query_name],
                dataset_config['other']['offset_time_scale'])

        # Open reference and query into EventCV
        reference = ecv.open(ref_info['hdf5_path'], dt_ms=timewindow, offset=ref_offset, hot_pixel_filter=True)
        query = ecv.open(query_info['hdf5_path'], dt_ms=timewindow, offset=qry_offset, hot_pixel_filter=True)

        batch_noburst = []
        for idx in tqdm(range(reference.n_slices), desc="Removing random bursts from reference"):
            # batch is already [B, H, W] float32, regardless of npy/h5 backend
            batch_noburst.append(remove_random_bursts(
                reference.slice(idx).count().numpy(),
                threshold=10).astype(np.float32)
                )

        # mean over all (burst-filtered) reference frames, used for saliency
        batch_mean = np.mean(np.stack(batch_noburst), axis=0).astype(np.float64)

        # ------------------------------------------------------------------
        # Compute probabilities and sample sparse pixels from the full mean
        # ------------------------------------------------------------------
        if self.baseline_config['use_saliency']:
            prob_to_draw_from = adjust_and_normalize_probabilities(batch_mean[0])
        else:
            prob_to_draw_from = None

        im_width  = dataset_config["dataset"]["resolution"][0]
        im_height = dataset_config["dataset"]["resolution"][1]

        random_pixels = np.array(
            get_random_pixels(
                self.baseline_config['num_target_pixels'],
                im_width=im_width,
                im_height=im_height,
                local_suppression_radius=7,
                prob_to_draw_from=prob_to_draw_from,
            )
        )


        # y, x for indexing
        y_coords = random_pixels[:, 0]
        x_coords = random_pixels[:, 1]

        # Remove random_pixels indices from reference data
        self.sparse_reference_data = np.stack(batch_noburst)[:, 0, y_coords, x_coords]

        query_noburst = []
        for idx in tqdm(range(query.n_slices), desc="Removing random bursts from query"):
            query_noburst.append(remove_random_bursts(
                query.slice(idx).count().numpy(),
                threshold=10).astype(np.float32)
            )

        # remove indices from query_noburst
        self.sparse_query_data = np.stack(query_noburst)[:, 0, y_coords, x_coords]

        # Create sparse_event dict for downstream distance computation
        self.frames_sets = {
            "subset": (self.sparse_reference_data, self.sparse_query_data),
        }

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
        pass

    def run(self):
        """
        Run the baseline.
        """
        print(f"Running baseline")
        from baselines.vpr_sparse_event.src.sparse_event_vpr.sparse_pixel_utils import compute_distance_matrices

        # Get the current device
        device = torch.device("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available()
                            else "cpu")

        distance_matrices = compute_distance_matrices(
            self.frames_sets,
            device,
            self.baseline_config['sequence_length']
        )

        # Save PNGs + npy as before
        import matplotlib.pyplot as plt
        for key, matrix in distance_matrices.items():
            if key.endswith("_seq"):
                # Only visualize the seq matrices, or keep as you had it
                pass

        for key, matrix in distance_matrices.items():
            if not key.endswith("_seq"):
                plt.figure(figsize=(10, 8))
                plt.imshow(matrix, cmap='viridis')
                plt.colorbar()
                plt.title(f'Distance Matrix - {key}')
                plt.xlabel('Query Frames')
                plt.ylabel('Reference Frames')
                plt.savefig(f"{self.output_dir}/distance_matrix_{key}.png")
                plt.close()

        # np.save(f"{self.output_dir}/all_pixels_seq.npy", distance_matrices['all_pixels_seq'])
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