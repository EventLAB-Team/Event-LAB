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

        # Apply temporal filtering
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

        # ----------------------------
        # Helper: load one frame as [H, W] (collapse polarity if needed)
        # ----------------------------
        def load_single_frame(path: Path) -> np.ndarray:
            arr = np.load(path)
            # collapse polarity channels if configured that way
            if (
                config['frame_generator'] == 'frames'
                and config['frame_accumulator'] in ('eventcount', 'polarity')
            ):
                # original code did arr.sum(axis=2) for (H, W, 2)
                if arr.ndim == 3:
                    arr = arr.sum(axis=2)
            return arr.astype(np.float32, copy=False)

        # ------------------------------------------------------------------
        # Pass 1: chunked over reference frames to compute reference_event_means
        # ------------------------------------------------------------------
        chunk_size = int(config.get("frames_chunk_size", 1000))

        ref_sum = None
        ref_count = 0
        H = W = None

        for start in range(0, len(ref_files), chunk_size):
            end = min(start + chunk_size, len(ref_files))
            batch_paths = ref_files[start:end]

            # load batch -> (B, H, W)
            batch = [load_single_frame(p) for p in batch_paths]
            if len(batch) == 0:
                continue
            batch = np.stack(batch, axis=0)  # (B, H, W)

            # Remove random bursts for this chunk
            batch_noburst = remove_random_bursts(batch, threshold=10).astype(np.float32, copy=False)

            if ref_sum is None:
                H, W = batch_noburst.shape[1:]
                ref_sum = batch_noburst.sum(axis=0, dtype=np.float64)
            else:
                ref_sum += batch_noburst.sum(axis=0, dtype=np.float64)

            ref_count += batch_noburst.shape[0]

        if ref_count == 0:
            raise ValueError("No reference frames left after filtering; check your config / time filter.")

        # mean over all (burst-filtered) reference frames, used for saliency
        self.reference_event_means = (ref_sum / float(ref_count)).astype(np.float32)

        # ------------------------------------------------------------------
        # Compute probabilities and sample sparse pixels from the full mean
        # ------------------------------------------------------------------
        if self.baseline_config['use_saliency']:
            prob_to_draw_from = adjust_and_normalize_probabilities(self.reference_event_means)
        else:
            prob_to_draw_from = None

        # NOTE: dataset_config should match actual (W, H), but we trust the data shape we saw.
        im_width  = dataset_config["dataset"]["resolution"][0]
        im_height = dataset_config["dataset"]["resolution"][1]
        if im_width != W or im_height != H:
            # Trust the actual data, override silently
            im_width, im_height = W, H

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
        num_pixels = random_pixels.shape[0]

        # ------------------------------------------------------------------
        # Pass 2: re-load ref & query in chunks, build full *_noburst and sparse arrays
        # ------------------------------------------------------------------
        num_ref = len(ref_files)
        num_qry = len(query_files)

        # Allocate final arrays
        self.reference_data_noburst = np.empty((num_ref, H, W), dtype=np.float32)
        self.query_data_noburst     = np.empty((num_qry, H, W), dtype=np.float32)
        self.sparse_reference_data  = np.empty((num_ref, num_pixels), dtype=np.float32)
        self.sparse_query_data      = np.empty((num_qry, num_pixels), dtype=np.float32)

        # Fill reference arrays
        ref_idx = 0
        for start in range(0, len(ref_files), chunk_size):
            end = min(start + chunk_size, len(ref_files))
            batch_paths = ref_files[start:end]
            if not batch_paths:
                continue

            batch = [load_single_frame(p) for p in batch_paths]
            batch = np.stack(batch, axis=0)  # (B, H, W)

            batch_noburst = remove_random_bursts(batch, threshold=10).astype(np.float32, copy=False)
            B = batch_noburst.shape[0]

            # self.reference_data_noburst[ref_idx:ref_idx + B] = batch_noburst
            # sparse sampling for this chunk
            self.sparse_reference_data[ref_idx:ref_idx + B] = batch_noburst[:, y_coords, x_coords]

            ref_idx += B

        # Fill query arrays
        qry_idx = 0
        for start in range(0, len(query_files), chunk_size):
            end = min(start + chunk_size, len(query_files))
            batch_paths = query_files[start:end]
            if not batch_paths:
                continue

            batch = [load_single_frame(p) for p in batch_paths]
            batch = np.stack(batch, axis=0)  # (B, H, W)

            batch_noburst = remove_random_bursts(batch, threshold=10).astype(np.float32, copy=False)
            B = batch_noburst.shape[0]

            # self.query_data_noburst[qry_idx:qry_idx + B] = batch_noburst
            self.sparse_query_data[qry_idx:qry_idx + B] = batch_noburst[:, y_coords, x_coords]

            qry_idx += B

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
            self.baseline_config['sequence_length'],
            # optional: override chunk_size here if you like
            chunk_size=1000,
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