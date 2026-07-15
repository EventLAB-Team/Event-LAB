import os, tempfile, subprocess, csv, yaml
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from datetime import datetime, timezone
from utils.utils import convert_offset
import eventcv as ecv

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

    @staticmethod
    def _open_reader(hdf5_path, timewindow, offset):
        """Open an EventCV count-frame stream with hot-pixel filtering."""
        kwargs = {"dt_ms": timewindow, "repr": "count", "hot_pixel_filter": True}
        if offset is not None:
            kwargs["offset"] = offset
        return ecv.open(hdf5_path, **kwargs)

    @staticmethod
    def _kept_place_indices(n_slices, timewindow_ms, min_gap_sec):
        """
        Greedy temporal thinning over fixed-duration frames.

        EventCV renders uniform dt_ms frames, so frame i sits at a deterministic
        time i * dt_sec (the absolute offset is constant and cancels in the
        pairwise gap). This reproduces the greedy "keep frames >= min_gap_sec
        apart" filter that used to live in make_frame_source, without needing any
        per-frame tick metadata.
        """
        if not min_gap_sec or min_gap_sec <= 0:
            return list(range(n_slices))
        dt_sec = float(timewindow_ms) / 1000.0
        if dt_sec <= 0:
            return list(range(n_slices))

        kept = []
        last_kept_t = None
        for i in range(n_slices):
            t = i * dt_sec
            if last_kept_t is None or (t - last_kept_t) >= min_gap_sec:
                kept.append(i)
                last_kept_t = t
        return kept

    @staticmethod
    def _frame_to_uint8(frame):
        """Collapse a single [C,H,W] (or [H,W]) count frame to a uint8 image."""
        frame = np.asarray(frame)
        if frame.ndim == 3:
            # EventCV yields channel-first frames; sum polarity/count channels.
            frame = frame.sum(axis=0)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _write_lens_sequence(self, reader, timewindow, min_gap_sec, out_dir, csv_path, desc,
                             batch_size=64):
        """
        Stream fixed-duration frames from an EventCV reader, thin them to distinct
        places, and materialize the PNGs + CSV manifest the external LENS
        dataloader expects. Returns the number of places written.
        """
        n_slices = int(reader.n_slices)
        kept = self._kept_place_indices(n_slices, timewindow, min_gap_sec)
        if len(kept) <= 0:
            raise ValueError(
                f"No frames available for LENS input (n_slices={n_slices}, "
                f"min_gap_sec={min_gap_sec})."
            )

        places = 0
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Image_name', 'index', 'gps_coordinate'])

            with tqdm(total=len(kept), desc=desc) as pbar:
                for start in range(0, len(kept), batch_size):
                    idx_chunk = kept[start:start + batch_size]
                    batch = reader.batch(idx_chunk)  # [B, C, H, W]
                    batch = batch.numpy() if hasattr(batch, "numpy") else np.asarray(batch)
                    for b in range(batch.shape[0]):
                        filename = f"{places:06d}.png"
                        Image.fromarray(self._frame_to_uint8(batch[b])).save(
                            os.path.join(out_dir, filename)
                        )
                        writer.writerow([filename, places, 0])
                        places += 1
                        pbar.update(1)

        return places

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Format the reference and query data for LENS baseline.

        LENS expects image folders plus CSV manifests. Frames are now streamed
        directly from the EventCV recording (fixed dt_ms count frames, hot-pixel
        filtered) and thinned to distinct places, materializing only the PNGs the
        external LENS dataloader needs.
        """
        self.config = config
        # Get experimental details
        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()
        ref_seq = ref_info['sequence_name']
        query_seq = query_info['sequence_name']
        # Resolve the timewindow-tagged sequence names LENS uses for its temp
        # folders, CSV manifests, model filenames and output paths.
        self.ref_key = [d for d in ref_info['file_path'] if ref_seq in d and str(timewindow) in d]
        self.query_key = [d for d in query_info['file_path'] if query_seq in d and str(timewindow) in d]
        self.ref_name = self.ref_key[0]
        self.query_name = self.query_key[0]

        # Offsets (consistent with the other EventCV baselines).
        if "other" in dataset_config and "offset" in dataset_config["other"]:
            ref_offset, qry_offset = convert_offset(
                dataset_config['other']['offset'][ref_seq],
                dataset_config['other']['offset'][query_seq],
                dataset_config['other']['offset_time_scale'])
        else:
            ref_offset, qry_offset = None, None

        min_gap_sec = float(config.get("filter_places_sec", 60))

        # Open EventCV streams directly instead of loading frames from disk.
        ref_reader = self._open_reader(ref_info['hdf5_path'], timewindow, ref_offset)
        query_reader = self._open_reader(query_info['hdf5_path'], timewindow, qry_offset)

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="lens_data_")
        self.ref_dir = os.path.join(self.temp_dir, self.ref_name)
        self.query_dir = os.path.join(self.temp_dir, self.query_name)
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.query_dir, exist_ok=True)

        ref_csv_path = os.path.join(self.temp_dir, f"{self.ref_name}.csv")
        query_csv_path = os.path.join(self.temp_dir, f"{self.query_name}.csv")

        self.reference_places = self._write_lens_sequence(
            ref_reader,
            timewindow,
            min_gap_sec,
            self.ref_dir,
            ref_csv_path,
            "Formatting reference data to LENS requirements",
        )
        self.query_places = self._write_lens_sequence(
            query_reader,
            timewindow,
            min_gap_sec,
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
        logger.info(f"Running LENS baseline with command: {' '.join(self.train_cmd)}")

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

            logger.info(f"Running LENS baseline with command: {' '.join(self.eval_cmd)}")
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
            logger.warning(f"No .npy result files found in {self.output_dir}")
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
