import os, tempfile, subprocess, yaml, platform
import numpy as np
from pathlib import Path
from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from datetime import datetime, timezone
import re
import utils.functional as FUNC

class ensemble_baseline(EventBaseline):
    def __init__(self):
        super().__init__()

        self.name = "ensemble"
        # Check if the baseline repository is already cloned
        self.repo_path = "./baselines/ensemble"
        # Baseline URL
        self.url = "https://github.com/Tobias-Fischer/ensemble-event-vpr.git"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)
        self.baseline_config_path = './baselines/ensemble.yaml'
        # Load the baseline configuration
        with open(self.baseline_config_path, 'r') as file:
            self.baseline_config = yaml.safe_load(file)
        # Check if the pytorch-NetVlad path exists
        if not os.path.exists(self.baseline_config['netvlad_path']):
            netvlad_url = "https://github.com/Nanne/pytorch-NetVlad.git"
            clone_repo(netvlad_url, destination=self.baseline_config['netvlad_path'])
        # Create the data output folder
        self.outdir = './output/ensemble'
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
        self.ref_name = ref_info['sequence_name']
        self.query_name = query_info['sequence_name']
        self.dataset = ref_info['dataset_name']

        # Obtain the reconstruction method to determine directory structures
        frame_generator = config['frame_generator']
        frame_accumulator = config['frame_accumulator']
        if frame_generator == 'reconstruction':
            self.image_subfolder = 'reconstruction'
            self.data_format = '.png'
        else:
            self.image_subfolder = ''
            self.data_format = '.npy'

        # Determine the timewindows to use based on the frame generator
        if frame_accumulator == 'eventcount':
            self.windows = config['num_events']
        else:
            self.windows = config['timewindows']
            
        # Construct a list of all reference and query directories for each timewindow and num_events from a loop
        self.ref_directories = []
        self.query_directories = []
        for window in self.windows:
            self.ref_directories.append(os.path.join(
                config['data_path'],
                ref_info['dataset_name'], 
                self.ref_name, 
                f'{self.ref_name}-{frame_generator}-{window}',
                f'{self.image_subfolder}'))
            
            self.query_directories.append(os.path.join(
                config['data_path'],
                query_info['dataset_name'], 
                self.query_name, 
                f'{self.query_name}-{frame_generator}-{window}',
                f'{self.image_subfolder}'))
        if self.num_events:
            for num in self.num_events:
                self.ref_directories.append(os.path.join(
                    config['data_path'],
                    ref_info['dataset_name'], 
                    self.ref_name, 
                    f'{self.ref_name}-{frame_generator}-{num}',
                    f'{self.image_subfolder}'))

                self.query_directories.append(os.path.join(
                    config['data_path'],
                    query_info['dataset_name'], 
                    self.query_name, 
                    f'{self.query_name}-{frame_generator}-{num}',
                    f'{self.image_subfolder}'))
                
        if self.data_format == '.png':
            _RX_FRAME = re.compile(r"^frame_(\d+)\.png$")
        else:
            _RX_FRAME = re.compile(r"^frame_(\d+)\.npy$")

        def list_frame_files(dirpath: str):
            paths = []
            for p in Path(dirpath).iterdir():
                m = _RX_FRAME.fullmatch(p.name)
                if m:
                    paths.append((int(m.group(1)), p))
            paths.sort(key=lambda t: t[0])  # numeric sort
            return [p for _, p in paths]

        def build_files_only_list(dirs):
            """Returns List[List[Path]], aligned with dirs."""
            return [list_frame_files(d) for d in dirs]

        # Build aligned lists:
        self.ref_files_list   = build_files_only_list(self.ref_directories)
        self.query_files_list = build_files_only_list(self.query_directories)

        # after you have ref_files, query_files and min_gap_sec
        min_gap_sec = float(config.get("filter_places_sec", 60))

        # Create data to store temporary data
        self.temp_dir = tempfile.mkdtemp(prefix="baseline_data_")
        
        for idx, timewindow in enumerate(self.windows):
            ref_res   = FUNC._apply_time_filter_to_files(self.ref_files_list[idx],   
                                                         self.ref_directories[idx],  
                                                         min_gap_sec, 
                                                         debug=False)
            
            query_res = FUNC._apply_time_filter_to_files(self.query_files_list[idx], 
                                                         self.query_directories[idx], 
                                                         min_gap_sec, 
                                                         debug=False)

            # Replace file lists with filtered ones
            ref_files   = ref_res['files']
            query_files = query_res['files']

            # OPTIONAL: Create temporary directory to store converted data, if not using numpy arrays
            self.ref_dir = os.path.join(self.temp_dir, 
                                        self.dataset,
                                        self.ref_name,
                                        f'{self.ref_name}-{frame_generator}-{timewindow}',
                                        self.image_subfolder)
            
            self.query_dir = os.path.join(self.temp_dir,
                                          self.dataset, 
                                          self.query_name,
                                          f'{self.query_name}-{frame_generator}-{timewindow}',
                                          self.image_subfolder)
            os.makedirs(self.ref_dir, exist_ok=True)
            os.makedirs(self.query_dir, exist_ok=True)

            # Copy files to temporary directory
            import shutil
            # Copy files to temporary directory using shutil
            for i, ref_file in enumerate(ref_files):
                shutil.copy(ref_file, os.path.join(self.ref_dir, f"frame_{i:06d}.png"))

            for i, query_file in enumerate(query_files):
                shutil.copy(query_file, os.path.join(self.query_dir, f"frame_{i:06d}.png"))

            # Additionally copy over `metadata.json` if it exists from the directory parent dir
            shutil.copy(os.path.join(os.path.dirname(self.ref_directories[idx]), 'metadata.json'), 
                            os.path.join(os.path.dirname(self.ref_dir), 'metadata.json'))
            shutil.copy(os.path.join(os.path.dirname(self.query_directories[idx]), 'metadata.json'), 
                            os.path.join(os.path.dirname(self.query_dir), 'metadata.json'))

            # write filtered timestamps to the temporary directory
            ref_timestamps = ref_res['timestamps_filtered_text']
            query_timestamps = query_res['timestamps_filtered_text']
            with open(os.path.join(self.ref_dir, 'timestamps.txt'), 'w') as f:
                f.writelines(ref_timestamps)
            with open(os.path.join(self.query_dir, 'timestamps.txt'), 'w') as f:
                f.writelines(query_timestamps)

        self.output_dir = os.path.join(self.outdir, f"{ref_info['dataset_name']}", f"{ref_info['sequence_name']}_{query_info['sequence_name']}",
                                       f"{config['frame_generator']}_{timewindow}")
        os.makedirs(self.output_dir, exist_ok=True)

    def build_execute(self, config, data_config, ground_truth):
        """
        Build a commandline execute for the baseline with the provided reference, query, and ground truth data.
        """

        # Build the command as a single string
        eval_cmd_str = (
                # Include command line arugments specific to the baseline
                f'python ./baselines/ensemble/run_eevpr.py '
                f'-ds "{self.dataset}" '
                f'-r {self.ref_name} '
                f'-q {self.query_name} '
                f'-w {self.windows} '
                f'-n {self.num_events} '
                f'-d "{self.temp_dir}" '
                f'-nv {self.baseline_config["netvlad_path"]} '
                f'-o {self.output_dir} '
            )
        
        # Wrap it with pixi run
        if platform.system() in ("Windows", "Darwin"):
            self.eval_cmd = ["pixi", "run", "bash", "-c", eval_cmd_str]
        else:
            self.eval_cmd = ["pixi", "run", "bash", "-c", eval_cmd_str]

    def run(self, timeout=None):
        """Start baseline, capture output, keep handle in self.proc."""
        print(f"Running baseline with command: {' '.join(self.eval_cmd)}")
        kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["preexec_fn"] = os.setsid
        self.proc = subprocess.Popen(self.eval_cmd, **kwargs)
        try:
            out, err = self.proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # timed out: attempt cleanup and return whatever we can
            self.cleanup()
            out, err = "", ""
        self.stdout = out
        self.stderr = err
        print("STDOUT:", out)
        if err:
            print("STDERR:", err)
        if getattr(self, "proc", None) and self.proc.returncode != 0:
            raise RuntimeError(f"Baseline evaluation failed with return code {self.proc.returncode}")

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
                tolerance=self.config.get('ground_truth_tolerance', 5)
        )

        # Save results to excel spreadsheet
        self.save_results(rows, pr_curves, self.name, f'{self.ref_name}_{self.query_name}')

    def cleanup(self):
        """Terminate only the process (and its children) started by run()."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)