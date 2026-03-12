import os, sys, yaml

from utils import utils
from datasets.get_data import get_dataset
from baselines.get_baseline import get_baseline 
from datasets.groundtruths import generate_ground_truth

def main():
    # Start the Event-LAB banner
    utils.start()
    # Load the main configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get the experimental details
    baseline_name = sys.argv[1]
    dataset_name = sys.argv[2]

    # Ensure dataset contains .py and .yaml configuration files
    yaml_path = os.path.join("datasets", f"{dataset_name}.yaml")
    with open(yaml_path, 'r') as file:
        dataset_config = yaml.safe_load(file)

    # Check the baseline argument if it is VPR or SLAM
    if baseline_name in config["VPR-Baselines"]:
        try:
            reference_name = sys.argv[3]
            query_name = sys.argv[4]

            # Print experimental details
            print(f"Running VPR baseline: {baseline_name}")
            print(f"Dataset: {dataset_name}")
            print(f"Reference sequence: {reference_name}")
            print(f"Query sequence: {query_name}")
            print('')

            # Load general config and set the dataset (will download, format, and construct data if not present)
            reference = get_dataset(config, dataset_name, reference_name)
            query = get_dataset(config, dataset_name, query_name)

            # Check for the existence of a ground truth for the reference and query datasets
            GT_file = (os.path.join(config['data_path'], dataset_name, 'ground_truth', f"{reference_name}_{query_name}_GT.npy"))
            
            # Generate ground truth
            if not os.path.exists(GT_file):
                generate_ground_truth(config, 
                                    dataset_config, 
                                    dataset_name, 
                                    reference_name, 
                                    query_name, 
                                    reference, 
                                    query,
                                    timewindow=reference.timewindow_list[0],
                                    gps_available=dataset_config['sequences'][f'{reference_name}']['ground_truth']['available'])

        except IndexError:
            raise ValueError("Not enough arguments provided. Expected: <baseline>, <dataset>, <reference>, <query>.")
        
    elif baseline_name in config["SLAM-Baselines"]: #TODO: Implement SLAM baselines
        pass

    else:
        raise ValueError(f"Baseline {baseline_name} is not recognized. Please check the configuration.")

    # Check that experimental details are valid
    baseline_path = os.path.join("baselines", f"{baseline_name}.py")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline file for {baseline_name} is missing.")
    
    # Get the baseline instance
    baseline = get_baseline(baseline_name, config, dataset_config, reference, query)

    for _, timewindow in enumerate(reference.timewindow_list):

        # Format the data for the baseline, if pre-generated
        if config['stream']:
            baseline.format_data(config, dataset_config, reference, query, timewindow)

        # Build the execution command for the baseline
        baseline.build_execute(config, dataset_config, GT_file)

        # Run the baseline
        baseline.run()

        # Run the evaluation
        baseline.parse_results(GT_file)

        # Clean up temporary files if necessary
        baseline.cleanup()

        if baseline_name == "ensemble":
            # Do not run any more iterations, ensemble handles all timewindows at once
            break

if __name__ == "__main__":
    main()