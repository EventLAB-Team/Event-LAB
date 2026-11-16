import os, sys, yaml
import argparse

from utils import utils
from loguru import logger
from datasets.get_data import get_dataset
from baselines.get_baseline import get_baseline 
from datasets.groundtruths import generate_ground_truth
from utils.configuration import validate_runtime

def main():
    parser = argparse.ArgumentParser(description='Run an Event-LAB experiment')
    for name in ('baseline', 'dataset', 'reference', 'query'):
        parser.add_argument(name)
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    # Load the main configuration file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    validate_runtime(config)
    os.environ['EVENTLAB_RUNTIME_CONFIG'] = os.path.abspath(args.config)
    # Standalone adapters run from both the project and upstream directories.
    root = os.path.dirname(os.path.abspath(__file__))
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, (root, os.environ.get('PYTHONPATH'))))

    # Get the experimental details
    baseline_name = args.baseline
    dataset_name = args.dataset

    # Start the Event-LAB banner
    utils.start(baseline_name, dataset_name)

    # Ensure dataset contains .py and .yaml configuration files
    yaml_path = os.path.join("datasets", f"{dataset_name}.yaml")
    with open(yaml_path, 'r') as file:
        dataset_config = yaml.safe_load(file)

    # Check the baseline argument if it is VPR or SLAM
    if baseline_name in config["VPR-Baselines"]:
        try:
            reference_name = args.reference
            query_name = args.query

            # Print experimental details
            logger.info(f"Running VPR baseline: {baseline_name}")
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"Reference sequence: {reference_name}")
            logger.info(f"Query sequence: {query_name}")
            logger.info('')

            # Load general config and set the dataset (downloads recordings if not present)
            reference = get_dataset(config, dataset_name, reference_name)
            query = get_dataset(config, dataset_name, query_name)

        except IndexError:
            raise ValueError("Not enough arguments provided. Expected: <baseline>, <dataset>, <reference>, <query>.")

    else:
        raise ValueError(f"Baseline {baseline_name} is not recognized. Please check the configuration.")

    # Check that experimental details are valid
    baseline_path = os.path.join("baselines", f"{baseline_name}.py")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline file for {baseline_name} is missing.")
    
    # Get the baseline instance
    baseline = get_baseline(baseline_name, config, dataset_config, reference, query)

    for _, timewindow in enumerate(reference.timewindow_list):

        # Regenerate for the current window/tolerance instead of reusing stale GT.
        GT_file = generate_ground_truth(
            config, dataset_config, dataset_name, reference_name, query_name,
            reference, query, timewindow=timewindow,
            gps_available=dataset_config['sequences'][reference_name]['ground_truth']['available'])
        try:
            baseline.format_data(config, dataset_config, reference, query, timewindow)
            baseline.build_execute(config, dataset_config, GT_file)
            baseline.run()
            baseline.parse_results(GT_file)
        finally:
            baseline.cleanup()

        if baseline_name == "ensemble":
            # Do not run any more iterations, ensemble handles all timewindows at once
            break

if __name__ == "__main__":
    main()