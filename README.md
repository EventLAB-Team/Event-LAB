# Event-LAB: Towards Standardized Evaluation of Neuromorphic Localization Methods
This repository contains a _demonstration version_ of the code for Event-LAB, a framework for performing easy and reliable localization methods using event-based localization methods and datasets. As this work is currently under review, the full and final code will be released upon acceptance.

In the demonstration version, we provide two localization methods and datasets. Details of which are below.

## Quick start :dizzy:
Event-LAB uses [Pixi](https://prefix.dev/docs/pixi/overview) by prefix.dev to manage packages and dependencies to achieve bit-for-bit reproducibility. Follow the instructions below to get started with Event-LAB.

### Installation :package:
If not already installed, [install Pixi](https://prefix.dev/docs/pixi/overview#installation) for your operating system by running the following command in your terminal:

#### Linux/MacOS
```console
curl -fsSL https://pixi.sh/install.sh | bash
```
#### Windows (Powershell)
```console
iwr -useb https://pixi.sh/install.ps1 | iex
```

### Get the repository :inbox_tray:
Download this reposistory and navigate to the project directory by running the following command in your terminal:
```console
git clone git@github.com:EventLAB-Team/Event-LAB.git && cd Event-LAB
```

### Run the demonstration :rocket:
That's it! You're ready to run Event-LAB, try it now with the demonstration by running the following command in your terminal:
```console
pixi run demo
```

## Using Event-LAB
Using simple command-line invocation, Event-LAB is designed to easily mix and match localization methods and datasets. To use Event-LAB, we simply run `pixi run eventlab <baseline_method> <dataset> <reference> <query>` in the command terminal. This triggers a series of processes starting from downloading the dataset and baseline, generating event frames, and then running the method to return Recall@1 and Precision-Recall metrics.

### Command-line invocation
Below are some examples of experiments that we can run:
```console
# Run Ensemble-Event-VPR on the Brisbane-Event-VPR with Sunset2 as the reference and Sunrise as the query
pixi run eventlab sparse_event brisbane_event sunset2 sunrise

# Run EventVLAD on the NSAVP with R0_FS0 as the reference and F0_FA0 as the query
pixi run eventlab ensemble nsavp R0_FS0 F0_FA0 
```

Event-LAB will handle the pre-processing of all individual datasets, format as required for the baseline, and then return results into an `.xslx` spreadsheet for further analysis.

### Modify the configuration
Parameters for generating event frames in Event-LAB are controlled using the `config.yaml` file in the main project directory:
```yaml
# Frame reconstruction parameters
timewindows: [33, 66, 99, 120] # The time window to collect events over
num_events: [25000, 50000, 75000, 100000] # The maximum number of events per frame, only used if frame_generator is "eventcount"
frame_generator: "reconstruction" # Options: "frames", "eventcount", "reconstruction"
frame_accumulator: "eventcount"    # Options: "count", "polarity" (default), "timestamp"
reconstruction_model: "e2vid"  # Options: "firenet", "e2vid (default)"
```
Modify the parameters to suit your experimental needs. Time windows and maximum number of events can be parsed as a list to process and analyze several conditions in a single experiment.

### Batch run experiments
It is often required and useful to run a variety of baseline methods and datasets across a standardized set of parameters. Event-LAB allows users to easily set-up and batch numerous experiments to run in series. The `batch_config.yaml` file in the main project directory can be set up to customize implementations:
```yaml
# Batch experiment parameters
batch_experiments:
  - dataset: brisbane_event
    reference: sunset2
    queries: [sunrise]
    baselines: [eventvlad, ensemble]
    config:
      frame_generator: frames
      frame_accumulator: polarity
      timewindows: [33, 66, 120, 250]
```
The above will run 3 baseline methods across 4 different time windows. Then to run the evaluation we simply run the following in the command terminal:
```console
pixi run bash
```
This will generate a `run_batch.sh` file and execute it.

### Implemented baseline methods and datasets
For the demonstration version of the repository, we have implemented two baseline methods and datasets. The full and final version of the code will be released upon acceptance which includes the other methods. Below is the list of implemented methods and their invocation name:

| Baseline Method | Link | Invocation | 
|:----------------|:------|:------------:|
| EventVLAD |  https://github.com/alexjunholee/EventVLAD | eventvlad|
|Ensemble-Event-VPR | https://github.com/Tobias-Fischer/ensemble-event-vpr | ensemble |

| Datasets | Link | Invocation | Traverses | 
|:----------------|:------|:------------|:-------:|
| Brisbane-Event-VPR |  https://huggingface.co/datasets/TobiasRobotics/brisbane-event-vpr | brisbane_event | sunset1, sunset2, sunrise, daytime, morning, night
|NSAVP | https://umautobots.github.io/nsavp | nsavp | R0_RN0, R0_RA0, R0_FS0, F0_FN0, RO_RS0, R1_DA0, R1_FA0, R1_RA0, R0_FA0 |

Any combination of implemented baseline methods, datasets and their traverses can be set-up for a reference/query pair to evaluate performance.

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/EventLAB-Team/Event-LAB/issues).