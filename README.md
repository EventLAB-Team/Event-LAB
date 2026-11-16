# Event-LAB: Towards Standardized Evaluation of Neuromorphic Localization Methods

<p align="center">
  <img src="./assets/logo.gif" alt="Event-LAB logo" width="220"/>
</p>

![Read the Docs](https://img.shields.io/badge/Read%20the%20Docs-%23000000?style=for-the-badge&logo=readthedocs&logoColor=white)
[![Documentation Status](https://readthedocs.org/projects/event-lab/badge/?version=latest&style=flat)](https://event-lab.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)
[![stars](https://img.shields.io/github/stars/EventLAB-Team/Event-LAB.svg?style=flat-square)](https://github.com/EventLAB-Team/Event-LAB/stargazers)
[![GitHub repo size](https://img.shields.io/github/repo-size/EventLAB-Team/Event-LAB.svg?style=flat-square)](./README.md)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/EventLAB/EventLAB)

**Event-LAB** is a framework for easy, reliable evaluation of event-based localization methods across standardized datasets and pipelines.
Using **single command-line invocation**, multiple different event-based methods can be implemented. See the [Event-LAB documentation](https://event-lab.readthedocs.io/en/latest/) for further details.

If you use this code in your work, please **cite our paper** (see [License and Citation](#license-and-citation)) and consider giving the repo a star! ⭐

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

### Terminal workbench

Run `pixi run tui` for an EventCV-inspired terminal interface. Edit runtime and
baseline configuration, select baselines/datasets with checkboxes, choose reference
and query sequences, and save or run experiment batches. The Downloads screen
prepares code, model weights, recordings, and ground truth without running inference.

Start from Home: choose Run an experiment, Manage downloads, Edit configuration,
View activity, or View results with arrows and Enter (or 1–5). Escape returns Home. Left/Right
cycles between work panels; Tab reaches buttons. Space toggles selections,
and A or the panel's Select all / Unselect all button toggles the whole list.
For trajectories, Select all keeps one reference and makes the rest queries.
Ctrl+S saves settings, and F5 reviews experiments. On the review dialog, Enter
starts the queue; Escape or Tab → Enter goes back. The queue continues after failures and provides live logs
and cancellation. Saved selections use current settings when the queue starts.

Downloads groups recordings by dataset. Space or a click selects a whole dataset;
Right opens its recordings for individual selection, and Left returns to the
dataset list. All downloads start unchecked. Counts show available/total recordings. Green means complete, orange means partial,
and red means missing. Availability follows saved `config.yaml`
changes automatically; Check files refreshes it without selecting missing files.
Before running or downloading a baseline, its external Git checkout is checked
for upstream changes and updated by fast-forward. Required submodules follow the
baseline's pinned revisions. Local changes and pinned checkouts are preserved;
offline update checks use the existing checkout and report the failure in the log.

View results browses saved matrices, precision–recall curves and Recall@K without
rerunning evaluation. Choose an experiment, then select a view; Enter focuses its
preview for scrolling. Open plot opens a full-size figure, and Refresh discovers
new outputs. New evaluations save `results.json` beside their matrices. Older
workbook metrics are shown only when they match one output unambiguously.

You can also edit runtime settings directly, using their exact `config.yaml` names:

```console
pixi run config --timewindows 100 200 300
pixi run config --request_input false
pixi run config --help
```

Running `pixi run config` displays the configuration. See
[configuration documentation](docs/config.rst) for batch and download details.

### Command-line invocation
Below are some examples of experiments that we can run:
```console
# Run Ensemble-Event-VPR on the Brisbane-Event-VPR with Sunset2 as the reference and Sunrise as the query
pixi run eventlab sparse_event brisbane_event sunset2 sunrise

# Run EventVLAD on the NSAVP with R0_FS0 as the reference and F0_FA0 as the query
pixi run eventlab ensemble nsavp R0_FS0 F0_FA0 
```

Event-LAB downloads recordings, streams the selected representation through EventCV, and writes results to an `.xlsx` spreadsheet for further analysis.

### Modify the configuration
Parameters for generating event frames in Event-LAB are controlled using the `config.yaml` file in the main project directory:
```yaml
# EventCV streaming parameters
timewindows: [33, 66, 99, 120] # Milliseconds
representation: polarity
representation_options: {} # EventCV reader.with_repr() parameters; {} uses defaults
hot_pixel_filter: true
gps_tolerance: 25.0 # Metres along the reference route
```
Every baseline uses the chosen EventCV representation. Recordings are streamed;
image-only methods use temporary images that are deleted after the run. Model
adapters average or repeat channels to fit pretrained inputs, so changing the
representation can change accuracy. See the [EventCV representation parameters](https://docs.eventcv.net/en/latest/api.html#eventcv.EventReader.with_repr)
for `representation_options`; these configure `reader.with_repr()`.
Reconstruction is disabled until named methods are available in EventCV.

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
      representation: polarity
      timewindows: [33, 66, 120, 250]
```
The above will run 3 baseline methods across 4 different time windows. Then to run the evaluation we simply run the following in the command terminal:
```console
pixi run batch
```
This runs the batch sequentially, continues after failed experiments, and saves logs without rewriting your configuration files.

### Implemented baseline methods and datasets
For the demonstration version of the repository, we have implemented two baseline methods and datasets. The full and final version of the code will be released upon acceptance which includes the other methods. Below is the list of implemented methods and their invocation name:

| Baseline | Source | Invocation |
| --- | --- | --- |
| EventVLAD | [alexjunholee/EventVLAD](https://github.com/alexjunholee/EventVLAD) | `eventvlad` |
| Ensemble-Event-VPR | [Tobias-Fischer/ensemble-event-vpr](https://github.com/Tobias-Fischer/ensemble-event-vpr) | `ensemble` |
| LENS | [AdamDHines/LENS](https://github.com/AdamDHines/LENS) | `lens` |
| Sparse-Event-VPR | [Tobias-Fischer/sparse-event-vpr](https://github.com/Tobias-Fischer/sparse-event-vpr) | `sparse_event` |
| VPR-Methods | [gmberton/VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) | `vprmethods` |
| SpikeVPR | [GeoffroyK/SpikeVPR](https://github.com/GeoffroyK/SpikeVPR) | `spikevpr` |
| MegaEvent | [AdamDHines/megaevent](https://github.com/AdamDHines/megaevent) | `megaevent` |
| Event-GeM | [AdamDHines/Event-GeM](https://github.com/AdamDHines/Event-GeM) | `eventgem` |

| Dataset | Source | Invocation |
| --- | --- | --- |
| Brisbane-Event-VPR | [TobiasRobotics/brisbane-event-vpr](https://huggingface.co/datasets/TobiasRobotics/brisbane-event-vpr) | `brisbane_event` |
| NSAVP | [umautobots.github.io/nsavp](https://umautobots.github.io/nsavp) | `nsavp` |
| Fast-and-Slow | [QVPR/QCR-Fast-Slow-Event-Dataset-Raw-Parquets](https://huggingface.co/datasets/QVPR/QCR-Fast-Slow-Event-Dataset-Raw-Parquets) | `fast_slow` |
| QCR-Event-VPR | [Zenodo record 10494919](https://zenodo.org/records/10494919) | `qcr_event` |

Any combination of implemented baseline methods, datasets and their traverses can be set-up for a reference/query pair to evaluate performance.

## License and Citation
This repository is licensed under the permissive [MIT License](./LICENSE). If you use our code, please cite our [paper](https://arxiv.org/abs/2509.14516):

```
@inproceedings{HinesEventLAB2026,
      title={Event-LAB: Towards Standardized Evaluation of Neuromorphic Localization Methods}, 
      author={Adam D. Hines and Alejandro Fontan and Michael Milford and Tobias Fischer},
      year={2026},
      booktitle={IEEE International Conference on Robotics and Automation},
      pages={16789-16796},
      doi={10.1109/ICRA57385.2026.11697007}     
}
```

Where using baselines or datasets from other authors in your evaluation as implemented in our code, please ensure you additionally cite the correct source material. 

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/EventLAB-Team/Event-LAB/issues).

### Gardens Point Event

`gardens_point_event` provides the public `left` and `right` walking traverses:

```console
pixi run prepare dataset gardens_point_event left right
pixi run eventlab eventvlad gardens_point_event left right
```

Recordings, GPS CSVs, and synchronization metadata come from
[Gardens-Point-Event](https://huggingface.co/datasets/AdamHines/Gardens-Point-Event).
GPS times are mapped onto the camera clock before ground-truth generation.
The existing route-completion matching and tolerance rules apply. Incomplete GPS
coverage produces a warning and retains the existing endpoint-clamping behavior.
The local-only `qut_event_walking` name remains available to legacy scripts but is
hidden from the TUI catalog. Cite the dataset's reference paper, *EventGeM:
Global-to-Local Feature Matching for Event-Based Visual Place Recognition*,
Hines et al., 2026 ([arXiv:2603.05807](https://arxiv.org/abs/2603.05807)).
