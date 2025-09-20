import os
from baselines.download_baseline import clone_repo

# Ensure the standardized metrics repo (VPR-Tutorial) is cloned
if not os.path.exists("./baselines/VPR_Tutorial"):
    clone_repo("https://github.com/stschubert/VPR_Tutorial.git", destination="./baselines/VPR_Tutorial")

from baselines.ensemble import ensemble_baseline
from baselines.eventvlad import eventvlad_baseline

def get_baseline_switcher():
    return {
        "ensemble": lambda: ensemble_baseline(),
        "eventvlad": lambda: eventvlad_baseline(),
    }

def get_baseline(baseline_name):
    baseline_name = baseline_name.lower()
    switcher = get_baseline_switcher()
    return switcher.get(baseline_name, lambda: "Invalid")()

def list_available_baselines():
    return list(get_baseline_switcher().keys())