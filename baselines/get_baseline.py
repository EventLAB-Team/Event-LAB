# Recall and precision/recall live in utils.metrics. Event-LAB used to clone
# stschubert/VPR_Tutorial here and call into it, but that project is GPL-3.0
# while Event-LAB is MIT, and its recallAtK resolved ties through an unstable
# sort. utils.metrics reimplements both metrics with an explicit tie policy.
from baselines.lens import LENS_baseline
from baselines.sparse_event import sparse_event_baseline
from baselines.ensemble import ensemble_baseline
from baselines.eventvlad import eventvlad_baseline
from baselines.vprmethods import vprmethods_baseline
from baselines.spikevpr import spikevpr_baseline
from baselines.megaevent import megaevent_baseline
from baselines.eventgem import eventgem_baseline

def get_baseline_switcher(config, dataset_config, reference, query):
    return {
        "lens": lambda: LENS_baseline(),
        "sparse_event": lambda: sparse_event_baseline(),
        "ensemble": lambda: ensemble_baseline(),
        "eventvlad": lambda: eventvlad_baseline(config, dataset_config, reference, query),
        "vprmethods": lambda: vprmethods_baseline(),
        "spikevpr": lambda: spikevpr_baseline(),
        "megaevent": lambda: megaevent_baseline(),
        "eventgem": lambda: eventgem_baseline(),
    }

def get_baseline(baseline_name, config, dataset_config, reference, query):
    baseline_name = baseline_name.lower()
    switcher = get_baseline_switcher(config, dataset_config, reference, query)
    return switcher.get(baseline_name, lambda: "Invalid")()