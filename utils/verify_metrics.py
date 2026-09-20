"""
Verify utils.metrics against the implementation it replaces.

Run:  pixi run python utils/verify_metrics.py

Checks, in order of what they would catch:

1. CHUNK INVARIANCE -- recall must be bit-identical across chunk widths and
   worker counts. The formulation guarantees this, so any difference is a bug.
2. EXACTNESS WITHOUT TIES -- where every column has distinct values, the two tie
   policies and VPR_Tutorial's recallAtK must agree exactly. This is the check
   that catches a genuine regression.
3. TIE BRACKET -- where ties exist, exact agreement is impossible (the old code
   resolved ties with an unstable sort). The old value must instead fall inside
   [pessimistic, optimistic], proving the new number is a valid resolution of
   the same ambiguity.
4. PR / AUPR PARITY -- against the old createPR, to floating-point tolerance.

The VPR_Tutorial comparison is skipped automatically once that clone is gone.
"""
import glob
import os
import sys
import time

import numpy as np

# Run as a script, sys.path[0] is this file's own directory, where utils.py would
# shadow the `utils` namespace package. Drop it and use the repo root instead.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

from utils.metrics import (aupr, conform_ground_truth, precision_recall_curve,
                           recall_at_k, resolve_chunk_cols, resolve_workers)

K_LIST = [1, 5, 10, 15, 20, 25]
GT_DIR = "datasets/brisbane_event/ground_truth"

# Whether each baseline's saved matrix is a distance (lower = better) and so must
# be inverted before scoring. Mirrors each wrapper's self.matrix_type.
SIMILARITY_BASELINES = {"lens", "spikevpr", "megaevent"}

try:
    from baselines.VPR_Tutorial.evaluation.metrics import createPR, recallAtK
    HAVE_REFERENCE = True
except Exception:
    HAVE_REFERENCE = False


def fail(msg):
    print(f"    FAIL  {msg}")
    return 1


def check_matrix(name, S, GT):
    """Run every check against one (similarity, ground-truth) pair."""
    problems = 0
    n_rows, n_cols = S.shape
    distinct = np.median([np.unique(S[:, c]).size
                          for c in np.linspace(0, n_cols - 1, min(64, n_cols)).astype(int)])
    tie_free = distinct >= n_rows
    print(f"\n{name}  S={S.shape} {S.dtype}  median distinct/col={distinct:.0f}/{n_rows}"
          f"  {'(tie-free)' if tie_free else '(has ties)'}")

    # --- 1. chunk invariance -------------------------------------------------
    t = time.perf_counter()
    baseline = recall_at_k(S, GT, K_LIST, chunk_cols=n_cols, workers=1)
    t_single = time.perf_counter() - t

    for cols, workers in ((1, 1), (7, 1), (128, 1), (None, None), (13, 4), (n_cols, 8)):
        got = recall_at_k(S, GT, K_LIST, chunk_cols=cols, workers=workers)
        if got != baseline:
            problems += fail(f"chunk_cols={cols} workers={workers} differs: {got} != {baseline}")
    if not problems:
        auto_w = resolve_workers()
        t = time.perf_counter()
        recall_at_k(S, GT, K_LIST)
        print(f"    ok    chunk-invariant over 6 configurations "
              f"(1 thread {t_single:.2f}s, auto {resolve_workers()}w {time.perf_counter()-t:.2f}s)")

    opt = baseline
    pes = recall_at_k(S, GT, K_LIST, tie_policy="pessimistic", workers=1)

    # --- 2 / 3. against the implementation being replaced --------------------
    if HAVE_REFERENCE:
        GTi = GT.astype(int)
        old = {k: float(recallAtK(S, GTi, K=k)) for k in K_LIST}
        if tie_free:
            bad = [k for k in K_LIST if not (abs(old[k] - opt[k]) < 1e-12
                                             and abs(old[k] - pes[k]) < 1e-12)]
            if bad:
                problems += fail(f"tie-free matrix must match exactly; differs at K={bad}\n"
                                 f"          old={old}\n          opt={opt}\n          pes={pes}")
            else:
                print(f"    ok    tie-free: optimistic == pessimistic == recallAtK exactly")
        else:
            bad = [k for k in K_LIST if not (pes[k] - 1e-12 <= old[k] <= opt[k] + 1e-12)]
            if bad:
                problems += fail(f"recallAtK outside [pessimistic, optimistic] at K={bad}\n"
                                 f"          old={old}\n          opt={opt}\n          pes={pes}")
            else:
                print(f"    ok    recallAtK inside the tie bracket at every K")
                print(f"          R@1  pessimistic {pes[1]:.4f} <= old {old[1]:.4f} <= optimistic {opt[1]:.4f}")

        # --- 4. PR / AUPR parity --------------------------------------------
        P_old, R_old = createPR(S, GTi, matching="single", n_thresh=100)
        P_new, R_new = precision_recall_curve(S, GT, n_thresh=100)
        if len(P_old) != len(P_new):
            problems += fail(f"PR length {len(P_new)} != {len(P_old)}")
        else:
            dp = np.max(np.abs(np.asarray(P_old) - np.asarray(P_new)))
            dr = np.max(np.abs(np.asarray(R_old) - np.asarray(R_new)))
            da = abs(float(np.trapz(np.asarray(P_old)[np.argsort(R_old)], np.sort(R_old)))
                     - aupr(P_new, R_new))
            if max(dp, dr, da) > 1e-9:
                problems += fail(f"PR parity: dP={dp:.3e} dR={dr:.3e} dAUPR={da:.3e}")
            else:
                print(f"    ok    PR/AUPR parity (dP={dp:.1e} dR={dr:.1e} dAUPR={da:.1e})")
    else:
        print("    --    VPR_Tutorial not present; comparison checks skipped")
    return problems


def main():
    print(f"workers={resolve_workers()}  "
          f"chunk_cols(14473 rows, f32, 12820 cols)="
          f"{resolve_chunk_cols(14473, 4, 12820, resolve_workers())}")
    problems = 0

    # Synthetic: guaranteed tie-free, and a guaranteed all-ties degenerate case.
    rng = np.random.default_rng(0)
    S = rng.random((300, 200), dtype=np.float32)
    GT = np.zeros((300, 200), bool)
    GT[rng.integers(0, 300, 200), np.arange(200)] = True
    problems += check_matrix("synthetic/tie-free", S, GT)

    Sc = np.ones((50, 40), dtype=np.float32)          # every value identical
    GTc = np.zeros((50, 40), bool); GTc[0, :] = True
    opt = recall_at_k(Sc, GTc, [1], workers=1)[1]
    pes = recall_at_k(Sc, GTc, [1], tie_policy="pessimistic", workers=1)[1]
    print(f"\nsynthetic/all-tied  optimistic R@1={opt:.1f} (expect 1.0), "
          f"pessimistic R@1={pes:.1f} (expect 0.0)")
    if not (opt == 1.0 and pes == 0.0):
        problems += fail("degenerate all-ties case wrong")
    else:
        print("    ok    tie policies behave as documented at the extremes")

    # Real matrices produced by the baselines.
    paths = sorted(glob.glob("output/*/brisbane_event/*/frames_*/*.npy"))
    paths = [p for p in paths if "_matches" not in p]
    for path in paths:
        arr = np.load(path)
        if arr.ndim != 2:
            continue
        parts = path.split(os.sep)
        baseline, ref_query = parts[1], parts[3]
        # Each output directory is named <reference>_<query>; use ITS ground
        # truth. Scoring everything against one pair's GT silently compares
        # against a transposed matrix and makes the reported numbers junk.
        gt_path = os.path.join(GT_DIR, f"{ref_query}_GT.npy")
        if not os.path.exists(gt_path):
            print(f"\n{path}\n    --    no ground truth at {gt_path}; skipped")
            continue
        # Score the matrix the same way its wrapper does, or the bracket is
        # computed on an inverted matrix and means nothing.
        S = arr if baseline in SIMILARITY_BASELINES else arr.max() - arr
        GT = conform_ground_truth(np.load(gt_path), S.shape)
        problems += check_matrix(f"{path}  [GT {ref_query}]", S, GT)

    print("\n" + ("ALL CHECKS PASSED" if problems == 0 else f"{problems} CHECK(S) FAILED"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
