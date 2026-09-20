"""
Retrieval metrics for Event-LAB.

Replaces the recall/PR functions Event-LAB previously imported from the cloned
VPR_Tutorial repository, which is GPL-3.0 while Event-LAB is MIT.

The recall implementation here is deliberately sort-free. Asking "is a correct
match among the top K" does not require ordering the candidates: it is true
exactly when fewer than K references score strictly better than the best correct
one. That turns an O(N log N) sort per K into two O(N) reductions shared across
every K, and -- because each query column is evaluated independently with no
state crossing columns -- makes the result identical for any chunk width or
worker count.

Ties matter here. Event-based similarity matrices are often heavily quantised
(the sparse-event matrices hold ~1000 distinct values across 14k rows), so a
correct match is frequently tied with incorrect ones. VPR_Tutorial's
`recallAtK` resolved such ties through numpy's *unstable* sort, which makes the
reported number arbitrary among the tied candidates and not reproducible across
runs or numpy versions. `tie_policy` makes the choice explicit instead:

    "optimistic"  a match tied with non-matches counts as retrieved (default)
    "pessimistic" a match tied with non-matches counts as missed

The two bracket the value any tie-breaking rule could produce, so computing both
reports the ambiguity rather than hiding it.
"""
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger
from skimage.transform import resize
from tqdm import tqdm

TIE_POLICIES = ("optimistic", "pessimistic")

# Per-chunk working-set target. Chunk width is derived from this and the matrix
# shape rather than being a fixed column count, so it adapts to tall matrices.
_CHUNK_TARGET_BYTES = 64 * 1024 * 1024


def resolve_workers(requested=None):
    """
    Decide how many worker threads to use, without hardcoding a count.

    Checked in order, first hit wins:
      1. an explicit argument
      2. EVENTLAB_WORKERS
      3. SLURM_CPUS_PER_TASK -- a cluster job must not grab the whole node
      4. os.sched_getaffinity -- honours cgroup/taskset/container limits on Linux,
         which os.cpu_count() ignores
      5. macOS performance-core count -- scheduling onto the efficiency cores
         hurts throughput; torch reaches the same conclusion independently
      6. os.cpu_count()
    """
    if requested:
        return max(1, int(requested))

    for var in ("EVENTLAB_WORKERS", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(var)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                logger.warning(f"Ignoring non-integer {var}={value!r}")

    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass

    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return max(1, int(out.stdout.strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    return max(1, os.cpu_count() or 1)


def resolve_chunk_cols(n_rows, itemsize, n_cols, workers):
    """Column block width that keeps each worker's slice near the memory target."""
    per_col = max(1, int(n_rows) * int(itemsize))
    cols = max(1, _CHUNK_TARGET_BYTES // per_col)
    # Prefer enough chunks to keep every worker busy, without going tiny.
    if workers > 1:
        cols = min(cols, max(1, n_cols // workers))
    return int(min(max(1, cols), n_cols))


def conform_ground_truth(GThard, target_shape):
    """
    Return GT as a bool array matching `target_shape`.

    Mirrors the behaviour Event-LAB has always had: nearest-neighbour resize and
    re-binarise when the shapes disagree, otherwise take the array as supplied.
    """
    if tuple(GThard.shape) == tuple(target_shape):
        return GThard.astype(bool)
    resized = resize(GThard, target_shape, order=0,
                     preserve_range=True, anti_aliasing=False)
    return resized > 0.5


def _neg_fill(dtype):
    """Value that loses every comparison, in the matrix's own dtype."""
    if np.issubdtype(dtype, np.floating):
        return np.array(-np.inf, dtype=dtype)
    return np.array(np.iinfo(dtype).min, dtype=dtype)


def _chunk_counts(S, GT, fill):
    """
    Per-column rank statistics for one column block.

    Returns (keep, n_better, n_tied_nonmatch) for the columns that have at least
    one ground-truth match. `best` is the highest similarity among a column's GT
    rows, so nothing scoring strictly above it can be a GT row -- which is why
    `n_better` counts exactly the incorrect references that outrank the match.
    """
    keep = GT.any(axis=0)
    if not keep.any():
        empty = np.zeros(0, dtype=np.int64)
        return 0, empty, empty

    Sk = S[:, keep]
    GTk = GT[:, keep]
    best = np.where(GTk, Sk, fill).max(axis=0)
    n_better = (Sk > best).sum(axis=0)
    n_tied_nonmatch = ((Sk == best) & ~GTk).sum(axis=0)
    return int(keep.sum()), n_better, n_tied_nonmatch


def recall_at_k(S, GT, k_list, *, tie_policy="optimistic", chunk_cols=None,
                workers=None, progress=False, desc="recall@K"):
    """
    Recall@K for every K in `k_list`, from a single sort-free pass.

    S is (references, queries), higher = more similar. GT is the same shape and
    binary. Queries with no ground-truth match are excluded, matching the
    convention Event-LAB has always used.

    The result does not depend on `chunk_cols` or `workers`: columns are
    independent and each chunk's contribution is an integer count.
    """
    if tie_policy not in TIE_POLICIES:
        raise ValueError(f"tie_policy must be one of {TIE_POLICIES}, got {tie_policy!r}")
    S = np.asarray(S)
    GT = np.asarray(GT)
    if S.shape != GT.shape:
        raise ValueError(f"S {S.shape} and GT {GT.shape} must have the same shape")
    if S.ndim != 2:
        raise ValueError("S and GT must be two-dimensional")
    k_list = [int(k) for k in k_list]
    if any(k < 1 for k in k_list):
        raise ValueError("every K must be >= 1")

    GT = GT.astype(bool, copy=False)
    fill = _neg_fill(S.dtype)
    n_rows, n_cols = S.shape

    workers = resolve_workers(workers)
    if chunk_cols is None:
        chunk_cols = resolve_chunk_cols(n_rows, S.dtype.itemsize, n_cols, workers)
    chunk_cols = max(1, min(int(chunk_cols), n_cols))
    bounds = [(a, min(a + chunk_cols, n_cols)) for a in range(0, n_cols, chunk_cols)]

    def work(bound):
        a, b = bound
        return _chunk_counts(S[:, a:b], GT[:, a:b], fill)

    if workers > 1 and len(bounds) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            it = pool.map(work, bounds)
            results = list(tqdm(it, total=len(bounds), desc=desc, disable=not progress))
    else:
        results = [work(b) for b in tqdm(bounds, desc=desc, disable=not progress)]

    kept = sum(r[0] for r in results)
    if kept == 0:
        logger.warning("No query has a ground-truth match; recall is undefined.")
        return {k: float("nan") for k in k_list}

    better = np.concatenate([r[1] for r in results]) if results else np.zeros(0, np.int64)
    if tie_policy == "pessimistic":
        better = better + np.concatenate([r[2] for r in results])

    return {k: float((better < k).sum() / kept) for k in k_list}


def recall_at_k_bracket(S, GT, k_list, **kwargs):
    """Both tie policies from the same data: (optimistic, pessimistic)."""
    kwargs.pop("tie_policy", None)
    return (recall_at_k(S, GT, k_list, tie_policy="optimistic", **kwargs),
            recall_at_k(S, GT, k_list, tie_policy="pessimistic", **kwargs))


def precision_recall_curve(S, GT, n_thresh=100):
    """
    Single-best-match precision/recall over `n_thresh` thresholds.

    Equivalent to VPR_Tutorial's `createPR(matching='single')`, minus its
    unconditional full copy of S -- that copy only existed to mask GTsoft
    entries, and Event-LAB never passes GTsoft.
    """
    S = np.asarray(S)
    GT = np.asarray(GT)
    if S.shape != GT.shape:
        raise ValueError(f"S {S.shape} and GT {GT.shape} must have the same shape")
    if n_thresh <= 1:
        raise ValueError("n_thresh must be > 1")

    GT = GT.astype(bool, copy=False)
    n_positive = int(np.count_nonzero(GT.any(axis=0)))

    best_row = np.argmax(S, axis=0)
    gt_best = GT[best_row, np.arange(GT.shape[1])]
    s_best = S[best_row, np.arange(S.shape[1])]

    P, R = [1.0], [0.0]
    for threshold in np.linspace(s_best.max(), s_best.min(), n_thresh):
        selected = s_best >= threshold
        tp = int(np.count_nonzero(gt_best & selected))
        fp = int(np.count_nonzero(~gt_best & selected))
        P.append(tp / (tp + fp) if (tp + fp) else 1.0)
        R.append(tp / n_positive if n_positive else float("nan"))
    return P, R


def aupr(P, R):
    """Area under the precision-recall curve, integrating over sorted recall."""
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)
    if P.size == 0 or R.size == 0:
        return float("nan")
    order = np.argsort(R)
    return float(np.trapz(P[order], R[order]))


def best_match_rows(S, chunk_cols=None):
    """Top-1 reference row per query column -- the predictions the overlay draws."""
    S = np.asarray(S)
    if chunk_cols is None:
        return np.argmax(S, axis=0)
    out = np.empty(S.shape[1], dtype=np.int64)
    for a in range(0, S.shape[1], chunk_cols):
        b = min(a + chunk_cols, S.shape[1])
        out[a:b] = np.argmax(S[:, a:b], axis=0)
    return out
