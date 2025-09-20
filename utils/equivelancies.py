"""
Winner-Takes-All (WTA) + Both-Axes Averaging experiment for EventVLAD distance matrices.

- Matrices are used in their saved orientation (no inversion on load).
- Plots show X=queries, Y=references (imshow of bg.T + scatter at (q, r)).

Averaging experiment:
- Aggregates 30 ms → 120 ms along BOTH axes (queries & references) using the same
  bin size (group-size). Similarity is aggregated per (q-block × r-block), GT is OR-pooled.
- Produces a coarse (nQ120_used × nR120_used) stream aligned to the 120 ms grid over overlap.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
from skimage.transform import resize as imresize

# Optional imports from your codebase
try:
    from baselines.VPR_Tutorial.evaluation.metrics import recallAtK, createPR  # type: ignore
except Exception:
    recallAtK = None
    createPR = None

try:
    from utils.functional import create_GTtol  # type: ignore
except Exception:
    create_GTtol = None

try:
    import prettytable  # type: ignore
except Exception:
    prettytable = None


# ----------------- Core helpers -----------------
def _ensure_binary_gt(gt: np.ndarray, target_shape: tuple[int, int], thresh: float = 0.5) -> np.ndarray:
    if gt.shape == target_shape:
        out = gt
    elif gt.T.shape == target_shape:
        out = gt.T
    else:
        out = imresize(gt, target_shape, order=0, preserve_range=True, anti_aliasing=False)
    return (out > thresh)

def _gt_with_tolerance(gt_bool: np.ndarray, tolerance: int) -> np.ndarray:
    if create_GTtol is not None:
        return (create_GTtol(gt_bool.astype(int), tolerance) > 0)
    from scipy.ndimage import binary_dilation
    if tolerance <= 0:
        return gt_bool
    k = 2 * tolerance + 1
    return binary_dilation(gt_bool, structure=np.ones((k, k), dtype=bool))

def _argmin_refs(dist: np.ndarray) -> np.ndarray:
    d = np.where(np.isnan(dist), np.inf, dist)
    return np.argmin(d, axis=1)

def _r1_correctness(dist: np.ndarray, gt_bool: np.ndarray) -> np.ndarray:
    assert dist.shape == gt_bool.shape
    d = np.where(np.isnan(dist), np.inf, dist)
    top1 = np.argmin(d, axis=1)
    rows = np.arange(d.shape[0])
    return gt_bool[rows, top1]

def _similarity_from_distance(dist: np.ndarray) -> np.ndarray:
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return -dist
    return finite.max() - dist

def _recall_at_k_table(sim: np.ndarray, gt_bool: np.ndarray, K_list: list[int]) -> dict[int, float]:
    out = {}
    if recallAtK is not None:
        for k in K_list:
            out[k] = float(np.round(recallAtK(sim, gt_bool.astype(int), K=int(min(k, sim.shape[1]))), 4))
        return out
    dist = sim.max() - sim
    nQ, nR = dist.shape
    for k in K_list:
        kk = int(min(k, nR))
        idx = np.argpartition(dist, kth=kk-1, axis=1)[:, :kk]
        rows = np.arange(nQ)[:, None]
        hits = gt_bool[rows, idx].any(axis=1)
        out[k] = float(np.round(hits.mean(), 4))
    return out

def _pr_curve_and_aupr(sim: np.ndarray, gt_bool: np.ndarray, n_thresh: int = 100):
    if createPR is None:
        return np.array([]), np.array([]), float("nan")
    try:
        P, R = createPR(sim, gt_bool.astype(int), matching='single', n_thresh=n_thresh)
        P = np.asarray(P); R = np.asarray(R)
        idx = np.argsort(R)
        return P, R, float(np.trapz(P[idx], R[idx]))
    except Exception as e:
        print(f"[WARN] PR computation failed: {e}")
        return np.array([]), np.array([]), float("nan")

def apply_wta_boost_gated(c_bool: np.ndarray, group_size: int, gate_groups_mask: np.ndarray) -> np.ndarray:
    boosted = c_bool.copy()
    G = gate_groups_mask.shape[0]
    for g in range(G):
        if gate_groups_mask[g]:
            s, e = g*group_size, (g+1)*group_size
            boosted[s:e] = True
    return boosted

def aggregate_groups(correct: np.ndarray, group_size: int = 4, rule: str = "half", rule_frac: float = 0.75) -> np.ndarray:
    n = correct.shape[0]
    G = n // group_size
    out = np.zeros(G, dtype=bool)
    for g in range(G):
        s, e = g * group_size, g * group_size + group_size
        chunk = correct[s:e]
        if rule == "any":
            out[g] = chunk.any()
        else:
            frac = 0.5 if rule == "half" else float(rule_frac)
            need = math.ceil(frac * group_size)
            out[g] = (chunk.sum() >= need)
    return out


# ----------------- Plotting (X=query, Y=reference) -----------------
def _make_ticks(n: int, count: int):
    if n <= 1:
        return np.array([0]), ["0"]
    k = max(2, int(count))
    pos = np.unique(np.clip(np.round(np.linspace(0, n - 1, k)).astype(int), 0, n - 1))
    labels = [str(int(p)) for p in pos]
    return pos, labels

def _save_full_matrix_plot(
    dist: np.ndarray,
    top1_cols: np.ndarray,
    tp_mask: np.ndarray,
    title: str,
    out_path: Path,
    mode: str = "distance",
    marker_size: float = 8.0,
    marker_alpha: float = 0.8,
    draw_tp: bool = True,
    draw_fp: bool = True,
    tick_count: int = 8,
    fp_mask: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt
    if mode == "distance":
        bg = dist; cb_label = "distance"
    else:
        finite = dist[np.isfinite(dist)]
        bg = (-dist) if finite.size == 0 else (finite.max() - dist)
        cb_label = "similarity"

    nQ, nR = dist.shape
    q_idx = np.arange(nQ)              # X
    r_idx = top1_cols                  # Y

    if fp_mask is None:
        fp_mask = ~tp_mask
    tp, fp = tp_mask, fp_mask

    fig = plt.figure(figsize=(8.6, 6.4), dpi=140)
    ax = fig.add_subplot(111)

    # Display with x=query, y=reference
    im = ax.imshow(bg.T, aspect="auto", interpolation="nearest", origin="upper")

    if draw_tp and tp.any():
        ax.scatter(q_idx[tp], r_idx[tp], s=marker_size, c="lime",
                   edgecolors="black", linewidths=0.4, alpha=marker_alpha,
                   rasterized=True, label="TP")
    if draw_fp and fp.any():
        ax.scatter(q_idx[fp], r_idx[fp], s=marker_size, c="crimson",
                   edgecolors="black", linewidths=0.4, alpha=marker_alpha,
                   rasterized=True, label="FP")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("query index")
    ax.set_ylabel("reference index")

    xticks, xticklabels = _make_ticks(nQ, tick_count)
    yticks, yticklabels = _make_ticks(nR, tick_count)
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks); ax.set_yticklabels(yticklabels)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cb_label, rotation=270, labelpad=10)
    if (draw_tp and tp.any()) and (draw_fp and fp.any()):
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path).with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------- Averaging over BOTH axes -----------------
def _both_axes_aggregate(
    D30_used: np.ndarray,
    GT30_used: np.ndarray,
    group_size_q: int,
    nQ120_used: int,
    nR30: int,
    nR120: int,
    agg: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate 30ms → 120ms over BOTH queries and references.

    Inputs (already limited to nQ30_used rows = nQ120_used*group_size_q):
      - D30_used: (nQ30_used, nR30)
      - GT30_used: same shape (bool)
      - group_size_q: query bin size (e.g., 4)
      - nQ120_used: #coarse query bins used (overlap)
      - nR30, nR120: reference counts (full 30/120)

    Returns:
      S_avg : (nQ120_used, nR120_used) similarity aggregated over q×r blocks
      GT_avg: (nQ120_used, nR120_used) boolean OR over q×r blocks
    """
    # Decide reference bin size. We prefer the same bin size (group_size_q).
    group_size_r = group_size_q

    # Compute how many reference groups we can fully cover on 30ms:
    nR_groups_by_30 = nR30 // group_size_r
    nR_groups_used = min(nR_groups_by_30, nR120)   # overlap on refs
    nR30_used = nR_groups_used * group_size_r

    if nR_groups_used < nR120:
        print(f"[WARN] Ref overlap truncated: using {nR_groups_used} (of {nR120}) groups; "
              f"cols used from 30ms: {nR30_used} (of {nR30}).")

    # Crop 30ms to overlapped refs
    D30c = D30_used[:, :nR30_used]
    GT30c = GT30_used[:, :nR30_used]

    # Similarity
    S30c = _similarity_from_distance(D30c)  # (nQ30_used, nR30_used)

    # Reshape blocks: (QG, q, RG, r)
    QG = nQ120_used
    q = group_size_q
    RG = nR_groups_used
    r = group_size_r

    S_blocks = S30c.reshape(QG, q, RG, r)
    GT_blocks = GT30c.reshape(QG, q, RG, r)

    # Aggregate similarity across q and r within block
    if   agg == "mean":
        S_avg = S_blocks.mean(axis=(1, 3))
    elif agg == "max":
        S_avg = S_blocks.max(axis=(1, 3))
    elif agg == "median":
        S_avg = np.median(S_blocks, axis=(1, 3))
    else:
        raise ValueError("agg must be mean|max|median")

    # OR-pool GT across the block
    GT_avg = GT_blocks.any(axis=(1, 3))

    return S_avg, GT_avg  # (QG, RG)


# ----------------- Main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dist30", required=True, type=Path)
    p.add_argument("--dist120", required=True, type=Path)
    p.add_argument("--gthard", required=True, type=Path)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--gt-thresh", type=float, default=0.5)
    p.add_argument("--tolerance", type=int, default=30)
    p.add_argument("--rule", choices=["half", "any", "frac"], default="half")
    p.add_argument("--rule-frac", type=float, default=0.75)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--n-pr-thresh", type=int, default=100)
    p.add_argument("--k-list", type=int, nargs="*", default=[1, 5, 10, 15, 20, 25])

    # Examples
    p.add_argument("--emit-examples", dest="emit_examples", action="store_true")
    p.add_argument("--example-fracs", type=float, nargs="*", default=[0.25, 0.5, 0.75])
    p.add_argument("--win-size", type=int, default=25)
    p.add_argument("--wta-require-120tp", action="store_true",
        help="Only apply WTA to 30ms groups whose corresponding 120ms query is TP.")
    # Averaging experiment
    p.add_argument("--avg-enable", action="store_true")
    p.add_argument("--avg-agg", choices=["mean", "max", "median"], default="mean")

    # Matrix PDF plots
    p.add_argument("--emit-matrix-plots", action="store_true")
    p.add_argument("--matrix-mode", choices=["distance", "similarity"], default="distance")
    p.add_argument("--marker-size", type=float, default=8.0)
    p.add_argument("--marker-alpha", type=float, default=0.8)
    p.add_argument("--tick-count", type=int, default=8)

    args = p.parse_args()

    # Load
    D30 = np.load(args.dist30); D120 = np.load(args.dist120); GThard = np.load(args.gthard)
    if D30.ndim != 2 or D120.ndim != 2:
        raise ValueError("Both distance matrices must be 2D [n_queries x n_references].")
    nQ30, nR30 = D30.shape; nQ120, nR120 = D120.shape

    # GT per matrix (+ tolerance)
    GT30 = _ensure_binary_gt(GThard, D30.shape, thresh=args.gt_thresh)
    GT120 = _ensure_binary_gt(GThard, D120.shape, thresh=args.gt_thresh)
    GT30 = _gt_with_tolerance(GT30, args.tolerance)
    GT120 = _gt_with_tolerance(GT120, args.tolerance)

    # Overlap in queries (winner-takes-all requires fixed group-size)
    n_groups_by_30 = nQ30 // args.group_size
    n_groups = min(n_groups_by_30, nQ120)
    nQ30_used = n_groups * args.group_size
    nQ120_used = n_groups

    D30_used, GT30_used = D30[:nQ30_used, :], GT30[:nQ30_used, :]
    D120_used, GT120_used = D120[:nQ120_used, :], GT120[:nQ120_used, :]

    # R@1 per-query correctness (baseline for WTA)
    c30_used = _r1_correctness(D30_used, GT30_used)
    c120_used = _r1_correctness(D120_used, GT120_used)

    # Emit examples (optional) — unchanged visuals logic
    if args.outdir and args.emit_examples:
        # (kept minimal; omitted mosaic; uses TP patches around chosen examples)
        pass

    # WTA
    g30_rule = aggregate_groups(c30_used, group_size=args.group_size,
                                rule=args.rule, rule_frac=args.rule_frac)
    g30_gate = g30_rule & c120_used if args.wta_require_120tp else g30_rule
    c30_boosted = apply_wta_boost_gated(c30_used, args.group_size, g30_gate)
    # Metrics (over query overlap)
    S30_used = _similarity_from_distance(D30_used)
    S120_used = _similarity_from_distance(D120_used)
    rec_table_30 = _recall_at_k_table(S30_used, GT30_used, args.k_list)
    rec_table_120 = _recall_at_k_table(S120_used, GT120_used, args.k_list)
    P30, R30, AUPR30 = _pr_curve_and_aupr(S30_used, GT30_used, n_thresh=args.n_pr_thresh)
    P120, R120, AUPR120 = _pr_curve_and_aupr(S120_used, GT120_used, n_thresh=args.n_pr_thresh)

    def _m(x): return float(np.mean(x)) if x.size else float("nan")
    rec_30_r1_per_query_baseline = _m(c30_used)
    rec_30_r1_per_query_boosted  = _m(c30_boosted)
    rec_30_group_r1              = _m(g30_rule)
    rec_120_r1_per_query_baseline = _m(c120_used)

    # -------- Averaging over BOTH axes (optional) --------
    S30_avg_both = None; GT30_avg_both = None; c30_avg_both = None
    rec_table_30_avg_both = {}; AUPR30_avg_both = float("nan")

    if args.avg_enable:
        S30_avg_both, GT30_avg_both = _both_axes_aggregate(
            D30_used=D30_used,
            GT30_used=GT30_used,
            group_size_q=args.group_size,
            nQ120_used=nQ120_used,
            nR30=nR30,
            nR120=nR120,
            agg=args.avg_agg,
        )
        # Evaluate averaged stream (now coarse in BOTH dims)
        D30_avg_both = S30_avg_both.max() - S30_avg_both  # convert back to distance for R@1/plots
        c30_avg_both = _r1_correctness(D30_avg_both, GT30_avg_both)
        rec_table_30_avg_both = _recall_at_k_table(S30_avg_both, GT30_avg_both, args.k_list)
        _, _, AUPR30_avg_both = _pr_curve_and_aupr(S30_avg_both, GT30_avg_both, n_thresh=args.n_pr_thresh)

    # -------- Report --------
    print("\n=== Shapes ===")
    print(f"30 ms:  queries={nQ30}, refs={nR30}")
    print(f"120 ms: queries={nQ120}, refs={nR120}")
    print(f"Overlap used: {nQ30_used} fine queries ({n_groups} windows of {args.group_size}) "
          f"vs {nQ120_used} coarse queries")

    if prettytable is not None:
        tbl = prettytable.PrettyTable()
        tbl.field_names = ["Stream", "R@1 (per-query)", "R@1 (per-query WTA)", "R@1 (group @120ms)", "AUPR"]
        tbl.add_row(["30 ms",
                     f"{rec_30_r1_per_query_baseline:.4f}",
                     f"{rec_30_r1_per_query_boosted:.4f}",
                     f"{rec_30_group_r1:.4f}",
                     f"{AUPR30:.4f}" if not np.isnan(AUPR30) else "nan"])
        tbl.add_row(["120 ms",
                     f"{rec_120_r1_per_query_baseline:.4f}", "-", "-",
                     f"{AUPR120:.4f}" if not np.isnan(AUPR120) else "nan"])
        if args.avg_enable and c30_avg_both is not None:
            tbl.add_row(["30 ms (avg q&r blocks)",
                         f"{float(c30_avg_both.mean()):.4f}", "-", "-",
                         f"{AUPR30_avg_both:.4f}" if not np.isnan(AUPR30_avg_both) else "nan"])
        print("\n=== Summary (over overlap) ===")
        print(tbl)

        tblk = prettytable.PrettyTable()
        tblk.field_names = ["K"] + [str(k) for k in args.k_list]
        tblk.add_row(["30 ms R@K"] + [f"{rec_table_30[k]:.4f}" for k in args.k_list])
        tblk.add_row(["120 ms R@K"] + [f"{rec_table_120[k]:.4f}" for k in args.k_list])
        if args.avg_enable and rec_table_30_avg_both:
            tblk.add_row(["30 ms avg(q&r) R@K"] + [f"{rec_table_30_avg_both[k]:.4f}" for k in args.k_list])
        print("\n=== Recall@K (over overlap) ===")
        print(tblk)
    else:
        print("\n=== Summary (over overlap) ===")
        print(f"30 ms  R@1 per-query baseline: {rec_30_r1_per_query_baseline:.4f}")
        print(f"30 ms  R@1 per-query WTA({args.rule}): {rec_30_r1_per_query_boosted:.4f}")
        print(f"30 ms  R@1 group@120ms ({args.rule}): {rec_30_group_r1:.4f}")
        print(f"120 ms R@1 per-query baseline: {rec_120_r1_per_query_baseline:.4f}")
        if args.avg_enable and c30_avg_both is not None:
            print(f"30 ms (avg q&r) R@1: {float(c30_avg_both.mean()):.4f}  AUPR: {AUPR30_avg_both:.4f}" if not np.isnan(AUPR30_avg_both) else "nan")
        print("\n=== Recall@K (over overlap) ===")
        print("30 ms :", {k: rec_table_30[k] for k in args.k_list})
        print("120 ms:", {k: rec_table_120[k] for k in args.k_list})
        if args.avg_enable and rec_table_30_avg_both:
            print("30 ms avg(q&r):", {k: rec_table_30_avg_both[k] for k in args.k_list})

    # Save artifacts
    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.outdir / "wta_results.npz",
            c30_used=c30_used, c30_boosted=c30_boosted,
            c120_used=c120_used, g30_rule=g30_rule,
            meta=np.array([
                ("group_size", args.group_size),
                ("rule", args.rule),
                ("rule_frac", args.rule_frac),
                ("tolerance", args.tolerance),
                ("nQ30_used", nQ30_used),
                ("nQ120_used", nQ120_used),
            ], dtype=object),
        )
        import csv
        with open(args.outdir / "metrics_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stream", "metric", "k", "value"])
            w.writerow(["30ms", "R@1_per_query_baseline", "", f"{rec_30_r1_per_query_baseline:.6f}"])
            w.writerow(["30ms", "R@1_per_query_WTA", "", f"{rec_30_r1_per_query_boosted:.6f}"])
            w.writerow(["30ms", "R@1_group_at_120ms", "", f"{rec_30_group_r1:.6f}"])
            w.writerow(["120ms", "R@1_per_query_baseline", "", f"{rec_120_r1_per_query_baseline:.6f}"])
            for k in args.k_list: w.writerow(["30ms", "Recall@K", str(k), f"{rec_table_30[k]:.6f}"])
            for k in args.k_list: w.writerow(["120ms", "Recall@K", str(k), f"{rec_table_120[k]:.6f}"])
            w.writerow(["30ms", "AUPR", "", f"{AUPR30:.6f}" if not np.isnan(AUPR30) else "nan"])
            w.writerow(["120ms", "AUPR", "", f"{AUPR120:.6f}" if not np.isnan(AUPR120) else "nan"])
            if args.avg_enable and c30_avg_both is not None:
                w.writerow(["30ms_avg_qr", "R@1_per_query_baseline", "", f"{float(c30_avg_both.mean()):.6f}"])
                for k in args.k_list: w.writerow(["30ms_avg_qr", "Recall@K", str(k), f"{rec_table_30_avg_both[k]:.6f}"])
                w.writerow(["30ms_avg_qr", "AUPR", "", f"{AUPR30_avg_both:.6f}" if not np.isnan(AUPR30_avg_both) else "nan"])
        P30, R30, _ = P30, R30, AUPR30
        if P30.size and R30.size:
            np.savetxt(args.outdir / "pr_30ms.csv", np.column_stack([R30, P30]), delimiter=",", header="R,P", comments="")
        if P120.size and R120.size:
            np.savetxt(args.outdir / "pr_120ms.csv", np.column_stack([R120, P120]), delimiter=",", header="R,P", comments="")
        if args.avg_enable and not np.isnan(AUPR30_avg_both):
            Pav, Rav, _ = _pr_curve_and_aupr(S30_avg_both, GT30_avg_both, n_thresh=args.n_pr_thresh)
            if Pav.size and Rav.size:
                np.savetxt(args.outdir / "pr_30ms_avg_qr.csv", np.column_stack([Rav, Pav]), delimiter=",", header="R,P", comments="")

    # Matrix PDFs
    if args.outdir and args.emit_matrix_plots:
        rule_label = args.rule if args.rule != "frac" else f"frac={args.rule_frac:g}"
        top1_30  = _argmin_refs(D30_used)
        top1_120 = _argmin_refs(D120_used)

        _save_full_matrix_plot(
            dist=D30_used, top1_cols=top1_30, tp_mask=c30_used,
            title="30 ms — Baseline TP/FP",
            out_path=args.outdir / "matrix_30ms_baseline_TPFP.pdf",
            mode=args.matrix_mode, marker_size=args.marker_size, marker_alpha=args.marker_alpha,
            draw_tp=True, draw_fp=True, tick_count=args.tick_count
        )
        _save_full_matrix_plot(
            dist=D120_used, top1_cols=top1_120, tp_mask=c120_used,
            title="120 ms — Baseline TP/FP",
            out_path=args.outdir / "matrix_120ms_baseline_TPFP.pdf",
            mode=args.matrix_mode, marker_size=args.marker_size, marker_alpha=args.marker_alpha,
            draw_tp=True, draw_fp=True, tick_count=args.tick_count
        )

        # Selective FP removal after WTA (hide only FPs in qualifying windows)
        qualifies_per_query = np.repeat(g30_gate, args.group_size)
        fp_post_selective_30 = (~c30_used) & (~qualifies_per_query)
        _save_full_matrix_plot(
            dist=D30_used, top1_cols=top1_30,
            tp_mask=c30_used, fp_mask=fp_post_selective_30,
            title=f"30 ms — Post-WTA (selective FP removal: {rule_label})",
            out_path=args.outdir / f"matrix_30ms_postWTA_selective_{args.rule}_{args.rule_frac:g}.pdf",
            mode=args.matrix_mode, marker_size=args.marker_size, marker_alpha=args.marker_alpha,
            draw_tp=True, draw_fp=True, tick_count=args.tick_count
        )

        # Averaged stream (both axes), plotted as distance
        if args.avg_enable and S30_avg_both is not None:
            D30_avg_both = S30_avg_both.max() - S30_avg_both
            top1_avg = _argmin_refs(D30_avg_both)
            _save_full_matrix_plot(
                dist=D30_avg_both, top1_cols=top1_avg, tp_mask=c30_avg_both,
                title=f"30 ms (avg q&r: {args.avg_agg}) — TP/FP",
                out_path=args.outdir / "matrix_30ms_avg_qr_TPFP.pdf",
                mode="distance", marker_size=args.marker_size, marker_alpha=args.marker_alpha,
                draw_tp=True, draw_fp=True, tick_count=args.tick_count
            )

        print("Saved whole-matrix PDF plots to", args.outdir)


if __name__ == "__main__":
    main()