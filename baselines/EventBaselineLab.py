import prettytable, os
import time

import numpy as np

from utils.metrics import (aupr, best_match_rows, conform_ground_truth,
                           precision_recall_curve, recall_at_k, resolve_workers)
from datasets.groundtruths import create_GTtol_by_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from loguru import logger

# Longest edge of the rendered overlay. A 14473x12820 matrix rasterised at
# dpi=200 is both unreadable and enormous; the figure is a diagnostic, so it is
# decimated to a bounded grid before anything allocates an RGB copy.
OVERLAY_MAX_SIDE = 2000


def overlay_matches_on_array(
    array,
    GThard,
    top_k=1,
    pred_mode="per_column",   # "per_column" or "per_row"
    marker_size=20,
    alpha_blend=0.7,
    save_path=None,
    array_name=None,
    pred_rows=None,
    max_side=OVERLAY_MAX_SIDE,
):
    """
    Render the similarity matrix with TP/FP/FN markers.

    - array: 2D numpy array (refs x queries), higher = more similar
    - GThard: 0/1 ground truth, resized to match if needed
    - pred_rows: top-1 reference row per query, if the caller already computed it
      (run_metrics does). Avoids re-deriving predictions from a second full sort.

    Everything is decimated to at most `max_side` on the long edge before the RGB
    buffer is built, so cost is bounded by the figure size rather than the matrix.
    """
    if array.ndim != 2:
        raise ValueError("array must be 2D (refs x queries)")
    if pred_mode not in ("per_column", "per_row"):
        raise ValueError("pred_mode must be 'per_column' or 'per_row'")

    GT = conform_ground_truth(GThard, array.shape)
    h, w = array.shape

    # Predictions: one reference row per query column (or transposed for per_row).
    if pred_mode == "per_column":
        rows = best_match_rows(array) if pred_rows is None else np.asarray(pred_rows)
        cols = np.arange(w)
    else:
        cols = np.argmax(array, axis=1)
        rows = np.arange(h)

    hit = GT[rows, cols]
    tp_r, tp_c = rows[hit], cols[hit]
    fp_r, fp_c = rows[~hit], cols[~hit]

    # False negatives: ground-truth cells that were not predicted. Derived from
    # the GT coordinates directly, so no full-size temporary is allocated.
    gt_r, gt_c = np.nonzero(GT)
    if pred_mode == "per_column":
        missed = rows[gt_c] != gt_r
    else:
        missed = cols[gt_r] != gt_c
    fn_r, fn_c = gt_r[missed], gt_c[missed]

    # Decimate to the display grid.
    step_r = max(1, int(np.ceil(h / max_side)))
    step_c = max(1, int(np.ceil(w / max_side)))
    small = np.asarray(array[::step_r, ::step_c], dtype=np.float32)
    sh, sw = small.shape

    # Normalise for display. The previous implementation clipped raw values into
    # [0, 1], which flattened any matrix not already in that range to a blank
    # image; percentile scaling keeps the structure visible.
    lo, hi = np.percentile(small, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(small.min()), float(small.max())
    grey = np.clip((small - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(small)
    overlay = np.repeat(grey[:, :, None], 3, axis=2)

    def blend(r, c, colour):
        if r.size == 0:
            return
        rr = np.clip(r // step_r, 0, sh - 1)
        cc = np.clip(c // step_c, 0, sw - 1)
        colour = np.asarray(colour, dtype=np.float32)
        overlay[rr, cc, :] = (1.0 - alpha_blend) * overlay[rr, cc, :] + alpha_blend * colour

    blend(fn_r, fn_c, (0.0, 0.4, 1.0))   # blue   - missed ground truth
    blend(fp_r, fp_c, (1.0, 0.0, 0.0))   # red    - wrong prediction
    blend(tp_r, tp_c, (0.0, 1.0, 0.0))   # green  - correct prediction

    overlay_rgb = (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(overlay_rgb, origin="upper", interpolation="nearest",
              extent=[0, w, h, 0], aspect="auto")
    ax.set_title(f"Matches overlay (mode={pred_mode}, top_k={top_k})")
    ax.set_xlabel("Query index (cols)")
    ax.set_ylabel("Reference index (rows)")
    # Filled swatches, because the cells are colour-blended rather than outlined.
    ax.legend(handles=[Patch(facecolor='lime', edgecolor='black', label=f'TP ({tp_r.size})'),
                       Patch(facecolor='red', edgecolor='black', label=f'FP ({fp_r.size})'),
                       Patch(facecolor='#0066ff', edgecolor='black', label=f'FN ({fn_r.size})')],
              loc='upper right', framealpha=0.9)
    plt.tight_layout()

    if save_path:
        fig.savefig(os.path.join(f'{save_path}', f'{array_name}_matches'), dpi=200)
    plt.close(fig)   # the previous version leaked one figure per scored array

    return fig, (rows, cols), (tp_r, tp_c), (fp_r, fp_c), (fn_r, fn_c)


class EventBaseline:
    def __init__(self):
        self.K_list = [1, 5, 10, 15, 20, 25]

    def run_metrics(self, all_names, all_arrays, GThard, timestamp, run_name, ref_query,
                    matrix_type="distance", outdir=None, tolerance=0, tie_policy="optimistic"):
        """
        Score each result matrix and return (rows, pr_curves).

        Recall for every K comes from one sort-free pass (utils.metrics.recall_at_k):
        a correct match is inside the top K exactly when fewer than K references
        score strictly better than it, so no ordering is needed. `tie_policy`
        decides whether a match tied with incorrect references counts as
        retrieved -- the matrices are often heavily quantised, and the previous
        implementation resolved such ties through an unstable sort, which made
        the reported number arbitrary among the tied candidates.
        """
        table = prettytable.PrettyTable()
        table.field_names = ["Recall@K"] + [f"@{k}" for k in self.K_list] + ["AUPR"]
        rows = []
        pr_curves = {}  # (ref_query, array_name) -> (P,R)

        workers = resolve_workers()
        logger.info(f"Scoring {len(all_arrays)} matri{'x' if len(all_arrays) == 1 else 'ces'} "
                    f"({tie_policy} ties, {workers} worker{'' if workers == 1 else 's'})")

        for name, array in zip(all_names, all_arrays):
            t_start = time.perf_counter()
            if matrix_type == "distance":
                array = array.max() - array  # convert to similarity

            # Conformed once per array. This used to sit inside the per-K loop,
            # so a large ground truth was resized six times over.
            target_shape = array.shape
            GT = conform_ground_truth(GThard, target_shape)
            t_gt = time.perf_counter()

            recalls_by_k = recall_at_k(array, GT, self.K_list, tie_policy=tie_policy,
                                       workers=workers, progress=True,
                                       desc=f"recall@K {name}")
            recalls = [np.round(recalls_by_k[k], 2) for k in self.K_list]
            t_recall = time.perf_counter()

            try:
                P, R = precision_recall_curve(array, GT, n_thresh=100)
                P = np.asarray(P); R = np.asarray(R)
                area = aupr(P, R)
            except Exception as e:
                logger.error(f"  -> Error computing PR for {name}: {e}")
                P, R, area = np.array([]), np.array([]), np.nan
            t_pr = time.perf_counter()

            # The overlay reuses these predictions rather than re-sorting the matrix.
            pred_rows = best_match_rows(array)
            overlay_matches_on_array(
                array=array,
                GThard=GT,
                top_k=1,
                pred_mode="per_column",
                marker_size=20,
                alpha_blend=0.6,
                save_path=outdir,
                array_name=name,
                pred_rows=pred_rows,
            )
            t_end = time.perf_counter()
            logger.info(f"  {name} {target_shape}: gt {t_gt - t_start:.2f}s | "
                        f"recall {t_recall - t_gt:.2f}s | pr {t_pr - t_recall:.2f}s | "
                        f"overlay {t_end - t_pr:.2f}s | total {t_end - t_start:.2f}s")

            table.add_row([name] + recalls + [np.round(area, 4)])

            rows.append({
                "timestamp_utc": timestamp,
                "run_name": run_name,
                "ref_query": ref_query,
                "array_name": name,
                "n_references": int(target_shape[0]) if len(target_shape) >= 1 else None,
                "n_queries": int(target_shape[1]) if len(target_shape) >= 2 else None,
                "R@1": recalls[0], "R@5": recalls[1], "R@10": recalls[2],
                "R@15": recalls[3], "R@20": recalls[4], "R@25": recalls[5],
                "aupr": np.round(area, 6)
            })
            pr_curves[(ref_query, name)] = (P, R)

        logger.info("\n{}", table.get_string())
        return rows, pr_curves

    def save_results(self, rows, pr_curves, run_name, ref_query):
        """
        rows: list[dict] with keys:
            timestamp_utc, run_name, ref_query, array_name, n_references, n_queries,
            R@1,R@5,R@10,R@15,R@20,R@25, aupr
        pr_curves: dict[(ref_query, array_name)] -> (P, R)
        """
        import os, re, time
        from datetime import datetime, timezone
        import openpyxl

        # ---------- paths ----------
        excel_path = "./output/eventlab_results.xlsx"
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)

        # ---------- workbook/sheet helpers ----------
        def load_wb(path):
            try:
                if os.path.exists(path):
                    wb_ = openpyxl.load_workbook(path)
                else:
                    wb_ = openpyxl.Workbook()
                # remove empty default "Sheet" if present
                if "Sheet" in wb_.sheetnames:
                    ws0 = wb_["Sheet"]
                    if ws0.max_row == 1 and ws0.max_column == 1 and ws0.cell(1,1).value in (None, ""):
                        del wb_["Sheet"]
                return wb_
            except Exception:
                bak = f"{path}.corrupt.{int(time.time())}.bak"
                try:
                    os.rename(path, bak)
                except Exception:
                    pass
                wb_ = openpyxl.Workbook()
                if "Sheet" in wb_.sheetnames:
                    del wb_["Sheet"]
                return wb_

        def ensure_sheet(wb, name):
            return wb[name] if name in wb.sheetnames else wb.create_sheet(name)

        def sheet_headers(ws, headers):
            existing = [ws.cell(row=1, column=i+1).value for i in range(len(headers))]
            if existing != headers:
                for i, h in enumerate(headers, 1):
                    ws.cell(row=1, column=i, value=h)
            return {h: i+1 for i, h in enumerate(headers)}

        # --- index helper used by Summary only (run sheet uses append-only upsert) ---
        def build_index(ws, key_cols):  # returns {key_tuple: row_idx}
            idx = {}
            r = 2
            while r <= ws.max_row:
                vals = [ws.cell(row=r, column=c).value for c in key_cols]
                if all(v in (None, "") for v in vals):
                    nxt = [ws.cell(row=r+1, column=c).value for c in key_cols] if r+1 <= ws.max_row else [None]
                    if all(v in (None, "") for v in nxt):
                        break
                else:
                    idx[tuple(vals)] = r
                r += 1
            return idx

        # ---------- Per-run sheet: append-only upsert (NO ROW INSERTIONS) ----------
        # --- new helper: ALWAYS APPEND (no matching/overwrite) ---
        def append_rows_no_dedupe(ws, headers, rows_to_write):
            hdr = sheet_headers(ws, headers)
            next_row = ws.max_row + 1 if ws.max_row >= 1 else 2
            for d in rows_to_write:
                for h in headers:
                    ws.cell(row=next_row, column=hdr[h], value=d.get(h, None))
                next_row += 1


        # ---------- PR columns on the SAME run sheet (no row sharing issues) ----------
        def pr_find_block_col(ws, run_headers, run_label, array_name):
            """Find starting column of an existing PR block on the run sheet."""
            hp = f"PR ({run_label}) [{array_name}] - Precision"
            hr = f"PR ({run_label}) [{array_name}] - Recall"
            for col in range(len(run_headers)+2, ws.max_column):  # search to the right of recall table
                if ws.cell(row=1, column=col).value == hp and ws.cell(row=1, column=col+1).value == hr:
                    return col
            return None

        def pr_next_free_col(ws, start_col):
            """Return first empty *pair* of columns at/after start_col (row 1 empty in both)."""
            col = max(1, start_col)
            while True:
                c1 = ws.cell(row=1, column=col).value
                c2 = ws.cell(row=1, column=col+1).value
                if (c1 in (None, "")) and (c2 in (None, "")):
                    return col
                col += 2

        def pr_clear_block(ws, start_col, top_row=2, n_rows=5000):
            """Clear a tall region under the headers (no row shifts)."""
            for r in range(top_row, top_row + n_rows):
                ws.cell(row=r, column=start_col,     value=None)
                ws.cell(row=r, column=start_col + 1, value=None)

        def pr_write_block_on_run_sheet(ws, run_headers, run_label, array_name, P, R):
            """
            Write/overwrite a 2-col PR block for (run_label, array_name) on the SAME run sheet.
            Headers at row 1; numeric data from row 2 down. No gaps, no inserts.
            """
            base_cols = len(run_headers)  # PR always starts to the right of these
            start_col = pr_find_block_col(ws, run_headers, run_label, array_name)
            if start_col is None:
                start_col = pr_next_free_col(ws, base_cols + 2)

            # headers on row 1
            ws.cell(row=1, column=start_col,     value=f"PR ({run_label}) [{array_name}] - Precision")
            ws.cell(row=1, column=start_col + 1, value=f"PR ({run_label}) [{array_name}] - Recall")

            # clear old contents below headers
            pr_clear_block(ws, start_col, top_row=2, n_rows=5000)

            # write numeric data (row 2..)
            n = max(len(P), len(R))
            for i in range(n):
                ws.cell(row=2 + i, column=start_col,     value=float(P[i]) if i < len(P) else None)
                ws.cell(row=2 + i, column=start_col + 1, value=float(R[i]) if i < len(R) else None)

        def pr_write_all_blocks_on_run_sheet(ws_run, pr_curves, run_headers, run_label):
            # deterministic by array_name
            items = sorted(pr_curves.items(), key=lambda x: x[0][1])  # ((ref_query, array_name), (P,R))
            for (rq, array_name), (P, R) in items:
                pr_write_block_on_run_sheet(ws_run, run_headers, run_label, array_name, P, R)

        # ---------- Aggregated Summary (across timewindows) ----------
        METRICS = ["R@1","R@5","R@10","R@15","R@20","R@25","aupr"]

        summary_headers = (
            ["timestamp_utc","run_name","ref_query","array_name","n_references","n_queries"] +
            [f"{m}_mean" for m in METRICS] +
            [f"{m}_std"  for m in METRICS] +
            [f"{m}_n"    for m in METRICS]
        )

        def clean_label(label: str) -> str:
            """Remove -frames-### / -reconstruction-### so timewindows aggregate."""
            if label is None:
                return ""
            return re.sub(r"-(?:frames|reconstruction)-\d+", "", str(label))

        def update_agg_cell(ws, row_idx, hdr_map, base, new_val):
            """Welford (sample std) incremental update."""
            if new_val is None:
                return
            c_mean, c_std, c_n = hdr_map[f"{base}_mean"], hdr_map[f"{base}_std"], hdr_map[f"{base}_n"]

            try: old_n = int(ws.cell(row=row_idx, column=c_n).value or 0)
            except Exception: old_n = 0
            try: old_mean = float(ws.cell(row=row_idx, column=c_mean).value)
            except Exception: old_mean = None
            try: old_std = float(ws.cell(row=row_idx, column=c_std).value)
            except Exception: old_std = None

            if old_n < 1 or old_mean is None or old_std is None:
                ws.cell(row=row_idx, column=c_n,    value=1)
                ws.cell(row=row_idx, column=c_mean, value=float(new_val))
                ws.cell(row=row_idx, column=c_std,  value=0.0)
                return

            M2 = (old_std ** 2) * (old_n - 1) if old_n > 1 else 0.0
            n_new = old_n + 1
            delta = float(new_val) - old_mean
            mean_new = old_mean + delta / n_new
            M2_new = M2 + delta * (float(new_val) - mean_new)
            std_new = (M2_new / (n_new - 1)) ** 0.5 if n_new > 1 else 0.0

            ws.cell(row=row_idx, column=c_n,    value=n_new)
            ws.cell(row=row_idx, column=c_mean, value=mean_new)
            ws.cell(row=row_idx, column=c_std,  value=std_new)

        def find_or_create_summary_row(ws_sum, hdr_map, clean_run, clean_rq, array_name, n_refs, n_qs):
            key_cols = [hdr_map["run_name"], hdr_map["ref_query"], hdr_map["array_name"]]
            existing = build_index(ws_sum, key_cols)
            key_tuple = (clean_run, clean_rq, array_name)
            r = existing.get(key_tuple)
            if r is None:
                r = (max(existing.values()) + 1) if existing else 2
                ws_sum.cell(row=r, column=hdr_map["timestamp_utc"], value=datetime.now(timezone.utc).isoformat())
                ws_sum.cell(row=r, column=hdr_map["run_name"],     value=clean_run)
                ws_sum.cell(row=r, column=hdr_map["ref_query"],    value=clean_rq)
                ws_sum.cell(row=r, column=hdr_map["array_name"],   value=array_name)
                ws_sum.cell(row=r, column=hdr_map["n_references"], value=n_refs)
                ws_sum.cell(row=r, column=hdr_map["n_queries"],    value=n_qs)
                for m in METRICS:
                    ws_sum.cell(row=r, column=hdr_map[f"{m}_n"], value=0)
            else:
                ws_sum.cell(row=r, column=hdr_map["timestamp_utc"], value=datetime.now(timezone.utc).isoformat())
                ws_sum.cell(row=r, column=hdr_map["n_references"], value=n_refs)
                ws_sum.cell(row=r, column=hdr_map["n_queries"],    value=n_qs)
            return r
        
        def append_rows_force(ws, headers, rows_to_write):
            hdr = sheet_headers(ws, headers)

            # Find the last *actually used* row across the first N columns
            def last_used_row(ws, ncols):
                r = ws.max_row or 1
                # scan upward until we hit a row that has any value in cols 1..ncols
                while r >= 2:
                    if any(ws.cell(row=r, column=c).value not in (None, "") for c in range(1, ncols+1)):
                        return r
                    r -= 1
                return 1  # only header remains

            start_row = last_used_row(ws, len(headers)) + 1
            r = start_row
            for d in rows_to_write:
                for h in headers:
                    ws.cell(row=r, column=hdr[h], value=d.get(h, None))
                r += 1


        # ---------- write workbook ----------
        wb = load_wb(excel_path)

        # Summary sheet (aggregated)
        ws_sum = ensure_sheet(wb, "Summary")
        hdr_sum = sheet_headers(ws_sum, summary_headers)

        # Per-run sheet (recall rows + PR columns on the right)
        safe_run = re.sub(r'[:\\/?*\[\]]', "_", str(run_name)).strip()[:31] or "run"
        ws_run = ensure_sheet(wb, safe_run)

        run_headers = [
            "timestamp_utc","run_name","ref_query","array_name","n_references","n_queries",
            "R@1","R@5","R@10","R@15","R@20","R@25","aupr"
        ]
        sheet_headers(ws_run, run_headers)

        # 1) Upsert recall rows WITHOUT INSERTING ROWS (so PR columns never shift)
        append_rows_force(ws_run, run_headers, rows)

        # 2) PR columns to the RIGHT on the SAME sheet (unique per run/timewindow & array)
        run_label_for_block = f"{run_name} :: {ref_query}"
        pr_write_all_blocks_on_run_sheet(ws_run, pr_curves, run_headers, run_label_for_block)

        # 3) Incremental aggregation into Summary (across timewindows)
        for row in rows:
            clean_rq  = clean_label(row["ref_query"])
            clean_run = clean_label(row["run_name"])
            r_idx = find_or_create_summary_row(
                ws_sum, hdr_sum, clean_run, clean_rq, row["array_name"],
                n_refs=row.get("n_references"), n_qs=row.get("n_queries")
            )
            for m in METRICS:
                update_agg_cell(ws_sum, r_idx, hdr_sum, m, row.get(m))

        wb.save(excel_path)
        logger.info(f"Saved Excel workbook to: {excel_path}")
