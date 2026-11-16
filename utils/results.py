"""Read saved evaluation artifacts without loading models or rerunning metrics."""
import json
from pathlib import Path

import numpy as np
from rich.text import Text


def save_details(directory, rows, curves, matrix_type):
    directory = Path(directory)
    results = []
    for row in rows:
        precision, recall = curves[(row['ref_query'], row['array_name'])]
        matrix = directory / (row['array_name'] + '.npy')
        results.append(dict(metrics=row, precision=list(precision), recall=list(recall),
                            matrix_mtime_ns=matrix.stat().st_mtime_ns if matrix.exists() else None))
    # Normalise numpy scalars and undefined metrics to portable JSON values.
    def normalise(value):
        if isinstance(value, dict):
            return {k: normalise(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalise(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        return None if isinstance(value, float) and not np.isfinite(value) else value
    target = directory / 'results.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(normalise(dict(matrix_type=matrix_type, results=results)), indent=2))
    temporary.replace(target)


def discover_results(root):
    """Latest saved evaluation per matrix, with conservative legacy workbook matching."""
    root = Path(root)
    entries, warnings = [], []
    for folder in sorted({p.parent for p in root.glob('*/*/*/*/*.npy')}):
        baseline, dataset, pair, window = folder.relative_to(root).parts
        if dataset in ('descriptors', 'features', 'keypoints'):
            continue
        details = {}
        manifest = folder / 'results.json'
        if manifest.exists():
            try:
                details = json.loads(manifest.read_text())
            except (ValueError, OSError) as error:
                warnings.append(f'{manifest}: {error}')
        for path in sorted(folder.glob('*.npy')):
            result = next((r for r in details.get('results', [])
                           if r['metrics']['array_name'] == path.stem), None)
            current = result and result.get('matrix_mtime_ns') == path.stat().st_mtime_ns
            entries.append(dict(path=path, baseline=baseline, pair=pair,
                label=f'{baseline} · {dataset} · {pair} · {window} · {path.stem}',
                matrix_type=details.get('matrix_type', 'similarity' if baseline in
                    ('lens', 'spikevpr', 'megaevent') else 'distance'),
                metrics=result['metrics'] if current else {},
                precision=result['precision'] if current else [], recall=result['recall'] if current else [],
                source='Saved evaluation' if current else 'Matrix only · no matching saved evaluation',
                has_manifest=manifest.exists()))
    workbook = root / 'eventlab_results.xlsx'
    if workbook.exists():
        try:
            _legacy_metrics(workbook, entries)
        except Exception as error:
            warnings.append(f'Could not read saved workbook: {error}')
    entries.sort(key=lambda e: e['path'].stat().st_mtime_ns, reverse=True)
    return entries, warnings


def _legacy_metrics(path, entries):
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in workbook:
            rows = sheet.iter_rows(values_only=True)
            headers = next(rows, ())
            if 'R@1' not in headers:
                continue
            rows = list(rows)
            latest = {}
            for values in rows:
                record = dict(zip(headers[:13], values[:13]))
                if record.get('array_name'):
                    latest[(record['run_name'], record['ref_query'], record['array_name'])] = record
            for (run, pair, array), metrics in latest.items():
                baseline = {'LENS': 'lens', 'sparse_event_vpr': 'sparse_event'}.get(run, run)
                matches = [e for e in entries if e['baseline'] == baseline and e['pair'] == pair
                           and e['path'].stem == array]
                # Old workbooks omit dataset/window. Never guess between multiple outputs.
                if len(matches) != 1 or matches[0]['has_manifest']:
                    continue
                entry = matches[0]
                matrix = np.load(entry['path'], mmap_mode='r', allow_pickle=False)
                if matrix.shape != (metrics['n_references'], metrics['n_queries']):
                    continue
                entry.update(metrics=metrics, source='Legacy workbook · latest saved evaluation')
                header = f'PR ({run} :: {pair}) [{array}] - Precision'
                if header in headers:
                    col = headers.index(header)
                    points = [(r[col], r[col + 1]) for r in rows
                              if isinstance(r[col], (float, int)) and isinstance(r[col + 1], (float, int))]
                    entry['precision'] = [p for p, _ in points]
                    entry['recall'] = [r for _, r in points]
    finally:
        workbook.close()


def sample_matrix(entry, width, height):
    matrix = np.load(entry['path'], mmap_mode='r', allow_pickle=False)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError('The saved matrix must be a non-empty 2D array.')
    rows = np.linspace(0, matrix.shape[0] - 1, min(height, matrix.shape[0]), dtype=int)
    cols = np.linspace(0, matrix.shape[1] - 1, min(width, matrix.shape[1]), dtype=int)
    values = np.asarray(matrix[np.ix_(rows, cols)], dtype=float)
    if entry['matrix_type'] == 'distance':
        values = -values  # Higher colour intensity always means a closer match.
    return values, matrix.shape


def preview(entry, kind, width=64, height=12):
    width, height = max(16, min(width, 100)), max(4, min(height, 24))
    text = Text()
    if kind == 'matrix':
        values, shape = sample_matrix(entry, width, height)
        finite = values[np.isfinite(values)]
        if not finite.size:
            return Text('No finite matrix values are available.')
        low, high = finite.min(), finite.max()
        palette = ['#172239', '#243e5c', '#315b7c', '#3d819a', '#4aafbf', '#56dce8', '#b3f4ee', '#ff718f']
        levels = np.clip((np.nan_to_num(values, nan=low, posinf=high, neginf=low) - low)
                         / (high - low or 1) * (len(palette) - 1), 0, len(palette) - 1).astype(int)
        text.append(f'{shape[0]:,} references × {shape[1]:,} queries · sampled preview\n', '#a5b4ce')
        for row in levels:
            for level in row:
                text.append(' ', f'on {palette[level]}')
            text.append('\n')
        text.append('Rows: reference · Columns: query · Dark → bright: closer match')
    elif kind == 'pr':
        points = [(r, p) for r, p in zip(entry['recall'], entry['precision'])
                  if r is not None and p is not None and np.isfinite(r) and np.isfinite(p)]
        if not points:
            return Text('No saved precision–recall curve is available for this matrix.')
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        for recall, precision in points:
            x = round(np.clip(recall, 0, 1) * (width - 1))
            y = round((1 - np.clip(precision, 0, 1)) * (height - 1))
            grid[y][x] = '●'
        text.append('Precision\n', 'bold')
        for y, row in enumerate(grid):
            text.append(('1.0' if y == 0 else '0.0' if y == height - 1 else '   ') + ' │', '#a5b4ce')
            text.append(''.join(row) + '\n', '#56dce8')
        text.append('    └' + '─' * width + '\n     0' + ' ' * (width - 2) + '1\n     Recall →')
    else:
        metrics = entry['metrics']
        if not metrics:
            return Text('No matching Recall@K evaluation is saved. The matrix can still be viewed.')
        text.append('Recall@K\n\n', 'bold #56dce8')
        for key in ('R@1', 'R@5', 'R@10', 'R@15', 'R@20', 'R@25'):
            value = metrics.get(key)
            if value is not None:
                bars = round(np.clip(value, 0, 1) * min(30, width - 15))
                text.append(f'{key:5}  {value:6.1%}  ', 'bold')
                text.append('━' * bars + '\n', '#56dce8')
        area = metrics.get('aupr')
        text.append(f'\nAUPR  {area:.4f}' if area is not None else '\nAUPR unavailable')
    return text


def export_plot(entry, kind):
    """Create a full-size, shareable figure using only saved artifacts."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(9, 6), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    if kind == 'matrix':
        overlay = entry['path'].with_name(entry['path'].stem + '_matches.png')
        if overlay.exists():
            return overlay
        values, shape = sample_matrix(entry, 1600, 1600)
        heatmap = axes.imshow(values, aspect='auto', extent=(0, shape[1], shape[0], 0))
        figure.colorbar(heatmap, ax=axes, label='Similarity' if entry['matrix_type'] == 'similarity' else 'Negative distance')
        axes.set(xlabel='Query', ylabel='Reference')
    elif kind == 'pr':
        if not entry['precision']:
            raise ValueError('No saved PR curve is available.')
        axes.plot(entry['recall'], entry['precision'])
        axes.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
        axes.grid(alpha=.25)
    else:
        metrics = entry['metrics']
        keys = [k for k in ('R@1', 'R@5', 'R@10', 'R@15', 'R@20', 'R@25') if metrics.get(k) is not None]
        if not keys:
            raise ValueError('No saved Recall@K results are available.')
        axes.bar(keys, [metrics[k] for k in keys], color='#289cac')
        axes.set(ylabel='Recall', ylim=(0, 1))
    axes.set_title(entry['label'], fontsize=9, wrap=True)
    folder = entry['path'].parent / 'previews'
    folder.mkdir(exist_ok=True)
    path = folder / f'{entry["path"].stem}-{kind}.png'
    figure.savefig(path, dpi=160)
    return path
