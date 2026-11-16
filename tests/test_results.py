import asyncio

import numpy as np
from openpyxl import Workbook

from utils.results import discover_results, export_plot, preview, save_details


def result_fixture(root, window='frames_50'):
    folder = root / 'megaevent/gardens_point_event/left_right' / window
    folder.mkdir(parents=True)
    np.save(folder / 'similarity_matrix.npy', np.eye(5))
    row = dict(timestamp_utc='2026-11-16T03:00:00', run_name='megaevent', ref_query='left_right',
               array_name='similarity_matrix', n_references=5, n_queries=5,
               **{'R@1': np.float64(.8), 'R@5': 1., 'R@10': 1., 'R@15': 1., 'R@20': 1., 'R@25': 1., 'aupr': .9})
    curves = {('left_right', 'similarity_matrix'): (np.array([1., .8, .2]), np.array([0., .5, 1.]))}
    return folder, row, curves


def test_saved_details_previews_and_export(tmp_path):
    folder, row, curves = result_fixture(tmp_path)
    save_details(folder, [row], curves, 'similarity')
    entries, warnings = discover_results(tmp_path)
    assert len(entries) == 1 and not warnings
    entry = entries[0]
    assert entry['metrics']['R@1'] == .8
    assert '80.0%' in preview(entry, 'recall').plain
    assert 'Precision' in preview(entry, 'pr').plain
    assert '5 references × 5 queries' in preview(entry, 'matrix').plain
    for kind in ('recall', 'pr', 'matrix'):
        assert export_plot(entry, kind).read_bytes().startswith(b'\x89PNG')
    # A new, unevaluated matrix must not inherit the previous run's metrics.
    np.save(folder / 'similarity_matrix.npy', np.ones((3, 3)))
    entries, _ = discover_results(tmp_path)
    assert not entries[0]['metrics']


def test_legacy_workbook_never_guesses_between_windows(tmp_path):
    folder, row, curves = result_fixture(tmp_path)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'megaevent'
    prefix = 'PR (megaevent :: left_right) [similarity_matrix] - '
    sheet.append(list(row) + [None, prefix + 'Precision', prefix + 'Recall'])
    sheet.append(list(row.values()) + [None, 1., 0.])
    sheet.append([None] * 14 + [.8, .5])
    workbook.save(tmp_path / 'eventlab_results.xlsx')
    entries, warnings = discover_results(tmp_path)
    assert not warnings
    assert entries[0]['metrics']['R@1'] == .8
    assert entries[0]['precision'] == [1., .8]
    result_fixture(tmp_path, 'frames_100')
    entries, _ = discover_results(tmp_path)
    assert all(not e['metrics'] for e in entries)
    # Exact metadata remains usable even with ambiguous historical workbook rows.
    save_details(folder, [row], curves, 'similarity')
    entries, _ = discover_results(tmp_path)
    assert sum(bool(e['metrics']) for e in entries) == 1


def test_missing_and_invalid_results(tmp_path):
    assert discover_results(tmp_path) == ([], [])
    folder, _, _ = result_fixture(tmp_path)
    (folder / 'results.json').write_text('{')
    entries, warnings = discover_results(tmp_path)
    assert len(entries) == 1 and warnings
    assert 'No matching' in preview(entries[0], 'recall').plain
    assert 'No saved' in preview(entries[0], 'pr').plain


def test_results_menu_keyboard_mouse_and_refresh(tmp_path, monkeypatch):
    from eventlab_tui import EventLabApp
    from utils.terminal_results import ResultsBrowser
    from textual.widgets import Select, Static, TabbedContent
    folder, row, curves = result_fixture(tmp_path)
    save_details(folder, [row], curves, 'similarity')
    async def scenario():
        app = EventLabApp()
        opened = []
        monkeypatch.setattr(app, 'open_url', opened.append)
        async with app.run_test(size=(80, 24)) as pilot:
            browser = app.query_one(ResultsBrowser)
            browser.root = tmp_path
            await pilot.press('5')
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert app.query_one(TabbedContent).active == 'results'
            assert app.focused.id == 'result-list'
            assert '80.0%' in str(app.query_one('#result-plot', Static).render())
            await pilot.press('enter')
            assert app.focused.id == 'result-preview'
            app.query_one('#result-kind', Select).value = 'matrix'
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert '5 references' in str(app.query_one('#result-plot', Static).render())
            app.query_one('#result-kind', Select).value = 'pr'
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.click('#open-result')
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert opened[0].endswith('similarity_matrix-pr.png')
            assert app.query_one('#result-preview').region.bottom <= 23
            await pilot.click('#refresh-results')
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert len(browser.entries) == 1
            await pilot.press('escape')
            assert app.query_one(TabbedContent).active == 'home'
    asyncio.run(scenario())
