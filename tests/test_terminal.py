import asyncio
from pathlib import Path
import sys

import pytest

from utils.configuration import CONFIG, FIELDS, parse_value, read_config, update_config
from utils.experiments import Queue, expand_batch
from utils.terminal_theme import COMPLETE, MISSING, PARTIAL


def batch():
    return {'batch_experiments': [dict(dataset='brisbane_event', reference='sunset2',
                                     queries=['sunrise', 'sunset1'], baselines=['lens', 'eventvlad'])]}


def test_config_roundtrip(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(CONFIG.read_text())
    original = read_config(path)
    update_config(path, {'timewindows': [100, 200, 300], 'request_input': True})
    data = read_config(path)
    assert data['timewindows'] == [100, 200, 300]
    assert data['request_input'] is True
    assert data['datasets'] == original['datasets']
    assert '# General configuration' in path.read_text()
    saved = path.read_bytes()
    with pytest.raises(ValueError):
        update_config(path, {'timewindows': [20], 'filter_places_sec': -1})
    assert path.read_bytes() == saved


@pytest.mark.parametrize('key,value', [('timewindows', '0 20'), ('timewindows', '1.5'),
    ('request_input', 'yes'), ('gps_tolerance', 'nan'), ('filter_places_sec', '-1'),
    ('representation', 'unknown')])
def test_invalid_config(key, value):
    with pytest.raises(ValueError):
        parse_value(key, value)


def test_batch_legacy_and_current():
    document = batch()
    assert len(expand_batch(document)) == 4
    document['batch_experiments'][0]['config'] = {'timewindows': {'list': [100, 200], 'gps_tolerance': 2.5}}
    document['batch_experiments'][0]['baseline_config'] = {'lens': {'method': ['a', 'b']}}
    jobs = expand_batch(document)
    assert len(jobs) == 6
    assert jobs[0]['config'] == {'timewindows': [100, 200], 'gps_tolerance': 2.5}
    document['batch_experiments'][0]['queries'] = ['sunset2']
    with pytest.raises(ValueError):
        expand_batch(document)


def test_queue_continues_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr('utils.experiments.ROOT', tmp_path)
    queue = Queue()
    commands = [('bad', [sys.executable, '-c', 'print("bad"); raise SystemExit(2)'], {}),
                ('good', [sys.executable, '-c', 'print("good")'], {})]
    assert asyncio.run(queue.run(commands, lambda _: None)) == 1
    assert queue.statuses == ['failed', 'succeeded']
    assert 'good' in (queue.log_dir / '002.log').read_text()


def test_queue_cancels(tmp_path, monkeypatch):
    monkeypatch.setattr('utils.experiments.ROOT', tmp_path)
    async def scenario():
        queue = Queue()
        command = [sys.executable, '-c', 'import time; print("ready", flush=True); time.sleep(60)']
        task = asyncio.create_task(queue.run([('slow', command, {}), ('next', command, {})], lambda _: None))
        while queue.process is None:
            await asyncio.sleep(.01)
        await queue.cancel()
        await asyncio.wait_for(task, 5)
        assert queue.statuses == ['cancelled', 'cancelled']
    asyncio.run(scenario())


def test_effective_configs_do_not_mutate_files(tmp_path, monkeypatch):
    config = tmp_path / 'config.yaml'
    config.write_text(CONFIG.read_text())
    before = config.read_bytes()
    document = batch()
    document['batch_experiments'][0]['config'] = {'timewindows': [33]}
    queue = Queue()
    async def inspect(commands, emit):
        for _, command, env in commands:
            effective = read_config(command[-1])
            assert effective['timewindows'] == [33]
            assert effective['request_input'] is False
            assert (Path(env['EVENTLAB_BASELINE_CONFIG_DIR']) / 'lens.yaml').exists()
        return 0
    monkeypatch.setattr(queue, 'run', inspect)
    asyncio.run(queue.experiments(document, config_path=config))
    assert config.read_bytes() == before




def test_baseline_settings_preserve_types(tmp_path):
    path = tmp_path / 'baseline.yaml'
    path.write_text('device: "auto" # device help\ndims: [28, 28]\nbatch_size: 16\n')
    data = read_config(path)
    assert parse_value('device', 'cpu', data['device'], False) == 'cpu'
    update_config(path, {'device': 'cpu', 'dims': [32, 32], 'batch_size': 8}, False)
    assert read_config(path)['dims'] == [32, 32]
    assert '# device help' in path.read_text()
    with pytest.raises(ValueError):
        update_config(path, {'batch_size': 'oops'}, False)


def test_cli_updates_and_rejects_unknown(tmp_path):
    import subprocess
    from utils.configuration import ROOT
    path = tmp_path / 'config.yaml'
    path.write_text(CONFIG.read_text())
    command = [sys.executable, str(ROOT / 'eventlab_config.py'), '--config', str(path)]
    result = subprocess.run(command + ['--timewindows', '100', '200', '300', '--request_input', 'false'], capture_output=True)
    assert result.returncode == 0, result.stderr
    assert read_config(path)['timewindows'] == [100, 200, 300]
    before = path.read_bytes()
    assert subprocess.run(command + ['--time-windows', '200'], capture_output=True).returncode == 2
    assert path.read_bytes() == before


class Response:
    def __init__(self, data=b'', status=200, headers=None):
        self.data, self.status_code, self.headers = data, status, headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def raise_for_status(self):
        pass
    def iter_content(self, chunk_size):
        yield self.data


@pytest.mark.parametrize('resume', [True, False])
def test_download_resume_or_restart(tmp_path, monkeypatch, resume):
    from datasets.download_data import download_file
    path = tmp_path / 'recording.hdf5'
    Path(str(path) + '.part').write_bytes(b'abc')
    monkeypatch.setattr('datasets.download_data.check_download_size', lambda _: 6)
    def get(url, headers, **kwargs):
        assert headers['Range'] == 'bytes=3-'
        return Response(b'def' if resume else b'abcdef', 206 if resume else 200,
                        {'Content-Range': 'bytes 3-5/6'} if resume else {})
    monkeypatch.setattr('datasets.download_data.requests.get', get)
    download_file('https://example.test/data', str(path))
    assert path.read_bytes() == b'abcdef'
    assert not Path(str(path) + '.part').exists()


def test_download_unknown_size_and_failure(tmp_path, monkeypatch):
    from datasets.download_data import download_file
    path = tmp_path / 'recording.hdf5'
    monkeypatch.setattr('datasets.download_data.check_download_size', lambda _: None)
    monkeypatch.setattr('datasets.download_data.requests.get', lambda *a, **k: Response(b'abc'))
    download_file('https://example.test/data', str(path))
    assert path.read_bytes() == b'abc'
    path.unlink()
    monkeypatch.setattr('datasets.download_data.check_download_size', lambda _: 6)
    monkeypatch.setattr('datasets.download_data.time.sleep', lambda _: None)
    with pytest.raises(IOError):
        download_file('https://example.test/data', str(path))
    assert not path.exists()
    assert Path(str(path) + '.part').exists()


def test_dataset_preparation_has_no_runtime_dependency(tmp_path, monkeypatch):
    from datasets.prepare import DatasetFiles
    called = []
    monkeypatch.setattr('datasets.prepare.download_sequence_data', lambda *args: called.append('raw'))
    monkeypatch.setattr('datasets.prepare.download_file', lambda *args, **kwargs: called.append('ground_truth'))
    monkeypatch.setattr(DatasetFiles, '_resolve_zenodo_file', lambda *args: ('https://example.test/gt', None, None))
    config = dict(read_config(), data_path=str(tmp_path))
    prepared = DatasetFiles(config, 'brisbane_event', 'sunset2')
    assert called == ['raw', 'ground_truth']
    assert not hasattr(prepared, 'timewindow_list')


def test_baseline_preparation_retries_assets_without_constructor(monkeypatch):
    from utils.preparation import prepare_baseline
    calls = []
    monkeypatch.setattr('utils.preparation.ensure_repo', lambda *a, **k: calls.append(('repo', a)))
    monkeypatch.setattr('utils.preparation.drive_file', lambda *a: calls.append(('weight', a)))
    prepare_baseline('eventvlad', weights=True)
    assert [kind for kind, _ in calls] == ['repo', 'repo', 'weight', 'weight']




@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX process group cancellation')
def test_cancel_reaches_children_after_parent_exits(tmp_path, monkeypatch):
    monkeypatch.setattr('utils.experiments.ROOT', tmp_path)
    async def scenario():
        queue = Queue()
        code = ('import subprocess, sys; '
                'subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); '
                'print("parent done", flush=True)')
        task = asyncio.create_task(queue.run([('parent', [sys.executable, '-c', code], {})], lambda _: None))
        try:
            async def wait_for_exit():
                while queue.process is None or queue.process.returncode is None:
                    await asyncio.sleep(.01)
            await asyncio.wait_for(wait_for_exit(), 5)
            await queue.cancel()
            await asyncio.wait_for(task, 5)
            assert queue.statuses == ['cancelled']
        finally:
            await queue.cancel()
    asyncio.run(scenario())


def test_compact_tui_keyboard_roles_and_no_io(tmp_path, monkeypatch):
    from eventlab_tui import EventLabApp
    from utils.terminal_widgets import SelectionList
    from textual.widgets import DataTable, TabbedContent, Input
    config = tmp_path / 'config.yaml'
    config.write_text(CONFIG.read_text())
    async def scenario():
        app = EventLabApp(config)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press('ctrl+right')
            await pilot.pause()
            table = app.query_one('#fields', DataTable)
            assert app.focused is table
            assert table.row_count == len(FIELDS)
            assert table.size.height >= 15
            table.move_cursor(row=app.field_keys.index('timewindows'))
            await pilot.press('enter')
            await pilot.pause()
            app.screen.query_one(Input).value = '100 200 300'
            await pilot.press('enter', 'ctrl+s')
            await pilot.pause()
            assert read_config(config)['timewindows'] == [100, 200, 300]
            assert not app.dirty()
            await pilot.press('ctrl+right')
            await pilot.pause()
            baselines = app.query_one('#baselines', SelectionList)
            assert app.focused is baselines
            def no_reads(*args, **kwargs):
                raise AssertionError('Interaction must not read YAML')
            monkeypatch.setattr('eventlab_tui.read_config', no_reads)
            monkeypatch.setattr('utils.experiments.read_config', no_reads)
            await pilot.press('space', 'right')
            assert baselines.selected == ['lens']
            await pilot.press('space', 'right')
            await pilot.pause()
            roles = app.query_one('#trajectories', DataTable)
            roles.focus()
            await pilot.press('r', 'down', 'q')
            await pilot.pause()
            name = app.active_dataset
            assert app.sequence_choices[name] == (app.datasets[name][0], [app.datasets[name][1]])
            await pilot.press('down', 'r')
            await pilot.pause()
            assert app.sequence_choices[name][0] == app.datasets[name][2]
            assert roles.get_cell_at((0, 1)) == ''
            await pilot.press('delete')
            assert app.sequence_choices[name][0] is None
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert roles.size.height > 0
    asyncio.run(scenario())


def test_gardens_camera_clock():
    from datasets.gardens_point import camera_seconds
    sync = {'anchor': {'delta_ns': 1_000_000_000},
            'camera': {'alpha': 1.001, 'beta_prime_ns': 2_000_000_000}}
    # The first camera event is 100 ms; GPS must retain that offset.
    assert camera_seconds([3_100_100_000, 4_101_100_000], sync, 100_000) == pytest.approx([0, 1])


def test_gardens_catalog_and_urls():
    from utils.experiments import catalog
    data = read_config(CONFIG.parent / 'datasets/gardens_point_event.yaml')
    assert catalog()[1]['gardens_point_event'] == ['left', 'right']
    for name, sequence in data['sequences'].items():
        assert sequence['data']['url'].endswith(f'{name}/events.h5')
        assert sequence['ground_truth']['sync_url'].endswith(f'{name}/derived/sync.json')


def test_gardens_pose_fixture(tmp_path):
    import h5py
    import json
    import numpy as np
    from datasets.gardens_point import load_pose
    with h5py.File(tmp_path / 'left.hdf5', 'w') as f:
        f['events/t'] = [100000, 2100000]
    (tmp_path / 'sync.json').write_text(json.dumps({'anchor': {'delta_ns': 1000000000},
        'camera': {'alpha': 1., 'beta_prime_ns': 2000000000}}))
    pose_file = tmp_path / 'left_ground_truth.csv'
    pose_file.write_text('time_laptop_ns,lat,lon,alt,in_gap\n3100000000,-27,153,0,0\n4100000000,-27.1,153,0,0\n5100000000,-27.2,153,0,0\n')
    pose, kind, centers = load_pose(pose_file, 1000)
    assert kind == 'geo'
    assert pose[:, 2] == pytest.approx([0, 1, 2])
    assert centers == pytest.approx([.5, 1.5])
    pose_file.write_text('time_laptop_ns,lat,lon,alt,in_gap\n3100000000,-27,153,0,0\n4100000000,-27.1,153,0,0\n')
    with pytest.warns(UserWarning, match='does not cover'):
        load_pose(pose_file, 1000)


def test_compact_mouse_batch_roundtrip(tmp_path):
    from eventlab_tui import EventLabApp
    from utils.terminal_widgets import SelectionList
    from textual.widgets import TabbedContent, Input, DataTable
    async def scenario():
        app = EventLabApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = 'experiments'
            await pilot.pause()
            await pilot.click('#baselines', offset=(4, 0))
            await pilot.click('#datasets', offset=(4, 0))
            await pilot.click('#trajectories', offset=(4, 1))
            table = app.query_one('#trajectories', DataTable)
            table.move_cursor(row=1)
            await pilot.press('q')
            await pilot.pause()
            assert app.document()['batch_experiments'][0]['reference'] == 'sunset1'
            app.query_one('#batch-path', Input).value = str(tmp_path / 'batch.yaml')
            await pilot.click('#save-batch')
            await pilot.pause()
            saved = app.document()
            assert (tmp_path / 'batch.yaml').exists()
            await pilot.click('#new-batch')
            await pilot.click('#load-batch')
            await pilot.pause()
            assert app.document() == saved
    asyncio.run(scenario())


def test_progress_stream_replaces_bars_and_keeps_log_lines():
    from utils.terminal_output import TerminalOutput
    output = TerminalOutput()
    assert output.feed('starting\n\rDownload: 1') == ['starting']
    assert output.progress == 'Download: 1'
    assert output.feed('0%\rDownload: 20%\x1b[') == []
    assert output.progress == 'Download: 20%'
    assert output.feed('K\rDownload: 100%\nfinished\n') == ['Download: 100%', 'finished']
    assert output.progress == ''
    assert output.feed('Windows\r\nlines\r\n') == ['Windows', 'lines']
    assert output.feed('partial') == []
    assert output.finish() == ['partial']
    assert output.feed('\rTask: 1%\rTask: 90%') == []
    assert output.finish() == ['Task: 90%']


def test_run_summary_is_compact_and_explains_overrides():
    from eventlab_tui import run_summary
    jobs = expand_batch(batch())
    text = run_summary(jobs, read_config())
    assert text.count('brisbane_event') == 1
    assert '4 experiments' in text
    assert 'sunset2 → sunrise, sunset1' in text
    assert 'lens, eventvlad' in text
    assert 'data_path' not in text and 'Runtime configuration' not in text
    assert len(text.splitlines()) < 12
    jobs[0]['config'] = {'timewindows': [33]}
    assert 'overrides apply to 1' in run_summary(jobs, read_config())


def test_progress_display_and_compact_review():
    from eventlab_tui import EventLabApp, Confirm, run_summary
    from textual.widgets import Static
    async def scenario():
        app = EventLabApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            for i in range(100):
                app.append_output(f'\rDownload: {i}%')
            app.flush_logs()
            assert app.terminal_output.progress == 'Download: 99%'
            assert app.log_buffer == []
            app.append_output('\rDownload: 100%\n')
            app.append_log('[1/1] SUCCEEDED download')
            assert app.log_buffer == ['Download: 100%', '[1/1] SUCCEEDED download']
            app.push_screen(Confirm(run_summary(expand_batch(batch()), read_config()), 'Run 4 experiments'))
            await pilot.pause()
            assert app.screen.query_one('#yes').region.bottom <= 24
            assert app.screen.query_one('#dialog').size.height < 24
    asyncio.run(scenario())


def test_home_navigation_and_panel_arrows_never_enter_actions():
    from eventlab_tui import EventLabApp
    from textual.widgets import TabbedContent, Input
    async def scenario():
        app = EventLabApp()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app.focused.id == 'home-menu'
            assert not app.query_one('ContentTabs').display
            assert not app.query_one('#topbar').display
            await pilot.press('enter')
            await pilot.pause()
            assert app.query_one(TabbedContent).active == 'experiments'
            assert app.focused.id == 'baselines'
            for _ in range(4):
                for expected in ('datasets', 'trajectories', 'baselines'):
                    await pilot.press('right')
                    assert app.focused.id == expected
            for expected in ('trajectories', 'datasets', 'baselines'):
                await pilot.press('left')
                assert app.focused.id == expected
            assert not app.query_one('#batch-path').display
            await pilot.click('#choose-batch')
            await pilot.pause()
            field = app.screen.query_one(Input)
            field.value = 'example.yaml'
            await pilot.press('left')
            assert app.focused is field
            await pilot.press('enter')
            await pilot.pause()
            assert app.query_one('#batch-path', Input).value == 'example.yaml'
            await pilot.press('escape')
            assert app.query_one(TabbedContent).active == 'home'
            await pilot.press('2')
            await pilot.pause()
            assert app.query_one(TabbedContent).active == 'downloads'
            for expected in ('download-datasets', 'download-sequences', 'download-baselines'):
                await pilot.press('right')
                assert app.focused.id == expected
            await pilot.click('#home-back')
            await pilot.pause()
            await pilot.resize_terminal(80,24)
            assert app.focused.id == 'home-menu'
            await pilot.press('3')
            await pilot.pause()
            assert app.query_one(TabbedContent).active == 'configuration'
            for key in ('left', 'right', 'right'):
                await pilot.press(key)
                assert app.focused.id == 'fields'
    asyncio.run(scenario())


def test_review_enter_runs_once_and_back_cancels(monkeypatch):
    from eventlab_tui import EventLabApp, Confirm
    from textual.widgets import Button
    async def scenario():
        app = EventLabApp()
        started = []
        monkeypatch.setattr(app, 'run_experiments', lambda document: started.append(document))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press('enter', 'space', 'right', 'space', 'right', 'r', 'down', 'q')
            await pilot.click('#run')
            await pilot.pause()
            assert isinstance(app.screen, Confirm)
            assert app.focused.id == 'yes'
            assert str(app.screen.query_one('#yes', Button).label) == 'Run 1 experiment'
            await pilot.press('enter')
            await pilot.pause()
            assert len(started) == 1
            assert started[0]['batch_experiments'][0]['queries']
            assert not isinstance(app.screen, Confirm)
            await pilot.press('f5')
            await pilot.pause()
            await pilot.press('tab')
            assert app.focused.id == 'no'
            await pilot.press('enter')
            assert not isinstance(app.screen, Confirm)
            assert len(started) == 1
            await pilot.press('f5')
            await pilot.pause()
            await pilot.press('escape')
            assert not isinstance(app.screen, Confirm)
            assert len(started) == 1
    asyncio.run(scenario())


def test_home_dashboard_animation_and_compact_layout():
    from eventlab_tui import EventLabApp
    from utils.terminal_widgets import EventLogo
    async def scenario():
        app = EventLabApp()
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.pause()
            welcome = app.query_one('#welcome')
            viewport = app.query_one('#home-scroll')
            assert abs(welcome.region.center[0] - viewport.region.center[0]) <= 1
            assert abs(welcome.region.center[1] - viewport.region.center[1]) <= 1
            logo = app.query_one(EventLogo)
            await pilot.press('enter')
            await pilot.pause()
            phase = logo.phase
            await pilot.pause(.5)
            assert logo.phase == phase
            await pilot.press('escape')
            await pilot.pause(.5)
            assert logo.phase != phase
            assert '█████' in str(logo.render())
            assert 'Localization' in str(app.query_one('#intro').render())
            assert not list(app.query('#dashboard-summary'))
            assert app.focused.id == 'home-menu'
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert app.query_one('#brand').region.y >= 0
            # The taller two-line wordmark makes small terminals scroll naturally.
            app.query_one('#home-scroll').scroll_end(animate=False)
            await pilot.pause()
            assert app.query_one('#credits').region.bottom <= 23
            menu = app.query_one('#home-menu')
            assert menu.virtual_size.height <= menu.scrollable_content_region.height
    asyncio.run(scenario())


def test_space_roles_preserve_reference_until_explicitly_reassigned():
    from eventlab_tui import EventLabApp
    async def scenario():
        app = EventLabApp()
        async with app.run_test() as pilot:
            await pilot.press('enter', 'right', 'space', 'right', 'space')
            dataset = app.active_dataset
            first, second = app.datasets[dataset][:2]
            assert app.sequence_choices[dataset] == (first, [])
            await pilot.press('down', 'space')
            assert app.sequence_choices[dataset] == (first, [second])
            await pilot.press('space')
            assert app.sequence_choices[dataset] == (second, [])
            await pilot.press('space')
            assert app.sequence_choices[dataset] == (None, [])
    asyncio.run(scenario())


@pytest.mark.parametrize('review', ['shortcut', 'mouse', 'keyboard'])
@pytest.mark.parametrize('confirm', ['keyboard', 'mouse'])
def test_confirmed_experiment_switches_to_activity(monkeypatch, review, confirm):
    from eventlab_tui import EventLabApp
    from textual.widgets import TabbedContent, ContentSwitcher
    async def scenario():
        app = EventLabApp()
        release = asyncio.Event()
        async def fake_queue(*args):
            app.append_log('[1/1] RUNNING example')
            await release.wait()
        monkeypatch.setattr(app.queue, 'experiments', fake_queue)
        async with app.run_test() as pilot:
            await pilot.press('enter', 'space', 'right', 'space', 'right', 'r', 'down', 'q')
            if review == 'shortcut':
                await pilot.press('f5')
            elif review == 'mouse':
                await pilot.click('#run')
            else:
                app.query_one('#run').focus()
                await pilot.press('enter')
            await pilot.pause()
            if confirm == 'mouse':
                await pilot.click('#yes')
            else:
                await pilot.press('enter')
            await pilot.pause()
            assert app.running
            assert app.query_one(TabbedContent).active == 'logs'
            assert app.query_one(ContentSwitcher).current == 'logs'
            assert app.focused.id == 'log'
            release.set()
            await app.workers.wait_for_complete()
            assert app.query_one(TabbedContent).active == 'logs'
            assert not app.running
    asyncio.run(scenario())


def test_tui_banner_is_compact(tmp_path, monkeypatch, capsys):
    from utils.utils import start
    from loguru import logger
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('EVENTLAB_TUI', '1')
    try:
        start('lens', 'brisbane_event')
        output = capsys.readouterr().out
        assert 'EventLAB - Localization · lens · brisbane_event' in output
        assert '█' not in output
    finally:
        logger.remove()


def test_dataset_download_detection_is_local_and_requires_all_assets(tmp_path, monkeypatch):
    from datasets.prepare import sequence_download_status
    def no_network(*args, **kwargs):
        pytest.fail('Download detection must not use the network')
    monkeypatch.setattr('requests.get', no_network)
    monkeypatch.setattr('requests.head', no_network)
    config = dict(read_config(), data_path=str(tmp_path))
    metadata = read_config(CONFIG.parent / 'datasets/gardens_point_event.yaml')
    status = lambda: sequence_download_status(config, metadata, 'gardens_point_event', 'left')
    folder = tmp_path / 'gardens_point_event/left'
    assert status() == 'Missing'
    assert not folder.exists()
    folder.mkdir(parents=True)
    (folder / 'left.hdf5.part').write_bytes(b'partial')
    assert status() == 'Partial'
    (folder / 'left.hdf5').write_bytes(b'recording')
    (folder / 'left_ground_truth.csv').touch()
    assert status() == 'Partial'
    (folder / 'left_ground_truth.csv').write_text('ground truth')
    assert status() == 'Partial'  # Required clock synchronisation is still missing.
    (folder / 'sync.json').write_text('{}')
    assert status() == 'Downloaded'
    (folder / 'left.hdf5').write_bytes(b'')
    assert status() == 'Partial'
    metadata['sequences']['left']['data']['url'] = ''
    assert status() == 'Local only'
    (folder / 'left.hdf5').write_bytes(b'recording')
    assert status() == 'Downloaded'
    metadata['sequences']['left']['ground_truth'] = {'available': False}
    (folder / 'left_ground_truth.csv').unlink()
    (folder / 'sync.json').unlink()
    assert status() == 'Downloaded'


def test_empty_download_is_replaced_even_without_content_length(tmp_path, monkeypatch):
    from datasets.download_data import download_file
    path = tmp_path / 'data.hdf5'
    path.touch()
    monkeypatch.setattr('datasets.download_data.check_download_size', lambda _: None)
    monkeypatch.setattr('datasets.download_data.requests.get', lambda *a, **k: Response(b'complete'))
    download_file('https://example.test/data', str(path))
    assert path.read_bytes() == b'complete'


def test_download_groups_and_config_changes(tmp_path):
    from eventlab_tui import EventLabApp
    from utils.configuration import write_config
    from utils.terminal_widgets import SelectionList
    config_path = tmp_path / 'config.yaml'
    config = read_config()
    config['data_path'] = str(tmp_path / 'empty')
    write_config(config_path, config)
    async def scenario():
        app = EventLabApp(config_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press('2')
            await app.workers.wait_for_complete()
            assert not app.download_selection
            await pilot.press('right', 'space', 'right')
            assert app.focused.id == 'download-sequences'
            assert app.download_selection == {('brisbane_event', s) for s in app.datasets['brisbane_event']}
            await pilot.press('space')
            assert len(app.download_selection) == len(app.datasets['brisbane_event']) - 1
            await pilot.press('left', 'down', 'space', 'right')
            assert app.download_dataset == 'fast_slow'
            assert all(('fast_slow', s) in app.download_selection for s in app.datasets['fast_slow'])
            assert len(app.query_one('#download-sequences', SelectionList).items) == len(app.datasets['fast_slow'])
            config['data_path'] = str(tmp_path / 'new-data')
            folder = Path(config['data_path']) / 'qcr_event/normal1'
            folder.mkdir(parents=True)
            (folder / 'normal1.hdf5').write_bytes(b'recording')
            write_config(config_path, config)
            await pilot.pause(1.2)
            await app.workers.wait_for_complete()
            assert app.download_root == config['data_path']
            assert app.download_states[('qcr_event', 'normal1')] == 'Downloaded'
            assert ('qcr_event', 'normal2') not in app.download_selection
            await pilot.click('#check-downloads')
            await app.workers.wait_for_complete()
            assert ('qcr_event', 'normal2') not in app.download_selection
    asyncio.run(scenario())


def test_bulk_selection_buttons_keyboard_and_roles(tmp_path):
    from eventlab_tui import EventLabApp
    from utils.terminal_widgets import SelectionList
    from utils.configuration import write_config
    from textual.widgets import Button, DataTable, Input
    async def scenario():
        app = EventLabApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press('enter')
            for name in ('baselines', 'datasets'):
                listing = app.query_one('#' + name, SelectionList)
                await pilot.click('#all-' + name)
                assert listing.selected == [item[1] for item in listing.items]
                await pilot.pause()
                assert str(app.query_one('#all-' + name, Button).label) == 'Unselect all'
                listing.focus()
                await pilot.press('a')
                assert listing.selected == []
            table = app.query_one('#trajectories', DataTable)
            table.focus()
            await pilot.press('down', 'r', 'a')
            sequences = app.datasets[app.active_dataset]
            assert app.sequence_choices[app.active_dataset] == (sequences[1], [s for s in sequences if s != sequences[1]])
            await pilot.click('#all-trajectories')
            assert app.sequence_choices[app.active_dataset] == (None, [])
            await pilot.pause(.3)
            await pilot.click('#all-trajectories')
            assert app.sequence_choices[app.active_dataset] == (sequences[0], sequences[1:])
            # Bulk controls cannot edit imported batches containing per-run overrides.
            document = batch()
            document['batch_experiments'][0]['config'] = {'timewindows': [20]}
            path = tmp_path / 'batch.yaml'
            write_config(path, document)
            app.query_one('#batch-path', Input).value = str(path)
            await pilot.click('#load-batch')
            await pilot.pause()
            assert all(button.disabled for button in app.query('#experiments .selection-all'))
            app.action_toggle_all_roles()
            assert app.document() == document
            await pilot.click('#new-batch')
            assert all(not button.disabled for button in app.query('#experiments .selection-all'))
            await pilot.press('escape', '2')
            await app.workers.wait_for_complete()
            for name in ('download-baselines', 'download-sequences'):
                listing = app.query_one('#' + name, SelectionList)
                listing.deselect_all()
                await pilot.click('#all-' + name)
                assert len(listing.selected) == len(listing.items)
                listing.focus()
                await pilot.press('a')
                assert not listing.selected
    asyncio.run(scenario())


def test_preparation_retries_empty_ground_truth(tmp_path, monkeypatch):
    from datasets.prepare import DatasetFiles
    folder = tmp_path / 'gardens_point_event/left'
    folder.mkdir(parents=True)
    (folder / 'left.hdf5').write_bytes(b'complete recording')
    (folder / 'sync.json').write_text('{}')
    ground_truth = folder / 'left_ground_truth.csv'
    ground_truth.touch()
    monkeypatch.setattr('datasets.download_data.check_download_size', lambda _: None)
    monkeypatch.setattr('datasets.download_data.requests.get', lambda *a, **k: Response(b'complete ground truth'))
    DatasetFiles(dict(read_config(), data_path=str(tmp_path)), 'gardens_point_event', 'left')
    assert ground_truth.read_bytes() == b'complete ground truth'


def test_dataset_folder_enter_saves_and_scans_without_save_button(tmp_path):
    from eventlab_tui import EventLabApp
    from textual.widgets import Input, DataTable
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(CONFIG.read_text())
    folder = tmp_path / 'new-data/qcr_event/normal1'
    folder.mkdir(parents=True)
    (folder / 'normal1.hdf5').write_bytes(b'recording')
    async def scenario():
        app = EventLabApp(config_path)
        async with app.run_test() as pilot:
            await pilot.press('3')
            await pilot.pause()
            # Saving the folder must not silently commit other draft edits.
            previous_stream = read_config(config_path)['request_input']
            app.pending['request_input'] = not previous_stream
            app.query_one('#fields', DataTable).move_cursor(row=app.field_keys.index('data_path'))
            await pilot.press('enter')
            await pilot.pause()
            app.screen.query_one('#value-input', Input).value = str(tmp_path / 'new-data')
            await pilot.press('enter')
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert read_config(config_path)['data_path'] == str(tmp_path / 'new-data')
            assert read_config(config_path)['request_input'] == previous_stream
            assert app.pending['request_input'] != previous_stream
            await pilot.press('escape', '2')
            await app.workers.wait_for_complete()
            assert app.download_root == str(tmp_path / 'new-data')
            assert app.download_states[('qcr_event', 'normal1')] == 'Downloaded'
            assert not app.download_selection
    asyncio.run(scenario())


@pytest.mark.parametrize('available,color', [(0, MISSING), (7, PARTIAL), (9, COMPLETE)])
def test_dataset_counts_show_availability_independent_of_selection(available, color):
    from eventlab_tui import EventLabApp
    from utils.terminal_widgets import SelectionList
    async def scenario():
        app = EventLabApp()
        async with app.run_test() as pilot:
            sequences = app.datasets['nsavp']
            assert len(sequences) == 9
            app.download_states = {('nsavp', seq): 'Downloaded' if i < available else 'Missing'
                                   for i, seq in enumerate(sequences)}
            for selected in (set(), {('nsavp', seq) for seq in sequences}):
                app.download_selection = selected
                app.render_download_groups()
                listing = app.query_one('#download-datasets', SelectionList)
                label = next(label for label, name in listing.items if name == 'nsavp')
                assert label.plain == f'nsavp ({available}/9)'
                assert label.style == color
                assert ('nsavp' in listing.selected) == bool(selected)
            app.download_states = {('nsavp', sequences[0]): 'Partial'}
            app.render_download_groups()
            label = next(label for label, name in listing.items if name == 'nsavp')
            assert label.plain == 'nsavp (0/9)'
            assert label.style == PARTIAL
            app.download_dataset = 'nsavp'
            app.render_download_recordings()
            assert app.query_one('#download-sequences', SelectionList).items[0][0].style == PARTIAL
    asyncio.run(scenario())


def test_baseline_availability_colors_do_not_select_downloads(monkeypatch):
    from eventlab_tui import EventLabApp
    from utils.terminal_widgets import SelectionList
    monkeypatch.setattr('utils.preparation.baseline_availability',
                        lambda name: (2, 2, []) if name == 'lens' else (1, 2, ['weights']))
    async def scenario():
        app = EventLabApp()
        async with app.run_test() as pilot:
            await pilot.press('2')
            await app.workers.wait_for_complete()
            listing = app.query_one('#download-baselines', SelectionList)
            assert not listing.selected
            labels = {name: label for label, name in listing.items}
            assert labels['lens'].plain == 'lens (2/2)'
            assert labels['lens'].style == COMPLETE
            assert labels['eventgem'].style == PARTIAL
            await pilot.press('space')
            assert listing.selected == ['lens']
            assert listing.items[0][0].plain == 'lens (2/2)'
    asyncio.run(scenario())


@pytest.mark.parametrize('key', ['std_format', 'max_frames', 'chunk_size', 'stream', 'num_events',
                                      'frame_generator', 'frame_accumulator', 'reconstruction_model',
                                      'ground_truth_tolerance'])
def test_retired_runtime_switches_are_not_editable(tmp_path, key):
    from utils.configuration import FIELDS
    assert key not in FIELDS
    assert key not in read_config()
    path = tmp_path / 'config.yaml'
    path.write_text(CONFIG.read_text())
    before = path.read_bytes()
    with pytest.raises(ValueError):
        update_config(path, {key: 'unused'})
    assert path.read_bytes() == before


def test_representation_dropdown_is_visible_and_opens_with_enter(tmp_path, monkeypatch):
    from eventlab_tui import EventLabApp
    from textual.widgets import DataTable, Select, Input, OptionList, Link
    config = tmp_path / 'config.yaml'
    config.write_text(CONFIG.read_text())
    async def scenario():
        app = EventLabApp(config)
        opened = []
        monkeypatch.setattr(app, 'open_url', opened.append)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press('ctrl+right')
            await pilot.pause()
            table = app.query_one('#fields', DataTable)
            row = app.field_keys.index('representation')
            assert table.get_cell_at((row, 1)).endswith(' ▾')
            assert 'Dropdown' not in table.get_cell_at((row, 1))
            table.move_cursor(row=row)
            await pilot.press('enter')
            await pilot.pause()
            dropdown = app.screen.query_one('#value-input', Select)
            assert dropdown.expanded
            option_list = dropdown.query_one(OptionList)
            assert app.field_choices('representation') == ('Events', 'Reconstruction')
            assert not option_list.get_option_at_index(1).disabled
            await pilot.press('end', 'enter')
            await pilot.pause()
            assert app.pending['representation'] == 'Reconstruction'
            await pilot.press('ctrl+s')
            assert read_config(config)['representation'] == 'Reconstruction'
            assert 'representation_options' not in app.field_keys
    asyncio.run(scenario())


def test_single_key_shortcuts_and_roles_select_dataset(tmp_path):
    from eventlab_tui import EventLabApp
    from utils.terminal_widgets import SelectionList
    from textual.widgets import DataTable, TabbedContent
    config = tmp_path / 'config.yaml'
    config.write_text(CONFIG.read_text())
    async def scenario():
        app = EventLabApp(config)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press('1')
            await pilot.pause()
            datasets = app.query_one('#datasets', SelectionList)
            roles = app.query_one('#trajectories', DataTable)
            assert not datasets.selected
            roles.focus()
            await pilot.press('r', 'down', 'q')
            await pilot.pause()
            assert datasets.selected == [app.active_dataset]
            ref, queries = app.sequence_choices[app.active_dataset]
            assert ref and queries
            datasets.deselect_all()
            await pilot.press('space')
            await pilot.pause()
            assert datasets.selected == [app.active_dataset]
            await pilot.press('h', '3')
            await pilot.pause()
            assert app.query_one('#tabs', TabbedContent).active == 'configuration'
            app.pending['gps_tolerance'] = 12.5
            await pilot.press('s')
            assert read_config(config)['gps_tolerance'] == 12.5
            await pilot.press('h', 'x')
            assert app.return_value is None
            assert not app.is_running
    asyncio.run(scenario())
