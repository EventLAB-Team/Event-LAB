"""Experiment selection, expansion, and subprocess execution (no ML imports)."""
import asyncio
import codecs
import copy
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from utils.configuration import ROOT, CONFIG, read_config, write_config, parse_value, dumps


def catalog(config=None):
    config = config or read_config()
    baselines = [name for name in config['VPR-Baselines']
                 if (ROOT / 'baselines' / f'{name}.py').is_file()]
    datasets = {}
    for path in sorted((ROOT / 'datasets').glob('*.yaml')):
        if 'template' in path.stem:
            continue
        data = read_config(path)
        if 'dataset' in data and 'sequences' in data:
            datasets[path.stem] = list(data['sequences'])
    return baselines, datasets


def expand_batch(document, config=None):
    baselines, datasets = catalog(config)
    jobs = []
    entries = document.get('batch_experiments', [])
    if not entries:
        raise ValueError('Select at least one experiment.')
    for entry in entries:
        dataset = entry['dataset']
        reference = entry['reference']
        if dataset not in datasets or reference not in datasets[dataset]:
            raise ValueError(f'Unknown dataset/reference: {dataset}/{reference}')
        if not entry.get('queries') or not entry.get('baselines'):
            raise ValueError(f'{dataset}: select baselines and query sequences.')
        overrides = copy.deepcopy(entry.get('config') or {})
        tw = overrides.get('timewindows')
        if isinstance(tw, dict):
            overrides.pop('timewindows')
            if tw.get('list'):
                overrides['timewindows'] = tw['list']
            if set(tw) - {'list', 'gps_tolerance'}:
                raise ValueError('timewindows: use a list of durations in milliseconds')
            if 'gps_tolerance' in tw:
                overrides.setdefault('gps_tolerance', tw['gps_tolerance'])
        for key, value in overrides.items():
            parse_value(key, value if isinstance(value, str) else dumps(value).strip().removesuffix('\n...'))
        baseline_overrides = entry.get('baseline_config') or {}
        if isinstance(baseline_overrides, list):
            baseline_overrides = {k: v for item in baseline_overrides for k, v in item.items()}
        for query in entry['queries']:
            if query == reference or query not in datasets[dataset]:
                raise ValueError(f'{dataset}: invalid query {query!r}')
            for baseline in entry['baselines']:
                if baseline not in baselines:
                    raise ValueError(f'Unknown baseline: {baseline}')
                settings = baseline_overrides.get(baseline) or {}
                methods = settings.get('method')
                variants = [dict(settings, method=m) for m in methods] if isinstance(methods, list) else [settings]
                for variant in variants:
                    jobs.append(dict(baseline=baseline, dataset=dataset, reference=reference,
                                     query=query, config=overrides, baseline_config=variant))
    if not jobs:
        raise ValueError('No experiments selected.')
    return jobs


class Queue:
    def __init__(self, output=None):
        self.output = output
        self.process = None
        self.cancelled = False
        self.statuses = []
        self.log_dir = None

    async def cancel(self):
        self.cancelled = True
        process = self.process
        if process is None or (os.name == 'nt' and process.returncode is not None):
            return
        if os.name == 'nt':
            await asyncio.to_thread(subprocess.run, ['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            # The parent may have exited before a child that ignored SIGTERM.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await process.wait()

    async def run(self, commands, emit=print):
        self.cancelled = False
        self.statuses = ['pending'] * len(commands)
        self.log_dir = ROOT / 'logs' / ('terminal-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        self.log_dir.mkdir(parents=True)
        emit(f'Logs: {self.log_dir}\nResults: {ROOT / "output"}')
        for index, (label, command, env) in enumerate(commands):
            if self.cancelled:
                self.statuses[index] = 'cancelled'
                continue
            self.statuses[index] = 'running'
            emit(f'[{index + 1}/{len(commands)}] RUNNING {label}')
            kwargs = {'start_new_session': True} if os.name != 'nt' else {
                'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
            with (self.log_dir / f'{index + 1:03d}.log').open('w') as logfile:
                try:
                    self.process = await asyncio.create_subprocess_exec(
                        *command, cwd=ROOT, env={**os.environ, **env, 'PYTHONUNBUFFERED': '1',
                                               **({'EVENTLAB_TUI': '1'} if self.output else {})},
                        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT, **kwargs)
                    if self.cancelled:
                        await self.cancel()
                    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                    while True:
                        chunk = await self.process.stdout.read(4096)
                        if not chunk:
                            break
                        text = decoder.decode(chunk)
                        logfile.write(text)
                        logfile.flush()
                        if self.output is not None:
                            self.output(text)
                        else:
                            emit(text.rstrip())
                    tail = decoder.decode(b'', final=True)
                    if tail:
                        logfile.write(tail)
                        (self.output or emit)(tail)
                    code = await self.process.wait()
                    self.statuses[index] = 'cancelled' if self.cancelled else 'succeeded' if code == 0 else 'failed'
                except asyncio.CancelledError:
                    await self.cancel()
                    self.statuses[index] = 'cancelled'
                    raise
                except (OSError, ValueError) as error:
                    logfile.write(str(error))
                    emit(str(error))
                    self.statuses[index] = 'failed'
                finally:
                    self.process = None
            emit(f'[{index + 1}/{len(commands)}] {self.statuses[index].upper()} {label}')
        emit('Queue finished: ' + ', '.join(f'{s}: {self.statuses.count(s)}' for s in dict.fromkeys(self.statuses)))
        return 1 if any(s != 'succeeded' for s in self.statuses) else 0

    async def experiments(self, document, emit=print, config_path=CONFIG):
        config = read_config(config_path)
        jobs = expand_batch(document, config)
        # Snapshot current settings at queue start; user configuration is never rewritten.
        with tempfile.TemporaryDirectory(prefix='eventlab-') as directory:
            directory = Path(directory)
            baseline_names, _ = catalog(config)
            baseline_settings = {name: read_config(ROOT / 'baselines' / f'{name}.yaml') for name in baseline_names}
            commands = []
            for index, job in enumerate(jobs):
                folder = directory / str(index)
                folder.mkdir()
                effective = copy.deepcopy(config)
                effective.update(job['config'])
                effective['request_input'] = False
                write_config(folder / 'config.yaml', effective)
                for name, settings in baseline_settings.items():
                    settings = copy.deepcopy(settings)
                    if name == job['baseline']:
                        settings.update(job['baseline_config'])
                    write_config(folder / f'{name}.yaml', settings)
                label = f'{job["baseline"]} · {job["dataset"]} · {job["reference"]} → {job["query"]}'
                if 'method' in job['baseline_config']:
                    label += f' · {job["baseline_config"]["method"]}'
                command = [sys.executable, '-u', str(ROOT / 'eventlab_run.py'),
                           job['baseline'], job['dataset'], job['reference'], job['query'],
                           '--config', str(folder / 'config.yaml')]
                commands.append((label, command, {'EVENTLAB_BASELINE_CONFIG_DIR': str(folder)}))
            return await self.run(commands, emit)
