"""Run helpers exactly as the adapters do, without importing them as packages."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('environment,script,args,expected', [
    ('megaevent', 'megaevent_descriptors.py', ['--help'], 0),
    ('spikevpr', 'spikevpr_descriptors.py', ['--help'], 0),
    ('eventgem', 'eventgem_run.py', ['--eventlab-device', 'invalid'], 2),
    ('default', 'eventvlad_denoiser.py', ['--help'], 0),
])
def test_direct_helper_imports_from_another_directory(tmp_path, environment, script, args, expected):
    python = ROOT / '.pixi' / 'envs' / environment / 'bin' / 'python'
    if not python.is_file():
        pytest.skip(f'{environment} Pixi environment is not installed')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([str(python), str(ROOT / 'utils' / script), *args],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == expected, result.stdout + result.stderr
    assert 'Traceback' not in result.stderr
    assert ('usage:' if expected == 0 else 'invalid choice') in result.stdout + result.stderr
