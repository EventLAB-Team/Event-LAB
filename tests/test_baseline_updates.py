"""Exercise the updater against local Git remotes, without network access."""
import subprocess

import pytest

from utils.preparation import ensure_repo, update_repo


def git(path, *args, check=True):
    result = subprocess.run(['git', '-c', 'protocol.file.allow=always', '-C', str(path), *args],
                            capture_output=True, text=True, check=check)
    return result.stdout.strip()


def commit(path, name, content):
    (path / name).write_text(content)
    git(path, 'add', name)
    git(path, '-c', 'user.name=EventLAB Test', '-c', 'user.email=test@example.invalid',
        'commit', '-m', name)
    return git(path, 'rev-parse', 'HEAD')


@pytest.fixture
def repository(tmp_path):
    remote = tmp_path / 'remote.git'
    remote.mkdir()
    git(remote, 'init', '--bare', '--initial-branch=main')
    source = tmp_path / 'source'
    git(tmp_path, 'clone', str(remote), str(source))
    commit(source, 'method.py', 'original\n')
    commit(source, '.gitignore', 'weights.bin\n')
    git(source, 'push', '-u', 'origin', 'main')
    checkout = tmp_path / 'external'
    git(tmp_path, 'clone', str(remote), str(checkout))
    return source, checkout, remote


def test_existing_baseline_fetches_and_fast_forwards(repository, capsys):
    source, checkout, _ = repository
    expected = commit(source, 'method.py', 'fixed\n')
    git(source, 'push')
    (checkout / 'local-note.txt').write_text('keep this untracked file')
    ensure_repo('unused/existing', checkout)
    assert git(checkout, 'rev-parse', 'HEAD') == expected
    assert (checkout / 'method.py').read_text() == 'fixed\n'
    assert (checkout / 'local-note.txt').read_text() == 'keep this untracked file'
    assert 'updated' in capsys.readouterr().out
    ensure_repo('unused/existing', checkout)
    assert 'already up to date' in capsys.readouterr().out


@pytest.mark.parametrize('state', ['modified', 'staged', 'diverged', 'ahead', 'detached', 'untracked', 'ignored', 'no-upstream'])
def test_local_changes_and_pins_are_preserved(repository, state, capsys):
    source, checkout, _ = repository
    if state in ('modified', 'staged'):
        (checkout / 'method.py').write_text('local fix\n')
        if state == 'staged':
            git(checkout, 'add', 'method.py')
        git(checkout, 'config', 'merge.autostash', 'true')
    elif state in ('diverged', 'ahead'):
        commit(checkout, 'local.txt', 'local commit')
    elif state == 'detached':
        git(checkout, 'checkout', '--detach')
    elif state == 'no-upstream':
        git(checkout, 'branch', '--unset-upstream')
    elif state in ('untracked', 'ignored'):
        name = 'weights.bin' if state == 'ignored' else 'new.py'
        (checkout / name).write_text('local contents')
        (source / name).write_text('upstream contents')
        git(source, 'add', '-f', name)
    old = git(checkout, 'rev-parse', 'HEAD')
    before = git(checkout, 'status', '--porcelain')
    if state != 'ahead':
        commit(source, 'method.py', 'upstream fix\n')
        git(source, 'push')
    assert update_repo(checkout) is False
    assert git(checkout, 'rev-parse', 'HEAD') == old
    assert git(checkout, 'status', '--porcelain') == before
    assert not git(checkout, 'stash', 'list')
    if state in ('modified', 'staged'):
        assert (checkout / 'method.py').read_text() == 'local fix\n'
    if state in ('untracked', 'ignored'):
        assert (checkout / name).read_text() == 'local contents'
    assert capsys.readouterr().out


def test_unreachable_remote_keeps_cached_checkout(repository, capsys):
    _, checkout, remote = repository
    remote.rename(remote.with_suffix('.offline'))
    old = git(checkout, 'rev-parse', 'HEAD')
    ensure_repo('unused/existing', checkout)
    assert git(checkout, 'rev-parse', 'HEAD') == old
    assert 'using existing checkout' in capsys.readouterr().out


def test_fetch_timeout_keeps_cached_checkout(repository, monkeypatch, capsys):
    import utils.preparation as preparation
    _, checkout, _ = repository
    original = preparation._git
    def timeout(path, *args, **kwargs):
        if args[0] == 'fetch':
            raise subprocess.TimeoutExpired('git fetch', 60)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(preparation, '_git', timeout)
    assert update_repo(checkout) is False
    assert 'connection timed out' in capsys.readouterr().out


def test_required_submodules_follow_updated_parent_pins(repository, tmp_path, monkeypatch):
    source, checkout, _ = repository
    child = tmp_path / 'child'
    child.mkdir()
    git(child, 'init', '--initial-branch=main')
    commit(child, 'child.py', 'original child')
    git(source, 'submodule', 'add', str(child), 'external/child')
    commit(source, 'method.py', 'with submodule')
    git(source, 'push')
    # Only these test child repositories use local file transport.
    monkeypatch.setenv('GIT_ALLOW_PROTOCOL', 'file')
    ensure_repo('unused/existing', checkout, submodules=('external/child',))
    old_child = git(checkout / 'external/child', 'rev-parse', 'HEAD')
    expected = commit(child, 'child.py', 'fixed child')
    git(source / 'external/child', 'fetch', 'origin')
    git(source / 'external/child', 'checkout', expected)
    git(source, 'add', 'external/child')
    commit(source, 'method.py', 'updated submodule')
    git(source, 'push')
    ensure_repo('unused/existing', checkout, submodules=('external/child',))
    assert git(checkout / 'external/child', 'rev-parse', 'HEAD') == expected != old_child
    assert not git(checkout, 'status', '--porcelain')
    (checkout / 'external/child/child.py').write_text('local child fix')
    commit(source, 'method.py', 'another upstream change')
    git(source, 'push')
    parent = git(checkout, 'rev-parse', 'HEAD')
    ensure_repo('unused/existing', checkout, submodules=('external/child',))
    assert git(checkout, 'rev-parse', 'HEAD') == parent
    assert (checkout / 'external/child/child.py').read_text() == 'local child fix'


def test_baseline_availability_checks_configured_assets(tmp_path, monkeypatch):
    import utils.preparation as preparation
    monkeypatch.setattr(preparation, 'ROOT', tmp_path)
    monkeypatch.setattr(preparation, 'read_config', lambda _: {'checkpoint': 'chosen.pth'})
    repo = tmp_path / 'baselines/SpikeVPR'
    assert preparation.baseline_availability('spikevpr')[:2] == (0, 2)
    (repo / '.git').mkdir(parents=True)
    assert preparation.baseline_availability('spikevpr')[:2] == (1, 2)
    weights = repo / 'src/weights/chosen.pth'
    weights.parent.mkdir(parents=True)
    weights.touch()
    assert preparation.baseline_availability('spikevpr')[:2] == (1, 2)
    weights.write_bytes(b'weights')
    assert preparation.baseline_availability('spikevpr') == (2, 2, [])
    weights.rename(weights.with_suffix('.part'))
    assert preparation.baseline_availability('spikevpr')[:2] == (1, 2)
