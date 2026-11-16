"""Asset preparation, independent of datasets, baseline objects, and inference."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from utils.configuration import ROOT, baseline_config_path, read_config

REPOSITORIES = {
    'lens': ('AdamDHines/LENS', 'LENS'),
    'sparse_event': ('Tobias-Fischer/sparse-event-vpr', 'vpr_sparse_event'),
    'ensemble': ('AdamDHines/ensemble-event-vpr', 'ensemble'),
    'eventvlad': ('alexjunholee/EventVLAD', 'EventVLAD'),
    'vprmethods': ('gmberton/VPR-methods-evaluation', 'vpr_methods'),
    'spikevpr': ('GeoffroyK/SpikeVPR', 'SpikeVPR'),
    'megaevent': ('AdamDHines/megaevent', 'megaevent'),
    'eventgem': ('AdamDHines/Event-GeM', 'Event-GeM'),
}
HUB_REPOS = {
    'cosplace': 'gmberton/cosplace', 'eigenplaces': 'gmberton/eigenplaces',
    'eigenplaces-indoor': 'Enrico-Chiavassa/Indoor-VPR', 'salad': 'serizba/salad',
    'salad-indoor': 'Enrico-Chiavassa/Indoor-VPR', 'cricavpr': 'Lu-Feng/CricaVPR',
    'megaloc': 'gmberton/MegaLoc', 'edtformer': 'Tong-Jin01/EDTformer',
}


def _git(path, *args, check=True):
    return subprocess.run(['git', '-C', str(path), *args], check=check,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}, timeout=60)


def update_repo(path):
    """Fast-forward the tracked branch; never stash, reset, or merge local work.

    Return whether it is safe to align submodules to the parent's pinned revisions.
    """
    def report(message):
        print(f'[baseline] {path.name}: {message}', flush=True)

    branch = _git(path, 'symbolic-ref', '--quiet', '--short', 'HEAD', check=False)
    if branch.returncode:
        report('keeping pinned/detached checkout; automatic update skipped.')
        return False
    upstream = _git(path, 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}', check=False)
    if upstream.returncode:
        report('no tracking branch; automatic update skipped.')
        return False
    remote = _git(path, 'config', '--get', f'branch.{branch.stdout.strip()}.remote').stdout.strip()
    report(f'checking {upstream.stdout.strip()} for updates…')
    try:
        _git(path, 'fetch', '--quiet', '--no-recurse-submodules', remote)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        reason = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else 'connection timed out'
        report(f'update check failed ({reason}); using existing checkout.')
        return False
    ahead, behind = map(int, _git(path, 'rev-list', '--left-right', '--count', 'HEAD...@{upstream}').stdout.split())
    old = _git(path, 'rev-parse', '--short', 'HEAD').stdout.strip()
    if _git(path, 'status', '--porcelain', '--untracked-files=no', '--ignore-submodules=none').stdout:
        report(f'{behind} upstream commit(s) available; local changes present, keeping {old}.')
        return False
    if ahead:
        report(f'{ahead} local / {behind} upstream commit(s); keeping local history at {old}.')
        return False
    if not behind:
        report(f'already up to date at {old}.')
        return True
    try:
        _git(path, '-c', 'merge.autostash=false', '-c', 'submodule.recurse=false',
             'merge', '--ff-only', '--no-edit', '--no-overwrite-ignore', '@{upstream}')
    except subprocess.CalledProcessError as error:
        report(f'fast-forward refused ({error.stderr.strip()}); keeping {old}.')
        return False
    new = _git(path, 'rev-parse', '--short', 'HEAD').stdout.strip()
    report(f'updated {old} → {new} ({behind} commit(s)).')
    return True


def ensure_repo(repository, path, recursive=False, submodules=()):
    path = Path(path)
    sync_modules = True
    if not path.exists():
        command = ['git', 'clone'] + (['--recursive'] if recursive else [])
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f'.{path.name}-', dir=path.parent) as staging:
            checkout = Path(staging) / 'checkout'
            subprocess.run(command + [f'https://github.com/{repository}.git', str(checkout)], check=True)
            os.replace(checkout, path)
    elif not (path / '.git').exists():
        raise RuntimeError(f'{path} exists but is not a Git checkout; move it aside and retry.')
    else:
        sync_modules = update_repo(path)
    if recursive or submodules:
        modules_file = path / '.gitmodules'
        modules = list(submodules) or ([line.split('=', 1)[1].strip() for line in modules_file.read_text().splitlines()
                    if line.strip().startswith('path =')] if modules_file.exists() else [])
        if not sync_modules:
            modules = [item for item in modules if not (path / item).is_dir() or not any((path / item).iterdir())]
        if modules:
            flags = ['--recursive'] if recursive else []
            subprocess.run(['git', '-C', str(path), 'submodule', 'sync', *flags, '--', *modules], check=True)
            subprocess.run(['git', '-C', str(path), 'submodule', 'update', '--init', '--checkout',
                            *flags, '--', *modules], check=True)


def drive_file(file_id, destination):
    import gdown
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = str(destination) + '.part'
    if not gdown.download(id=file_id, output=partial, quiet=False, resume=True):
        raise RuntimeError(f'Download failed: {destination}')
    if not Path(partial).stat().st_size:
        raise RuntimeError(f'Empty download: {destination}')
    os.replace(partial, destination)


def trust_hub(method, settings):
    repo = 'AnyLoc/DINO' if method.startswith('anyloc') else HUB_REPOS.get(method)
    if repo is None:
        return
    import torch.hub
    listing = Path(torch.hub.get_dir()) / 'trusted_list'
    entry = repo.replace('/', '_')
    existing = listing.read_text().splitlines() if listing.exists() else []
    if entry in existing:
        return
    if not settings.get('trust_torch_hub', True):
        raise RuntimeError(f'{repo} is not trusted; enable trust_torch_hub to prepare this method.')
    listing.parent.mkdir(parents=True, exist_ok=True)
    with listing.open('a') as handle:
        handle.write(entry + '\n')


def prepare_baseline(name, *, weights=False):
    repository, folder = REPOSITORIES[name]
    repo = ROOT / 'baselines' / folder
    settings = read_config(baseline_config_path(name))
    modules = ('eventgem/external/superevent', 'eventgem/external/vprtutorial') if name == 'eventgem' else ()
    ensure_repo(repository, repo, recursive=name == 'vprmethods', submodules=modules)
    if name in ('ensemble', 'eventvlad'):
        ensure_repo('Nanne/pytorch-NetVlad', settings['netvlad_path'])
    if name == 'eventvlad':
        drive_file('1xdoGI7vmNelaR_D9-FUk5SbB3webqa5c', repo / 'denoiser_brisbane')
        drive_file('1rSIhH1pk8ADxfqYQXoos_hTuWyfiWSu3', repo / 'vgg16_eventvlad.tar')
    elif name == 'spikevpr':
        destination = repo / 'src' / 'weights' / settings['checkpoint']
        drive_file(settings['gdrive_id'], destination)
        expected = settings.get('sha256')
        if expected:
            digest = hashlib.sha256()
            with destination.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                # Preserve the adapter's existing warning policy.
                print(f'WARNING: Checksum mismatch for {destination}; verify the configured sha256.', flush=True)
    elif name == 'eventgem':
        if weights:
            from huggingface_hub import hf_hub_download
            hf_hub_download('AdamHines/eventgem', 'eventgem.pt')
    if not weights:
        return
    if name == 'ensemble':
        from torchvision.models import VGG16_Weights
        VGG16_Weights.IMAGENET1K_FEATURES.get_state_dict(progress=True)
    elif name == 'megaevent':
        sys.path.insert(0, str(repo / 'src'))
        import numpy._core.multiarray  # numpy 1.x compatibility used by the runtime adapter
        from megaevent.checkpoints import resolve_model
        resolve_model(settings['model'], os.path.abspath(settings['ckpt_dir']))
    elif name == 'vprmethods':
        # Use upstream's parser and factory: they encode each method's weight
        # locations and backbone overrides. Construct on CPU, never run forward.
        trust_hub(settings['method'], settings)
        sys.path.insert(0, str(repo))
        from parser import parse_arguments
        from vpr_models import get_model
        previous = sys.argv
        previous_directory = Path.cwd()
        try:
            os.chdir(repo)
            sys.argv = ['prepare', '--database_folder', '.', '--queries_folder', '.', '--device', 'cpu']
            for key in ('method', 'backbone', 'descriptors_dimension'):
                if settings.get(key) is not None:
                    sys.argv.extend([f'--{key}', str(settings[key])])
            args = parse_arguments()
            if args.method.startswith('anyloc'):
                import torch.hub
                model = torch.hub.load('AnyLoc/DINO', 'get_vlad_model', backbone='DINOv2',
                                       domain=args.method.split('-', 1)[1], device='cpu')
            else:
                model = get_model(args.method, args.backbone, args.descriptors_dimension)
            if model is None:
                raise ValueError(f'Unsupported VPR method: {args.method}')
            del model
        finally:
            sys.argv = previous
            os.chdir(previous_directory)
    print(f'Prepared {name}: repository and configured weights (no inference).', flush=True)


def baseline_availability(name):
    """Count known local assets without importing models or contacting a server."""
    import json
    repo = ROOT / 'baselines' / REPOSITORIES[name][1]
    settings = read_config(baseline_config_path(name))
    checks = []
    def file(label, path):
        path = Path(path)
        checks.append((label, path.is_file() and path.stat().st_size > 0))
    def checkout(label, path):
        checks.append((label, (Path(path) / '.git').exists()))
    def hub_file(repository, filename):
        home = Path(os.environ.get('HF_HOME', Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'huggingface'))
        cache = Path(os.environ.get('HF_HUB_CACHE', os.environ.get('HUGGINGFACE_HUB_CACHE', home / 'hub')))
        folder = cache / ('models--' + repository.replace('/', '--'))
        ref = folder / 'refs/main'
        if ref.is_file():
            file(filename, folder / 'snapshots' / ref.read_text().strip() / filename)
        else:
            checks.append((filename, False))
    checkout('repository', repo)
    torch_home = Path(os.environ.get('TORCH_HOME', Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'torch'))
    if name in ('ensemble', 'eventvlad'):
        checkout('NetVLAD', ROOT / Path(settings['netvlad_path']).expanduser())
    if name == 'ensemble':
        file('VGG16 weights', torch_home / 'hub/checkpoints/vgg16_features-amdegroot-88682ab5.pth')
    elif name == 'eventvlad':
        for filename in ('denoiser_brisbane', 'vgg16_eventvlad.tar'):
            file(filename, repo / filename)
    elif name == 'spikevpr':
        file(settings['checkpoint'], repo / 'src/weights' / settings['checkpoint'])
    elif name == 'megaevent':
        registry = repo / 'src/megaevent/models.json'
        model = json.loads(registry.read_text()).get(settings['model'], {}) if registry.is_file() else {}
        filename = model.get('filename')
        if filename:
            file(filename, ROOT / Path(settings['ckpt_dir']).expanduser() / filename)
        else:
            checks.append(('configured model weights', False))
    elif name == 'eventgem':
        for module in ('superevent', 'vprtutorial'):
            checkout(module, repo / 'eventgem/external' / module)
        file('SuperEvent weights', repo / 'eventgem/external/superevent/saved_models/super_event_weights.pth')
        hub_file('AdamHines/eventgem', 'eventgem.pt')
    elif name == 'vprmethods':
        checkout('deep-image-retrieval', repo / 'third_party/deep-image-retrieval')
        if settings['method'] == 'megaloc':
            hub_file('gberton/MegaLoc', 'model.safetensors')
            file('MegaLoc model code', torch_home / 'hub/gmberton_MegaLoc_main/hubconf.py')
        else:
            checks.append((f"{settings['method']} model cache (not verified)", False))
    return sum(bool(present) for _, present in checks), len(checks), [label for label, present in checks if not present]
