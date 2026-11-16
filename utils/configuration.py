"""Lightweight, comment-preserving configuration shared by CLI and TUI."""
import argparse
import io
import math
import os
from pathlib import Path
import tempfile

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config.yaml'
REPRESENTATION_DOCS = 'https://docs.eventcv.net/en/latest/api.html#eventcv.EventReader.with_repr'
RECONSTRUCTION_UNAVAILABLE = 'Reconstruction is unavailable until named methods are supported by EventCV.'
# (type, minimum, choices, help). Names deliberately match the YAML keys.
REPRESENTATIONS = ('polarity', 'binary', 'count', 'voxel', 'tsurf', 'atsurf',
                   'tencode', 'redblue', 'countmask', 'mcts', 'flow', 'reconstruction')
FIELDS = {
    'data_path': (str, None, None, 'Dataset directory'),
    'request_input': (bool, None, None, 'Ask before downloading in the CLI'),
    'gps_tolerance': (float, 0, None, 'Ground truth tolerance in metres along the reference route'),
    'filter_places_sec': (float, 0, None, 'Minimum separation in seconds for image-folder adapters'),
    'timewindows': (list, 1, None, 'Positive integer time windows in milliseconds'),
    'representation': (str, None, REPRESENTATIONS, 'Choose how EventCV turns events into model inputs; polarity is the default'),
    'representation_options': (dict, None, None, f'EventCV representation parameters passed to reader.with_repr(). {{}} uses defaults. Docs: {REPRESENTATION_DOCS}'),
    'hot_pixel_filter': (bool, None, None, 'Filter hot pixels before rendering'),
    'reconstruction': (dict, None, None, RECONSTRUCTION_UNAVAILABLE),
}



BASELINE_CHOICES = {
    'vprmethods': {'device': ('auto', 'cpu', 'cuda')},
    'spikevpr': {'encoder': ('sew_resnet18', 'sew_resnet34'),
                 'channel_order': ('off_on', 'on_off'), 'device': ('auto', 'cpu', 'cuda', 'mps')},
    'megaevent': {'model': ('megaevent_vits14', 'megaevent_vitb14'),
                  'device': ('auto', 'cpu', 'cuda', 'mps')},
    'eventgem': {'match_filter': ('mutual', 'ratio'), 'device': ('auto', 'cpu', 'cuda', 'mps')},
}


def yaml_codec():
    codec = YAML()
    codec.preserve_quotes = True
    return codec


def read_config(path=CONFIG):
    with Path(path).open() as handle:
        value = yaml_codec().load(handle)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a YAML mapping')
    return value


def dumps(value):
    out = io.StringIO()
    yaml_codec().dump(value, out)
    return out.getvalue()


def write_config(path, value):
    """Replace only after serialization succeeds; never leave a truncated YAML."""
    path = Path(path)
    contents = dumps(value)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(contents)
        if path.exists():
            os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_value(key, text, current=None, runtime=True):
    if runtime:
        if key == 'reconstruction' or (key == 'representation' and text == 'reconstruction'):
            raise ValueError(RECONSTRUCTION_UNAVAILABLE)
        if key not in FIELDS:
            raise ValueError(f'Unknown runtime field: {key}')
        kind, minimum, choices, _ = FIELDS[key]
    else:
        kind = (bool if isinstance(current, bool) else int if isinstance(current, int)
                else float if isinstance(current, float) else str if isinstance(current, str)
                else list if isinstance(current, list) else type(current))
        minimum, choices = None, None
    if kind is str:
        value = text
        if not value.strip():
            raise ValueError(f'{key}: must not be empty')
    elif kind is bool:
        if text.lower() not in ('true', 'false'):
            raise ValueError(f'{key}: use true or false')
        value = text.lower() == 'true'
    elif kind in (int, float):
        try:
            value = kind(text)
        except ValueError:
            raise ValueError(f'{key}: expected {kind.__name__}') from None
        if not math.isfinite(value):
            raise ValueError(f'{key}: must be finite')
    elif runtime and kind is dict:
        value = YAML(typ='safe').load(text)
        if not isinstance(value, dict):
            raise ValueError(f'{key}: expected a YAML mapping')
    else:
        # ruamel's round-trip scalar/list subclasses are handled by value shape.
        if runtime and kind is list:
            try:
                if text.strip().startswith(('[', '- ')):
                    value = YAML(typ='safe').load(text)
                else:
                    value = [int(part) for part in text.replace(',', ' ').split()]
            except (ValueError, TypeError):
                raise ValueError(f'{key}: expected a list of integers') from None
            if (not isinstance(value, list) or not value
                    or any(type(item) is not int or item < minimum for item in value)):
                raise ValueError(f'{key}: provide one or more integers >= {minimum}')
        else:
            value = YAML(typ='safe').load(text)
            expected = (bool if isinstance(current, bool) else int if isinstance(current, int)
                        else float if isinstance(current, float) else str if isinstance(current, str)
                        else list if isinstance(current, list) else dict if isinstance(current, dict)
                        else type(None))
            if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
                raise ValueError(f'{key}: expected {expected.__name__}')
    if minimum is not None and isinstance(value, (int, float)) and value < minimum:
        raise ValueError(f'{key}: must be >= {minimum}')
    if choices and value not in choices:
        raise ValueError(f'{key}: choose {", ".join(choices)}')
    return value


def update_config(path, changes, runtime=True):
    if not changes:
        return
    data = read_config(path)
    for key, value in changes.items():
        if runtime and key not in FIELDS:
            raise ValueError(f'Unknown runtime field: {key}')
        # Validate before changing anything on disk.
        text = dumps(value).strip().removesuffix('\n...') if not isinstance(value, str) else value
        parsed = parse_value(key, text, data.get(key), runtime)
        choices = BASELINE_CHOICES.get(Path(path).stem, {}).get(key) if not runtime else None
        if choices and parsed not in choices:
            raise ValueError(f'{key}: choose {choices}')
        data[key] = parsed
    write_config(path, data)


def baseline_config_path(name):
    directory = Path(os.environ.get('EVENTLAB_BASELINE_CONFIG_DIR', ROOT / 'baselines'))
    return directory / f'{name}.yaml'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Show or update config.yaml runtime settings.')
    parser.add_argument('--config', type=Path, default=CONFIG, help='Configuration file (default: config.yaml)')
    for key, (kind, _, choices, help_text) in FIELDS.items():
        parser.add_argument(f'--{key}', nargs='+' if kind is list else None,
                            help=help_text + (f' ({", ".join(choices)})' if choices else ''))
    args = vars(parser.parse_args(argv))
    path = args.pop('config')
    try:
        changes = {key: parse_value(key, ' '.join(value) if isinstance(value, list) else value)
                   for key, value in args.items() if value is not None}
        if changes:
            update_config(path, changes)
        print(dumps(read_config(path)), end='')
    except Exception as error:
        # YAML parser failures also become concise command-line errors.
        parser.exit(2, f'config: {error}\n')


def validate_runtime(config):
    for key in FIELDS:
        if key in config and key != 'reconstruction':
            value = config[key]
            parse_value(key, value if isinstance(value, str) else dumps(value).strip().removesuffix('\n...'))
    retired = {'std_format', 'max_frames', 'chunk_size', 'stream', 'num_events',
               'frame_generator', 'frame_accumulator', 'reconstruction_model', 'ground_truth_tolerance'}
    if retired.intersection(config):
        raise ValueError(f"Retired configuration settings: {', '.join(sorted(retired.intersection(config)))}. "
                         'Use representation, representation_options and gps_tolerance (metres).')
