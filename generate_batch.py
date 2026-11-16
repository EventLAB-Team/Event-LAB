"""Compatibility entry point for generating a portable-to-POSIX batch launcher."""
import argparse
from pathlib import Path
import shlex

from utils.configuration import ROOT, read_config
from utils.experiments import expand_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch')
    parser.add_argument('config')
    parser.add_argument('output')
    args = parser.parse_args()
    expand_batch(read_config(args.batch), read_config(args.config))
    command = ['python', str(ROOT / 'eventlab_batch.py'), str(Path(args.batch).resolve()),
               '--config', str(Path(args.config).resolve())]
    output = Path(args.output)
    output.write_text('#!/usr/bin/env bash\nexec ' + shlex.join(command) + '\n')
    output.chmod(output.stat().st_mode | 0o111)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
