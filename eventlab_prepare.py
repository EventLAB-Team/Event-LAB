"""Download assets without running an experiment."""
import argparse
import os
from utils.configuration import ROOT, CONFIG, read_config
from utils.experiments import catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(CONFIG))
    commands = parser.add_subparsers(dest='kind', required=True)
    baseline = commands.add_parser('baseline')
    baseline.add_argument('name')
    dataset = commands.add_parser('dataset')
    dataset.add_argument('name')
    dataset.add_argument('sequences', nargs='+')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = read_config(args.config)
    config['request_input'] = False
    baselines, datasets = catalog(config)
    try:
        if args.kind == 'baseline':
            if args.name not in baselines:
                raise ValueError(f'Unknown baseline: {args.name}')
            from utils.preparation import prepare_baseline
            prepare_baseline(args.name, weights=True)
        else:
            if args.name not in datasets or any(s not in datasets[args.name] for s in args.sequences):
                raise ValueError('Unknown dataset or sequence')
            from datasets.prepare import DatasetFiles
            for sequence in args.sequences:
                DatasetFiles(config, args.name, sequence)
                print(f'Prepared {args.name}/{sequence}', flush=True)
    except Exception as error:
        parser.exit(1, f'prepare: {error}\n')


if __name__ == '__main__':
    main()
