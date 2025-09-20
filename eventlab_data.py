import sys
import yaml
from datasets.get_data import get_dataset

def main():
    config_name = sys.argv[1]
    # Load the main configuration file
    with open(config_name, 'r') as f:
        config = yaml.safe_load(f) or {}

    datasets = config.get('datasets', [])

    # Allow either a single mapping or a list of mappings for "datasets"
    if isinstance(datasets, dict):
        datasets = [datasets]
    elif not isinstance(datasets, list):
        print("ERROR: 'datasets' must be a mapping or a list of mappings in data_config.yaml", file=sys.stderr)
        sys.exit(1)

    total = 0
    for ds in datasets:
        if not isinstance(ds, dict):
            print("WARNING: Skipping non-mapping entry under 'datasets'.", file=sys.stderr)
            continue

        name = ds.get('name')
        seqs = ds.get('sequences', [])

        if not name:
            print("WARNING: Skipping a dataset entry with no 'name'.", file=sys.stderr)
            continue

        # Normalize sequences to a list
        if isinstance(seqs, str):
            seqs = [seqs]
        elif not isinstance(seqs, list):
            print(f"WARNING: 'sequences' for dataset '{name}' is neither a list nor a string. Skipping.", file=sys.stderr)
            continue

        for sequence_name in seqs:
            if not isinstance(sequence_name, str):
                print(f"WARNING: Skipping non-string sequence in dataset '{name}'.", file=sys.stderr)
                continue
            print(f"→ Getting dataset='{name}', sequence='{sequence_name}'")
            get_dataset(config, name, sequence_name)
            total += 1

    print(f"Done. Processed {total} (dataset, sequence) pairs.")

if __name__ == "__main__":
    main()
