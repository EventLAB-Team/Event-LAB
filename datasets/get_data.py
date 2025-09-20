from datasets.dataloader import EventDataset

def get_dataset(config, dataset_name, sequence_name):
    
    dataset_name = dataset_name.lower()
    dataset= EventDataset(config, dataset_name, sequence_name)

    return dataset