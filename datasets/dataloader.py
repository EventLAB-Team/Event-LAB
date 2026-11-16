"""Recording metadata from EventCV; representations are streamed by the adapters."""
import eventcv as ecv

from datasets.prepare import DatasetFiles


class EventDataset(DatasetFiles):
    def __init__(self, config, dataset_name, sequence_name):
        super().__init__(config, dataset_name, sequence_name)
        self.timewindow_list = list(config['timewindows'])

    def get_dataset_info(self):
        reader = ecv.open(self.dataset_raw,
                          sensor_size=tuple(self.data_config['dataset']['resolution']))
        start, end = reader.time_span_ms
        duration = reader.duration_ms / 1000
        return {
            'dataset_name': self.dataset_name,
            'sequence_name': self.sequence_name,
            'hdf5_path': self.dataset_raw,
            'events_count': reader.n_events,
            'sensor_resolution': reader.sensor_size,
            'time_range': (start, end),
            'duration_seconds': duration,
            'event_rate': reader.n_events / duration if duration > 0 else 0,
        }
