from ts_datasets.base import BaseDataset
from merlion.utils.time_series import TimeSeries

class DatasetLoader:
    def __init__(self, dataset: BaseDataset, test_split_ratio: float = 0.2):
        self.dataset = dataset
        self.test_split_ratio = test_split_ratio
    
    def load(self):
        """Returns train/test split as single TimeSeries objects"""
        t = 7 
        # Calculate split timestamp based on ratio
        total_length = len(self.dataset)
        split_idx = int(total_length * (1 - self.test_split_ratio))
        split_timestamp = self.dataset.time_stamps[split_idx]
        
        # Use Merlion's bisect method to split at timestamp
        train_data, test_data = self.dataset.bisect(split_timestamp, t_in_left=False)
        
        return train_data, test_data
