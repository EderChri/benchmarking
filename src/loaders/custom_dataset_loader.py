from ts_datasets.base import BaseDataset
from merlion.utils.time_series import TimeSeries
import pandas as pd

class CustomDatasetLoader(BaseDataset):
    def __init__(self, filepath, test_split=0.2):
        self.filepath = filepath
        self.test_split = test_split
        self._load_data()
    
    def _load_data(self):
        df = pd.read_csv(self.filepath, index_col=0, parse_dates=True)
        split_idx = int(len(df) * (1 - self.test_split))
        df['trainval'] = df.index < df.index[split_idx]
        
        self.time_series = [df]
        self.metadata = [pd.DataFrame({'trainval': df['trainval']})]
    
    def __len__(self):
        return len(self.time_series)
    
    def __getitem__(self, i):
        return self.time_series[i], self.metadata[i]
