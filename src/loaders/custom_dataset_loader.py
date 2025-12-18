from loaders.base_loader import BaseDataLoader, register_loader
from merlion.utils import TimeSeries
import pandas as pd

@register_loader("custom")
@register_loader("csv")  # Can register multiple sources
class CustomDatasetLoader(BaseDataLoader):
    def load(self):
        df = pd.read_csv(self.config['file_path'])
        df[self.config.get('time_column', 'timestamp')] = pd.to_datetime(df[self.config.get('time_column', 'timestamp')])
        df = df.set_index(self.config.get('time_column', 'timestamp'))[[self.config.get('target_column', 'value')]]
        
        split_idx = int(len(df) * (1 - self.test_split_ratio))
        train_data = TimeSeries.from_pd(df.iloc[:split_idx])
        test_data = TimeSeries.from_pd(df.iloc[split_idx:])
        
        return train_data, test_data
