from loaders.base_loader import BaseDataLoader, register_loader
from merlion.utils import TimeSeries
import pandas as pd
from datetime import datetime

@register_loader("energy_power")
class EnergyPowerDatasetLoader(BaseDataLoader):
    def load(self):
        file_path = self.config.get('file_path')
        target_column = self.config.get('target_column', 'Global_active_power')
        test_split_ratio = self.config.get('test_split_ratio', 0.2)
        
        df = pd.read_csv(file_path, na_values='?')
        
        # Combine Date and Time columns into a single Datetime column
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        df = df.set_index('Datetime')
        
        df = df[[target_column]].astype(float)
        
        # Sort by index
        df = df.sort_index()
        
        if self.test_mode:
            df = df.head(100)
        
        # Split train/test
        split_idx = int(len(df) * (1 - test_split_ratio))
        train_data = TimeSeries.from_pd(df.iloc[:split_idx])
        test_data = TimeSeries.from_pd(df.iloc[split_idx:])
        
        return train_data, test_data
