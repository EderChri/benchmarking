from loaders.base_loader import BaseDataLoader, register_loader
from merlion.utils import TimeSeries
import pandas as pd
from datetime import datetime

@register_loader("energy_power")
class EnergyPowerDatasetLoader(BaseDataLoader):
    def load(self):
        file_path = self.config.get('file_path')
        target_column = self.config.get('target_column', 'Global_active_power')
        
        df = pd.read_csv(file_path, na_values='?')
        
        # Combine Date and Time columns into a single Datetime column
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        df = df.set_index('Datetime')
        
        df = df[[target_column]].astype(float)
        
        # Sort by index
        df = df.sort_index()
        
        if self.test_mode:
            df = df.head(100)
        
        train_split_idx = int(len(df) * (1 - self.test_split_ratio - self.validation_split_ratio))
        test_split_idx = int(len(df) * (1 - self.test_split_ratio))
        train_data = TimeSeries.from_pd(df.iloc[:train_split_idx])
        val_data = TimeSeries.from_pd(df.iloc[train_split_idx:test_split_idx])
        test_data = TimeSeries.from_pd(df.iloc[test_split_idx:])
        
        return train_data, val_data, test_data