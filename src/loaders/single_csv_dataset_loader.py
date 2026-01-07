from loaders.base_loader import BaseDataLoader, register_loader
import pandas as pd
from merlion.utils import TimeSeries


@register_loader("single_csv")
class SingleCSVDatasetLoader(BaseDataLoader):
    """Loader for datasets with a single CSV file"""
    
    def load(self):
        file_path = self.config.get('file_path')
        target_columns = self.config.get('target_columns', None) 
        index_column = self.config.get('index_column', None)
        datetime_columns = self.config.get('datetime_columns', None) 
        datetime_format = self.config.get('datetime_format', None)
        na_values = self.config.get('na_values', None)
        
        # Read CSV
        df = pd.read_csv(file_path, na_values=na_values)
        
        # Handle datetime index creation if needed
        if datetime_columns:
            if len(datetime_columns) > 1:
                # Combine multiple columns (e.g., Date + Time)
                df['Datetime'] = pd.to_datetime(
                    df[datetime_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1),
                    format=datetime_format
                )
            else:
                df['Datetime'] = pd.to_datetime(df[datetime_columns[0]], format=datetime_format)
            df = df.set_index('Datetime')
            df = df.drop(columns=datetime_columns, errors='ignore')
        elif index_column:
            df = df.set_index(index_column)
            df.index = pd.to_datetime(df.index)
        
        # Select target columns
        if target_columns:
            df = df[target_columns]
        
        df = df.astype(float)
        df = df.sort_index()
        
        if self.test_mode:
            df = df.head(100)
        
        # Split data
        train_split_idx = int(len(df) * (1 - self.test_split_ratio - self.validation_split_ratio))
        test_split_idx = int(len(df) * (1 - self.test_split_ratio))
        
        train_data = TimeSeries.from_pd(df.iloc[:train_split_idx])
        val_data = TimeSeries.from_pd(df.iloc[train_split_idx:test_split_idx])
        test_data = TimeSeries.from_pd(df.iloc[test_split_idx:])
        
        return train_data, val_data, test_data