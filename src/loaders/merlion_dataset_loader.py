from loaders.base_loader import BaseDataLoader, register_loader
from merlion.utils import TimeSeries
import importlib

import pandas as pd

@register_loader("merlion")
class MerlionDatasetLoader(BaseDataLoader):
    
    def load(self):
        module = importlib.import_module(f"ts_datasets.{self.config['module']}")
        dataset_cls = getattr(module, self.config['class'])
        
        # Build init params
        params = {}
        for key in ['subset', 'rootdir', 'num_columns', 'test_frac']:
            if key in self.config:
                params[key] = self.config[key]
        
        # Handle granularity -> subset mapping
        if 'granularity' in self.config and 'subset' not in params:
            params['subset'] = self.config['granularity']
            
        dataset = dataset_cls(**params)
        
        # Get time series by index
        series_idx = self.config.get('series_id', 0)
        if isinstance(series_idx, str) and series_idx[0].isalpha():
            series_idx = int(series_idx[1:]) - 1
        
        time_series, metadata = dataset[series_idx]
        
        if self.test_mode:
            time_series = time_series.iloc[:100]
            split_idx = len(time_series) // 2
            train_mask = pd.Series([True] * split_idx + [False] * (len(time_series) - split_idx), 
                                   index=time_series.index)
            val_mask = pd.Series([False] * len(time_series), index=time_series.index)
        else:
            # Get train/test split from metadata
            trainval_mask = metadata.trainval.reindex(time_series.index).astype(bool)
            
            n = len(time_series)
            val_size = int(n * self.validation_split_ratio)
            
            # Split trainval into train and validation
            trainval_indices = time_series[trainval_mask].index
            train_end_idx = len(trainval_indices) - val_size
            
            train_mask = pd.Series(False, index=time_series.index)
            val_mask = pd.Series(False, index=time_series.index)
            
            train_mask.loc[trainval_indices[:train_end_idx]] = True
            val_mask.loc[trainval_indices[train_end_idx:]] = True
        
        train_data = TimeSeries.from_pd(time_series[train_mask])
        val_data = TimeSeries.from_pd(time_series[val_mask])
        test_data = TimeSeries.from_pd(time_series[~trainval_mask])
        
        return train_data, val_data, test_data
