from loaders.base_loader import BaseDataLoader, register_loader
from merlion.utils import TimeSeries
import importlib

import pandas as pd

@register_loader("merlion")
class MerlionDatasetLoader(BaseDataLoader):
    
    def load(self):
        module = importlib.import_module(f"ts_datasets.{self.config['module']}")
        dataset_cls = getattr(module, self.config['class'])
        
        # Build init params (subset, rootdir, num_columns, etc.)
        params = {}
        for key in ['subset', 'rootdir', 'num_columns', 'test_frac']:
            if key in self.config:
                params[key] = self.config[key]
        
        # Handle granularity -> subset mapping
        if 'granularity' in self.config and 'subset' not in params:
            params['subset'] = self.config['granularity']
            
        dataset = dataset_cls(**params)
        
        # Get time series by index (default to first one)
        series_idx = self.config.get('series_id', 0)
        if isinstance(series_idx, str) and series_idx[0].isalpha():
            # Convert "D1" to 0, "D2" to 1, etc.
            series_idx = int(series_idx[1:]) - 1
        
        time_series, metadata = dataset[series_idx]
        
        if self.test_mode:
            time_series = time_series.iloc[:100]
            split_idx = len(time_series) // 2
            train_mask = pd.Series([True] * split_idx + [False] * (len(time_series) - split_idx), index=time_series.index)
        else:
            train_mask = metadata.trainval.reindex(time_series.index).astype(bool)
        train_data = TimeSeries.from_pd(time_series[train_mask])
        test_data = TimeSeries.from_pd(time_series[~train_mask])
        
        return train_data, test_data
