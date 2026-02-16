import os
import pickle
import hashlib
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from merlion.utils import TimeSeries


class PreprocessingCache:
    """Manages caching of preprocessed time series data."""
    
    def __init__(self, cache_dir: Path = Path("src/data/.cache/")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_key = None
    
    def _compute_cache_key(self, data_cfg: Dict[str, Any], 
                          preproc_cfg: Dict[str, Any],
                          task: str,
                          target: Optional[int] = None):
        """Compute and store cache key based on dataset, preprocessing, task, and target."""
        cache_components = {
            'data': data_cfg,
            'preprocessing': preproc_cfg,
            'task': task,
            'target': target
        }
        
        cache_str = yaml.dump(cache_components, sort_keys=True)
        cache_hash = hashlib.sha256(cache_str.encode()).hexdigest()
        self.cache_key = cache_hash[:16]
    
    def _get_cache_dir(self) -> Path:
        """Get cache directory for current cache_key"""
        return self.cache_dir / self.cache_key
    
    def _get_preprocessing_dir(self) -> Path:
        """Get preprocessing subdirectory"""
        return self._get_cache_dir() / "preprocessing"
    
    def _get_cache_path(self, split: str) -> Path:
        """Get path for cached split (train/val/test)"""
        return self._get_preprocessing_dir() / f"{split}.pkl"
    
    def _get_labels_path(self, split: str) -> Path:
        """Get path for cached labels"""
        return self._get_preprocessing_dir() / f"{split}_labels.pkl"
    
    def _get_metadata_path(self) -> Path:
        """Get path for cache metadata"""
        return self._get_preprocessing_dir() / "meta.yaml"
    
    def exists(self, data_cfg: Dict[str, Any], 
               preproc_cfg: Dict[str, Any],
               task: str,
               target: Optional[int] = None) -> bool:
        """Check if preprocessed data exists in cache"""
        self._compute_cache_key(data_cfg, preproc_cfg, task, target)
        
        train_path = self._get_cache_path("train")
        test_path = self._get_cache_path("test")
        meta_path = self._get_metadata_path()
        
        return train_path.exists() and test_path.exists() and meta_path.exists()
    
    def load(self, data_cfg: Dict[str, Any], 
             preproc_cfg: Dict[str, Any],
             task: str,
             target: Optional[int] = None) -> Tuple:
        """Load preprocessed data from cache.
                Returns:
            Tuple matching the format expected by main.py:
            - Unsupervised with val: (train_data, val_data, test_data)
            - Unsupervised without val: (train_data, test_data)
            - Supervised with val: ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels))
            - Supervised without val: ((train_data, train_labels), (test_data, test_labels))
        """
        self._compute_cache_key(data_cfg, preproc_cfg, task, target)
        
        # Load metadata
        with open(self._get_metadata_path(), 'r') as f:
            metadata = yaml.safe_load(f)
        
        has_labels = metadata['has_labels']
        has_validation = metadata['has_validation']
        
        # Load data splits
        train_data = self._load_timeseries("train")
        test_data = self._load_timeseries("test")
        val_data = self._load_timeseries("val") if has_validation else None
        
        # Load labels if supervised
        if has_labels:
            train_labels = self._load_timeseries("train", is_label=True)
            test_labels = self._load_timeseries("test", is_label=True)
            
            if has_validation:
                val_labels = self._load_timeseries("val", is_label=True)
                return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
            else:
                return (train_data, train_labels), (test_data, test_labels)
        else:
            if has_validation:
                return train_data, val_data, test_data
            else:
                return train_data, test_data
    
    def save(self, data_cfg: Dict[str, Any],
             preproc_cfg: Dict[str, Any],
             task: str,
             train_data: TimeSeries,
             test_data: TimeSeries,
             val_data: Optional[TimeSeries] = None,
             train_labels: Optional[TimeSeries] = None,
             test_labels: Optional[TimeSeries] = None,
             val_labels: Optional[TimeSeries] = None,
             target: Optional[int] = None):
        """Save preprocessed data to cache"""
        self._compute_cache_key(data_cfg, preproc_cfg, task, target)
        
        # Create directories
        self._get_preprocessing_dir().mkdir(parents=True, exist_ok=True)
        
        # Save data splits
        self._save_timeseries(train_data, "train")
        self._save_timeseries(test_data, "test")
        
        has_validation = val_data is not None
        has_labels = train_labels is not None
        
        if has_validation:
            self._save_timeseries(val_data, "val")
        
        # Save labels if supervised
        if has_labels:
            self._save_timeseries(train_labels, "train", is_label=True)
            self._save_timeseries(test_labels, "test", is_label=True)
            
            if has_validation and val_labels is not None:
                self._save_timeseries(val_labels, "val", is_label=True)
        
        # Save metadata
        metadata = {
            'cache_key': self.cache_key,
            'has_labels': has_labels,
            'has_validation': has_validation,
            'task': task,
            'target': target,
            'data_config': data_cfg,
            'preprocessing_config': preproc_cfg
        }
        
        with open(self._get_metadata_path(), 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
    
    def _save_timeseries(self, ts: TimeSeries, split: str, is_label: bool = False):
        """Save TimeSeries object using pickle"""
        path = self._get_labels_path(split) if is_label else self._get_cache_path(split)
        
        with open(path, 'wb') as f:
            pickle.dump(ts, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_timeseries(self, split: str, is_label: bool = False) -> TimeSeries:
        """Load TimeSeries object from pickle"""
        path = self._get_labels_path(split) if is_label else self._get_cache_path(split)
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def clear_cache(self, data_cfg: Optional[Dict[str, Any]] = None,
                   preproc_cfg: Optional[Dict[str, Any]] = None,
                   task: Optional[str] = None,
                   target: Optional[int] = None):
        """Clear cache. If configs provided, clear specific entry. Otherwise, clear all."""
        if data_cfg and preproc_cfg and task:
            self._compute_cache_key(data_cfg, preproc_cfg, task, target)
            cache_dir = self._get_cache_dir()
            
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                print(f"Cleared cache for key: {self.cache_key}")
        else:
            import shutil
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
            print("Cleared entire preprocessing cache")
