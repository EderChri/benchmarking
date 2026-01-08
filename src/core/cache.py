import os
import pickle
import hashlib
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from merlion.utils import TimeSeries


class PreprocessingCache:
    """Manages caching of preprocessed time series data."""
    
    def __init__(self, cache_dir: Path = Path("src/data/.cache/preprocessed")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_cache_key(self, data_cfg: Dict[str, Any], 
                          preproc_cfg: Dict[str, Any],
                          task: str,
                          target: Optional[int] = None) -> str:
        """
        Compute unique cache key based on dataset, preprocessing, task, and target.
        Model config is explicitly excluded to allow different models to share cache.
        """
        cache_components = {
            'data': data_cfg,
            'preprocessing': preproc_cfg,
            'task': task,
            'target': target
        }
        
        # Create deterministic string representation
        cache_str = yaml.dump(cache_components, sort_keys=True)
        
        # Generate hash
        cache_hash = hashlib.sha256(cache_str.encode()).hexdigest()
        return cache_hash[:16]  # Use first 16 chars for readability
    
    def _get_cache_path(self, cache_key: str, split: str) -> Path:
        """Get path for cached split (train/val/test)"""
        return self.cache_dir / f"{cache_key}_{split}.pkl"
    
    def _get_labels_path(self, cache_key: str, split: str) -> Path:
        """Get path for cached labels"""
        return self.cache_dir / f"{cache_key}_{split}_labels.pkl"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get path for cache metadata"""
        return self.cache_dir / f"{cache_key}_meta.yaml"
    
    def exists(self, data_cfg: Dict[str, Any], 
               preproc_cfg: Dict[str, Any],
               task: str,
               target: Optional[int] = None) -> bool:
        """Check if preprocessed data exists in cache"""
        cache_key = self._compute_cache_key(data_cfg, preproc_cfg, task, target)
        
        # Check if all required files exist
        train_path = self._get_cache_path(cache_key, "train")
        test_path = self._get_cache_path(cache_key, "test")
        meta_path = self._get_metadata_path(cache_key)
        
        return train_path.exists() and test_path.exists() and meta_path.exists()
    
    def load(self, data_cfg: Dict[str, Any], 
             preproc_cfg: Dict[str, Any],
             task: str,
             target: Optional[int] = None) -> Tuple:
        """
        Load preprocessed data from cache.
        
        Returns:
            Tuple matching the format expected by main.py:
            - Unsupervised with val: (train_data, val_data, test_data)
            - Unsupervised without val: (train_data, test_data)
            - Supervised with val: ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels))
            - Supervised without val: ((train_data, train_labels), (test_data, test_labels))
        """
        cache_key = self._compute_cache_key(data_cfg, preproc_cfg, task, target)
        
        # Load metadata
        meta_path = self._get_metadata_path(cache_key)
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        has_labels = metadata['has_labels']
        has_validation = metadata['has_validation']
        
        # Load data splits
        train_data = self._load_timeseries(cache_key, "train")
        test_data = self._load_timeseries(cache_key, "test")
        
        if has_validation:
            val_data = self._load_timeseries(cache_key, "val")
        else:
            val_data = None
        
        # Load labels if supervised
        if has_labels:
            train_labels = self._load_timeseries(cache_key, "train", is_label=True)
            test_labels = self._load_timeseries(cache_key, "test", is_label=True)
            
            if has_validation:
                val_labels = self._load_timeseries(cache_key, "val", is_label=True)
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
        cache_key = self._compute_cache_key(data_cfg, preproc_cfg, task, target)
        
        # Save data splits
        self._save_timeseries(train_data, cache_key, "train")
        self._save_timeseries(test_data, cache_key, "test")
        
        has_validation = val_data is not None
        has_labels = train_labels is not None
        
        if has_validation:
            self._save_timeseries(val_data, cache_key, "val")
        
        # Save labels if supervised
        if has_labels:
            self._save_timeseries(train_labels, cache_key, "train", is_label=True)
            self._save_timeseries(test_labels, cache_key, "test", is_label=True)
            
            if has_validation and val_labels is not None:
                self._save_timeseries(val_labels, cache_key, "val", is_label=True)
        
        # Save metadata
        metadata = {
            'cache_key': cache_key,
            'has_labels': has_labels,
            'has_validation': has_validation,
            'task': task,
            'target': target,
            'data_config': data_cfg,
            'preprocessing_config': preproc_cfg
        }
        
        meta_path = self._get_metadata_path(cache_key)
        with open(meta_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
    
    def _save_timeseries(self, ts: TimeSeries, cache_key: str, 
                        split: str, is_label: bool = False):
        """Save TimeSeries object using pickle"""
        if is_label:
            path = self._get_labels_path(cache_key, split)
        else:
            path = self._get_cache_path(cache_key, split)
        
        with open(path, 'wb') as f:
            pickle.dump(ts, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_timeseries(self, cache_key: str, split: str, 
                        is_label: bool = False) -> TimeSeries:
        """Load TimeSeries object from pickle"""
        if is_label:
            path = self._get_labels_path(cache_key, split)
        else:
            path = self._get_cache_path(cache_key, split)
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def clear_cache(self, data_cfg: Optional[Dict[str, Any]] = None,
                   preproc_cfg: Optional[Dict[str, Any]] = None,
                   task: Optional[str] = None,
                   target: Optional[int] = None):
        """
        Clear cache. If configs are provided, clear specific cache entry.
        Otherwise, clear entire cache directory.
        """
        if data_cfg and preproc_cfg and task:
            cache_key = self._compute_cache_key(data_cfg, preproc_cfg, task, target)
            # Remove all files with this cache key
            for file in self.cache_dir.glob(f"{cache_key}*"):
                file.unlink()
            print(f"Cleared cache for key: {cache_key}")
        else:
            # Clear entire cache
            for file in self.cache_dir.glob("*"):
                file.unlink()
            print("Cleared entire preprocessing cache")
