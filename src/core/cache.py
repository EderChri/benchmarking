import os
import pickle
import hashlib
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from merlion.utils import TimeSeries
from core.data_splits import DataSplits

class PreprocessingCache:
    def __init__(self, cache_dir: Path = Path("src/data/.cache/")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_key = None

    def _compute_cache_key(self, data_cfg, preproc_cfg, task, target=None):
        cache_str = yaml.dump({"data": data_cfg, "preprocessing": preproc_cfg, "task": task, "target": target}, sort_keys=True)
        self.cache_key = hashlib.sha256(cache_str.encode()).hexdigest()[:16]

    def _preproc_dir(self) -> Path:
        return self.cache_dir / self.cache_key / "preprocessing"

    def _path(self, split: str, is_label: bool = False) -> Path:
        suffix = "_labels.pkl" if is_label else ".pkl"
        return self._preproc_dir() / f"{split}{suffix}"

    def exists(self, data_cfg, preproc_cfg, task, target=None) -> bool:
        self._compute_cache_key(data_cfg, preproc_cfg, task, target)
        return all(self._path(s).exists() for s in ["train", "test"]) and (self._preproc_dir() / "meta.yaml").exists()

    def load(self, data_cfg, preproc_cfg, task, target=None) -> DataSplits:
        self._compute_cache_key(data_cfg, preproc_cfg, task, target)
        with open(self._preproc_dir() / "meta.yaml") as f:
            meta = yaml.safe_load(f)
        train_data = self._load(self._path("train"))
        test_data = self._load(self._path("test"))
        val_data = self._load(self._path("val")) if meta["has_validation"] else None
        if meta["has_labels"]:
            return DataSplits(
                train_data, test_data, val_data,
                train_labels=self._load(self._path("train", is_label=True)),
                test_labels=self._load(self._path("test", is_label=True)),
                val_labels=self._load(self._path("val", is_label=True)) if meta["has_validation"] else None,
                has_labels=True,
            )
        return DataSplits(train_data, test_data, val_data, has_labels=False)

    def save(self, data_cfg, preproc_cfg, task, splits: DataSplits, target=None):
        self._compute_cache_key(data_cfg, preproc_cfg, task, target)
        self._preproc_dir().mkdir(parents=True, exist_ok=True)
        for split, data in [("train", splits.train_data), ("test", splits.test_data)]:
            self._save(data, self._path(split))
        if splits.val_data is not None:
            self._save(splits.val_data, self._path("val"))
        if splits.has_labels:
            for split, labels in [("train", splits.train_labels), ("test", splits.test_labels)]:
                self._save(labels, self._path(split, is_label=True))
            if splits.val_labels is not None:
                self._save(splits.val_labels, self._path("val", is_label=True))
        yaml.dump({
            "cache_key": self.cache_key, "has_labels": splits.has_labels,
            "has_validation": splits.val_data is not None, "task": task, "target": target,
        }, open(self._preproc_dir() / "meta.yaml", "w"), default_flow_style=False)

    def _save(self, obj, path: Path):
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self, path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def clear_cache(self, data_cfg=None, preproc_cfg=None, task=None, target=None):
        import shutil
        if data_cfg and preproc_cfg and task:
            self._compute_cache_key(data_cfg, preproc_cfg, task, target)
            shutil.rmtree(self.cache_dir / self.cache_key, ignore_errors=True)
        else:
            for d in self.cache_dir.iterdir():
                if d.is_dir():
                    shutil.rmtree(d)
