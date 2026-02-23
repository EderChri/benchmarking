import logging
from typing import Dict, Any
from core.data_splits import DataSplits
from core.factory import ComponentFactory
from loaders import LOADER_REGISTRY

logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, factory: ComponentFactory, cache=None, test_mode: bool = False):
        self.factory = factory
        self.cache = cache
        self.test_mode = test_mode

    def get_data(self, run: Dict[str, Any], configs: Dict[str, Any]) -> DataSplits:
        if self.cache and self.cache.exists(configs["data"], configs["preprocessing"], run["task"], configs.get("target")):
            logger.info("Loading preprocessed data from cache...")
            return self.cache.load(configs["data"], configs["preprocessing"], run["task"], configs.get("target"))
        return self._preprocess_and_cache(run, configs)

    def _preprocess_and_cache(self, run: Dict[str, Any], configs: Dict[str, Any]) -> DataSplits:
        logger.info("Processing data (not in cache)...")
        loader_cls = LOADER_REGISTRY[configs["data"].get("source", "custom")]
        loader = loader_cls(configs["data"], configs["data"].get("test_split_ratio", 0.2), test_mode=self.test_mode)
        splits = self._parse_raw(loader.load())
        splits = self._apply_transforms(configs["preprocessing"], splits)
        if self.cache:
            self.cache.save(configs["data"], configs["preprocessing"], run["task"], splits, configs.get("target"))
        return splits

    def _parse_raw(self, raw) -> DataSplits:
        if isinstance(raw[0], tuple):
            (train_data, train_labels), (test_data, test_labels) = raw[:2]
            val_data = val_labels = None
            if len(raw) == 3:
                val_data, val_labels = raw[2]
            return DataSplits(train_data, test_data, val_data, train_labels, test_labels, val_labels, has_labels=True)
        train_data, test_data = raw[0], raw[-1]
        val_data = raw[1] if len(raw) == 3 else None
        return DataSplits(train_data, test_data, val_data, has_labels=False)

    def _apply_transforms(self, preproc_cfg: Dict[str, Any], splits: DataSplits) -> DataSplits:
        t = self.factory.load_preprocessor(preproc_cfg)
        t.train(splits.train_data)
        splits.train_data = t(splits.train_data)
        splits.test_data = t(splits.test_data)
        if splits.val_data is not None:
            splits.val_data = t(splits.val_data)
        return splits
