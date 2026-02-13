import os
from loaders.base_loader import BaseDataLoader, register_loader
import pandas as pd
import torch
from merlion.utils import TimeSeries


@register_loader("torch_dataset")
class TorchDatasetLoader(BaseDataLoader):
    """Loader for datasets with pre-split .pt files (train.pt, val.pt, test.pt)

    Expects .pt files to contain either:
    - Tensors directly (unsupervised)
    - Dictionaries with 'samples' and 'labels' keys (supervised)
    """

    def load(self):
        dataset_dir = self.config.get("dataset_dir")
        has_labels = self.config.get("has_labels", False)

        train_data = self._load_split(dataset_dir, "train.pt")
        val_data = self._load_split(dataset_dir, "val.pt")
        test_data = self._load_split(dataset_dir, "test.pt")

        if has_labels:
            return self._load_supervised(train_data, val_data, test_data)
        return self._load_unsupervised(train_data, val_data, test_data)

    def _load_split(self, dataset_dir: str, filename: str):
        """Load a single .pt file"""
        return torch.load(os.path.join(dataset_dir, filename))

    def _load_supervised(self, train_data, val_data, test_data):
        """Load supervised datasets with labels"""
        train_ts, train_labels = self._create_timeseries_with_labels(train_data)
        val_ts, val_labels = self._create_timeseries_with_labels(val_data)
        test_ts, test_labels = self._create_timeseries_with_labels(test_data)

        if self.test_mode:
            train_ts, train_labels = self._truncate_pair(train_ts, train_labels)
            val_ts, val_labels = self._truncate_pair(val_ts, val_labels)
            test_ts, test_labels = self._truncate_pair(test_ts, test_labels)

        print(f"Loaded supervised dataset with {len(train_ts)} training samples, "
              f"{len(val_ts)} validation samples, and {len(test_ts)} test samples.")

        return (train_ts, train_labels), (val_ts, val_labels), (test_ts, test_labels)

    def _load_unsupervised(self, train_data, val_data, test_data):
        """Load unsupervised datasets without labels"""
        train_ts = self._tensor_to_timeseries(train_data)
        val_ts = self._tensor_to_timeseries(val_data)
        test_ts = self._tensor_to_timeseries(test_data)

        if self.test_mode:
            train_ts = train_ts[:100]
            val_ts = val_ts[:100]
            test_ts = test_ts[:100]

        return train_ts, val_ts, test_ts

    def _create_timeseries_with_labels(self, data_dict):
        """Convert dictionary with samples and labels to TimeSeries pair"""
        samples = data_dict["samples"]
        labels = data_dict["labels"]

        ts = self._tensor_to_timeseries(samples)
        label_ts = self._tensor_to_timeseries(labels)

        return ts, label_ts

    def _tensor_to_timeseries(self, tensor, column_name=None):
        """Convert tensor to TimeSeries object"""
        array = tensor.numpy() if torch.is_tensor(tensor) else tensor
        
        # Handle 3D tensors by squeezing dimension 1
        if array.ndim == 3:
            if array.shape[1] != 1:
                raise ValueError(
                    f"Cannot squeeze dimension 1: expected size 1, got {array.shape[1]}. "
                    f"Full shape: {array.shape}"
                )
            array = array.squeeze(1)
        
        df = pd.DataFrame(array)

        if column_name:
            df.columns = [column_name]

        return TimeSeries.from_pd(df)


    def _truncate_pair(self, ts, label_ts, size=100):
        """Truncate both TimeSeries objects for test mode"""
        return ts[:size], label_ts[:size]
