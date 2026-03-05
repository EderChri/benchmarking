import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from merlion.utils import TimeSeries

from loaders.base_loader import BaseDataLoader, register_loader


@register_loader("multi_view_dataset")
class MultiViewContrastiveLoader(BaseDataLoader):
    """Loader for DA preprocessed pickle files.

    Expected pickle layout (same as original_paper/run_pretrain.py):
      [
        X_train_intp, X_train_shirink, X_train_forecast, y_train,
        X_val_intp,   X_val_shirink,   X_val_forecast,   y_val,
        X_test_intp,  X_test_shirink,  X_test_forecast,  y_test
      ]

    This loader uses the *_intp tensors and labels, matching run_pretrain.py.
    """

    def load(self):
        file_path = self._resolve_file_path()

        with open(file_path, "rb") as f:
            raw = pickle.load(f)

        if not isinstance(raw, (list, tuple)) or len(raw) != 12:
            raise ValueError(
                f"Expected a 12-item list/tuple in {file_path}, got {type(raw).__name__} with length "
                f"{len(raw) if isinstance(raw, (list, tuple)) else 'N/A'}."
            )

        (
            x_train_intp,
            _,
            _,
            y_train,
            x_val_intp,
            _,
            _,
            y_val,
            x_test_intp,
            _,
            _,
            y_test,
        ) = raw

        train_ts = self._array_to_timeseries(x_train_intp)
        val_ts = self._array_to_timeseries(x_val_intp)
        test_ts = self._array_to_timeseries(x_test_intp)

        train_labels = self._labels_to_timeseries(y_train)
        val_labels = self._labels_to_timeseries(y_val)
        test_labels = self._labels_to_timeseries(y_test)

        if self.test_mode:
            train_ts, train_labels = self._truncate_pair(train_ts, train_labels)
            val_ts, val_labels = self._truncate_pair(val_ts, val_labels)
            test_ts, test_labels = self._truncate_pair(test_ts, test_labels)

        return (train_ts, train_labels), (test_ts, test_labels), (val_ts, val_labels)

    def _resolve_file_path(self) -> str:
        dataset_dir = self.config.get("dataset_dir")
        if not dataset_dir:
            raise ValueError("'dataset_dir' is required for source=multi_view_dataset (same style as ecg.yaml).")

        file_name = self.config.get("file_name") or self.config.get("data_name")
        if not file_name:
            raise ValueError("Provide 'file_name' (or 'data_name') in the data config.")

        file_name = f"{file_name}.pkl" if not str(file_name).endswith(".pkl") else str(file_name)
        file_path = str(Path(dataset_dir) / file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DA pretrain pickle not found: {file_path}")

        return file_path

    def _array_to_timeseries(self, array) -> TimeSeries:
        arr = np.asarray(array)

        if arr.ndim == 3:
            if arr.shape[1] == 1:
                arr = arr[:, 0, :]
            else:
                arr = np.transpose(arr, (0, 2, 1)).reshape(arr.shape[0], -1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"Unsupported sample array shape: {arr.shape}")

        return TimeSeries.from_pd(pd.DataFrame(arr))

    def _labels_to_timeseries(self, labels) -> TimeSeries:
        y = np.asarray(labels).reshape(-1)
        return TimeSeries.from_pd(pd.DataFrame(y))

    @staticmethod
    def _truncate_pair(ts, label_ts, size=100):
        return ts[:size], label_ts[:size]
