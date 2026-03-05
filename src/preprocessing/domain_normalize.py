from typing import List

import numpy as np

from merlion.transform.base import TransformBase
from merlion.transform.normalize import MeanVarNormalize
from merlion.utils import TimeSeries


class NormalizeColumnsExcludingSuffixesTransform(TransformBase):
    """Apply MeanVarNormalize to columns that do not end with any excluded suffix."""

    def __init__(self, excluded_suffixes: List[str] = None, samplewise_mode: bool = False, num_feature: int = 1):
        super().__init__()
        self.excluded_suffixes = excluded_suffixes or ["_derivative", "_fft"]
        self.samplewise_mode = samplewise_mode
        self.num_feature = int(num_feature)
        self._norm = MeanVarNormalize()
        self._cols = None
        self._mean = None
        self._std = None

    @property
    def requires_inversion_state(self):
        return False

    def train(self, time_series: TimeSeries):
        df = time_series.to_pd()
        self._cols = [
            c for c in df.columns
            if not any(str(c).endswith(sfx) for sfx in self.excluded_suffixes)
        ]
        if self._cols and self.samplewise_mode:
            x = self._reshape_nld(df[self._cols].values)
            self._mean = x.mean(axis=(0, 1), keepdims=True)
            self._std = x.std(axis=(0, 1), keepdims=True)
            self._std = np.clip(self._std, 1e-8, None)
        elif self._cols:
            self._norm.train(TimeSeries.from_pd(df[self._cols]))

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        df = time_series.to_pd().copy()
        if not self._cols:
            return TimeSeries.from_pd(df)
        if self.samplewise_mode:
            x = self._reshape_nld(df[self._cols].values)
            x = (x - self._mean) / self._std
            df.loc[:, self._cols] = self._flatten_nld(x)
        else:
            normed = self._norm(TimeSeries.from_pd(df[self._cols])).to_pd()
            df.loc[:, self._cols] = normed.values
        return TimeSeries.from_pd(df)

    def _reshape_nld(self, values_2d: np.ndarray) -> np.ndarray:
        arr = np.asarray(values_2d, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array [N, L*D], got {arr.shape}")
        n, cols = arr.shape
        d = max(1, self.num_feature)
        if cols % d != 0:
            raise ValueError(f"Column count {cols} not divisible by num_feature {d}")
        l = cols // d
        return arr.reshape(n, l, d)

    @staticmethod
    def _flatten_nld(values_nld: np.ndarray) -> np.ndarray:
        n, l, d = values_nld.shape
        return values_nld.reshape(n, l * d)


class NormalizeColumnsBySuffixTransform(TransformBase):
    """Apply MeanVarNormalize to columns with a target suffix."""

    def __init__(self, suffix: str, samplewise_mode: bool = False, num_feature: int = 1):
        super().__init__()
        self.suffix = suffix
        self.samplewise_mode = samplewise_mode
        self.num_feature = int(num_feature)
        self._norm = MeanVarNormalize()
        self._cols = None
        self._mean = None
        self._std = None

    @property
    def requires_inversion_state(self):
        return False

    def train(self, time_series: TimeSeries):
        df = time_series.to_pd()
        self._cols = [c for c in df.columns if str(c).endswith(self.suffix)]
        if self._cols and self.samplewise_mode:
            x = self._reshape_nld(df[self._cols].values)
            self._mean = x.mean(axis=(0, 1), keepdims=True)
            self._std = x.std(axis=(0, 1), keepdims=True)
            self._std = np.clip(self._std, 1e-8, None)
        elif self._cols:
            self._norm.train(TimeSeries.from_pd(df[self._cols]))

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        df = time_series.to_pd().copy()
        if not self._cols:
            return TimeSeries.from_pd(df)
        if self.samplewise_mode:
            x = self._reshape_nld(df[self._cols].values)
            x = (x - self._mean) / self._std
            df.loc[:, self._cols] = self._flatten_nld(x)
        else:
            normed = self._norm(TimeSeries.from_pd(df[self._cols])).to_pd()
            df.loc[:, self._cols] = normed.values
        return TimeSeries.from_pd(df)

    def _reshape_nld(self, values_2d: np.ndarray) -> np.ndarray:
        arr = np.asarray(values_2d, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array [N, L*D], got {arr.shape}")
        n, cols = arr.shape
        d = max(1, self.num_feature)
        if cols % d != 0:
            raise ValueError(f"Column count {cols} not divisible by num_feature {d}")
        l = cols // d
        return arr.reshape(n, l, d)

    @staticmethod
    def _flatten_nld(values_nld: np.ndarray) -> np.ndarray:
        n, l, d = values_nld.shape
        return values_nld.reshape(n, l * d)
