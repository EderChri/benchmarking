"""Log-signature and time-augmentation preprocessing transforms.

Column naming convention (consistent with existing transforms):
    Domain columns are identified by the presence of the marker string ("_MV_").
    Base columns contain no marker.  Specifically:
        - Base columns:    marker not in col
        - Logsig columns:  marker in col and col.endswith("logsig")
        - Logsig names:    f"ls_{i}{marker}logsig"  →  "ls_0_MV_logsig"

WindowedLogSigTransform
    Computes a rolling log-signature and appends logsig_dim columns.

    Non-samplewise (matrix) mode — series shape [T, D]:
        For each timestep t, computes log_signature(x[max(0,t-w):t+1], depth)
        and appends logsig_dim columns  ls_0_MV_logsig … ls_{k}_MV_logsig.
        Directly analogous to FFTTransform in non-samplewise mode.

    Samplewise mode — each row is a flattened [L*D] sample:
        For each sample, computes rolling log-sig over the L timesteps and stores
        the result as L*logsig_dim columns  ls_0_MV_logsig … ls_{L*k-1}_MV_logsig.
        The NView model detects these via the _logsig suffix and reshapes them to
        [n, L, logsig_dim], matching the shape contract of _derivative/_fft columns.

TimeAugmentTransform
    Appends a normalised time index as extra base columns (part of xt).
    Increment num_feature by 1 when using this transform.

    Non-samplewise: the DataFrame index (Unix timestamps or integer positions) is
        converted to float and normalised as (t - t_min) / (t_max - t_min) where
        t_min/t_max are stored during train().  Test data uses the same scale.

    Samplewise: each row is a full sample with no per-timestep timestamps.
        A within-sample [0..1] linspace of length L is used — the best available
        approximation to a true per-timestep timestamp.
"""

import logging

import numpy as np
import pandas as pd
import torch
from merlion.transform.base import TransformBase
from merlion.utils import TimeSeries

from utils.config import marker

logger = logging.getLogger(__name__)

try:
    from log_signatures_pytorch import log_signature, logsigdim as _compute_logsigdim
    _LOGSIG_AVAILABLE = True
except ImportError:
    _LOGSIG_AVAILABLE = False


def _require_logsig():
    if not _LOGSIG_AVAILABLE:
        raise ImportError(
            "log_signatures_pytorch is required for log-signature preprocessing. "
            "Install with: pip install log-signatures-pytorch"
        )


def _base_cols(df: pd.DataFrame) -> list:
    """Columns that are not domain-derived (contain no marker)."""
    return [c for c in df.columns if marker not in str(c)]


def _logsig_cols(df: pd.DataFrame) -> list:
    """Columns that are log-signature domain columns."""
    return [c for c in df.columns if marker in str(c) and str(c).endswith("logsig")]


def _make_logsig_col_names(n: int, offset: int = 0) -> list:
    return [f"ls_{offset + i}{marker}logsig" for i in range(n)]


def _index_to_float(index: pd.Index) -> np.ndarray:
    """Convert a pandas Index to a float64 array of numeric values.

    DatetimeIndex → Unix timestamps (seconds).
    Anything else → cast to float64 directly.
    """
    if isinstance(index, pd.DatetimeIndex):
        return index.astype(np.int64).astype(np.float64) / 1e9
    try:
        return index.astype(np.float64).values
    except (TypeError, ValueError):
        return np.arange(len(index), dtype=np.float64)


# ---------------------------------------------------------------------------
# WindowedLogSigTransform
# ---------------------------------------------------------------------------

class WindowedLogSigTransform(TransformBase):
    """Rolling log-signature stored as per-timestep / per-sample columns.

    Parameters
    ----------
    depth : int
        Truncation depth.  Depth 2 captures pairwise Lévy-area (Lie bracket)
        terms between channels.  logsig_dim grows combinatorially with depth.
    inner_window : int
        Number of timesteps in each rolling window.
    use_time_augmentation : bool
        Prepend a within-window [0..1] time channel before computing the
        log-signature.  Makes univariate paths non-trivial by introducing
        cross-terms between the time channel and the feature channel.
    samplewise_mode : bool
        If True, each DataFrame row is a flattened [L*D] sample.
    num_feature : int
        Feature dimension D per timestep (required for samplewise reshape).
    numeric_only : bool
        Restrict log-sig to numeric base columns (default True).
    """

    def __init__(
        self,
        depth: int = 2,
        inner_window: int = 24,
        use_time_augmentation: bool = False,
        samplewise_mode: bool = False,
        num_feature: int = 1,
        numeric_only: bool = True,
        **kwargs,
    ):
        _require_logsig()
        super().__init__()
        self.depth = int(depth)
        self.inner_window = int(inner_window)
        self.use_time_augmentation = bool(use_time_augmentation)
        self.samplewise_mode = bool(samplewise_mode)
        self.num_feature = int(num_feature)
        self.numeric_only = bool(numeric_only)
        self.kwargs = {
            "depth": self.depth,
            "inner_window": self.inner_window,
            "use_time_augmentation": self.use_time_augmentation,
            "samplewise_mode": self.samplewise_mode,
            "num_feature": self.num_feature,
            "numeric_only": self.numeric_only,
        }
        self._source_cols = None
        self._logsig_dim = None

    def to_dict(self):
        d = super().to_dict()
        d["name"] = f"{type(self).__module__}:{type(self).__name__}"
        return d

    @property
    def requires_inversion_state(self):
        return False

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def train(self, time_series: TimeSeries):
        df = time_series.to_pd()
        cols = _base_cols(df)
        if self.numeric_only:
            cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        self._source_cols = cols

        if self.samplewise_mode:
            path_dim = self.num_feature + (1 if self.use_time_augmentation else 0)
        else:
            path_dim = len(cols) + (1 if self.use_time_augmentation else 0)

        self._logsig_dim = _compute_logsigdim(path_dim, self.depth)
        logger.info(
            f"WindowedLogSigTransform: path_dim={path_dim}, depth={self.depth}, "
            f"logsig_dim={self._logsig_dim}, samplewise={self.samplewise_mode}"
        )

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        if self.samplewise_mode:
            return self._call_samplewise(time_series)
        return self._call_matrix(time_series)

    # --- non-samplewise -----------------------------------------------

    def _call_matrix(self, time_series: TimeSeries) -> TimeSeries:
        df = time_series.to_pd()
        drop = _logsig_cols(df)
        if drop:
            df = df.drop(columns=drop)

        vals = df[self._source_cols].values.astype(np.float32)  # [T, D]
        T, D = vals.shape
        w = self.inner_window
        path_dim = D + (1 if self.use_time_augmentation else 0)
        time_template = np.linspace(0.0, 1.0, w, dtype=np.float32)

        windows = np.zeros((T, w, path_dim), dtype=np.float32)
        feat_start = 0
        if self.use_time_augmentation:
            windows[:, :, 0] = time_template
            feat_start = 1

        for t in range(T):
            seg = vals[max(0, t - w + 1): t + 1]
            pad = w - len(seg)
            windows[t, pad:, feat_start:] = seg

        logsig_np = self._batch_logsig(windows)  # [T, logsig_dim]
        col_names = _make_logsig_col_names(self._logsig_dim)
        logsig_df = pd.DataFrame(logsig_np, index=df.index, columns=col_names)
        return TimeSeries.from_pd(pd.concat([df, logsig_df], axis=1))

    # --- samplewise ---------------------------------------------------

    def _call_samplewise(self, time_series: TimeSeries) -> TimeSeries:
        df = time_series.to_pd()
        drop = _logsig_cols(df)
        if drop:
            df = df.drop(columns=drop)

        base = _base_cols(df)
        arr = df[base].values.astype(np.float32)  # [n, L*D]
        n, total = arr.shape
        D = max(1, self.num_feature)
        if total % D != 0:
            raise ValueError(
                f"Column count {total} not divisible by num_feature {D}"
            )
        L = total // D
        samples = arr.reshape(n, L, D)  # [n, L, D]

        w = self.inner_window
        path_dim = D + (1 if self.use_time_augmentation else 0)
        time_template = (
            np.linspace(0.0, 1.0, w, dtype=np.float32)
            if self.use_time_augmentation else None
        )

        # Output stored as [n, L*logsig_dim]; model reshapes to [n, L, logsig_dim].
        logsig_flat = np.zeros((n, L * self._logsig_dim), dtype=np.float32)

        for s in range(n):
            x = samples[s]  # [L, D]
            windows = np.zeros((L, w, path_dim), dtype=np.float32)
            feat_start = 0
            if self.use_time_augmentation:
                windows[:, :, 0] = time_template
                feat_start = 1
            for t in range(L):
                seg = x[max(0, t - w + 1): t + 1]
                pad = w - len(seg)
                windows[t, pad:, feat_start:] = seg
            ls = self._batch_logsig(windows)  # [L, logsig_dim]
            logsig_flat[s] = ls.reshape(L * self._logsig_dim)

        col_names = _make_logsig_col_names(L * self._logsig_dim)
        logsig_df = pd.DataFrame(logsig_flat, index=df.index, columns=col_names)
        return TimeSeries.from_pd(pd.concat([df, logsig_df], axis=1))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _batch_logsig(self, windows: np.ndarray) -> np.ndarray:
        """Compute log_signature for a batch of windows → [N, logsig_dim]."""
        with torch.no_grad():
            ls = log_signature(torch.tensor(windows, dtype=torch.float32), self.depth)
        return ls.numpy().astype(np.float32)

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        df = time_series.to_pd()
        return TimeSeries.from_pd(df.drop(columns=_logsig_cols(df)))


# ---------------------------------------------------------------------------
# TimeAugmentTransform
# ---------------------------------------------------------------------------

class TimeAugmentTransform(TransformBase):
    """Append a normalised time index as extra base (xt) columns.

    Non-samplewise:
        The DataFrame index (Unix timestamps or integer positions) is converted
        to float and normalised as (t - t_min) / (t_max - t_min), where t_min
        and t_max are stored during train().  Test data is mapped with the same
        scale so values outside [0, 1] are possible and correct.
        Adds one column  "time_global"  with no marker — it becomes part of xt.

    Samplewise:
        Each row is a full sample; there are no per-timestep timestamps.
        A within-sample linspace [0..1] of length L is used, which is the best
        available approximation to a true per-timestep position signal.
        Adds L columns  "time_t0" … "time_t{L-1}"  with no marker.

    Increment num_feature by 1 when using this transform.
    """

    def __init__(self, samplewise_mode: bool = False, num_feature: int = 1, **kwargs):
        super().__init__()
        self.samplewise_mode = bool(samplewise_mode)
        self.num_feature = int(num_feature)
        self.kwargs = {
            "samplewise_mode": self.samplewise_mode,
            "num_feature": self.num_feature,
        }
        self._t_min = None
        self._t_max = None

    def to_dict(self):
        d = super().to_dict()
        d["name"] = f"{type(self).__module__}:{type(self).__name__}"
        return d

    @property
    def requires_inversion_state(self):
        return False

    def train(self, time_series: TimeSeries):
        if not self.samplewise_mode:
            t = _index_to_float(time_series.to_pd().index)
            self._t_min = float(t.min())
            self._t_max = float(t.max())
            if self._t_max == self._t_min:
                self._t_max = self._t_min + 1.0   # degenerate single-row series

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        df = time_series.to_pd()
        if self.samplewise_mode:
            return self._call_samplewise(df)
        return self._call_matrix(df)

    def _call_matrix(self, df: pd.DataFrame) -> TimeSeries:
        t = _index_to_float(df.index)
        t_norm = (t - self._t_min) / (self._t_max - self._t_min)
        df = df.copy()
        # Plain name, no marker — this is a base feature column part of xt.
        df["time_global"] = t_norm.astype(np.float32)
        return TimeSeries.from_pd(df)

    def _call_samplewise(self, df: pd.DataFrame) -> TimeSeries:
        base = _base_cols(df)
        n, total = df[base].shape
        D = max(1, self.num_feature)
        L = total // D
        time_arr = np.tile(np.linspace(0.0, 1.0, L, dtype=np.float32), (n, 1))
        col_names = [f"time_t{t}" for t in range(L)]
        time_df = pd.DataFrame(time_arr, index=df.index, columns=col_names)
        return TimeSeries.from_pd(pd.concat([df, time_df], axis=1))

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        df = time_series.to_pd()
        drop = [c for c in df.columns if str(c).startswith("time_")]
        return TimeSeries.from_pd(df.drop(columns=drop))
