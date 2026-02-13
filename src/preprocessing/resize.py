from merlion.transform.base import InvertibleTransformBase
from merlion.utils import TimeSeries
import numpy as np
import pandas as pd


class ResizeTransform(InvertibleTransformBase):
    """
    Resizes time series to a target length using linear interpolation.
    """
    
    def __init__(self, target_length: int, axis: int = 0):
        """
        Args:
            target_len: Target sequence length after resizing
            axis: Axis along which to resize (0=rows/time, 1=columns)
        """
        super().__init__()
        self.target_length = target_length
        self.original_lengths = {}  # Store original lengths for inversion
        self.axis = axis
    
    @property
    def requires_inversion_state(self) -> bool:
        """Requires state to restore original length"""
        return True
    
    def train(self, time_series: TimeSeries):
        """No training required for resizing"""
        pass
    
    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        """Resize time series to target length"""
        df = time_series.to_pd()
        
        # Store original length for inversion
        orig_length = len(df)
        self.inversion_state = {'original_len': orig_length}
        
        if orig_length == self.target_length:
            return time_series
        
        # Resize each column
        resized_data = {}
        for col in df.columns:
            resized_data[col] = self._resize_series(df[col].values, self.target_length)
        
        # Create new index (interpolated timestamps)
        new_index = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            periods=self.target_length
        ) if isinstance(df.index, pd.DatetimeIndex) else np.arange(self.target_length)
        
        resized_df = pd.DataFrame(resized_data, index=new_index)
        return TimeSeries.from_pd(resized_df)
    
    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        """Resize back to original length"""
        if not self.inversion_state:
            return time_series
        
        orig_length = self.inversion_state['original_len']
        df = time_series.to_pd()
        
        if len(df) == orig_length:
            return time_series
        
        # Resize back
        resized_data = {}
        for col in df.columns:
            resized_data[col] = self._resize_series(df[col].values, orig_length)
        
        new_index = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            periods=orig_length
        ) if isinstance(df.index, pd.DatetimeIndex) else np.arange(orig_length)
        
        resized_df = pd.DataFrame(resized_data, index=new_index)
        return TimeSeries.from_pd(resized_df)
    
    @staticmethod
    def _resize_series(s: np.ndarray, target_length: int) -> np.ndarray:
        """Resize 1D array using linear interpolation"""
        orig_len = len(s)
        return np.interp(
            np.arange(0, target_length),
            np.linspace(0, target_length, num=orig_len),
            s
        )
