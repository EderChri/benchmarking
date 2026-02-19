from merlion.transform.base import InvertibleTransformBase
from merlion.utils import TimeSeries
import numpy as np
import pandas as pd

class ResizeTransform(InvertibleTransformBase):
    """
    Resizes time series to a target length using linear interpolation.
    axis=0: Resizes the number of time steps (rows).
    axis=1: Resizes the number of features/metrics (columns).
    """
    
    def __init__(self, target_length: int, axis: int = 0):
        super().__init__()
        self.target_length = target_length
        self.axis = axis
    
    @property
    def requires_inversion_state(self) -> bool:
        return True
    
    def train(self, time_series: TimeSeries):
        pass
    
    def _apply_resize(self, df: pd.DataFrame, target: int) -> pd.DataFrame:
        """Helper to handle interpolation logic for both __call__ and _invert"""
        # Case 1: Resize along Time (Rows)
        if self.axis == 0:
            if len(df) == target:
                return df
            
            # Interpolate each column to new length
            resized_data = {
                col: self._resize_series(df[col].values, target) 
                for col in df.columns
            }
            
            # Generate new index
            new_index = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                periods=target
            ) if isinstance(df.index, pd.DatetimeIndex) else np.linspace(0, 1, target)
            
            return pd.DataFrame(resized_data, index=new_index)

        # Case 2: Resize along Features (Columns)
        elif self.axis == 1:
            if df.shape[1] == target:
                return df
            
            # Interpolate each row across the columns
            # We transpose to use the same _resize_series logic easily
            arr = df.values # Shape (N, D)
            new_values = np.array([
                self._resize_series(row, target) for row in arr
            ])
            
            new_cols = [f"metric_{i}" for i in range(target)]
            return pd.DataFrame(new_values, index=df.index, columns=new_cols)
        
        else:
            raise ValueError("Axis must be 0 (time) or 1 (metrics).")

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        df = time_series.to_pd()
        # Store original dimension for inversion
        self.inversion_state = {'original_dim': df.shape[self.axis]}
        
        resized_df = self._apply_resize(df, self.target_length)
        return TimeSeries.from_pd(resized_df)
    
    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        if not self.inversion_state:
            return time_series
        
        orig_dim = self.inversion_state['original_dim']
        df = time_series.to_pd()
        
        inverted_df = self._apply_resize(df, orig_dim)
        return TimeSeries.from_pd(inverted_df)
    
    @staticmethod
    def _resize_series(s: np.ndarray, target_length: int) -> np.ndarray:
        orig_len = len(s)
        old_time = np.linspace(0, 1, num=orig_len)
        new_time = np.linspace(0, 1, num=target_length)
        return np.interp(new_time, old_time, s)