import pandas as pd
import numpy as np
from merlion.transform.base import TransformBase
from merlion.utils import TimeSeries


class FFTTransform(TransformBase):
    """Add FFT magnitude features to time series.
    
    Computes absolute value of FFT along sequence dimension for numeric columns.
    """
    
    def __init__(self, numeric_only=True):
        super().__init__()
        self.numeric_only = numeric_only
        self._original_names = None
        self._numeric_columns = None
    
    @property
    def requires_inversion_state(self):
        return True
    
    def train(self, time_series: TimeSeries):
        df = time_series.to_pd()
        self._original_names = list(df.columns)
        
        if self.numeric_only:
            self._numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        else:
            self._numeric_columns = self._original_names
    
    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        df = time_series.to_pd()
        
        fft_dfs = []
        for col in self._numeric_columns:
            if col not in df.columns:
                continue
            
            # Compute FFT magnitude along sequence dimension
            fft_values = np.abs(np.fft.fft(df[col].values))
            fft_dfs.append(pd.DataFrame({f"{col}_fft": fft_values}, index=df.index))
        
        if fft_dfs:
            result_df = pd.concat([df] + fft_dfs, axis=1)
        else:
            result_df = df
        
        self.inversion_state = {
            'original_columns': self._original_names,
            'fft_columns': [f"{col}_fft" for col in self._numeric_columns if col in df.columns]
        }
        
        return TimeSeries.from_pd(result_df)
    
    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        if self.inversion_state is None:
            raise RuntimeError("Transform must be applied before inversion")
        
        df = time_series.to_pd()
        original_cols = self.inversion_state['original_columns']
        result_df = df[[col for col in original_cols if col in df.columns]]
        return TimeSeries.from_pd(result_df)
