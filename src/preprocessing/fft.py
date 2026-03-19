import pandas as pd
import numpy as np
from merlion.transform.base import TransformBase
from merlion.utils import TimeSeries
from utils.config import marker


class FFTTransform(TransformBase):
    """Add FFT magnitude features to time series.
    
    Computes absolute value of FFT along sequence dimension for numeric columns.
    """
    
    def __init__(self, numeric_only=True, samplewise_mode: bool = False, num_feature: int = 1, **kwargs):
        if "paper_mode" in kwargs:
            samplewise_mode = kwargs.pop("paper_mode")
        super().__init__()
        self.numeric_only = numeric_only
        self.samplewise_mode = samplewise_mode
        self.num_feature = int(num_feature)
        self.kwargs = {
            "numeric_only": numeric_only,
            "samplewise_mode": samplewise_mode,
            "num_feature": num_feature,
        }
        self._original_names = None
        self._numeric_columns = None

    def to_dict(self):
        d = super().to_dict()
        d["name"] = f"{type(self).__module__}:{type(self).__name__}"
        return d
    
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

        if self.samplewise_mode:
            base_cols = [c for c in df.columns if not str(c).endswith("_derivative") and not str(c).endswith("_fft")]
            arr = df[base_cols].values.astype(float)
            n, cols = arr.shape
            d = max(1, self.num_feature)
            if cols % d != 0:
                raise ValueError(
                    f"Column count {cols} not divisible by num_feature {d} in paper_mode"
                )
            l = cols // d
            x = arr.reshape(n, l, d)
            xf = np.abs(np.fft.fft(x, axis=1)).reshape(n, cols)

            fft_cols = [f"{c}_fft" for c in base_cols]
            existing = [c for c in fft_cols if c in df.columns]
            if existing:
                df = df.drop(columns=existing)

            fft_df = pd.DataFrame(xf, index=df.index, columns=fft_cols)
            self.inversion_state = {
                'original_columns': list(df.columns),
                'fft_columns': fft_cols,
            }
            return TimeSeries.from_pd(pd.concat([df, fft_df], axis=1))
        
        fft_dfs = []
        for col in self._numeric_columns:
            if col not in df.columns or marker in col:
                continue
            
            col_data = df[col].values

            if col_data.ndim == 1:
                fft_values = np.abs(np.fft.fft(col_data))
            else:
                fft_values = np.abs(np.fft.fft(col_data, axis=0))
            
            fft_dfs.append(pd.DataFrame({f"{col}{marker}fft": fft_values}, index=df.index))

        if fft_dfs:
            result_df = pd.concat([df] + fft_dfs, axis=1)
        else:
            result_df = df
        
        self.inversion_state = {
            'original_columns': self._original_names,
            'fft_columns': [f"{col}{marker}fft" for col in self._numeric_columns if col in df.columns]
        }
        
        return TimeSeries.from_pd(result_df)
    
    @property
    def requires_inversion_state(self):
        return False

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        df = time_series.to_pd()
        if self.inversion_state is not None:
            original_cols = self.inversion_state['original_columns']
        else:
            original_cols = [c for c in df.columns if not str(c).endswith("_fft")]
        result_df = df[[col for col in original_cols if col in df.columns]]
        return TimeSeries.from_pd(result_df)
