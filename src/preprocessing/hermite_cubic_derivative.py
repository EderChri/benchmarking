import pandas as pd
from merlion.transform.base import TransformBase
from merlion.utils import TimeSeries
import torchcde
import torch
import numpy as np


class HermiteCubicDerivativeTransform(TransformBase):
    """Add first-order derivative using Hermite cubic splines with backward differences.
    
    For multivariate series, computes derivatives for numeric columns only.
    """
    
    def __init__(self, numeric_only=True):
        """
        Args:
            numeric_only: If True, only compute derivatives for numeric columns
        """
        super().__init__()
        self.numeric_only = numeric_only
        self._original_names = None
        self._numeric_columns = None
    
    @property
    def requires_inversion_state(self):
        return True
    
    def train(self, time_series: TimeSeries):
        """Identify numeric columns during training"""
        df = time_series.to_pd()
        self._original_names = list(df.columns)
        
        if self.numeric_only:
            self._numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        else:
            self._numeric_columns = self._original_names
    
    def _compute_hermite_derivative(self, values: np.ndarray) -> np.ndarray:
        """
        Compute derivative using Hermite cubic splines with backward differences.
        Taken from https://github.com/yongkyung-oh/Multi-View_Contrastive_Learning/blob/main/src/dataloader.py#L51
        
        Args:
            values: Array of shape (L, D) where L is sequence length, D is dimensions
            
        Returns:
            Derivative array of shape (L, D)
        """
        L, D = values.shape
        
        X = torch.tensor(values, dtype=torch.float32).unsqueeze(0)  # Shape: (1, L, D)
        
        # Create normalized time vector [0, 1]
        t = torch.linspace(0, 1, L, dtype=torch.float32)
        
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
        spline = torchcde.CubicSpline(coeffs, t)
        
        dx = spline.derivative(t)  # Shape: (1, L, D)
        
        return dx.squeeze(0).numpy()
    
    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        """Apply Hermite cubic derivative transform and stack with original series"""
        df = time_series.to_pd()
        
        # Collect numeric columns data
        numeric_data = []
        numeric_cols_present = []
        
        for col in self._numeric_columns:
            if col in df.columns:
                numeric_data.append(df[col].values)
                numeric_cols_present.append(col)
        
        if not numeric_data:
            # No numeric columns, return original
            return time_series
        
        # Stack into (L, D) array
        X = np.column_stack(numeric_data)
        
        # Compute derivatives using Hermite cubic splines
        dx = self._compute_hermite_derivative(X)
        
        derivative_cols = {f"{col}_derivative": dx[:, i] 
                          for i, col in enumerate(numeric_cols_present)}
        derivative_df = pd.DataFrame(derivative_cols, index=df.index)
        
        result_df = pd.concat([df, derivative_df], axis=1)
        
        # Store metadata for inversion
        self.inversion_state = {
            'original_columns': self._original_names,
            'derivative_columns': list(derivative_cols.keys())
        }
        
        return TimeSeries.from_pd(result_df)
    
    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        """Remove derivative columns to recover original series"""
        if self.inversion_state is None:
            raise RuntimeError("Transform must be applied before inversion")
        
        df = time_series.to_pd()
        original_cols = self.inversion_state['original_columns']
        
        # Keep only original columns
        result_df = df[[col for col in original_cols if col in df.columns]]
        return TimeSeries.from_pd(result_df)