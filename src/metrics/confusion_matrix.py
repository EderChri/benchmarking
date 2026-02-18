import numpy as np
import pandas as pd
from typing import Any, Dict, Union, Optional, List
from merlion.evaluate.base import EvaluatorBase, EvaluatorConfig
from merlion.utils import TimeSeries


class ConfusionMatrixConfig(EvaluatorConfig):
    """Configuration for Confusion Matrix metric."""

    def __init__(self, threshold: float = 0.5, **kwargs):
        """
        Args:
            threshold: Classification threshold for converting scores to predictions
        """
        super().__init__(**kwargs)
        self.threshold = threshold


class ConfusionMatrix():
    """
    Confusion Matrix evaluator for classification tasks.

    Returns a dictionary with TP, TN, FP, FN counts.
    """

    config_class = ConfusionMatrixConfig

    def __init__(self, config: Optional[ConfusionMatrixConfig] = None, threshold: float = 0.5, **kwargs):
        """
        Args:
            config: Configuration object (optional)
            threshold: Classification threshold if config not provided
        """
        if config is None:
            config = ConfusionMatrixConfig(threshold=threshold, **kwargs)
        self.config = config

    def _call_model(
        self,
        ground_truth: Union[TimeSeries, pd.DataFrame, np.ndarray],
        predict: Union[TimeSeries, pd.DataFrame, np.ndarray],
    ) -> dict:
        """
        Compute confusion matrix.

        Args:
            ground_truth: Ground truth labels (0 or 1)
            predict: Predicted scores or labels

        Returns:
            Dictionary with confusion matrix values
        """
        y_true = self._to_array(ground_truth).flatten()
        y_pred_scores = self._to_array(predict).flatten()

        # Convert scores to binary predictions using threshold
        y_pred = (y_pred_scores >= self.config.threshold).astype(int)

        # Ensure same length
        assert len(y_true) == len(y_pred), \
            f"Length mismatch: ground_truth={len(y_true)}, predict={len(y_pred)}"

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Create 2x2 confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        self.results = {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'confusion_matrix': cm.tolist(),
            'total': int(tp + tn + fp + fn)
        }

        return self

    def evaluate(
        self,
        ground_truth: Union[TimeSeries, List[TimeSeries]],
        predict: Union[TimeSeries, List[TimeSeries]],
        metric_name: Optional[str] = None
    ) -> Union[float, dict]:
        """
        Evaluate confusion matrix for single or multiple time series.

        Args:
            ground_truth: Ground truth labels
            predict: Predictions
            metric_name: Optional specific metric to return

        Returns:
            Confusion matrix dictionary or specific metric value
        """
        # Handle list of time series (aggregate results)
        if isinstance(ground_truth, list) and isinstance(predict, list):
            all_true = []
            all_pred = []
            for gt, pred in zip(ground_truth, predict):
                all_true.append(self._to_array(gt).flatten())
                all_pred.append(self._to_array(pred).flatten())

            ground_truth = np.concatenate(all_true)
            predict = np.concatenate(all_pred)

        # Compute confusion matrix
        result = self._call_model(ground_truth, predict)

        # Return specific metric if requested
        if metric_name:
            if metric_name in result:
                return result[metric_name]
            elif metric_name.lower() == 'accuracy':
                return (result['TP'] + result['TN']) / result['total'] if result['total'] > 0 else 0.0
            else:
                raise ValueError(f"Unknown metric name: {metric_name}")

        return result

    def __call__(
        self,
        ground_truth: Union[TimeSeries, pd.DataFrame, np.ndarray],
        predict: Union[TimeSeries, pd.DataFrame, np.ndarray],
    ) -> dict:
        """Direct call interface."""
        return self._call_model(ground_truth, predict)

    def _to_array(self, data: Union[TimeSeries, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert various input types to numpy array."""
        if isinstance(data, TimeSeries):
            return data.to_pd().values
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

    def print_results(self) -> None:
        """Prints a formatted 2x2 confusion matrix to the console."""
        if not self.results:
            print("No results calculated yet.")
            return

        res = self.results
        # Formatting variables for alignment
        title = "CONFUSION MATRIX"
        width = 30

        print(f"\n{'-'*width}")
        print(f"{title.center(width)}")
        print(f"{'-'*width}")

        # Grid layout:
        #             Predicted 0    Predicted 1
        # Actual 0      TN             FP
        # Actual 1      FN             TP

        print(f"{'':<12} | {'Pred 0':<7} | {'Pred 1':<7}")
        print(f"{'-'*width}")
        print(f"{'Actual 0':<12} | {res['TN']:<7} | {res['FP']:<7}")
        print(f"{'Actual 1':<12} | {res['FN']:<7} | {res['TP']:<7}")
        print(f"{'-'*width}")
        print(f"Total Samples: {res['total']}")
        print(f"{'-'*width}\n")

    def to_dict(self) -> Dict[str, Any]:
        """Returns the raw dictionary for YAML serialization."""
        return self.results if self.results else {}


class NormalizedConfusionMatrix(ConfusionMatrix):
    """Returns row-normalized confusion matrix (percentages per class)."""

    def _call_model(
        self,
        ground_truth: Union[TimeSeries, pd.DataFrame, np.ndarray],
        predict: Union[TimeSeries, pd.DataFrame, np.ndarray],
    ) -> "NormalizedConfusionMatrix":
        """Return normalized confusion matrix."""
        cm_dict = super()._call_model(ground_truth, predict).results
        cm = np.array(cm_dict['confusion_matrix'])

        # Row-wise normalization
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)

        self.results = {
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'TP': cm_dict['TP'],
            'TN': cm_dict['TN'],
            'FP': cm_dict['FP'],
            'FN': cm_dict['FN'],
            'total': cm_dict['total']
        }

        return self
