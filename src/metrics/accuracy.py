import numpy as np
from typing import Optional, Union
from merlion.evaluate.base import EvaluatorConfig
from utils.utils import _to_flat_array

class AccuracyConfig(EvaluatorConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Accuracy:
    config_class = AccuracyConfig

    def __init__(self, config: Optional[AccuracyConfig] = None, **kwargs):
        self.config = config or AccuracyConfig(**kwargs)

    def __call__(
        self,
        ground_truth: None,
        predict: None,
    ) -> float:
        y_true = _to_flat_array(ground_truth)
        y_pred = _to_flat_array(predict)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: [{y_true.shape[0]}, {y_pred.shape[0]}]"
            )
        return float((y_true == y_pred).mean())
