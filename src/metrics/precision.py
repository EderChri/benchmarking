import numpy as np
from typing import Optional
from merlion.evaluate.base import EvaluatorConfig
from sklearn.metrics import precision_score
from utils.utils import _to_flat_array

class PrecisionConfig(EvaluatorConfig):
    def __init__(self, average: str = "weighted", **kwargs):
        super().__init__(**kwargs)
        self.average = average

class Precision:
    config_class = PrecisionConfig

    def __init__(self, config: Optional[PrecisionConfig] = None, **kwargs):
        self.config = config or PrecisionConfig(**kwargs)
        self.results = {}

    def __call__(self, ground_truth: None, predict: None) -> float:
        y_true = _to_flat_array(ground_truth)
        y_pred = _to_flat_array(predict)
        return float(precision_score(y_true, y_pred, average=self.config.average, zero_division=0))
