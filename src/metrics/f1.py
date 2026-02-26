import numpy as np
from typing import Optional
from merlion.evaluate.base import EvaluatorConfig
from sklearn.metrics import f1_score
from utils.utils import _to_flat_array

class F1Config(EvaluatorConfig):
    def __init__(self, average: str = "weighted", **kwargs):
        super().__init__(**kwargs)
        self.average = average

class F1:
    config_class = F1Config

    def __init__(self, config: Optional[F1Config] = None, **kwargs):
        self.config = config or F1Config(**kwargs)

    def __call__(self, ground_truth: None, predict: None) -> float:
        y_true = _to_flat_array(ground_truth)
        y_pred = _to_flat_array(predict)
        return float(f1_score(y_true, y_pred, average=self.config.average, zero_division=0))
