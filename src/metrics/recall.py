import numpy as np
from typing import Optional
from merlion.evaluate.base import EvaluatorConfig
from sklearn.metrics import recall_score
from utils.utils import _to_flat_array

class RecallConfig(EvaluatorConfig):
    def __init__(self, average: str = "weighted", **kwargs):
        super().__init__(**kwargs)
        self.average = average

class Recall:
    config_class = RecallConfig

    def __init__(self, config: Optional[RecallConfig] = None, **kwargs):
        self.config = config or RecallConfig(**kwargs)

    def __call__(self, ground_truth: None, predict: None) -> float:
        y_true = _to_flat_array(ground_truth)
        y_pred = _to_flat_array(predict)
        return float(recall_score(y_true, y_pred, average=self.config.average, zero_division=0))
