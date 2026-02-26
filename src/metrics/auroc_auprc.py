import numpy as np
import warnings
from typing import Optional, Union
from merlion.evaluate.base import EvaluatorConfig
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from utils.utils import _to_flat_array, _to_2d_array

class AUROCAUPRCConfig(EvaluatorConfig):
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

class AUROCAUPRC:
    config_class = AUROCAUPRCConfig

    def __init__(self, config: Optional[AUROCAUPRCConfig] = None, **kwargs):
        self.config = config or AUROCAUPRCConfig(**kwargs)
        self.results = {}

    def __call__(
        self,
        ground_truth: None,
        predict: None,  # shape (N, num_classes) — softmax scores
    ) -> "AUROCAUPRC":
        y_true = _to_flat_array(ground_truth, dtype=np.int64)
        y_score = _to_2d_array(predict, dtype=np.float64)

        unique = np.unique(y_true)
        if len(unique) <= 1:
            print(f"Warning: Only one class present (Class {unique[0]}). AUROC/AUPRC undefined.")
            self.results = {"auroc": None, "auprc": None}
            return self

        try:
            if self.config.num_classes == 2:
                if len(unique) != 2:
                    self.results = {"auroc": None, "auprc": None}
                    return self

                neg_label, pos_label = int(unique[0]), int(unique[1])
                y_true_bin = (y_true == pos_label).astype(int)

                if y_score.shape[1] >= 2:
                    if pos_label < y_score.shape[1]:
                        pos_score = y_score[:, pos_label]
                    else:
                        pos_score = y_score[:, 1]
                else:
                    pos_score = y_score[:, 0]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auroc = roc_auc_score(y_true_bin, pos_score)
                    auprc = average_precision_score(y_true_bin, pos_score)
            else:
                if y_score.shape[1] < self.config.num_classes:
                    y_pred = _to_flat_array(predict, dtype=np.int64)
                    if y_pred.shape[0] != y_true.shape[0]:
                        raise ValueError(
                            f"Expected {y_true.shape[0]} predictions, got {y_pred.shape[0]}"
                        )

                    if np.any((y_pred < 0) | (y_pred >= self.config.num_classes)):
                        raise ValueError(
                            f"Predicted class indices out of range [0, {self.config.num_classes - 1}]"
                        )

                    warnings.warn(
                        "AUROC/AUPRC received hard class labels instead of probability scores; "
                        "using one-hot fallback scores. For reliable AUROC/AUPRC, pass class probabilities.",
                        RuntimeWarning,
                    )
                    class_scores = np.zeros((y_pred.shape[0], self.config.num_classes), dtype=np.float64)
                    class_scores[np.arange(y_pred.shape[0]), y_pred] = 1.0
                else:
                    class_scores = y_score[:, :self.config.num_classes]

                y_bin = label_binarize(y_true, classes=range(self.config.num_classes))

                auroc_vals = []
                auprc_vals = []
                for class_idx in range(self.config.num_classes):
                    y_c = y_bin[:, class_idx]
                    if np.unique(y_c).size < 2:
                        continue
                    s_c = class_scores[:, class_idx]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        auroc_vals.append(roc_auc_score(y_c, s_c))
                        auprc_vals.append(average_precision_score(y_c, s_c))

                auroc = float(np.mean(auroc_vals)) if len(auroc_vals) > 0 else None
                auprc = float(np.mean(auprc_vals)) if len(auprc_vals) > 0 else None
            self.results = {
                "auroc": float(auroc) if auroc is not None else None,
                "auprc": float(auprc) if auprc is not None else None,
            }
        except Exception as e:
            print(f"Warning: Could not compute AUROC/AUPRC: {e}")
            self.results = {"auroc": None, "auprc": None}

        return self

    def print_results(self) -> None:
        w = 30
        print(f"\n{'AUROC / AUPRC':^{w}}\n{'-'*w}")
        for k, v in self.results.items():
            val = f"{v:.4f}" if v is not None else "N/A"
            print(f"  {k:<8}: {val}")
        print(f"{'-'*w}\n")

    def to_dict(self) -> dict:
        return self.results
