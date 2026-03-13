import logging
from typing import Dict, Any
from core.data_splits import DataSplits
from core.factory import ComponentFactory
from metrics.printable_metric import PrintableMetric
from merlion.utils import TimeSeries
from utils.config import marker

logger = logging.getLogger(__name__)

class MetricEvaluator:
    def __init__(self, factory: ComponentFactory):
        self.factory = factory

    def evaluate(
        self,
        metric_names: list,
        task: str,
        splits: DataSplits,
        predictions,
        metric_configs: Dict[str, Any] = None,
        prediction_scores=None,
    ) -> Dict[str, Any]:
        results = {}
        for name in metric_names:
            try:
                result = self._evaluate_one(
                    name,
                    task,
                    splits,
                    predictions,
                    metric_configs=metric_configs,
                    prediction_scores=prediction_scores,
                )
                if isinstance(result, float):
                    results[name] = float(result)
                    logger.info(f"{name.upper()}: {result:.4f}")
                elif isinstance(result, PrintableMetric):
                    result.print_results()
                    results[name] = result.to_dict()
                else:
                    results[name] = result
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                results[name] = None
        return results

    def _evaluate_one(
        self,
        metric_name: str,
        task: str,
        splits: DataSplits,
        predictions,
        metric_configs: Dict[str, Any] = None,
        prediction_scores=None,
    ):
        metric_cfg = None
        if metric_configs is not None:
            metric_cfg = metric_configs.get(metric_name)
        if metric_cfg is None:
            metric_cfg = self.factory.get_component_by_name(metric_name, "metrics")
        metric_fn = self.factory.instantiate(metric_cfg, "metrics")
        ground_truth = splits.test_labels if task in ["anomaly_detection", "anomaly", "classification"] else splits.test_data
        ground_truth = self._filter_marked_columns(ground_truth)
        if task in ["forecasting", "forecast"]:
            ground_truth = self._align_forecast_ground_truth(ground_truth, predictions)
        if ground_truth is None:
            raise ValueError(f"Metric {metric_name} requires labels for task '{task}'")
        metric_input = predictions
        if metric_name.lower() == "auroc_auprc" and prediction_scores is not None:
            metric_input = prediction_scores
        if hasattr(metric_fn, "value"):
            return metric_fn.value(ground_truth=ground_truth, predict=metric_input)
        return metric_fn(ground_truth=ground_truth, predict=metric_input)
    
    @staticmethod
    def _filter_marked_columns(ts):
        if ts is None:
            return None
        keep_cols = [name for name in ts.names if marker not in str(name)]
        if not keep_cols or keep_cols == ts.names:
            return ts
        return TimeSeries.from_pd(ts.to_pd()[keep_cols])

    @staticmethod
    def _align_forecast_ground_truth(ground_truth, predictions):
        if ground_truth is None or predictions is None:
            return ground_truth

        gt_cols = ground_truth.names
        pred_cols = predictions.names
        if not gt_cols or not pred_cols:
            return ground_truth

        shared_cols = [c for c in pred_cols if c in gt_cols]
        if shared_cols:
            return TimeSeries.from_pd(ground_truth.to_pd()[shared_cols])

        n = min(len(gt_cols), len(pred_cols))
        gt_df = ground_truth.to_pd().iloc[:, :n].copy()
        gt_df.columns = pred_cols[:n]
        return TimeSeries.from_pd(gt_df)