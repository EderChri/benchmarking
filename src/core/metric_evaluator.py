import logging
from typing import Dict, Any
from core.data_splits import DataSplits
from core.factory import ComponentFactory
from metrics.printable_metric import PrintableMetric

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
        if ground_truth is None:
            raise ValueError(f"Metric {metric_name} requires labels for task '{task}'")
        metric_input = predictions
        if metric_name.lower() == "auroc_auprc" and prediction_scores is not None:
            metric_input = prediction_scores
        if hasattr(metric_fn, "value"):
            return metric_fn.value(ground_truth=ground_truth, predict=metric_input)
        return metric_fn(ground_truth=ground_truth, predict=metric_input)
