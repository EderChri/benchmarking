import logging
from merlion.models.base import ModelBase
from merlion.utils import TimeSeries
from core.data_splits import DataSplits

logger = logging.getLogger(__name__)

class TaskExecutor:
    def train(self, model: ModelBase, task: str, splits: DataSplits) -> ModelBase:
        if task in ["anomaly_detection", "anomaly", "classification"] and splits.train_labels is not None:
            model.train(splits.train_data, train_labels=splits.train_labels, val_data=splits.val_data, val_labels=splits.val_labels)
        else:
            model.train(splits.train_data)
        return model

    def predict(self, task: str, model: ModelBase, test_data: TimeSeries):
        dispatch = {
            "forecasting": self._forecast, "forecast": self._forecast,
            "anomaly_detection": self._anomaly, "anomaly": self._anomaly,
            "classification": self._classify,
        }
        method = dispatch.get(task.lower())
        if not method:
            raise ValueError(f"Unknown task type: {task}")
        return method(model, test_data)

    def _forecast(self, model, test_data):
        result = model.forecast(time_stamps=test_data.time_stamps)
        return result[0] if isinstance(result, tuple) else result

    def _anomaly(self, model, test_data):
        return model.get_anomaly_score(test_data)

    def _classify(self, model, test_data):
        return model.predict(test_data)

    def classification_scores(self, model: ModelBase, test_data: TimeSeries):
        if hasattr(model, "get_classification_score"):
            return model.get_classification_score(test_data)
        return None
