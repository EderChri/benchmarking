import contextlib
import logging
import re
from typing import Any, Dict, Optional

from merlion.models.base import ModelBase
from merlion.utils import TimeSeries
from core.data_splits import DataSplits

logger = logging.getLogger(__name__)


class _EpochMetricsLogHandler(logging.Handler):
    """Parse epoch logs and stream extracted metrics to MLflow."""

    def __init__(self, pattern: str, tracker: Any):
        super().__init__(level=logging.INFO)
        self._regex = re.compile(pattern)
        self._tracker = tracker

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        match = self._regex.search(message)
        if not match:
            return

        groups = match.groupdict()
        if not groups:
            return

        epoch_raw = groups.pop("epoch", None)
        if epoch_raw is None:
            return

        try:
            step = int(epoch_raw)
        except (TypeError, ValueError):
            return

        metrics = {}
        for key, value in groups.items():
            if value is None:
                continue
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue

        if not metrics:
            return

        try:
            self._tracker.log_metrics(metrics, step=step)
        except Exception as exc:
            logger.warning(f"Failed to log parsed epoch metrics to mlflow: {exc}")


class TaskExecutor:
    def train(
        self,
        model: ModelBase,
        task: str,
        splits: DataSplits,
        tracker: Optional[Any] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> ModelBase:
        with self._mlflow_epoch_logging(tracker=tracker, model_config=model_config):
            if task in ["anomaly_detection", "anomaly", "classification"] and splits.train_labels is not None:
                model.train(
                    splits.train_data,
                    train_labels=splits.train_labels,
                    val_data=splits.val_data,
                    val_labels=splits.val_labels,
                )
            else:
                model.train(splits.train_data)
        return model

    @contextlib.contextmanager
    def _mlflow_epoch_logging(
        self,
        tracker: Optional[Any],
        model_config: Optional[Dict[str, Any]],
    ):
        if not tracker or not getattr(tracker, "enabled", False):
            yield
            return

        mlflow_cfg = (model_config or {}).get("mlflow") or {}
        pattern = mlflow_cfg.get("epoch_log_pattern")
        if not pattern:
            yield
            return

        handler = _EpochMetricsLogHandler(pattern=pattern, tracker=tracker)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            yield
        finally:
            root_logger.removeHandler(handler)

    def predict(self, task: str, model: ModelBase, splits: "DataSplits", model_config: dict = None):
        dispatch = {
            "forecasting": self._forecast, "forecast": self._forecast,
            "anomaly_detection": self._anomaly, "anomaly": self._anomaly,
            "classification": self._classify,
        }
        method = dispatch.get(task.lower())
        if not method:
            raise ValueError(f"Unknown task type: {task}")
        return method(model, splits, model_config)

    def _forecast(self, model, splits, model_config):
        from lightning import LightningModule
        if isinstance(model, LightningModule):
            result = model.forecast(time_stamps=splits.test_data.time_stamps, time_series_prev=splits.test_data)
            return result[0] if isinstance(result, tuple) else result
        if (model_config or {}).get("params", {}).get("rolling_forecast", False):
            return self._rolling_forecast(model, splits)
        result = model.forecast(time_stamps=splits.test_data.time_stamps, time_series_prev=None)
        return result[0] if isinstance(result, tuple) else result

    def _rolling_forecast(self, model, splits):
        import pandas as pd
        step = getattr(model.config, "max_forecast_steps", None) or 1
        # Build initial context from train + val
        ctx_df = splits.train_data.to_pd()
        if splits.val_data is not None:
            ctx_df = pd.concat([ctx_df, splits.val_data.to_pd()])
        ctx = TimeSeries.from_pd(ctx_df)

        test_df = splits.test_data.to_pd()
        chunks = []
        i = 0
        while i < len(test_df):
            chunk_df = test_df.iloc[i:i + step]
            chunk_ts = TimeSeries.from_pd(chunk_df)
            result = model.forecast(time_stamps=chunk_ts.time_stamps, time_series_prev=ctx)
            pred = result[0] if isinstance(result, tuple) else result
            chunks.append(pred.to_pd())
            ctx = TimeSeries.from_pd(pd.concat([ctx_df, test_df.iloc[: i + step]]))
            i += step
        return TimeSeries.from_pd(pd.concat(chunks))

    def _anomaly(self, model, splits, model_config):
        return model.get_anomaly_score(splits.test_data)

    def _classify(self, model, splits, model_config):
        return model.predict(splits.test_data)

    def classification_scores(self, model: ModelBase, test_data: TimeSeries):
        if hasattr(model, "get_classification_score"):
            return model.get_classification_score(test_data)
        return None
