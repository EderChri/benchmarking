"""ClassifierBase and SupervisedClassifierBase.

ClassifierBase inherits CustomLightningBase and provides the Protocol-facing
interface (train / predict / get_classification_score) that task_executor.py
and metric_evaluator.py rely on.  Merlion models satisfy the same interface
natively without inheriting from here.

The key subtlety: LightningModule (via nn.Module) defines train(mode: bool)
to switch training/eval state.  PL calls this internally.  Our task_executor
calls train(train_data, ...) with actual data as the first argument.

train() discriminates the two call sites:
    isinstance(first_arg, bool)  → nn.Module.train(mode) — delegated to super()
    otherwise                    → our training method — delegates to _train()
"""
from abc import abstractmethod
import logging
from typing import Optional

from merlion.utils import TimeSeries

from models.base import CustomLightningBase

logger = logging.getLogger(__name__)


class ClassifierBase(CustomLightningBase):
    """Protocol-facing base class for classification models."""

    # ------------------------------------------------------------------
    # Public interface (Protocol methods)
    # ------------------------------------------------------------------

    def train(
        self,
        mode_or_data=True,
        train_config=None,
        train_labels: Optional[TimeSeries] = None,
        val_data: Optional[TimeSeries] = None,
        val_labels: Optional[TimeSeries] = None,
    ):
        """Discriminate nn.Module.train(bool) from our training call."""
        if isinstance(mode_or_data, bool):
            # Called internally by PL / nn.Module to set train/eval state.
            return super().train(mode_or_data)

        train_data = mode_or_data
        logger.info(
            f"Training {type(self).__name__} — "
            f"{len(train_data)} train samples, "
            f"{len(val_data) if val_data is not None else 0} val samples"
        )
        return self._train(train_data, train_config, train_labels, val_data, val_labels)

    @abstractmethod
    def _train(
        self,
        train_data: TimeSeries,
        train_config,
        train_labels: Optional[TimeSeries],
        val_data: Optional[TimeSeries] = None,
        val_labels: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """Core training logic — runs Trainer.fit() and returns training scores."""

    def predict(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        self._ensure_ready()
        return self._predict(time_series, time_series_prev)

    @abstractmethod
    def _predict(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        pass

    def get_classification_score(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        self._ensure_ready()
        return self._get_classification_score(time_series, time_series_prev)

    @abstractmethod
    def _get_classification_score(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        pass


class SupervisedClassifierBase(ClassifierBase):
    """Enforces that training labels are always provided."""

    def train(
        self,
        mode_or_data=True,
        train_config=None,
        train_labels: Optional[TimeSeries] = None,
        val_data: Optional[TimeSeries] = None,
        val_labels: Optional[TimeSeries] = None,
    ):
        if not isinstance(mode_or_data, bool) and train_labels is None:
            raise ValueError(
                f"{type(self).__name__} requires training labels (train_labels=...)"
            )
        return super().train(mode_or_data, train_config, train_labels, val_data, val_labels)
