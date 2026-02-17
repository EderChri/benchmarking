from abc import abstractmethod
from typing import Optional, Union, Tuple
from models.classifier_base.config import ClassifierConfig
import pandas as pd
from merlion.utils import TimeSeries, UnivariateTimeSeries
from merlion.models.base import ModelBase, Config


class ClassifierBase(ModelBase):
    """
    Base class for time series sequence classification models.

    For compatibility with Merlion's evaluation framework, the classification
    score is broadcast to all timestamps in the sequence.
    """

    config_class = ClassifierConfig

    @property
    @abstractmethod
    def require_even_sampling(self) -> bool:
        """Whether the model requires evenly sampled time series."""
        pass

    @property
    @abstractmethod
    def require_univariate(self) -> bool:
        """Whether the model only works with univariate time series."""
        pass

    def __init__(self, config: ClassifierConfig):
        super().__init__(config)

    def train(
        self,
        train_data: TimeSeries,
        train_config=None,
        train_labels: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """
        Train the classifier on time series data.

        Args:
            train_data: Training time series data. Can be:
                - Single sequence: shape [seq_len, features]
                - Multiple sequences: concatenated with proper indexing
            train_config: Additional training configuration
            train_labels: Classification labels for training data.
                Should be a TimeSeries with integer class labels.

        Returns:
            Training classification scores as TimeSeries
        """
        if train_labels is None:
            raise ValueError(
                f"{type(self).__name__} requires labels for training")

        return self._train(train_data, train_config, train_labels)

    @abstractmethod
    def _train(
        self,
        train_data: TimeSeries,
        train_config=None,
        train_labels: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """
        Core training logic. Must be implemented by subclasses.

        Returns:
            Classification scores for training data
        """
        pass

    def predict(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """
        Predict class labels for time series sequences.

        Args:
            time_series: Time series to classify
            time_series_prev: Optional previous context (not used by most classifiers)

        Returns:
            TimeSeries with predicted class labels broadcast to all timestamps
        """
        return self._predict(time_series, time_series_prev)

    @abstractmethod
    def _predict(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """
        Core prediction logic. Must be implemented by subclasses.

        Should return class predictions as integers.
        """
        pass

    def get_classification_score(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """
        Get classification confidence scores (probabilities) for time series.

        Args:
            time_series: Time series to score
            time_series_prev: Optional previous context

        Returns:
            TimeSeries with classification scores (e.g., probability of positive class)
            broadcast to all timestamps for Merlion compatibility
        """
        return self._get_classification_score(time_series, time_series_prev)

    @abstractmethod
    def _get_classification_score(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """
        Core scoring logic. Must be implemented by subclasses.

        Should return a univariate TimeSeries with classification scores
        (typically probability of positive class for binary classification).
        Score should be broadcast to all timestamps in the sequence.
        """
        pass

    def _broadcast_score_to_timestamps(
        self,
        sequence_score: float,
        time_series: TimeSeries
    ) -> TimeSeries:
        """
        Broadcast a single sequence-level score to all timestamps.

        Helper method for compatibility with Merlion's per-timestamp paradigm.

        Args:
            sequence_score: Single score for the entire sequence
            time_series: Original time series (for index/length)

        Returns:
            Univariate TimeSeries with score repeated for each timestamp
        """
        import numpy as np

        ts_df = time_series.to_pd()
        num_timestamps = len(ts_df)

        scores_np = np.full(num_timestamps, sequence_score)

        score_df = pd.DataFrame(
            scores_np.reshape(-1, 1),
            index=ts_df.index,
            columns=["class_score"]
        )

        return TimeSeries.from_pd(score_df)


class SupervisedClassifierBase(ClassifierBase):
    """
    Base class for supervised time series classifiers that require labels.

    This is a convenience class for models that always need labels during training.
    """

    def train(
        self,
        train_data: TimeSeries,
        train_config=None,
        train_labels: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """Train with mandatory labels."""
        if train_labels is None:
            raise ValueError(
                f"{type(self).__name__} is a supervised classifier and requires training labels"
            )
        return super().train(train_data, train_config, train_labels)
