from merlion.models.base import Config


class ClassifierConfig(Config):
    """
    Configuration class for time series classifiers.
    """

    def __init__(
        self,
        num_classes: int = 2,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of target classes for classification
            **kwargs: Additional parameters passed to parent Config
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
