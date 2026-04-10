from typing import Optional
from merlion.models.anomaly.base import DetectorConfig


class RawClassifierConfig(DetectorConfig):
    """Config for RawClassifier.

    input_mode:
        "raw"       — only the time domain (xt) is used.
        "multiview" — time, derivative, and FFT domains (xt, dx, xf) are used.
    """

    def __init__(
        self,
        num_feature: int = 1,
        num_target: int = 2,

        batch_size: int = 32,
        num_epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 20,
        l2_scale: float = 0.01,

        input_mode: str = "raw",   # "raw" | "multiview"
        finetune_monitor_metric: str = "loss",
        use_gpu: bool = True,

        max_score: float = 1.0,
        threshold=None,
        enable_calibrator: bool = True,
        enable_threshold: bool = True,
        transform=None,
        **kwargs,
    ):
        self.num_feature = num_feature
        self.num_target = num_target

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.l2_scale = l2_scale

        self.input_mode = input_mode
        self.finetune_monitor_metric = finetune_monitor_metric
        self.use_gpu = use_gpu

        super().__init__(
            max_score=max_score,
            threshold=threshold,
            enable_calibrator=enable_calibrator,
            enable_threshold=enable_threshold,
            transform=transform,
            **kwargs,
        )
