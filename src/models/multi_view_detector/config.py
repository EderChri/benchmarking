from typing import Optional
from merlion.models.anomaly.base import DetectorConfig


class MultiViewDetectorConfig(DetectorConfig):
    """Configuration for self-supervised multi-view anomaly detector."""

    def __init__(
        self,
        num_feature: int = 1,
        num_embedding: int = 128,
        num_hidden: int = 256,
        num_head: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        feature: str = "latent",
        loss_type: str = "ALL",
        projection_dim: int = 128,
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 20,
        l1_scale: float = 0.0,
        l2_scale: float = 0.01,
        temperature: float = 0.5,
        augmentation_strength: float = 0.1,
        threshold_quantile: float = 0.99,
        use_gpu: bool = True,
        checkpoint_path: Optional[str] = None,
        max_score: float = 1.0,
        threshold=None,
        enable_calibrator: bool = True,
        enable_threshold: bool = True,
        transform=None,
        **kwargs,
    ):
        self.num_feature = num_feature
        self.num_embedding = num_embedding
        self.num_hidden = num_hidden
        self.num_head = num_head
        self.num_layers = num_layers
        self.dropout = dropout

        self.feature = feature
        self.loss_type = loss_type
        self.projection_dim = projection_dim

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_scale = l1_scale
        self.l2_scale = l2_scale
        self.temperature = temperature
        self.augmentation_strength = augmentation_strength

        self.threshold_quantile = threshold_quantile
        self.use_gpu = use_gpu
        self.checkpoint_path = checkpoint_path

        super().__init__(
            max_score=max_score,
            threshold=threshold,
            enable_calibrator=enable_calibrator,
            enable_threshold=enable_threshold,
            transform=transform,
            **kwargs,
        )
