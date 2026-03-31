from typing import Optional

from merlion.models.forecast.base import ForecasterConfig
from utils.config import marker


class RawForecasterConfig(ForecasterConfig):
    def __init__(
        self,
        num_feature: int = 1,
        num_out_features: int = 1,
        num_embedding: int = 128,
        num_hidden: int = 256,
        dropout: float = 0.1,
        window_size: int = 48,
        forecast_horizon: int = 1,
        target_seq_index: int = 0,
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 20,
        train_stride: int = 1,
        max_train_windows: Optional[int] = None,
        marker: str = marker,
        use_gpu: bool = True,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        self.num_feature = num_feature
        self.num_out_features = num_out_features
        self.num_embedding = num_embedding
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.target_seq_index = target_seq_index
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.train_stride = train_stride
        self.max_train_windows = max_train_windows
        self.marker = marker
        self.use_gpu = use_gpu
        self.checkpoint_path = checkpoint_path
        super().__init__(target_seq_index=target_seq_index, **kwargs)