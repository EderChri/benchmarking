from typing import Optional

from merlion.models.forecast.base import ForecasterConfig


class MultiViewForecasterConfig(ForecasterConfig):
	def __init__(
		self,
		num_feature: int = 1,
		num_out_features: int = 1,
		num_embedding: int = 128,
		num_hidden: int = 256,
		num_head: int = 8,
		num_layers: int = 2,
		dropout: float = 0.1,
		feature: str = "latent",
		loss_type: str = "ALL",
		window_size: int = 48,
		forecast_horizon: int = 1,
		target_seq_index: int = 0,
		batch_size: int = 32,
		num_epochs: int = 10,
		lr: float = 0.001,
		weight_decay: float = 0.01,
		patience: int = 20,
		l1_scale: float = 0.0,
		l2_scale: float = 0.01,
		temperature: float = 0.5,
		lam: float = 0.1,
		mode: str = "finetune",
		pretrain_validate_on_train: bool = False,
		finetune_monitor_metric: str = "loss",
		augmentation_strength: float = 0.1,
		train_stride: int = 1,
		max_train_windows: Optional[int] = None,
		samplewise_windows_per_sample: int = 1,
		samplewise_train_sampling: str = "random",
		samplewise_eval_sampling: str = "center",
		samplewise_stride: int = 1,
		use_gpu: bool = True,
		checkpoint_path: Optional[str] = None,
		**kwargs,
	):
		self.num_feature = num_feature
		self.num_out_features = num_out_features
		self.num_embedding = num_embedding
		self.num_hidden = num_hidden
		self.num_head = num_head
		self.num_layers = num_layers
		self.dropout = dropout

		self.feature = feature
		self.loss_type = loss_type

		self.window_size = window_size
		self.forecast_horizon = forecast_horizon
		self.target_seq_index = target_seq_index

		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.lr = lr
		self.weight_decay = weight_decay
		self.patience = patience
		self.l1_scale = l1_scale
		self.l2_scale = l2_scale

		self.use_gpu = use_gpu
		self.checkpoint_path = checkpoint_path

		self.temperature = temperature
		self.lam = lam
		self.mode = mode
		self.pretrain_validate_on_train = pretrain_validate_on_train
		self.finetune_monitor_metric = finetune_monitor_metric
		self.augmentation_strength = augmentation_strength
		self.train_stride = train_stride
		self.max_train_windows = max_train_windows
		self.samplewise_windows_per_sample = samplewise_windows_per_sample
		self.samplewise_train_sampling = samplewise_train_sampling
		self.samplewise_eval_sampling = samplewise_eval_sampling
		self.samplewise_stride = samplewise_stride
		super().__init__(target_seq_index=target_seq_index, **kwargs)
