from argparse import Namespace
from typing import Optional

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from merlion.models.forecast.base import ForecasterBase
from merlion.utils import TimeSeries

from models.hash_checkpoint_model import HashCheckpointModel
from models.multi_view_core import MultiViewCoreMixin
from models.multi_view_forecaster.config import MultiViewForecasterConfig
from models.multi_view_forecaster.model import LinearForecastHead


logger = logging.getLogger(__name__)


class MultiViewForecaster(MultiViewCoreMixin, HashCheckpointModel, ForecasterBase):
	config_class = MultiViewForecasterConfig

	@property
	def require_even_sampling(self) -> bool:
		return False

	@property
	def require_univariate(self) -> bool:
		return False

	def __init__(self, config: MultiViewForecasterConfig, save_dir: Optional[str] = None):
		ForecasterBase.__init__(self, config)
		HashCheckpointModel.__init__(self, config, save_dir)

		if not hasattr(self, "current_epoch") or self.current_epoch is None:
			self.current_epoch = 0

		self.device = self._get_device()
		self.encoder = self._create_encoder().to(self.device)
		self.forecaster_head = self._create_head().to(self.device)

		self.encoder_optimizer = torch.optim.Adam(
			self.encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
		)
		self.head_optimizer = torch.optim.Adam(
			self.forecaster_head.parameters(), lr=config.lr, weight_decay=config.weight_decay
		)

		self.criterion = nn.MSELoss()
		self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

		self._context_xt = None
		self._context_dx = None
		self._context_xf = None

		if not self._try_load_existing_checkpoint():
			if config.checkpoint_path:
				self._load_pretrained_checkpoint(config.checkpoint_path)

	def _load_checkpoint_state(self, loaded_model):
		self.encoder.load_state_dict(loaded_model.encoder.state_dict())
		self.forecaster_head.load_state_dict(loaded_model.forecaster_head.state_dict())
		self.encoder_optimizer.load_state_dict(loaded_model.encoder_optimizer.state_dict())
		self.head_optimizer.load_state_dict(loaded_model.head_optimizer.state_dict())
		self._context_xt = getattr(loaded_model, "_context_xt", None)
		self._context_dx = getattr(loaded_model, "_context_dx", None)
		self._context_xf = getattr(loaded_model, "_context_xf", None)

		loaded_epoch = getattr(loaded_model, "current_epoch", 0)
		self.current_epoch = 0 if loaded_epoch is None else int(loaded_epoch)

	def _create_head(self):
		args = Namespace(
			num_feature=self.config.num_feature,
			num_embedding=self.config.num_embedding,
			num_hidden=self.config.num_hidden,
			num_head=self.config.num_head,
			num_layers=self.config.num_layers,
			dropout=self.config.dropout,
			loss_type=self.config.loss_type,
			feature=self.config.feature,
			forecast_horizon=self.config.forecast_horizon,
		)
		return LinearForecastHead(args)

	def _load_pretrained_checkpoint(self, checkpoint_path: str):
		super()._load_pretrained_checkpoint(checkpoint_path)
		logger.info(f"Loaded pre-trained encoder weights from {checkpoint_path}")

	def _extract_domains(self, time_series: pd.DataFrame):
		return self._extract_domains_matrix(time_series)

	def _build_window_dataset(self, xt, dx, xf, target_col: int):
		window = int(self.config.window_size)
		horizon = int(self.config.forecast_horizon)
		n = xt.shape[0]
		usable = n - window - horizon + 1
		if usable <= 0:
			raise ValueError(
				f"Not enough points ({n}) for window_size={window} and forecast_horizon={horizon}."
			)

		xw_t, xw_d, xw_f, y = [], [], [], []
		for start in range(usable):
			xw_t.append(xt[start : start + window])
			xw_d.append(dx[start : start + window])
			xw_f.append(xf[start : start + window])
			y.append(xt[start + window : start + window + horizon, target_col])

		return (
			np.asarray(xw_t, dtype=np.float32),
			np.asarray(xw_d, dtype=np.float32),
			np.asarray(xw_f, dtype=np.float32),
			np.asarray(y, dtype=np.float32),
		)

	def _make_loader(self, xt_w, dx_w, xf_w, y_w, shuffle: bool):
		dataset = TensorDataset(
			torch.FloatTensor(xt_w),
			torch.FloatTensor(dx_w),
			torch.FloatTensor(xf_w),
			torch.FloatTensor(y_w),
		)
		return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

	def _forward(self, batch_xt, batch_dx, batch_xf):
		ht, hd, hf, zt, zd, zf = self.encoder(batch_xt, batch_dx, batch_xf)
		if self.config.feature == "latent":
			return self.forecaster_head(zt, zd, zf)
		return self.forecaster_head(ht, hd, hf)

	def _train_epoch(self, loader: DataLoader):
		self.encoder.train()
		self.forecaster_head.train()

		total_loss = 0.0
		total_samples = 0

		for batch_xt, batch_dx, batch_xf, batch_y in loader:
			batch_xt = batch_xt.to(self.device)
			batch_dx = batch_dx.to(self.device)
			batch_xf = batch_xf.to(self.device)
			batch_y = batch_y.to(self.device)

			self.encoder_optimizer.zero_grad()
			self.head_optimizer.zero_grad()

			with autocast("cuda", enabled=self.device.type == "cuda"):
				preds = self._forward(batch_xt, batch_dx, batch_xf)
				loss = self.criterion(preds, batch_y)
				loss = (
					loss
					+ self._weight_regularization(self.encoder)
					+ self._weight_regularization(self.forecaster_head)
				)

			self.scaler.scale(loss).backward()
			self.scaler.step(self.encoder_optimizer)
			self.scaler.step(self.head_optimizer)
			self.scaler.update()

			total_loss += loss.item() * batch_xt.size(0)
			total_samples += batch_xt.size(0)

		return total_loss / total_samples if total_samples > 0 else 0.0

	def _validate(self, loader: DataLoader):
		self.encoder.eval()
		self.forecaster_head.eval()

		val_loss = 0.0
		total_samples = 0
		with torch.no_grad():
			for batch_xt, batch_dx, batch_xf, batch_y in loader:
				batch_xt = batch_xt.to(self.device)
				batch_dx = batch_dx.to(self.device)
				batch_xf = batch_xf.to(self.device)
				batch_y = batch_y.to(self.device)

				with autocast("cuda", enabled=self.device.type == "cuda"):
					preds = self._forward(batch_xt, batch_dx, batch_xf)
					loss = self.criterion(preds, batch_y)

				val_loss += loss.item() * batch_xt.size(0)
				total_samples += batch_xt.size(0)

		return val_loss / total_samples if total_samples > 0 else 0.0

	def _train(self, train_data: pd.DataFrame, train_config=None):
		if self.current_epoch is None:
			self.current_epoch = 0
		if self.config.num_epochs is None:
			raise ValueError("num_epochs must be set for training")

		xt, dx, xf, target_col = self._extract_domains(train_data)
		xt_w, dx_w, xf_w, y_w = self._build_window_dataset(xt, dx, xf, target_col)
		train_loader = self._make_loader(xt_w, dx_w, xf_w, y_w, shuffle=True)

		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.head_optimizer, mode="min", factor=0.5, patience=10
		)

		best_val = float("inf")
		early_stop_counter = 0

		for epoch in range(int(self.current_epoch), int(self.config.num_epochs)):
			train_loss = self._train_epoch(train_loader)
			val_loss = self._validate(train_loader)
			scheduler.step(val_loss)

			logger.info(
				f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
			)

			if val_loss < best_val:
				best_val = val_loss
				early_stop_counter = 0
				self.current_epoch = epoch + 1
				self.save(save_config=True)
			else:
				early_stop_counter += 1

			if early_stop_counter >= self.config.patience:
				logger.info(f"Early stopping triggered at epoch {epoch + 1}")
				break

		self._context_xt = xt.copy()
		self._context_dx = dx.copy()
		self._context_xf = xf.copy()

		target_col = int(min(max(self.config.target_seq_index, 0), xt.shape[1] - 1))
		train_fit = pd.DataFrame(xt[:, target_col], index=train_data.index, columns=["forecast"])
		return train_fit, None

	def forecast(
		self,
		time_stamps,
		time_series_prev: Optional[TimeSeries] = None,
		exog_data: Optional[TimeSeries] = None,
		return_iqr: bool = False,
		return_prev: bool = False,
	):
		if isinstance(time_stamps, (int, float)):
			if self._context_xt is None:
				raise RuntimeError("Forecaster has no context. Train the model before forecasting.")
			n_steps = int(time_stamps)
			time_stamps = list(range(n_steps))
		elif hasattr(time_stamps, "tolist"):
			time_stamps = time_stamps.tolist()
		else:
			time_stamps = list(time_stamps)

		forecast_df, err_df = self._forecast(
			time_stamps=time_stamps,
			time_series_prev=None if time_series_prev is None else time_series_prev.to_pd(),
			return_prev=return_prev,
		)
		forecast_ts = TimeSeries.from_pd(forecast_df)
		err_ts = None if err_df is None else TimeSeries.from_pd(err_df)

		if return_iqr:
			return forecast_ts, err_ts, err_ts
		return forecast_ts

	def _forecast(self, time_stamps, time_series_prev: Optional[pd.DataFrame] = None, return_prev: bool = False):
		if self._context_xt is None or self._context_dx is None or self._context_xf is None:
			raise RuntimeError("Forecaster has no context. Train the model before forecasting.")

		self.encoder.eval()
		self.forecaster_head.eval()

		xt_hist = self._context_xt.copy()
		dx_hist = self._context_dx.copy()
		xf_hist = self._context_xf.copy()

		horizon = len(time_stamps)
		window = int(self.config.window_size)
		target_col = int(min(max(self.config.target_seq_index, 0), xt_hist.shape[1] - 1))

		preds = []
		with torch.no_grad():
			for _ in range(horizon):
				if len(xt_hist) < window:
					raise RuntimeError(
						f"Insufficient context length {len(xt_hist)} for window_size={window}."
					)

				xt_window = torch.FloatTensor(xt_hist[-window:]).unsqueeze(0).to(self.device)
				dx_window = torch.FloatTensor(dx_hist[-window:]).unsqueeze(0).to(self.device)
				xf_window = torch.FloatTensor(xf_hist[-window:]).unsqueeze(0).to(self.device)

				next_vector = self._forward(xt_window, dx_window, xf_window)
				next_value = float(next_vector[0, 0].detach().cpu().item())
				preds.append(next_value)

				next_xt_row = xt_hist[-1].copy()
				next_xt_row[target_col] = next_value
				xt_hist = np.vstack([xt_hist, next_xt_row.astype(np.float32)])
				dx_hist = np.vstack([dx_hist, np.zeros((1, dx_hist.shape[1]), dtype=np.float32)])
				xf_hist = np.vstack([xf_hist, np.zeros((1, xf_hist.shape[1]), dtype=np.float32)])

		index = pd.to_datetime(time_stamps, unit="s", errors="coerce")
		if pd.isna(index).all():
			index = pd.to_datetime(time_stamps, errors="coerce")
		pred_df = pd.DataFrame(np.asarray(preds).reshape(-1, 1), index=index, columns=["forecast"])
		return pred_df, None
