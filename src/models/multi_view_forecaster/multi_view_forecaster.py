from argparse import Namespace
from typing import Optional

import logging
import numpy as np
import pandas as pd
import torch
import torchcde
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from pytorch_metric_learning import losses

from merlion.models.forecast.base import ForecasterBase
from merlion.utils import TimeSeries

from models.hash_checkpoint_model import HashCheckpointModel
from models.multi_view_core import MultiViewCoreMixin
from models.multi_view_forecaster.config import MultiViewForecasterConfig
from models.multi_view_forecaster.model import ForecastHead


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
		info_loss = losses.NTXentLoss(temperature=self.config.temperature)
		self.info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)
		self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

		self._context_xt = None
		self._context_dx = None
		self._context_xf = None
		self._samplewise_mode = False
		self._feature_names = None

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
		self._samplewise_mode = bool(getattr(loaded_model, "_samplewise_mode", False))
		self._feature_names = getattr(loaded_model, "_feature_names", None)

		loaded_epoch = getattr(loaded_model, "current_epoch", 0)
		self.current_epoch = 0 if loaded_epoch is None else int(loaded_epoch)

	def _create_head(self):
		args = Namespace(
			num_feature=self.config.num_feature,
			num_out_features=self.config.num_out_features,
			num_embedding=self.config.num_embedding,
			num_hidden=self.config.num_hidden,
			num_head=self.config.num_head,
			num_layers=self.config.num_layers,
			dropout=self.config.dropout,
			loss_type=self.config.loss_type,
			feature=self.config.feature,
			forecast_horizon=self.config.forecast_horizon,
		)
		return ForecastHead(args)

	def _load_pretrained_checkpoint(self, checkpoint_path: str):
		super()._load_pretrained_checkpoint(checkpoint_path)
		logger.info(f"Loaded pre-trained encoder weights from {checkpoint_path}")

	def _extract_domains(self, time_series: pd.DataFrame):
		try:
			xt, dx, xf, target_col = self._extract_domains_matrix(time_series)
			self._samplewise_mode = False
			return xt, dx, xf, target_col
		except ValueError as exc:
			if "Expected" not in str(exc) or "primary features" not in str(exc):
				raise

			# Samplewise datasets (e.g. DA preprocessed classification-style inputs)
			# arrive as [n_samples, seq_len] and need sequence-domain extraction.
			xt, dx, xf = self._extract_domains_sequence(time_series)
			if xt.ndim != 3:
				raise
			self._samplewise_mode = True
			target_col = int(min(max(self.config.target_seq_index, 0), xt.shape[2] - 1))
			return xt.astype(np.float32), dx.astype(np.float32), xf.astype(np.float32), target_col

	def _build_window_dataset(self, xt, dx, xf, target_col: int):
		if self._samplewise_mode:
			return self._build_window_dataset_samplewise(
				xt,
				dx,
				xf,
				target_col,
				windows_per_sample=int(max(1, getattr(self.config, "samplewise_windows_per_sample", 1))),
				sampling_mode=str(getattr(self.config, "samplewise_train_sampling", "random")).lower(),
			)

		window = int(self.config.window_size)
		horizon = int(self.config.forecast_horizon)
		n = xt.shape[0]
		usable = n - window - horizon + 1
		if usable <= 0:
			raise ValueError(
				f"Not enough points ({n}) for window_size={window} and forecast_horizon={horizon}."
			)

		stride = int(max(1, getattr(self.config, "train_stride", 1)))
		max_windows = getattr(self.config, "max_train_windows", None)
		starts = list(range(0, usable, stride))
		if max_windows is not None:
			starts = starts[: int(max_windows)]

		xw_t, xw_d, xw_f, y = [], [], [], []
		for start in starts:
			xw_t.append(xt[start : start + window])
			xw_d.append(dx[start : start + window])
			xw_f.append(xf[start : start + window])
			if self.config.num_out_features > 1:
				y.append(xt[start + window : start + window + horizon, :])
			else:
				y.append(xt[start + window : start + window + horizon, target_col])

		return (
			np.asarray(xw_t, dtype=np.float32),
			np.asarray(xw_d, dtype=np.float32),
			np.asarray(xw_f, dtype=np.float32),
			np.asarray(y, dtype=np.float32),
		)

	def _build_window_dataset_samplewise(
		self,
		xt,
		dx,
		xf,
		target_col: int,
		windows_per_sample: int,
		sampling_mode: str,
	):
		window = int(self.config.window_size)
		horizon = int(self.config.forecast_horizon)
		n_samples, seq_len, _ = xt.shape

		usable = seq_len - window - horizon + 1
		if usable <= 0:
			logger.warning(
				f"Skipping all {n_samples} samples: seq_len={seq_len} too short for "
				f"window_size={window} and forecast_horizon={horizon}."
			)
			empty = np.empty((0,), dtype=np.float32)
			return empty, empty, empty, empty

		max_windows = min(int(max(1, windows_per_sample)), usable)

		xw_t, xw_d, xw_f, y = [], [], [], []
		stride = int(max(1, getattr(self.config, "samplewise_stride", 1)))

		for sample_idx in range(n_samples):
			if sampling_mode == "random":
				starts = np.random.choice(usable, size=max_windows, replace=False)
			elif sampling_mode == "center":
				center = (usable - 1) // 2
				starts = np.array([center], dtype=np.int64)
			elif sampling_mode == "first":
				starts = np.arange(max_windows, dtype=np.int64)
			elif sampling_mode == "stride":
				all_starts = np.arange(0, usable, stride, dtype=np.int64)
				starts = all_starts[:max_windows]
			else:
				raise ValueError(
					f"Unknown samplewise sampling_mode '{sampling_mode}'. "
					"Use one of: random, center, first, stride."
				)

			for start in starts:
				xw_t.append(xt[sample_idx, start : start + window, :])
				xw_d.append(dx[sample_idx, start : start + window, :])
				xw_f.append(xf[sample_idx, start : start + window, :])
				if self.config.num_out_features > 1:
					y.append(xt[sample_idx, start + window : start + window + horizon, :])
				else:
					y.append(xt[sample_idx, start + window : start + window + horizon, target_col])

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

	def _set_training_mode(self):
		self._set_mode_for_encoder_and_head(self.forecaster_head)

	def _train_epoch(self, loader: DataLoader):
		self._set_training_mode()

		total_loss = 0.0
		total_loss_c = 0.0
		total_samples = 0

		for batch_xt, batch_dx, batch_xf, batch_y in loader:
			batch_xt = batch_xt.to(self.device)
			batch_dx = batch_dx.to(self.device)
			batch_xf = batch_xf.to(self.device)
			batch_y = batch_y.to(self.device)

			batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(
				batch_xt, batch_dx, batch_xf
			)

			self.encoder_optimizer.zero_grad()
			if self.config.mode != "pretrain":
				self.head_optimizer.zero_grad()

			with autocast("cuda", enabled=self.device.type == "cuda"):
				(ht, hd, hf, zt, zd, zf), loss = self._compute_contrastive_encoder_loss(
					batch_xt,
					batch_dx,
					batch_xf,
					batch_xt_aug,
					batch_dx_aug,
					batch_xf_aug,
				)

				loss_c_val = 0.0
				if self.config.mode != "pretrain":
					preds = (
						self.forecaster_head(zt, zd, zf)
						if self.config.feature == "latent"
						else self.forecaster_head(ht, hd, hf)
					)
					loss_c = self.criterion(preds, batch_y)
					loss = self._compose_finetune_loss(loss, loss_c, self.forecaster_head)
					loss_c_val = loss_c.item()

			self.scaler.scale(loss).backward()
			if self.config.mode != "freeze":
				self.scaler.step(self.encoder_optimizer)
			if self.config.mode != "pretrain":
				self.scaler.step(self.head_optimizer)
			self.scaler.update()

			total_loss += loss.item() * batch_xt.size(0)
			total_loss_c += loss_c_val * batch_xt.size(0)
			total_samples += batch_xt.size(0)

		if total_samples <= 0:
			return 0.0, 0.0
		return total_loss / total_samples, total_loss_c / total_samples

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

				batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(
					batch_xt, batch_dx, batch_xf
				)

				with autocast("cuda", enabled=self.device.type == "cuda"):
					(ht, hd, hf, zt, zd, zf), loss = self._compute_contrastive_encoder_loss(
						batch_xt,
						batch_dx,
						batch_xf,
						batch_xt_aug,
						batch_dx_aug,
						batch_xf_aug,
					)

					if self.config.mode != "pretrain":
						preds = (
							self.forecaster_head(zt, zd, zf)
							if self.config.feature == "latent"
							else self.forecaster_head(ht, hd, hf)
						)
						loss_c = self.criterion(preds, batch_y)
						loss = self._compose_finetune_loss(loss, loss_c, self.forecaster_head)

				val_loss += loss.item() * batch_xt.size(0)
				total_samples += batch_xt.size(0)

		return val_loss / total_samples if total_samples > 0 else 0.0

	def _train(self, train_data: pd.DataFrame, train_config=None):
		if self.current_epoch is None:
			self.current_epoch = 0
		if self.config.num_epochs is None:
			raise ValueError("num_epochs must be set for training")

		self._feature_names = list(train_data.columns[:self.config.num_feature])
		xt, dx, xf, target_col = self._extract_domains(train_data)

		self._set_training_mode()

		val_loader = None
		if self._samplewise_mode:
			if self.config.mode == "pretrain" and self.config.pretrain_validate_on_train:
				xt_v, dx_v, xf_v, y_v = self._build_window_dataset_samplewise(
					xt,
					dx,
					xf,
					target_col,
					windows_per_sample=int(max(1, getattr(self.config, "samplewise_windows_per_sample", 1))),
					sampling_mode=str(getattr(self.config, "samplewise_eval_sampling", "center")).lower(),
				)
				if xt_v.size > 0:
					val_loader = self._make_loader(xt_v, dx_v, xf_v, y_v, shuffle=False)
		else:
			xt_w, dx_w, xf_w, y_w = self._build_window_dataset(xt, dx, xf, target_col)
			if len(xt_w) == 0:
				logger.warning("No valid windows produced; skipping training.")
				return pd.DataFrame(columns=["forecast"]), None
			train_loader = self._make_loader(xt_w, dx_w, xf_w, y_w, shuffle=True)
			val_loader = train_loader if self.config.mode == "pretrain" and self.config.pretrain_validate_on_train else None

		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.encoder_optimizer, mode="min", factor=0.5, patience=10
		)

		best_val = float("inf")
		early_stop_counter = 0

		for epoch in range(int(self.current_epoch), int(self.config.num_epochs)):
			if self._samplewise_mode:
				xt_w, dx_w, xf_w, y_w = self._build_window_dataset_samplewise(
					xt,
					dx,
					xf,
					target_col,
					windows_per_sample=int(max(1, getattr(self.config, "samplewise_windows_per_sample", 1))),
					sampling_mode=str(getattr(self.config, "samplewise_train_sampling", "random")).lower(),
				)
				if xt_w.size == 0:
					logger.warning("No valid windows produced; skipping training.")
					break
				train_loader = self._make_loader(xt_w, dx_w, xf_w, y_w, shuffle=True)

			train_loss, train_loss_c = self._train_epoch(train_loader)
			val_loss = self._validate(val_loader) if val_loader is not None else train_loss
			scheduler.step(val_loss)

			logger.info(
				f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, train_loss_c={train_loss_c:.6f}, val_loss={val_loss:.6f}"
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

		if self._samplewise_mode:
			return self._forecast_samplewise(time_stamps, time_series_prev=time_series_prev)

		self.encoder.eval()
		self.forecaster_head.eval()

		if time_series_prev is not None:
			df_prev = time_series_prev if isinstance(time_series_prev, pd.DataFrame) else time_series_prev.to_pd()
			xt_test, dx_test, xf_test, _ = self._extract_domains_matrix(df_prev)
			xt_full = np.vstack([self._context_xt, xt_test])
			dx_full = np.vstack([self._context_dx, dx_test])
			xf_full = np.vstack([self._context_xf, xf_test])
		else:
			logger.warning("No previous time series provided for forecasting; using only training context.")
			xt_full = self._context_xt
			dx_full = self._context_dx
			xf_full = self._context_xf

		horizon = len(time_stamps)
		window = int(self.config.window_size)
		forecast_horizon = int(self.config.forecast_horizon)
		pos = len(self._context_xt)

		if pos < window:
			raise RuntimeError(
				f"Insufficient context length {pos} for window_size={window}."
			)

		chunks = []
		n_collected = 0
		with torch.no_grad():
			while n_collected < horizon:
				xt_window = torch.FloatTensor(xt_full[pos - window : pos]).unsqueeze(0).to(self.device)
				dx_window = torch.FloatTensor(dx_full[pos - window : pos]).unsqueeze(0).to(self.device)
				xf_window = torch.FloatTensor(xf_full[pos - window : pos]).unsqueeze(0).to(self.device)
				out = self._forward(xt_window, dx_window, xf_window)[0].detach().cpu().numpy()
				n = min(forecast_horizon, horizon - n_collected)
				chunks.append(out[:n])
				n_collected += n
				pos = min(pos + forecast_horizon, len(xt_full))

		all_preds = np.concatenate(chunks, axis=0)  # [horizon] or [horizon, num_out]
		index = pd.to_datetime(time_stamps, unit="s", errors="coerce")
		if pd.isna(index).all():
			index = pd.to_datetime(time_stamps, errors="coerce")
		if all_preds.ndim == 1:
			pred_df = pd.DataFrame(all_preds.reshape(-1, 1), index=index, columns=["forecast"])
		else:
			cols = self._feature_names or [f"feat_{i}" for i in range(all_preds.shape[1])]
			pred_df = pd.DataFrame(all_preds, index=index, columns=cols)
		return pred_df, None

	def _forecast_samplewise(self, time_stamps, time_series_prev: Optional[pd.DataFrame] = None):
		self.encoder.eval()
		self.forecaster_head.eval()

		if time_series_prev is not None:
			xt_src, dx_src, xf_src = self._extract_domains_sequence(time_series_prev)
			xt_src = xt_src.astype(np.float32)
			dx_src = dx_src.astype(np.float32)
			xf_src = xf_src.astype(np.float32)
		else:
			xt_src = self._context_xt
			dx_src = self._context_dx
			xf_src = self._context_xf

		if xt_src.ndim != 3:
			raise RuntimeError("Samplewise forecasting expects 3D context tensors [n_samples, seq_len, num_feature].")

		horizon = len(time_stamps)
		window = int(self.config.window_size)
		target_col = int(min(max(self.config.target_seq_index, 0), xt_src.shape[2] - 1))
		n_samples = xt_src.shape[0]
		if n_samples == 0:
			raise RuntimeError("No samples available for samplewise forecasting.")
		if xt_src.shape[1] < window:
			raise RuntimeError(
				f"Insufficient sample sequence length {xt_src.shape[1]} for window_size={window}."
			)

		preds = []
		with torch.no_grad():
			for i in range(horizon):
				sample_idx = i if i < n_samples else n_samples - 1
				xt_window = torch.FloatTensor(xt_src[sample_idx, -window:, :]).unsqueeze(0).to(self.device)
				dx_window = torch.FloatTensor(dx_src[sample_idx, -window:, :]).unsqueeze(0).to(self.device)
				xf_window = torch.FloatTensor(xf_src[sample_idx, -window:, :]).unsqueeze(0).to(self.device)

				next_vector = self._forward(xt_window, dx_window, xf_window)
				next_value = float(next_vector[0, 0].detach().cpu().item())
				preds.append(next_value)

		index = pd.to_datetime(time_stamps, unit="s", errors="coerce")
		if pd.isna(index).all():
			index = pd.to_datetime(time_stamps, errors="coerce")
		pred_df = pd.DataFrame(np.asarray(preds).reshape(-1, 1), index=index, columns=["forecast"])
		return pred_df, None
