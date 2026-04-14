"""MultiViewForecaster — multi-domain contrastive + supervised forecaster.

Training modes mirror MultiViewClassifier:
    pretrain  — encoder only, NTXentLoss
    finetune  — encoder + head; contrastive + MSE loss
    freeze    — input-projection layers + head only

Samplewise datasets (each row = full flattened sample) are supported.
Windows are re-sampled randomly every epoch in samplewise mode.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from merlion.utils import TimeSeries
from pytorch_metric_learning import losses

from models.base import CustomForecasterBase
from models.multi_view_core import MultiViewCoreMixin
from models.multi_view_forecaster.config import MultiViewForecasterConfig
from models.multi_view_forecaster.model import ForecastHead

logger = logging.getLogger(__name__)


class MultiViewForecaster(MultiViewCoreMixin, CustomForecasterBase):
    config_class = MultiViewForecasterConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: MultiViewForecasterConfig, save_dir: Optional[str] = None):
        CustomForecasterBase.__init__(self, config, save_dir)

        self.encoder = self._create_encoder().to(self.device)
        self.forecaster_head = self._create_head().to(self.device)

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.head_optimizer = torch.optim.Adam(
            self.forecaster_head.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()
        info_loss = losses.NTXentLoss(temperature=config.temperature)
        self.info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        # Context stored after training for use during forecasting
        self._context_xt: Optional[np.ndarray] = None
        self._context_dx: Optional[np.ndarray] = None
        self._context_xf: Optional[np.ndarray] = None
        self._samplewise_mode: bool = False
        self._feature_names: Optional[list] = None

        # Raw domain arrays stored during _train() for train_dataloader()
        self._xt_raw: Optional[np.ndarray] = None
        self._dx_raw: Optional[np.ndarray] = None
        self._xf_raw: Optional[np.ndarray] = None
        self._target_col: int = 0
        self._val_loader_cache: Optional[DataLoader] = None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _create_head(self):
        from argparse import Namespace
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

    # ------------------------------------------------------------------
    # PL monitor metric
    # ------------------------------------------------------------------

    def _monitor_metric(self) -> str:
        return "val_loss"

    def _monitor_mode(self) -> str:
        return "min"

    # ------------------------------------------------------------------
    # Checkpoint — persist context arrays
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["context_xt"] = self._context_xt
        checkpoint["context_dx"] = self._context_dx
        checkpoint["context_xf"] = self._context_xf
        checkpoint["samplewise_mode"] = self._samplewise_mode
        checkpoint["feature_names"] = self._feature_names

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._context_xt = checkpoint.get("context_xt")
        self._context_dx = checkpoint.get("context_dx")
        self._context_xf = checkpoint.get("context_xf")
        self._samplewise_mode = bool(checkpoint.get("samplewise_mode", False))
        self._feature_names = checkpoint.get("feature_names")

    # ------------------------------------------------------------------
    # Domain extraction
    # ------------------------------------------------------------------

    def _extract_domains(self, time_series: pd.DataFrame):
        try:
            xt, dx, xf, target_col = self._extract_domains_matrix(time_series)
            self._samplewise_mode = False
            return xt, dx, xf, target_col
        except ValueError as exc:
            if "Expected" not in str(exc) or "primary features" not in str(exc):
                raise
            xt, dx, xf = self._extract_domains_sequence(time_series)
            if xt.ndim != 3:
                raise
            self._samplewise_mode = True
            target_col = int(min(max(self.config.target_seq_index, 0), xt.shape[2] - 1))
            return xt.astype(np.float32), dx.astype(np.float32), xf.astype(np.float32), target_col

    # ------------------------------------------------------------------
    # Window building
    # ------------------------------------------------------------------

    def _build_window_dataset(self, xt, dx, xf, target_col: int):
        if self._samplewise_mode:
            return self._build_window_dataset_samplewise(
                xt, dx, xf, target_col,
                windows_per_sample=int(max(1, getattr(self.config, "samplewise_windows_per_sample", 1))),
                sampling_mode=str(getattr(self.config, "samplewise_train_sampling", "random")).lower(),
            )

        window = int(self.config.window_size)
        horizon = int(self.config.forecast_horizon)
        n = xt.shape[0]
        usable = n - window - horizon + 1
        if usable <= 0:
            raise ValueError(
                f"Not enough points ({n}) for window_size={window}, forecast_horizon={horizon}."
            )

        stride = int(max(1, getattr(self.config, "train_stride", 1)))
        max_windows = getattr(self.config, "max_train_windows", None)
        starts = list(range(0, usable, stride))
        if max_windows is not None:
            starts = starts[:int(max_windows)]

        xw_t, xw_d, xw_f, y = [], [], [], []
        for start in starts:
            xw_t.append(xt[start: start + window])
            xw_d.append(dx[start: start + window])
            xw_f.append(xf[start: start + window])
            if self.config.num_out_features > 1:
                y.append(xt[start + window: start + window + horizon, :])
            else:
                y.append(xt[start + window: start + window + horizon, target_col])

        return (
            np.asarray(xw_t, dtype=np.float32),
            np.asarray(xw_d, dtype=np.float32),
            np.asarray(xw_f, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
        )

    def _build_window_dataset_samplewise(self, xt, dx, xf, target_col, windows_per_sample, sampling_mode):
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
        stride = int(max(1, getattr(self.config, "samplewise_stride", 1)))
        xw_t, xw_d, xw_f, y = [], [], [], []

        for sample_idx in range(n_samples):
            if sampling_mode == "random":
                starts = np.random.choice(usable, size=max_windows, replace=False)
            elif sampling_mode == "center":
                starts = np.array([(usable - 1) // 2], dtype=np.int64)
            elif sampling_mode == "first":
                starts = np.arange(max_windows, dtype=np.int64)
            elif sampling_mode == "stride":
                starts = np.arange(0, usable, stride, dtype=np.int64)[:max_windows]
            else:
                raise ValueError(f"Unknown samplewise_train_sampling '{sampling_mode}'.")

            for start in starts:
                xw_t.append(xt[sample_idx, start: start + window, :])
                xw_d.append(dx[sample_idx, start: start + window, :])
                xw_f.append(xf[sample_idx, start: start + window, :])
                if self.config.num_out_features > 1:
                    y.append(xt[sample_idx, start + window: start + window + horizon, :])
                else:
                    y.append(xt[sample_idx, start + window: start + window + horizon, target_col])

        return (
            np.asarray(xw_t, dtype=np.float32),
            np.asarray(xw_d, dtype=np.float32),
            np.asarray(xw_f, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
        )

    def _make_loader(self, xt_w, dx_w, xf_w, y_w, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(
                torch.FloatTensor(xt_w),
                torch.FloatTensor(dx_w),
                torch.FloatTensor(xf_w),
                torch.FloatTensor(y_w),
            ),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )

    # ------------------------------------------------------------------
    # PL dataloader hooks (samplewise: train_dataloader called every epoch)
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        xt_w, dx_w, xf_w, y_w = self._build_window_dataset(
            self._xt_raw, self._dx_raw, self._xf_raw, self._target_col
        )
        return self._make_loader(xt_w, dx_w, xf_w, y_w, shuffle=True)

    def val_dataloader(self):
        return self._val_loader_cache

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, batch_xt, batch_dx, batch_xf):
        ht, hd, hf, zt, zd, zf = self.encoder(batch_xt, batch_dx, batch_xf)
        if self.config.feature == "latent":
            return self.forecaster_head(zt, zd, zf)
        return self.forecaster_head(ht, hd, hf)

    def _set_training_mode(self):
        self._set_mode_for_encoder_and_head(self.forecaster_head)

    # ------------------------------------------------------------------
    # PL hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        batch_xt, batch_dx, batch_xf, batch_y = batch
        self._set_training_mode()

        batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(batch_xt, batch_dx, batch_xf)

        self.encoder_optimizer.zero_grad()
        if self.config.mode != "pretrain":
            self.head_optimizer.zero_grad()

        with autocast("cuda", enabled=self.device.type == "cuda"):
            (ht, hd, hf, zt, zd, zf), loss = self._compute_contrastive_encoder_loss(
                batch_xt, batch_dx, batch_xf,
                batch_xt_aug, batch_dx_aug, batch_xf_aug,
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

        self.log("train_loss", loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_loss_c", loss_c_val, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_xt, batch_dx, batch_xf, batch_y = batch
        self.encoder.eval()
        self.forecaster_head.eval()

        batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(batch_xt, batch_dx, batch_xf)
        with autocast("cuda", enabled=self.device.type == "cuda"):
            (ht, hd, hf, zt, zd, zf), loss = self._compute_contrastive_encoder_loss(
                batch_xt, batch_dx, batch_xf,
                batch_xt_aug, batch_dx_aug, batch_xf_aug,
            )
            if self.config.mode != "pretrain":
                preds = (
                    self.forecaster_head(zt, zd, zf)
                    if self.config.feature == "latent"
                    else self.forecaster_head(ht, hd, hf)
                )
                loss = self._compose_finetune_loss(loss, self.criterion(preds, batch_y), self.forecaster_head)

        self.log("val_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        m = self.trainer.callback_metrics
        epoch = self.current_epoch + 1
        train_loss = m.get("train_loss", torch.tensor(0.0)).item()
        train_loss_c = m.get("train_loss_c", torch.tensor(0.0)).item()
        val_loss = m.get("val_loss", torch.tensor(0.0)).item()
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, "
            f"train_loss_c={train_loss_c:.6f}, val_loss={val_loss:.6f}"
        )

    def configure_optimizers(self):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode="min", factor=0.5, patience=10
        )
        self._lr_scheduler = sched
        return (
            [self.encoder_optimizer, self.head_optimizer],
            [],  # stepped manually in on_train_epoch_end
        )

    # ------------------------------------------------------------------
    # Protocol-facing training
    # ------------------------------------------------------------------

    def _train(self, train_data, train_config=None):
        if not isinstance(train_data, pd.DataFrame):
            train_data = train_data.to_pd()
        self._feature_names = list(train_data.columns[:self.config.num_feature])
        xt, dx, xf, target_col = self._extract_domains(train_data)

        # Store for train_dataloader() / forecasting context
        self._xt_raw = xt
        self._dx_raw = dx
        self._xf_raw = xf
        self._target_col = target_col

        # Build val loader when pretrain validates on training data
        self._val_loader_cache = None
        if self.config.pretrain_validate_on_train:
            eval_mode = str(getattr(self.config, "samplewise_eval_sampling", "center")).lower()
            val_ws = int(max(1, getattr(self.config, "samplewise_windows_per_sample", 1)))
            if self._samplewise_mode:
                xt_v, dx_v, xf_v, y_v = self._build_window_dataset_samplewise(
                    xt, dx, xf, target_col,
                    windows_per_sample=val_ws,
                    sampling_mode=eval_mode,
                )
            else:
                xt_v, dx_v, xf_v, y_v = self._build_window_dataset(xt, dx, xf, target_col)
            if xt_v.size > 0:
                self._val_loader_cache = self._make_loader(xt_v, dx_v, xf_v, y_v, shuffle=False)

        if self.config.checkpoint_path and not self._checkpoint_loaded:
            self._load_pretrained_checkpoint(self.config.checkpoint_path)

        # Samplewise: PL calls train_dataloader() every epoch for random re-sampling.
        # Non-samplewise: build static windows and pass as explicit loader.
        if self._samplewise_mode:
            self._fit_and_restore_best(
                train_loader=None,
                val_loader=None,  # PL calls val_dataloader()
                reload_dataloaders_every_n_epochs=1,
            )
        else:
            xt_w, dx_w, xf_w, y_w = self._build_window_dataset(xt, dx, xf, target_col)
            if len(xt_w) == 0:
                logger.warning("No valid windows produced; skipping training.")
                return
            train_loader = self._make_loader(xt_w, dx_w, xf_w, y_w, shuffle=True)
            self._fit_and_restore_best(
                train_loader=train_loader,
                val_loader=self._val_loader_cache,
            )

        # Persist context for forecasting
        self._context_xt = xt.copy()
        self._context_dx = dx.copy()
        self._context_xf = xf.copy()

    def set_context(self, train_data) -> None:
        """Initialise forecasting context without training (skip_training=True)."""
        self._ensure_ready()
        if not isinstance(train_data, pd.DataFrame):
            train_data = train_data.to_pd()
        xt, dx, xf, target_col = self._extract_domains(train_data)
        self._context_xt = xt
        self._context_dx = dx
        self._context_xf = xf
        self._feature_names = list(train_data.columns[:self.config.num_feature])

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        time_stamps,
        time_series_prev: Optional[TimeSeries] = None,
        exog_data=None,
        return_iqr: bool = False,
        return_prev: bool = False,
    ):
        self._ensure_ready()
        if isinstance(time_stamps, (int, float)):
            time_stamps = list(range(int(time_stamps)))
        elif hasattr(time_stamps, "tolist"):
            time_stamps = time_stamps.tolist()
        else:
            time_stamps = list(time_stamps)

        forecast_df, err_df = self._forecast(
            time_stamps=time_stamps,
            time_series_prev=None if time_series_prev is None else time_series_prev.to_pd(),
        )
        forecast_ts = TimeSeries.from_pd(forecast_df)
        err_ts = None if err_df is None else TimeSeries.from_pd(err_df)
        if return_iqr:
            return forecast_ts, err_ts, err_ts
        return forecast_ts

    def _forecast(self, time_stamps, time_series_prev: Optional[pd.DataFrame] = None, return_prev=False):
        if self._context_xt is None:
            raise RuntimeError("No context — train the model before forecasting.")

        if self._samplewise_mode:
            return self._forecast_samplewise(time_stamps, time_series_prev)

        self.encoder.eval()
        self.forecaster_head.eval()

        if time_series_prev is not None:
            df_prev = time_series_prev if isinstance(time_series_prev, pd.DataFrame) else time_series_prev.to_pd()
            xt_test, dx_test, xf_test, _ = self._extract_domains_matrix(df_prev)
            xt_full = np.vstack([self._context_xt, xt_test])
            dx_full = np.vstack([self._context_dx, dx_test])
            xf_full = np.vstack([self._context_xf, xf_test])
        else:
            logger.warning("No time_series_prev; using only training context.")
            xt_full = self._context_xt
            dx_full = self._context_dx
            xf_full = self._context_xf

        horizon = len(time_stamps)
        window = int(self.config.window_size)
        forecast_horizon = int(self.config.forecast_horizon)
        pos = len(self._context_xt)
        if pos < window:
            raise RuntimeError(f"Insufficient context length {pos} for window_size={window}.")

        chunks, n_collected = [], 0
        with torch.no_grad():
            while n_collected < horizon:
                xt_w = torch.FloatTensor(xt_full[pos - window: pos]).unsqueeze(0).to(self.device)
                dx_w = torch.FloatTensor(dx_full[pos - window: pos]).unsqueeze(0).to(self.device)
                xf_w = torch.FloatTensor(xf_full[pos - window: pos]).unsqueeze(0).to(self.device)
                out = self._forward(xt_w, dx_w, xf_w)[0].detach().cpu().numpy()
                n = min(forecast_horizon, horizon - n_collected)
                chunks.append(out[:n])
                n_collected += n
                pos = min(pos + forecast_horizon, len(xt_full))

        all_preds = np.concatenate(chunks, axis=0)
        index = pd.to_datetime(time_stamps, unit="s", errors="coerce")
        if pd.isna(index).all():
            index = pd.to_datetime(time_stamps, errors="coerce")
        if all_preds.ndim == 1:
            return pd.DataFrame(all_preds.reshape(-1, 1), index=index, columns=["forecast"]), None
        cols = self._feature_names or [f"feat_{i}" for i in range(all_preds.shape[1])]
        return pd.DataFrame(all_preds, index=index, columns=cols), None

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
            raise RuntimeError("Samplewise forecasting expects 3D context tensors.")

        horizon = len(time_stamps)
        window = int(self.config.window_size)
        target_col = int(min(max(self.config.target_seq_index, 0), xt_src.shape[2] - 1))
        n_samples = xt_src.shape[0]
        if n_samples == 0 or xt_src.shape[1] < window:
            raise RuntimeError("Insufficient samples / sequence length for samplewise forecasting.")

        preds = []
        with torch.no_grad():
            for i in range(horizon):
                idx = min(i, n_samples - 1)
                xt_w = torch.FloatTensor(xt_src[idx, -window:, :]).unsqueeze(0).to(self.device)
                dx_w = torch.FloatTensor(dx_src[idx, -window:, :]).unsqueeze(0).to(self.device)
                xf_w = torch.FloatTensor(xf_src[idx, -window:, :]).unsqueeze(0).to(self.device)
                preds.append(float(self._forward(xt_w, dx_w, xf_w)[0, 0].cpu().item()))

        index = pd.to_datetime(time_stamps, unit="s", errors="coerce")
        if pd.isna(index).all():
            index = pd.to_datetime(time_stamps, errors="coerce")
        return pd.DataFrame(np.asarray(preds).reshape(-1, 1), index=index, columns=["forecast"]), None
