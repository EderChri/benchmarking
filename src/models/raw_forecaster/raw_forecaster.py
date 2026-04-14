"""RawForecaster — supervised window-based forecaster on raw time series.

No multi-view preprocessing or contrastive learning; intended as a baseline
counterpart to MultiViewForecaster.  Any domain-transformed columns
(containing config.marker) are dropped automatically with a warning.
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

from models.base import CustomForecasterBase
from models.raw_forecaster.config import RawForecasterConfig
from models.raw_forecaster.model import RawForecastModel

logger = logging.getLogger(__name__)


class RawForecaster(CustomForecasterBase):
    """Supervised forecaster on raw (un-encoded) input windows."""

    config_class = RawForecasterConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: RawForecasterConfig, save_dir: Optional[str] = None):
        CustomForecasterBase.__init__(self, config, save_dir)

        from argparse import Namespace
        args = Namespace(
            num_feature=config.num_feature,
            num_out_features=config.num_out_features,
            num_embedding=config.num_embedding,
            num_hidden=config.num_hidden,
            dropout=config.dropout,
            forecast_horizon=config.forecast_horizon,
        )
        self.model = RawForecastModel(args).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        self._context_xt: Optional[np.ndarray] = None
        self._feature_names: Optional[list] = None

    # ------------------------------------------------------------------
    # Checkpoint — persist context
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["context_xt"] = self._context_xt
        checkpoint["feature_names"] = self._feature_names

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._context_xt = checkpoint.get("context_xt")
        self._feature_names = checkpoint.get("feature_names")

    # ------------------------------------------------------------------
    # Domain filtering
    # ------------------------------------------------------------------

    def _filter_domain_columns(self, df: pd.DataFrame) -> np.ndarray:
        domain_cols = [c for c in df.columns if self.config.marker in str(c)]
        if domain_cols:
            logger.warning(
                f"RawForecaster received {len(domain_cols)} domain column(s) "
                f"containing '{self.config.marker}' — ignoring: {domain_cols}. "
                "Use a non-multi-view preprocessor (e.g. normalization, no_preproc)."
            )
            df = df.drop(columns=domain_cols)
        return df.values.astype(np.float32)

    # ------------------------------------------------------------------
    # Window building
    # ------------------------------------------------------------------

    def _build_windows(self, xt: np.ndarray):
        window = int(self.config.window_size)
        horizon = int(self.config.forecast_horizon)
        usable = xt.shape[0] - window - horizon + 1
        if usable <= 0:
            raise ValueError(
                f"Not enough points ({xt.shape[0]}) for window_size={window}, "
                f"forecast_horizon={horizon}."
            )

        stride = int(max(1, self.config.train_stride))
        starts = list(range(0, usable, stride))
        if self.config.max_train_windows is not None:
            starts = starts[:int(self.config.max_train_windows)]

        target_col = int(min(max(self.config.target_seq_index, 0), xt.shape[1] - 1))
        xw, y = [], []
        for s in starts:
            xw.append(xt[s: s + window])
            y.append(
                xt[s + window: s + window + horizon, :]
                if self.config.num_out_features > 1
                else xt[s + window: s + window + horizon, target_col]
            )
        return np.asarray(xw, dtype=np.float32), np.asarray(y, dtype=np.float32)

    # ------------------------------------------------------------------
    # PL hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        self.model.train()
        self.optimizer.zero_grad()

        with autocast("cuda", enabled=self.device.type == "cuda"):
            loss = self.criterion(self.model(batch_x), batch_y)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.log("train_loss", loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        self.model.eval()

        with torch.no_grad(), autocast("cuda", enabled=self.device.type == "cuda"):
            loss = self.criterion(self.model(batch_x), batch_y)

        self.log("val_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        m = self.trainer.callback_metrics
        epoch = self.current_epoch + 1
        train_loss = m.get("train_loss", torch.tensor(0.0)).item()
        val_loss = m.get("val_loss", torch.tensor(0.0)).item()
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    def configure_optimizers(self):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self._lr_scheduler = sched
        return [self.optimizer], []  # stepped manually in on_train_epoch_end

    # ------------------------------------------------------------------
    # Protocol-facing training
    # ------------------------------------------------------------------

    def _train(self, train_data, train_config=None):
        if not isinstance(train_data, pd.DataFrame):
            train_data = train_data.to_pd()
        xt = self._filter_domain_columns(train_data)
        self._feature_names = list(train_data.columns[:xt.shape[1]])
        xw, y = self._build_windows(xt)

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(xw), torch.FloatTensor(y)),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )
        if self.config.checkpoint_path and not self._checkpoint_loaded:
            ckpt = torch.load(self.config.checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded model weights from {self.config.checkpoint_path}")

        # Validate on train (same loader) — just to drive the LR scheduler and early stop
        self._fit_and_restore_best(train_loader=loader, val_loader=loader)
        self._context_xt = xt.copy()

    def set_context(self, train_data) -> None:
        """Initialise forecasting context without running training (skip_training=True)."""
        self._ensure_ready()
        if not isinstance(train_data, pd.DataFrame):
            train_data = train_data.to_pd()
        xt = self._filter_domain_columns(train_data)
        self._context_xt = xt
        self._feature_names = list(train_data.columns[:xt.shape[1]])

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        time_stamps,
        time_series_prev=None,
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
            time_stamps,
            time_series_prev=None if time_series_prev is None else time_series_prev.to_pd(),
        )
        forecast_ts = TimeSeries.from_pd(forecast_df)
        err_ts = None if err_df is None else TimeSeries.from_pd(err_df)
        if return_iqr:
            return forecast_ts, err_ts, err_ts
        return forecast_ts

    def _forecast(self, time_stamps, time_series_prev=None, return_prev=False):
        if self._context_xt is None:
            raise RuntimeError("No context — train the model before forecasting.")
        self.model.eval()

        if time_series_prev is not None:
            df_prev = time_series_prev if isinstance(time_series_prev, pd.DataFrame) else time_series_prev.to_pd()
            xt_full = np.vstack([self._context_xt, self._filter_domain_columns(df_prev)])
        else:
            xt_full = self._context_xt

        horizon = len(time_stamps)
        window = int(self.config.window_size)
        forecast_horizon = int(self.config.forecast_horizon)
        pos = len(self._context_xt)
        if pos < window:
            raise RuntimeError(f"Insufficient context length {pos} for window_size={window}.")

        chunks, n_collected = [], 0
        with torch.no_grad():
            while n_collected < horizon:
                x_window = torch.FloatTensor(xt_full[pos - window: pos]).unsqueeze(0).to(self.device)
                out = self.model(x_window)[0].detach().cpu().numpy()
                n = min(forecast_horizon, horizon - n_collected)
                chunks.append(out[:n] if out.ndim > 0 else np.array([float(out)]))
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
