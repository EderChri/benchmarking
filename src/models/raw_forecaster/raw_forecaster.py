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
from models.raw_forecaster.config import RawForecasterConfig
from models.raw_forecaster.model import RawForecastModel


logger = logging.getLogger(__name__)


class RawForecaster(HashCheckpointModel, ForecasterBase):
    """
    Purely supervised forecaster that operates on raw (un-encoded) input windows.
    No multi-view preprocessing or contrastive learning — intended as a baseline
    counterpart to MultiViewForecaster.

    Pair with a non-multi-view preprocessor (e.g. normalization, no_preproc).
    If domain-transformed columns are present (containing config.marker), a
    warning is issued and they are dropped automatically.
    """

    config_class = RawForecasterConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: RawForecasterConfig, save_dir: Optional[str] = None):
        ForecasterBase.__init__(self, config)
        HashCheckpointModel.__init__(self, config, save_dir)
        if not hasattr(self, "current_epoch") or self.current_epoch is None:
            self.current_epoch = 0

        self.device = (
            torch.device("cuda") if config.use_gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = self._create_model().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")
        self._context_xt: Optional[np.ndarray] = None
        self._feature_names: Optional[list] = None

        if not self._try_load_existing_checkpoint():
            if config.checkpoint_path:
                self._load_pretrained_checkpoint(config.checkpoint_path)

    def _create_model(self) -> RawForecastModel:
        args = Namespace(
            num_feature=self.config.num_feature,
            num_out_features=self.config.num_out_features,
            num_embedding=self.config.num_embedding,
            num_hidden=self.config.num_hidden,
            dropout=self.config.dropout,
            forecast_horizon=self.config.forecast_horizon,
        )
        return RawForecastModel(args)

    def _load_checkpoint_state(self, loaded_model):
        self.model.load_state_dict(loaded_model.model.state_dict())
        self.optimizer.load_state_dict(loaded_model.optimizer.state_dict())
        self._context_xt = getattr(loaded_model, "_context_xt", None)
        self._feature_names = getattr(loaded_model, "_feature_names", None)
        loaded_epoch = getattr(loaded_model, "current_epoch", 0)
        self.current_epoch = 0 if loaded_epoch is None else int(loaded_epoch)

    def _load_pretrained_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded model weights from {checkpoint_path}")

    def _filter_domain_columns(self, df: pd.DataFrame) -> np.ndarray:
        """Drop any column containing the configured marker and warn if any are found."""
        domain_cols = [c for c in df.columns if self.config.marker in c]
        if domain_cols:
            logger.warning(
                f"RawForecaster received {len(domain_cols)} domain column(s) containing "
                f"'{self.config.marker}' — these will be ignored: {domain_cols}. "
                "Use a non-multi-view preprocessor (e.g. normalization, no_preproc)."
            )
            df = df.drop(columns=domain_cols)
        return df.values.astype(np.float32)

    def set_context(self, train_data: pd.DataFrame):
        """Initialize forecasting context from training data without running training.
        Required when skip_training=True so that _forecast has a valid context."""
        xt = self._filter_domain_columns(train_data)
        self._context_xt = xt
        self._feature_names = list(train_data.columns[: xt.shape[1]])

    def _build_windows(self, xt: np.ndarray):
        window, horizon = int(self.config.window_size), int(self.config.forecast_horizon)
        usable = xt.shape[0] - window - horizon + 1
        if usable <= 0:
            raise ValueError(f"Not enough points ({xt.shape[0]}) for window={window}, horizon={horizon}.")

        stride = int(max(1, self.config.train_stride))
        starts = list(range(0, usable, stride))
        if self.config.max_train_windows is not None:
            starts = starts[: int(self.config.max_train_windows)]

        target_col = int(min(max(self.config.target_seq_index, 0), xt.shape[1] - 1))
        xw, y = [], []
        for s in starts:
            xw.append(xt[s : s + window])
            y.append(
                xt[s + window : s + window + horizon, :]
                if self.config.num_out_features > 1
                else xt[s + window : s + window + horizon, target_col]
            )
        return np.asarray(xw, dtype=np.float32), np.asarray(y, dtype=np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            with autocast("cuda", enabled=self.device.type == "cuda"):
                loss = self.criterion(self.model(batch_x), batch_y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * batch_x.size(0)
            n += batch_x.size(0)
        return total_loss / n if n > 0 else 0.0

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                with autocast("cuda", enabled=self.device.type == "cuda"):
                    val_loss += self.criterion(self.model(batch_x), batch_y).item() * batch_x.size(0)
                n += batch_x.size(0)
        return val_loss / n if n > 0 else 0.0

    def _train(self, train_data: pd.DataFrame, train_config=None):
        if self.current_epoch is None:
            self.current_epoch = 0
        if self.config.num_epochs is None:
            raise ValueError("num_epochs must be set for training")

        xt = self._filter_domain_columns(train_data)
        self._feature_names = list(train_data.columns[: xt.shape[1]])
        xw, y = self._build_windows(xt)
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(xw), torch.FloatTensor(y)),
            batch_size=self.config.batch_size, shuffle=True,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        best_val, patience_count = float("inf"), 0
        for epoch in range(int(self.current_epoch), int(self.config.num_epochs)):
            train_loss = self._train_epoch(loader)
            val_loss = self._validate(loader)
            scheduler.step(val_loss)
            logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            if val_loss < best_val:
                best_val, patience_count = val_loss, 0
                self.current_epoch = epoch + 1
                self.save(save_config=True)
            else:
                patience_count += 1
            if patience_count >= self.config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        self._context_xt = xt.copy()
        target_col = int(min(max(self.config.target_seq_index, 0), xt.shape[1] - 1))
        return pd.DataFrame(xt[:, target_col], index=train_data.index, columns=["forecast"]), None

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, time_stamps, time_series_prev=None, exog_data=None,
                 return_iqr=False, return_prev=False):
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
            raise RuntimeError("No context. Train the model before forecasting.")
        self.model.eval()

        if time_series_prev is not None:
            df_prev = time_series_prev if isinstance(time_series_prev, pd.DataFrame) else time_series_prev.to_pd()
            xt_full = np.vstack([self._context_xt, self._filter_domain_columns(df_prev)])
        else:
            xt_full = self._context_xt

        horizon = len(time_stamps)
        window, forecast_horizon = int(self.config.window_size), int(self.config.forecast_horizon)
        pos = len(self._context_xt)
        if pos < window:
            raise RuntimeError(f"Insufficient context length {pos} for window_size={window}.")

        chunks, n_collected = [], 0
        with torch.no_grad():
            while n_collected < horizon:
                x_window = torch.FloatTensor(xt_full[pos - window : pos]).unsqueeze(0).to(self.device)
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
