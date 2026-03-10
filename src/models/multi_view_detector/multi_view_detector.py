from argparse import Namespace
from typing import Optional

import logging
import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from pytorch_metric_learning import losses

from merlion.models.anomaly.base import DetectorBase
from merlion.utils import TimeSeries

from models.hash_checkpoint_model import HashCheckpointModel
from models.multi_view_detector.config import MultiViewDetectorConfig
from .model import Encoder, AnomalyHead


logger = logging.getLogger(__name__)


class MultiViewDetector(HashCheckpointModel, DetectorBase):
    """Self-supervised multi-view anomaly detector."""

    config_class = MultiViewDetectorConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: MultiViewDetectorConfig, save_dir: Optional[str] = None):
        DetectorBase.__init__(self, config)
        HashCheckpointModel.__init__(self, config, save_dir)

        if not hasattr(self, "current_epoch") or self.current_epoch is None:
            self.current_epoch = 0

        self.device = self._get_device()
        self.encoder = self._create_encoder().to(self.device)
        self.detector_head = self._create_head().to(self.device)

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.head_optimizer = torch.optim.Adam(
            self.detector_head.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        info_loss = losses.NTXentLoss(temperature=self.config.temperature)
        self.info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        self.center = None
        self.score_threshold = None

        if not self._try_load_existing_checkpoint():
            if config.checkpoint_path:
                self._load_pretrained_checkpoint(config.checkpoint_path)

    def train(
        self,
        train_data: TimeSeries,
        train_config=None,
        train_labels: Optional[TimeSeries] = None,
        val_data: Optional[TimeSeries] = None,
        val_labels: Optional[TimeSeries] = None,
        anomaly_labels: Optional[TimeSeries] = None,
        post_rule_train_config=None,
    ) -> TimeSeries:
        return DetectorBase.train(
            self,
            train_data=train_data,
            train_config=train_config,
            anomaly_labels=anomaly_labels,
            post_rule_train_config=post_rule_train_config,
        )

    def _load_checkpoint_state(self, loaded_model):
        self.encoder.load_state_dict(loaded_model.encoder.state_dict())
        self.detector_head.load_state_dict(loaded_model.detector_head.state_dict())
        self.encoder_optimizer.load_state_dict(loaded_model.encoder_optimizer.state_dict())
        self.head_optimizer.load_state_dict(loaded_model.head_optimizer.state_dict())

        loaded_center = getattr(loaded_model, "center", None)
        self.center = loaded_center.to(self.device) if loaded_center is not None else None
        self.score_threshold = getattr(loaded_model, "score_threshold", None)

        loaded_epoch = getattr(loaded_model, "current_epoch", 0)
        self.current_epoch = 0 if loaded_epoch is None else int(loaded_epoch)

    def _get_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _create_encoder(self):
        args = Namespace(
            num_feature=self.config.num_feature,
            num_embedding=self.config.num_embedding,
            num_hidden=self.config.num_hidden,
            num_head=self.config.num_head,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        return Encoder(args)

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
            projection_dim=self.config.projection_dim,
        )
        return AnomalyHead(args)

    def _load_pretrained_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("encoder_state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pre-trained encoder weights from {checkpoint_path}")

    def _extract_domains(self, time_series: pd.DataFrame):
        num_feature = max(1, int(getattr(self.config, "num_feature", 1)))

        def _reshape_domain(values_2d: np.ndarray) -> np.ndarray:
            if values_2d.ndim != 2:
                return values_2d
            n, cols = values_2d.shape
            if cols % num_feature == 0:
                return values_2d.reshape(n, cols // num_feature, num_feature)
            return values_2d[:, :, np.newaxis]

        original_cols = [
            c
            for c in time_series.columns
            if not c.endswith("_derivative") and not c.endswith("_fft")
        ]
        derivative_cols = [c for c in time_series.columns if c.endswith("_derivative")]
        fft_cols = [c for c in time_series.columns if c.endswith("_fft")]

        xt = (
            time_series[original_cols].values
            if original_cols
            else np.zeros((len(time_series), 1), dtype=np.float32)
        )
        dx = (
            time_series[derivative_cols].values
            if derivative_cols
            else np.zeros_like(xt)
        )
        xf = time_series[fft_cols].values if fft_cols else np.zeros_like(xt)

        xt = _reshape_domain(xt)
        dx = _reshape_domain(dx)
        xf = _reshape_domain(xf)

        return xt, dx, xf

    def _augment_data(self, xt, dx, xf):
        sigma = self.config.augmentation_strength
        perturb_ratio = 0.05

        xt_aug = xt + torch.normal(mean=0.0, std=sigma, size=xt.shape, device=xt.device)
        dx_aug = dx + torch.normal(mean=0.0, std=sigma, size=dx.shape, device=dx.device)

        mask_remove = (torch.rand(xf.shape, device=xf.device) > perturb_ratio).to(xf.dtype)
        xf_removed = xf * mask_remove

        mask_add = (torch.rand(xf.shape, device=xf.device) > (1 - perturb_ratio)).to(xf.dtype)
        max_amplitude = xf.max()
        random_am = torch.rand(xf.shape, device=xf.device) * (max_amplitude * 0.1)
        xf_added = xf + (mask_add * random_am)
        xf_aug = xf_removed + xf_added

        return xt_aug, dx_aug, xf_aug

    def _compute_loss_by_type(self, loss_t, loss_d, loss_f):
        loss_dict = {
            "ALL": loss_t + loss_d + loss_f,
            "TDF": loss_t + loss_d + loss_f,
            "TD": loss_t + loss_d,
            "TF": loss_t + loss_f,
            "DF": loss_d + loss_f,
            "T": loss_t,
            "D": loss_d,
            "F": loss_f,
        }
        return loss_dict.get(self.config.loss_type, loss_t + loss_d + loss_f)

    def _weight_regularization(self, model):
        l1_reg, l2_reg = 0.0, 0.0
        for param in model.parameters():
            if param.requires_grad:
                l1_reg += self.config.l1_scale * param.abs().sum()
                l2_reg += self.config.l2_scale * param.pow(2).sum()
        return l1_reg + l2_reg

    def _forward_embedding(self, xt, dx, xf):
        ht, hd, hf, zt, zd, zf = self.encoder(xt, dx, xf)
        if self.config.feature == "latent":
            emb = self.detector_head(zt, zd, zf)
        else:
            emb = self.detector_head(ht, hd, hf)
        return ht, hd, hf, zt, zd, zf, emb

    def _train_epoch(self, loader: DataLoader):
        self.encoder.train()
        self.detector_head.train()

        total_loss = 0.0
        total_samples = 0

        for batch_xt, batch_dx, batch_xf in loader:
            batch_xt = batch_xt.to(self.device)
            batch_dx = batch_dx.to(self.device)
            batch_xf = batch_xf.to(self.device)

            batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(
                batch_xt, batch_dx, batch_xf
            )

            self.encoder_optimizer.zero_grad()
            self.head_optimizer.zero_grad()

            with autocast("cuda", enabled=self.device.type == "cuda"):
                _, _, _, zt, zd, zf, emb = self._forward_embedding(
                    batch_xt, batch_dx, batch_xf
                )
                _, _, _, zt_aug, zd_aug, zf_aug, emb_aug = self._forward_embedding(
                    batch_xt_aug, batch_dx_aug, batch_xf_aug
                )

                loss_t = self.info_criterion(zt, zt_aug)
                loss_d = self.info_criterion(zd, zd_aug)
                loss_f = self.info_criterion(zf, zf_aug)
                loss_views = self._compute_loss_by_type(loss_t, loss_d, loss_f)

                loss_emb = self.info_criterion(emb, emb_aug)

                loss = (
                    loss_views
                    + loss_emb
                    + self._weight_regularization(self.encoder)
                    + self._weight_regularization(self.detector_head)
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
        self.detector_head.eval()

        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_xt, batch_dx, batch_xf in loader:
                batch_xt = batch_xt.to(self.device)
                batch_dx = batch_dx.to(self.device)
                batch_xf = batch_xf.to(self.device)

                batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(
                    batch_xt, batch_dx, batch_xf
                )

                with autocast("cuda", enabled=self.device.type == "cuda"):
                    _, _, _, zt, zd, zf, emb = self._forward_embedding(
                        batch_xt, batch_dx, batch_xf
                    )
                    _, _, _, zt_aug, zd_aug, zf_aug, emb_aug = self._forward_embedding(
                        batch_xt_aug, batch_dx_aug, batch_xf_aug
                    )

                    loss_t = self.info_criterion(zt, zt_aug)
                    loss_d = self.info_criterion(zd, zd_aug)
                    loss_f = self.info_criterion(zf, zf_aug)
                    loss_views = self._compute_loss_by_type(loss_t, loss_d, loss_f)
                    loss_emb = self.info_criterion(emb, emb_aug)

                    loss = (
                        loss_views
                        + loss_emb
                        + self._weight_regularization(self.encoder)
                        + self._weight_regularization(self.detector_head)
                    )

                val_loss += loss.item() * batch_xt.size(0)
                total_samples += batch_xt.size(0)

        return val_loss / total_samples if total_samples > 0 else 0.0

    def _fit_reference_distribution(self, loader: DataLoader):
        self.encoder.eval()
        self.detector_head.eval()

        all_embeddings = []
        with torch.no_grad():
            for batch_xt, batch_dx, batch_xf in loader:
                batch_xt = batch_xt.to(self.device)
                batch_dx = batch_dx.to(self.device)
                batch_xf = batch_xf.to(self.device)

                _, _, _, _, _, _, emb = self._forward_embedding(batch_xt, batch_dx, batch_xf)
                all_embeddings.append(emb)

        if not all_embeddings:
            raise RuntimeError("No embeddings were produced to fit anomaly reference distribution.")

        embeddings = torch.cat(all_embeddings, dim=0)
        center = embeddings.mean(dim=0)
        scores = torch.norm(embeddings - center.unsqueeze(0), p=2, dim=-1)

        self.center = center.detach()
        self.score_threshold = float(
            torch.quantile(scores, q=float(self.config.threshold_quantile)).item()
        )

        logger.info(
            f"Fitted anomaly reference center with threshold={self.score_threshold:.6f} "
            f"(quantile={self.config.threshold_quantile})"
        )

    def _make_loader(self, xt, dx, xf, shuffle: bool):
        dataset = TensorDataset(
            torch.FloatTensor(xt),
            torch.FloatTensor(dx),
            torch.FloatTensor(xf),
        )
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        if self.current_epoch is None:
            self.current_epoch = 0
        if self.config.num_epochs is None:
            raise ValueError("num_epochs must be set for training")

        xt, dx, xf = self._extract_domains(train_data)
        train_loader = self._make_loader(xt, dx, xf, shuffle=True)

        best_val = float("inf")
        early_stop_counter = 0

        for epoch in range(int(self.current_epoch), int(self.config.num_epochs)):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(train_loader)

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

        self._fit_reference_distribution(train_loader)
        return self._get_anomaly_score(train_data)

    def _get_anomaly_score(
        self,
        time_series: pd.DataFrame,
        time_series_prev: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if self.center is None:
            raise RuntimeError("Detector has no fitted center. Train the model before scoring.")

        self.encoder.eval()
        self.detector_head.eval()

        xt, dx, xf = self._extract_domains(time_series)
        loader = self._make_loader(xt, dx, xf, shuffle=False)

        all_scores = []
        with torch.no_grad():
            for batch_xt, batch_dx, batch_xf in loader:
                batch_xt = batch_xt.to(self.device)
                batch_dx = batch_dx.to(self.device)
                batch_xf = batch_xf.to(self.device)

                _, _, _, _, _, _, emb = self._forward_embedding(batch_xt, batch_dx, batch_xf)
                batch_scores = torch.norm(
                    emb - self.center.unsqueeze(0), p=2, dim=-1
                )
                all_scores.append(batch_scores.detach().cpu().numpy())

        scores_np = np.concatenate(all_scores, axis=0)
        return pd.DataFrame(
            scores_np.reshape(-1, 1),
            index=time_series.index,
            columns=["anom_score"],
        )

    def predict(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        score_ts = self.get_anomaly_score(time_series, time_series_prev)
        score_df = score_ts.to_pd()

        threshold = (
            self.score_threshold
            if self.score_threshold is not None
            else float(score_df["anom_score"].quantile(self.config.threshold_quantile))
        )
        pred_df = (score_df["anom_score"] > threshold).astype(int).to_frame("prediction")
        return TimeSeries.from_pd(pred_df)
