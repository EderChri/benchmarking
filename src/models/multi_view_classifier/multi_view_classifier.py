"""MultiViewClassifier — three-domain contrastive + supervised classifier.

Training modes:
    pretrain  — only encoder updated via NTXentLoss
    finetune  — encoder + classifier updated; contrastive + supervised loss
    freeze    — encoder input-projection layers + classifier updated

Uses automatic_optimization=False so both optimisers can be stepped
independently with a single GradScaler.
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

from models.classifier_base.classifier_base import SupervisedClassifierBase
from models.multi_view_classifier.config import MultiViewClassifierConfig
from models.multi_view_core import MultiViewCoreMixin
from .model import Classifier

logger = logging.getLogger(__name__)


class MultiViewClassifier(MultiViewCoreMixin, SupervisedClassifierBase):
    """Multi-View Transformer Classifier."""

    config_class = MultiViewClassifierConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: MultiViewClassifierConfig, save_dir: Optional[str] = None):
        SupervisedClassifierBase.__init__(self, config, save_dir)

        self.encoder = self._create_encoder().to(self.device)
        self.classifier = self._create_classifier().to(self.device)

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.clf_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        info_loss = losses.NTXentLoss(temperature=config.temperature)
        self.info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # PL monitor metric
    # ------------------------------------------------------------------

    def _monitor_metric(self) -> str:
        if self.config.mode in ("finetune", "freeze"):
            metric = str(getattr(self.config, "finetune_monitor_metric", "loss")).lower()
            return "val_acc" if metric == "accuracy" else "val_loss"
        return "val_loss"

    def _monitor_mode(self) -> str:
        return "max" if self._monitor_metric() == "val_acc" else "min"

    # ------------------------------------------------------------------
    # Model construction helpers
    # ------------------------------------------------------------------

    def _create_classifier(self):
        from argparse import Namespace
        args = Namespace(
            num_feature=self.config.num_feature,
            num_embedding=self.config.num_embedding,
            num_hidden=self.config.num_hidden,
            num_head=self.config.num_head,
            num_layers=self.config.num_layers,
            num_target=self.config.num_target,
            dropout=self.config.dropout,
            loss_type=self.config.loss_type,
            feature=self.config.feature,
        )
        return Classifier(args)

    # ------------------------------------------------------------------
    # Training-mode setup
    # ------------------------------------------------------------------

    def _set_training_mode(self):
        self._set_mode_for_encoder_and_head(self.classifier)

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def _make_loader(self, time_series, labels, shuffle: bool) -> DataLoader:
        xt, dx, xf = self._extract_domains(time_series)
        y = labels.to_pd().values.flatten()
        ds = TensorDataset(
            torch.FloatTensor(xt),
            torch.FloatTensor(dx),
            torch.FloatTensor(xf),
            torch.LongTensor(y),
        )
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )

    def _extract_domains(self, time_series):
        return self._extract_domains_sequence(time_series)

    # ------------------------------------------------------------------
    # PL hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        batch_xt, batch_dx, batch_xf, batch_y = batch
        self._set_training_mode()

        batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(
            batch_xt, batch_dx, batch_xf
        )

        self.encoder_optimizer.zero_grad()
        if self.config.mode != "pretrain":
            self.clf_optimizer.zero_grad()

        with autocast("cuda", enabled=self.device.type == "cuda"):
            (ht, hd, hf, zt, zd, zf), loss = self._compute_contrastive_encoder_loss(
                batch_xt, batch_dx, batch_xf,
                batch_xt_aug, batch_dx_aug, batch_xf_aug,
            )
            loss_c_val = 0.0
            if self.config.mode != "pretrain":
                logits = (
                    self.classifier(zt, zd, zf)
                    if self.config.feature == "latent"
                    else self.classifier(ht, hd, hf)
                )
                loss_c = self.criterion(logits, batch_y.long())
                loss = self._compose_finetune_loss(loss, loss_c, self.classifier)
                loss_c_val = loss_c.item()

        self.scaler.scale(loss).backward()
        if self.config.mode != "freeze":
            self.scaler.step(self.encoder_optimizer)
        if self.config.mode != "pretrain":
            self.scaler.step(self.clf_optimizer)
        self.scaler.update()

        self.log("train_loss", loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_loss_c", loss_c_val, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_xt, batch_dx, batch_xf, batch_y = batch
        self.encoder.eval()
        self.classifier.eval()

        batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(
            batch_xt, batch_dx, batch_xf
        )
        with autocast("cuda", enabled=self.device.type == "cuda"):
            (ht, hd, hf, zt, zd, zf), loss = self._compute_contrastive_encoder_loss(
                batch_xt, batch_dx, batch_xf,
                batch_xt_aug, batch_dx_aug, batch_xf_aug,
            )
            if self.config.mode != "pretrain":
                logits = (
                    self.classifier(zt, zd, zf)
                    if self.config.feature == "latent"
                    else self.classifier(ht, hd, hf)
                )
                loss_c = self.criterion(logits, batch_y.long())
                loss = self._compose_finetune_loss(loss, loss_c, self.classifier)
            else:
                logits = (
                    self.classifier(zt, zd, zf)
                    if self.config.feature == "latent"
                    else self.classifier(ht, hd, hf)
                )

        preds = logits.argmax(dim=-1)
        acc = (preds == batch_y.long()).float().mean()

        self.log("val_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc.item(), prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        enc_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer,
            mode=self._monitor_mode(),
            factor=0.5,
            patience=10,
        )
        self._lr_scheduler = enc_sched
        return (
            [self.encoder_optimizer, self.clf_optimizer],
            [],  # stepped manually in on_train_epoch_end
        )

    # ------------------------------------------------------------------
    # Protocol-facing training
    # ------------------------------------------------------------------

    def _train(self, train_data, train_config, train_labels, val_data=None, val_labels=None):
        train_loader = self._make_loader(train_data, train_labels, shuffle=True)

        if self.config.mode == "pretrain" and self.config.pretrain_validate_on_train:
            val_loader = self._make_loader(train_data, train_labels, shuffle=False)
        elif val_data is not None and val_labels is not None:
            val_loader = self._make_loader(val_data, val_labels, shuffle=False)
        else:
            val_loader = None

        if self.config.checkpoint_path and not self._checkpoint_loaded:
            self._load_pretrained_checkpoint(self.config.checkpoint_path)

        self._fit_and_restore_best(train_loader, val_loader)
        return self.get_classification_score(TimeSeries.from_pd(train_data))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_batched(self, time_series):
        self.encoder.eval()
        self.classifier.eval()
        xt, dx, xf = self._extract_domains(time_series)
        n = xt.shape[0]
        bs = self.config.batch_size
        all_logits = []

        with torch.no_grad():
            for i in range(0, n, bs):
                xt_b = torch.FloatTensor(xt[i:i+bs]).to(self.device)
                dx_b = torch.FloatTensor(dx[i:i+bs]).to(self.device)
                xf_b = torch.FloatTensor(xf[i:i+bs]).to(self.device)
                ht, hd, hf, zt, zd, zf = self.encoder(xt_b, dx_b, xf_b)
                logits = (
                    self.classifier(zt, zd, zf)
                    if self.config.feature == "latent"
                    else self.classifier(ht, hd, hf)
                )
                all_logits.append(logits.cpu().numpy())

        return np.concatenate(all_logits, axis=0)

    def _get_classification_score(self, time_series, time_series_prev=None):
        logits = self._infer_batched(time_series)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        df = pd.DataFrame(
            probs,
            index=time_series.to_pd().index,
            columns=[f"class_score_{i}" for i in range(probs.shape[1])],
        )
        return TimeSeries.from_pd(df)

    def _predict(self, time_series, time_series_prev=None):
        logits = self._infer_batched(time_series)
        preds = np.argmax(logits, axis=-1)
        df = pd.DataFrame(
            preds.reshape(-1, 1),
            index=time_series.to_pd().index,
            columns=["prediction"],
        )
        return TimeSeries.from_pd(df)

    def train_post_process(self, train_result: TimeSeries) -> TimeSeries:
        return train_result
