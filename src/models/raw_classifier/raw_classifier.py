"""RawClassifier — supervised classifier on raw (or multi-view) time series.

No contrastive encoder beneath — feeds time-domain (and optionally derivative +
FFT) representations directly into a projection head.

input_mode="raw"        — only xt used.
input_mode="multi_view" — xt, dx, xf are each projected and concatenated.
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

from models.classifier_base.classifier_base import SupervisedClassifierBase
from models.multi_view_core import MultiViewCoreMixin
from models.raw_classifier.config import RawClassifierConfig
from models.raw_classifier.model import RawHead

logger = logging.getLogger(__name__)


class RawClassifier(MultiViewCoreMixin, SupervisedClassifierBase):
    """Classifier that feeds raw (or multi-view) time series directly into a head."""

    config_class = RawClassifierConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: RawClassifierConfig, save_dir: Optional[str] = None):
        SupervisedClassifierBase.__init__(self, config, save_dir)

        self.head = RawHead(
            num_feature=config.num_feature,
            num_target=config.num_target,
            input_mode=config.input_mode,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.head.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # PL monitor metric
    # ------------------------------------------------------------------

    def _monitor_metric(self) -> str:
        metric = str(getattr(self.config, "finetune_monitor_metric", "loss")).lower()
        return "val_acc" if metric == "accuracy" else "val_loss"

    def _monitor_mode(self) -> str:
        return "max" if self._monitor_metric() == "val_acc" else "min"

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def _extract_domains(self, time_series):
        return self._extract_domains_sequence(time_series)

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, batch_xt, batch_dx, batch_xf):
        if self.config.input_mode == "multi_view":
            return self.head(batch_xt, batch_dx, batch_xf)
        return self.head(batch_xt)

    # ------------------------------------------------------------------
    # PL hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        batch_xt, batch_dx, batch_xf, batch_y = batch
        self.head.train()
        self.optimizer.zero_grad()

        with autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self._forward(batch_xt, batch_dx, batch_xf)
            loss = self.criterion(logits, batch_y.long())
            if self.config.l2_scale > 0:
                l2 = sum(p.pow(2).sum() for p in self.head.parameters() if p.requires_grad)
                loss = loss + self.config.l2_scale * l2

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.log("train_loss", loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_xt, batch_dx, batch_xf, batch_y = batch
        self.head.eval()

        with torch.no_grad(), autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self._forward(batch_xt, batch_dx, batch_xf)
            loss = self.criterion(logits, batch_y.long())

        preds = logits.argmax(dim=-1)
        acc = (preds == batch_y.long()).float().mean()
        self.log("val_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc.item(), prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self._monitor_mode(),
            factor=0.5,
            patience=10,
        )
        self._lr_scheduler = sched
        return (
            [self.optimizer],
            [],  # stepped manually in on_train_epoch_end
        )

    # ------------------------------------------------------------------
    # Protocol-facing training
    # ------------------------------------------------------------------

    def train_post_process(self, train_result: TimeSeries) -> TimeSeries:
        return train_result

    def _train(self, train_data, train_config, train_labels, val_data=None, val_labels=None):
        train_loader = self._make_loader(train_data, train_labels, shuffle=True)
        val_loader = (
            self._make_loader(val_data, val_labels, shuffle=False)
            if val_data is not None and val_labels is not None
            else None
        )
        self._fit_and_restore_best(train_loader, val_loader)
        return self.get_classification_score(TimeSeries.from_pd(train_data))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_batched(self, time_series: TimeSeries) -> np.ndarray:
        self.head.eval()
        xt, dx, xf = self._extract_domains(time_series)
        n = xt.shape[0]
        bs = self.config.batch_size
        all_logits = []

        with torch.no_grad():
            for i in range(0, n, bs):
                xt_b = torch.FloatTensor(xt[i:i+bs]).to(self.device)
                dx_b = torch.FloatTensor(dx[i:i+bs]).to(self.device)
                xf_b = torch.FloatTensor(xf[i:i+bs]).to(self.device)
                all_logits.append(self._forward(xt_b, dx_b, xf_b).cpu().numpy())

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
