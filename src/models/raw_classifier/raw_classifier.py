from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from merlion.utils import TimeSeries

from models.classifier_base.classifier_base import SupervisedClassifierBase
from models.hash_checkpoint_model import HashCheckpointModel
from models.multi_view_core import MultiViewCoreMixin
from models.raw_classifier.config import RawClassifierConfig
from models.raw_classifier.model import RawHead

import logging

logger = logging.getLogger(__name__)


class RawClassifier(MultiViewCoreMixin, HashCheckpointModel, SupervisedClassifierBase):
    """Classifier that feeds raw (or multi-view) time series directly into a
    projection head — no contrastive encoder beneath.

    input_mode="raw"       : only xt is used.
    input_mode="multiview" : xt, dx, xf are each projected and concatenated.
    """

    config_class = RawClassifierConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: RawClassifierConfig, save_dir: Optional[str] = None):
        SupervisedClassifierBase.__init__(self, config)
        HashCheckpointModel.__init__(self, config, save_dir)
        if not hasattr(self, "current_epoch") or self.current_epoch is None:
            self.current_epoch = 0

        self.device = self._get_device()

        self.head = RawHead(
            num_feature=config.num_feature,
            num_target=config.num_target,
            input_mode=config.input_mode,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.head.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler("cuda")

        self._try_load_existing_checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_checkpoint_state(self, loaded_model):
        self.head.load_state_dict(loaded_model.head.state_dict())
        self.optimizer.load_state_dict(loaded_model.optimizer.state_dict())
        loaded_epoch = getattr(loaded_model, "current_epoch", 0)
        self.current_epoch = 0 if loaded_epoch is None else int(loaded_epoch)

    # ------------------------------------------------------------------
    # Domain extraction
    # ------------------------------------------------------------------

    def _extract_domains(self, time_series):
        return self._extract_domains_sequence(time_series)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _forward(self, batch_xt, batch_dx, batch_xf):
        if self.config.input_mode == "multiview":
            return self.head(batch_xt, batch_dx, batch_xf)
        return self.head(batch_xt)

    def _train_epoch(self, loader):
        self.head.train()
        total_loss, total_samples = 0.0, 0

        for batch_xt, batch_dx, batch_xf, batch_y in loader:
            batch_xt, batch_dx, batch_xf, batch_y = [
                b.to(self.device) for b in [batch_xt, batch_dx, batch_xf, batch_y]
            ]
            self.optimizer.zero_grad()

            with autocast("cuda", enabled=True):
                logits = self._forward(batch_xt, batch_dx, batch_xf)
                loss = self.criterion(logits, batch_y.long())
                if self.config.l2_scale > 0:
                    l2 = sum(p.pow(2).sum() for p in self.head.parameters() if p.requires_grad)
                    loss = loss + self.config.l2_scale * l2

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * batch_xt.size(0)
            total_samples += batch_xt.size(0)

        return total_loss / total_samples

    def _validate(self, loader) -> float:
        self.head.eval()
        total_loss, total_samples = 0.0, 0

        with torch.no_grad():
            for batch_xt, batch_dx, batch_xf, batch_y in loader:
                batch_xt, batch_dx, batch_xf, batch_y = [
                    b.to(self.device) for b in [batch_xt, batch_dx, batch_xf, batch_y]
                ]
                with autocast("cuda", enabled=True):
                    logits = self._forward(batch_xt, batch_dx, batch_xf)
                    loss = self.criterion(logits, batch_y.long())

                total_loss += loss.item() * batch_xt.size(0)
                total_samples += batch_xt.size(0)

        return total_loss / total_samples if total_samples > 0 else 0.0

    def _evaluate_accuracy(self, loader) -> float:
        self.head.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch_xt, batch_dx, batch_xf, batch_y in loader:
                batch_xt, batch_dx, batch_xf, batch_y = [
                    b.to(self.device) for b in [batch_xt, batch_dx, batch_xf, batch_y]
                ]
                logits = self._forward(batch_xt, batch_dx, batch_xf)
                correct += (logits.argmax(dim=-1) == batch_y.long()).sum().item()
                total += batch_y.size(0)

        return correct / total if total > 0 else 0.0

    def _make_loader(self, time_series, labels, shuffle: bool):
        xt, dx, xf = self._extract_domains(time_series)
        y = labels.to_pd().values.flatten()
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(xt),
            torch.FloatTensor(dx),
            torch.FloatTensor(xf),
            torch.LongTensor(y),
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=shuffle
        )

    def train_post_process(self, train_result: TimeSeries) -> TimeSeries:
        return train_result

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def _train(self, train_data, train_config, train_labels,
               val_data=None, val_labels=None):
        if self.current_epoch is None:
            self.current_epoch = 0
        if self.config.num_epochs is None:
            raise ValueError("num_epochs must be set")

        train_loader = self._make_loader(train_data, train_labels, shuffle=True)
        val_loader = (
            self._make_loader(val_data, val_labels, shuffle=False)
            if val_data is not None else None
        )

        monitor_accuracy = str(
            getattr(self.config, "finetune_monitor_metric", "loss")
        ).lower() == "accuracy"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max" if monitor_accuracy else "min",
            factor=0.5,
            patience=10,
        )
        best = -float("inf") if monitor_accuracy else float("inf")
        early_stop_counter = 0

        for epoch in range(int(self.current_epoch), int(self.config.num_epochs)):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader) if val_loader else train_loss
            val_acc = self._evaluate_accuracy(val_loader) if monitor_accuracy and val_loader else None

            monitor_value = val_acc if monitor_accuracy else val_loss
            scheduler.step(monitor_value)

            if val_acc is not None:
                logger.info(
                    f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            is_better = monitor_value > best if monitor_accuracy else monitor_value < best
            if is_better:
                best = monitor_value
                early_stop_counter = 0
                self.current_epoch = epoch + 1
                self.save(save_config=True)
                logger.info(f"  --> Best model saved ({best:.4f})")
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return self.get_classification_score(TimeSeries.from_pd(train_data))

    def _get_classification_score(self, time_series, time_series_prev=None):
        self.head.eval()
        xt, dx, xf = self._extract_domains(time_series)
        num_samples = xt.shape[0]
        all_scores = []

        with torch.no_grad():
            for i in range(0, num_samples, 16):
                end = min(i + 16, num_samples)
                xt_b = torch.FloatTensor(xt[i:end]).to(self.device)
                dx_b = torch.FloatTensor(dx[i:end]).to(self.device)
                xf_b = torch.FloatTensor(xf[i:end]).to(self.device)
                logits = self._forward(xt_b, dx_b, xf_b)
                all_scores.append(torch.softmax(logits, dim=-1).cpu().numpy())

        scores_np = np.concatenate(all_scores, axis=0)
        if scores_np.ndim == 1:
            scores_np = scores_np.reshape(-1, 1)

        score_df = pd.DataFrame(
            scores_np,
            index=time_series.to_pd().index,
            columns=[f"class_score_{i}" for i in range(scores_np.shape[1])],
        )
        return TimeSeries.from_pd(score_df)

    def _predict(self, time_series, time_series_prev=None):
        self.head.eval()
        xt, dx, xf = self._extract_domains(time_series)
        num_samples = xt.shape[0]
        all_preds = []

        with torch.no_grad():
            for i in range(0, num_samples, 16):
                end = min(i + 16, num_samples)
                xt_b = torch.FloatTensor(xt[i:end]).to(self.device)
                dx_b = torch.FloatTensor(dx[i:end]).to(self.device)
                xf_b = torch.FloatTensor(xf[i:end]).to(self.device)
                logits = self._forward(xt_b, dx_b, xf_b)
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())

        predictions_np = np.concatenate(all_preds)
        pred_df = pd.DataFrame(
            predictions_np.reshape(-1, 1),
            index=time_series.to_pd().index,
            columns=["prediction"],
        )
        return TimeSeries.from_pd(pred_df)
