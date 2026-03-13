from typing import Optional
from models.classifier_base.classifier_base import SupervisedClassifierBase
from models.multi_view_classifier.config import MultiViewClassifierConfig
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from merlion.utils import TimeSeries
import numpy as np
import pandas as pd
from pytorch_metric_learning import losses

from .model import Classifier
from models.hash_checkpoint_model import HashCheckpointModel
from models.multi_view_core import MultiViewCoreMixin
import logging


logger = logging.getLogger(__name__)


class MultiViewClassifier(MultiViewCoreMixin, HashCheckpointModel, SupervisedClassifierBase):
    """
    Multi-View Transformer Classifier for Merlion.

    Processes time series in three domains (time, derivative, frequency)
    using contrastive learning and supervised classification.
    """

    config_class = MultiViewClassifierConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: MultiViewClassifierConfig, save_dir: Optional[str] = None):
        SupervisedClassifierBase.__init__(self, config)
        HashCheckpointModel.__init__(self, config, save_dir)
        if not hasattr(self, 'current_epoch') or self.current_epoch is None:
            self.current_epoch = 0

        self.device = self._get_device()

        # Initialize encoder and classifier
        self.encoder = self._create_encoder().to(self.device)
        self.classifier = self._create_classifier().to(self.device)

        # Initialize optimizers
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.clf_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        info_loss = losses.NTXentLoss(temperature=self.config.temperature)
        self.info_criterion = losses.SelfSupervisedLoss(
            info_loss, symmetric=True)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler("cuda")

        # Try to load existing checkpoint, otherwise load pre-trained weights
        if not self._try_load_existing_checkpoint():
            if config.checkpoint_path:
                self._load_pretrained_checkpoint(config.checkpoint_path)

    @property
    def require_univariate(self) -> bool:
        """Whether the model requires univariate time series."""
        return False

    def train_post_process(self, train_result: TimeSeries) -> TimeSeries:
        """
        Post-process training results.

        Args:
            train_result: Result from _train() method

        Returns:
            Processed training result
        """
        # No post-processing needed for this model
        return train_result

    def _load_checkpoint_state(self, loaded_model):
        """Transfer state from loaded model to current instance"""
        self.encoder.load_state_dict(loaded_model.encoder.state_dict())
        self.classifier.load_state_dict(loaded_model.classifier.state_dict())
        self.encoder_optimizer.load_state_dict(
            loaded_model.encoder_optimizer.state_dict())
        self.clf_optimizer.load_state_dict(
            loaded_model.clf_optimizer.state_dict())
        loaded_epoch = getattr(loaded_model, "current_epoch", 0)
        self.current_epoch = 0 if loaded_epoch is None else int(loaded_epoch)

    def _create_classifier(self):
        """Create classifier from config"""
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

    def _load_pretrained_checkpoint(self, checkpoint_path: str):
        """Load pre-trained encoder weights (different from resuming training)."""
        super()._load_pretrained_checkpoint(checkpoint_path)
        logger.info(f"Loaded pre-trained weights from {checkpoint_path}")

    def _extract_domains(self, time_series: pd.DataFrame):
        """Extract time, derivative, and frequency domains from dataframe."""
        return self._extract_domains_sequence(time_series)

    def _set_training_mode(self):
        """Set models to appropriate training mode"""
        self._set_mode_for_encoder_and_head(self.classifier)

    def _train_epoch(self, loader):
        """Modified to take a loader directly for reuse with val_loader"""
        self._set_training_mode()

        total_loss = 0.0
        total_loss_c = 0.0
        total_samples = 0

        for batch_xt, batch_dx, batch_xf, batch_y in loader:
            batch_xt, batch_dx, batch_xf, batch_y = [
                b.to(self.device) for b in [batch_xt, batch_dx, batch_xf, batch_y]
            ]

            # Create augmented versions
            batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(
                batch_xt, batch_dx, batch_xf
            )

            self.encoder_optimizer.zero_grad()
            if self.config.mode != "pretrain":
                self.clf_optimizer.zero_grad()

            with autocast("cuda", enabled=True):
                # Forward passes and contrastive base loss
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
                    logits = self.classifier(
                        zt, zd, zf) if self.config.feature == "latent" else self.classifier(ht, hd, hf)
                    loss_c = nn.CrossEntropyLoss()(logits, batch_y.long())
                    loss = self._compose_finetune_loss(loss, loss_c, self.classifier)
                    loss_c_val = loss_c.item()

            self.scaler.scale(loss).backward()
            if self.config.mode != "freeze":
                self.scaler.step(self.encoder_optimizer)
            if self.config.mode != "pretrain":
                self.scaler.step(self.clf_optimizer)
            self.scaler.update()

            total_loss += loss.item() * batch_xt.size(0)
            total_loss_c += loss_c_val * batch_xt.size(0)
            total_samples += batch_xt.size(0)

        return total_loss / total_samples, total_loss_c / total_samples

    def _validate(self, loader):
        """Evaluation loop for validation set mirroring training logic."""
        self.encoder.eval()
        self.classifier.eval()

        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_xt, batch_dx, batch_xf, batch_y in loader:
                batch_xt, batch_dx, batch_xf, batch_y = [
                    b.to(self.device) for b in [batch_xt, batch_dx, batch_xf, batch_y]
                ]

                # 1. Create augmented versions for contrastive validation
                batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(
                    batch_xt, batch_dx, batch_xf
                )

                # 2. Forward pass with mixed precision (consistent with _train_epoch)
                with autocast("cuda", enabled=True):
                    # Original and Augmented forward passes + contrastive base loss
                    (ht, hd, hf, zt, zd, zf), loss = self._compute_contrastive_encoder_loss(
                        batch_xt,
                        batch_dx,
                        batch_xf,
                        batch_xt_aug,
                        batch_dx_aug,
                        batch_xf_aug,
                    )

                    # 5. Add Supervised loss if not in pretrain mode
                    if self.config.mode != "pretrain":
                        logits = self.classifier(
                            zt, zd, zf) if self.config.feature == "latent" else self.classifier(ht, hd, hf)
                        loss_c = self.criterion(logits, batch_y.long())
                        loss = self._compose_finetune_loss(loss, loss_c, self.classifier)

                val_loss += loss.item() * batch_xt.size(0)
                total_samples += batch_xt.size(0)

        # Avoid division by zero if loader is empty
        return val_loss / total_samples if total_samples > 0 else 0.0

    def _evaluate_accuracy(self, loader) -> float:
        self.encoder.eval()
        self.classifier.eval()

        total = 0
        correct = 0

        with torch.no_grad():
            for batch_xt, batch_dx, batch_xf, batch_y in loader:
                batch_xt = batch_xt.to(self.device)
                batch_dx = batch_dx.to(self.device)
                batch_xf = batch_xf.to(self.device)
                batch_y = batch_y.to(self.device)

                ht, hd, hf, zt, zd, zf = self.encoder(batch_xt, batch_dx, batch_xf)
                logits = self.classifier(zt, zd, zf) if self.config.feature == "latent" else self.classifier(ht, hd, hf)
                preds = logits.argmax(dim=-1)

                correct += (preds == batch_y.long()).sum().item()
                total += batch_y.size(0)

        return (correct / total) if total > 0 else 0.0

    def _train(self, train_data, train_config, train_labels, val_data=None, val_labels=None):
        if self.current_epoch is None:
            self.current_epoch = 0
        if self.config.num_epochs is None:
            raise ValueError("num_epochs must be set for training")

        # 1. Prepare DataLoaders
        xt, dx, xf = self._extract_domains(train_data)
        y_train = train_labels.to_pd().values.flatten()
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.FloatTensor(xt), torch.FloatTensor(dx),
                                           torch.FloatTensor(xf), torch.LongTensor(y_train)),
            batch_size=self.config.batch_size, shuffle=True
        )

        val_loader = None
        if val_data is not None:
            v_xt, v_dx, v_xf = self._extract_domains(val_data)
            y_val = val_labels.to_pd().values.flatten()
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.FloatTensor(v_xt), torch.FloatTensor(v_dx),
                                               torch.FloatTensor(v_xf), torch.LongTensor(y_val)),
                batch_size=self.config.batch_size, shuffle=False
            )

        if self.config.mode == "pretrain" and self.config.pretrain_validate_on_train:
            val_loader = train_loader

        # 2. Setup Scheduler & Early Stopping
        monitor_metric = (
            str(getattr(self.config, "finetune_monitor_metric", "loss")).lower()
            if self.config.mode in ["finetune", "freeze"]
            else "loss"
        )
        monitor_accuracy = monitor_metric == "accuracy"

        scheduler_mode = "max" if monitor_accuracy else "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode=scheduler_mode, factor=0.5, patience=10
        )
        early_stop_counter = 0
        best_monitor_value = -float('inf') if monitor_accuracy else float('inf')

        # 3. Training Loop
        for epoch in range(int(self.current_epoch), int(self.config.num_epochs)):
            train_loss, train_loss_c = self._train_epoch(train_loader)

            # Validation Step
            current_val_loss = self._validate(
                val_loader) if val_loader else train_loss
            current_val_acc = None
            if monitor_accuracy and val_loader is not None:
                current_val_acc = self._evaluate_accuracy(val_loader)

            monitor_value = current_val_acc if monitor_accuracy else current_val_loss

            # Step Scheduler
            scheduler.step(monitor_value)

            if current_val_acc is not None:
                logger.info(
                    f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {current_val_loss:.4f}, Val Acc: {current_val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {current_val_loss:.4f}"
                )

            # Early Stopping & Saving
            is_better = monitor_value > best_monitor_value if monitor_accuracy else monitor_value < best_monitor_value
            if is_better:
                best_monitor_value = monitor_value
                early_stop_counter = 0
                self.current_epoch = epoch + 1

                self.save(save_config=True)  # Save best model
                if monitor_accuracy:
                    logger.info(f"  --> Best model saved (Val Acc: {best_monitor_value:.4f})")
                else:
                    logger.info(f"  --> Best model saved (Val Loss: {best_monitor_value:.4f})")
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        return self.get_classification_score(TimeSeries.from_pd(train_data))

    def _get_classification_score(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """
        Get classification probabilities for each class.

        Returns class probabilities for each sequence.
        """
        self.encoder.eval()
        self.classifier.eval()

        xt, dx, xf = self._extract_domains(time_series)
        num_patients = xt.shape[0]

        # Process in smaller batches to avoid OOM
        inference_batch_size = 16  # Or smaller if still OOM
        all_scores = []

        with torch.no_grad():
            for i in range(0, num_patients, inference_batch_size):
                end_idx = min(i + inference_batch_size, num_patients)

                # Get batch of patients
                xt_batch = torch.FloatTensor(xt[i:end_idx]).to(self.device)
                dx_batch = torch.FloatTensor(dx[i:end_idx]).to(self.device)
                xf_batch = torch.FloatTensor(xf[i:end_idx]).to(self.device)

                # Forward pass
                ht, hd, hf, zt, zd, zf = self.encoder(
                    xt_batch, dx_batch, xf_batch)

                if self.config.feature == "latent":
                    logits = self.classifier(zt, zd, zf)
                else:
                    logits = self.classifier(ht, hd, hf)

                probs = torch.softmax(logits, dim=-1)

                batch_scores = probs.cpu().numpy()

                all_scores.append(batch_scores)

                # Clean up
                del xt_batch, dx_batch, xf_batch, ht, hd, hf, zt, zd, zf, logits, probs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all batch scores -> [num_patients, num_classes]
        scores_np = np.concatenate(all_scores, axis=0)

        if scores_np.ndim == 1:
            scores_np = scores_np.reshape(-1, 1)

        class_cols = [f"class_score_{i}" for i in range(scores_np.shape[1])]
        score_df = pd.DataFrame(
            scores_np,
            index=time_series.to_pd().index,
            columns=class_cols,
        )

        return TimeSeries.from_pd(score_df)

    def _predict(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """
        Predict class labels for sequences.

        Returns:
            TimeSeries with predicted class for each patient
        """
        self.encoder.eval()
        self.classifier.eval()

        xt, dx, xf = self._extract_domains(time_series)

        # Handle batch of patients
        num_patients = xt.shape[0]

        # Process in batches
        inference_batch_size = 16
        all_predictions = []

        with torch.no_grad():
            for i in range(0, num_patients, inference_batch_size):
                end_idx = min(i + inference_batch_size, num_patients)

                xt_batch = torch.FloatTensor(xt[i:end_idx]).to(self.device)
                dx_batch = torch.FloatTensor(dx[i:end_idx]).to(self.device)
                xf_batch = torch.FloatTensor(xf[i:end_idx]).to(self.device)

                ht, hd, hf, zt, zd, zf = self.encoder(
                    xt_batch, dx_batch, xf_batch)

                if self.config.feature == "latent":
                    logits = self.classifier(zt, zd, zf)
                else:
                    logits = self.classifier(ht, hd, hf)

                predictions = logits.argmax(dim=-1).cpu().numpy()
                all_predictions.append(predictions)

                del xt_batch, dx_batch, xf_batch, ht, hd, hf, zt, zd, zf, logits

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate predictions
        predictions_np = np.concatenate(all_predictions)

        # Return as TimeSeries with same index as input
        pred_df = pd.DataFrame(
            predictions_np.reshape(-1, 1),
            index=time_series.to_pd().index,
            columns=["prediction"]
        )

        return TimeSeries.from_pd(pred_df)
