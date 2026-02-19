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

from .model import Encoder, Classifier
from models.hash_checkpoint_model import HashCheckpointModel


class MultiViewClassifier(HashCheckpointModel, SupervisedClassifierBase):
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
        if not hasattr(self, 'current_epoch'):
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
        self.current_epoch = loaded_model.current_epoch

    def _get_device(self):
        """Determine device for training"""
        if self.config.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _create_encoder(self):
        """Create encoder from config"""
        from argparse import Namespace

        args = Namespace(
            num_feature=self.config.num_feature,
            num_embedding=self.config.num_embedding,
            num_hidden=self.config.num_hidden,
            num_head=self.config.num_head,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        return Encoder(args)

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
        """Load pre-trained encoder weights (different from resuming training)"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "encoder_state_dict" in checkpoint:
            state_dict = checkpoint["encoder_state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if exists (from DataParallel)
        state_dict = {k.replace("module.", ""): v for k,
                      v in state_dict.items()}

        self.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded pre-trained weights from {checkpoint_path}")

    def _extract_domains(self, time_series: pd.DataFrame):
        """Extract time, derivative, and frequency domains from dataframe"""
        if type(time_series) != pd.DataFrame:
            time_series = time_series.to_pd()

        # Identify columns by domain
        original_cols = [
            c
            for c in time_series.columns
            if not c.endswith("_derivative") and not c.endswith("_fft")
        ]
        derivative_cols = [
            c for c in time_series.columns if c.endswith("_derivative")]
        fft_cols = [c for c in time_series.columns if c.endswith("_fft")]

        # Extract data for each domain
        xt = time_series[original_cols].values if original_cols else np.zeros(
            (len(time_series), 1))
        dx = time_series[derivative_cols].values if derivative_cols else np.zeros_like(
            xt)
        xf = time_series[fft_cols].values if fft_cols else np.zeros_like(xt)

        if xt.ndim == 2:
            xt = xt[:, :, np.newaxis]
            dx = dx[:, :, np.newaxis]
            xf = xf[:, :, np.newaxis]

        return xt, dx, xf

    def _augment_data(self, xt, dx, xf):
        """Apply data augmentation (jittering)"""
        strength = self.config.augmentation_strength

        xt_aug = xt + torch.randn_like(xt) * strength
        dx_aug = dx + torch.randn_like(dx) * strength
        xf_aug = xf + torch.randn_like(xf) * strength

        return xt_aug, dx_aug, xf_aug

    def _compute_loss_by_type(self, loss_t, loss_d, loss_f):
        """Compute combined loss based on loss_type"""
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
        """Compute L1/L2 weight regularization"""
        l1_reg, l2_reg = 0.0, 0.0

        for param in model.parameters():
            if param.requires_grad:
                l1_reg += self.config.l1_scale * param.abs().sum()
                l2_reg += self.config.l2_scale * param.pow(2).sum()

        return l1_reg + l2_reg

    def _set_training_mode(self):
        """Set models to appropriate training mode"""
        if self.config.mode == "pretrain":
            self.encoder.train()
            for param in self.encoder.parameters():
                param.requires_grad = True

        elif self.config.mode == "finetune":
            self.encoder.train()
            for param in self.encoder.parameters():
                param.requires_grad = True
            self.classifier.train()

        elif self.config.mode == "freeze":
            self.encoder.eval()
            for name, param in self.encoder.named_parameters():
                if "input_layer" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.classifier.train()

    def _train_epoch(self, loader):
        """Modified to take a loader directly for reuse with val_loader"""
        self.encoder.train()
        if self.config.mode != "pretrain":
            self.classifier.train()

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
                # Forward passes
                ht, hd, hf, zt, zd, zf = self.encoder(
                    batch_xt, batch_dx, batch_xf)
                _, _, _, zt_aug, zd_aug, zf_aug = self.encoder(
                    batch_xt_aug, batch_dx_aug, batch_xf_aug)

                # Contrastive losses
                loss_t = self.info_criterion(zt, zt_aug)
                loss_d = self.info_criterion(zd, zd_aug)
                loss_f = self.info_criterion(zf, zf_aug)

                loss = self._compute_loss_by_type(loss_t, loss_d, loss_f) + \
                    self._weight_regularization(self.encoder)

                loss_c_val = 0.0
                if self.config.mode != "pretrain":
                    logits = self.classifier(
                        zt, zd, zf) if self.config.feature == "latent" else self.classifier(ht, hd, hf)
                    loss_c = nn.CrossEntropyLoss()(logits, batch_y.long())
                    loss = self.config.lam * loss + loss_c + \
                        self._weight_regularization(self.classifier)
                    loss_c_val = loss_c.item()

            self.scaler.scale(loss).backward()
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
                    # Original and Augmented forward passes
                    ht, hd, hf, zt, zd, zf = self.encoder(
                        batch_xt, batch_dx, batch_xf)
                    _, _, _, zt_aug, zd_aug, zf_aug = self.encoder(
                        batch_xt_aug, batch_dx_aug, batch_xf_aug
                    )

                    # 3. Calculate Contrastive Losses using self.info_criterion
                    loss_t = self.info_criterion(zt, zt_aug)
                    loss_d = self.info_criterion(zd, zd_aug)
                    loss_f = self.info_criterion(zf, zf_aug)

                    # 4. Combine domain losses using the helper method
                    loss = self._compute_loss_by_type(loss_t, loss_d, loss_f) + \
                        self._weight_regularization(self.encoder)

                    # 5. Add Supervised loss if not in pretrain mode
                    if self.config.mode != "pretrain":
                        logits = self.classifier(
                            zt, zd, zf) if self.config.feature == "latent" else self.classifier(ht, hd, hf)
                        loss_c = self.criterion(logits, batch_y.long())
                        loss = self.config.lam * loss + loss_c + \
                            self._weight_regularization(self.classifier)

                val_loss += loss.item() * batch_xt.size(0)
                total_samples += batch_xt.size(0)

        # Avoid division by zero if loader is empty
        return val_loss / total_samples if total_samples > 0 else 0.0

    def _train(self, train_data, train_config, train_labels, val_data=None, val_labels=None):
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

        # 2. Setup Scheduler & Early Stopping
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode='min', factor=0.5, patience=10
        )
        early_stop_counter = 0
        best_val_loss = float('inf')

        # 3. Training Loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            train_loss, train_loss_c = self._train_epoch(train_loader)

            # Validation Step
            current_val_loss = self._validate(
                val_loader) if val_loader else train_loss

            # Step Scheduler
            scheduler.step(current_val_loss)

            print(
                f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {current_val_loss:.4f}")

            # Early Stopping & Saving
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                early_stop_counter = 0
                self.save(save_config=True)  # Save best model
                print(f"  --> Best model saved (Loss: {best_val_loss:.4f})")
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        return self.get_classification_score(TimeSeries.from_pd(train_data))

    def _get_classification_score(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """
        Get classification probability for a single sequence.

        Returns probability of positive class, broadcast to all timestamps.
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

                # Get scores for this batch
                if probs.shape[1] > 1:
                    batch_scores = probs[:, 1].cpu().numpy()
                else:
                    batch_scores = probs[:, 0].cpu().numpy()

                all_scores.append(batch_scores)

                # Clean up
                del xt_batch, dx_batch, xf_batch, ht, hd, hf, zt, zd, zf, logits, probs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all batch scores
        scores_np = np.concatenate(all_scores)  # [num_patients]

        score_df = pd.DataFrame(
            scores_np.reshape(-1, 1),
            index=time_series.to_pd().index,
            columns=["class_score"]
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
