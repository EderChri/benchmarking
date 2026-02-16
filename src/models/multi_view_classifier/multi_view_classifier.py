from typing import Optional
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


class MultiViewClassifier(HashCheckpointModel):
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
        super().__init__(config, save_dir)
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

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler("cuda")

        # Try to load existing checkpoint, otherwise load pre-trained weights
        if not self._try_load_existing_checkpoint():
            if config.checkpoint_path:
                self._load_pretrained_checkpoint(config.checkpoint_path)

    def _load_checkpoint_state(self, loaded_model):
        """Transfer state from loaded model to current instance"""
        self.encoder.load_state_dict(loaded_model.encoder.state_dict())
        self.classifier.load_state_dict(loaded_model.classifier.state_dict())
        self.encoder_optimizer.load_state_dict(loaded_model.encoder_optimizer.state_dict())
        self.clf_optimizer.load_state_dict(loaded_model.clf_optimizer.state_dict())
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
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

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
        derivative_cols = [c for c in time_series.columns if c.endswith("_derivative")]
        fft_cols = [c for c in time_series.columns if c.endswith("_fft")]

        # Extract data for each domain
        xt = time_series[original_cols].values if original_cols else np.zeros(
            (len(time_series), 1))
        dx = time_series[derivative_cols].values if derivative_cols else np.zeros_like(xt)
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

    def _train_epoch(self, xt, dx, xf, labels):
        """Train for one epoch"""
        self._set_training_mode()

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(xt),
            torch.FloatTensor(dx),
            torch.FloatTensor(xf),
            torch.LongTensor(labels),
        )

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True
        )

        info_loss = losses.NTXentLoss(temperature=self.config.temperature)
        info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_loss_c = 0.0
        total_samples = 0

        for batch_xt, batch_dx, batch_xf, batch_y in loader:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_xt = batch_xt.to(self.device)
            batch_dx = batch_dx.to(self.device)
            batch_xf = batch_xf.to(self.device)
            batch_y = batch_y.to(self.device)

            # Create augmented versions
            batch_xt_aug, batch_dx_aug, batch_xf_aug = self._augment_data(
                batch_xt, batch_dx, batch_xf
            )

            self.encoder_optimizer.zero_grad()
            if self.config.mode != "pretrain":
                self.clf_optimizer.zero_grad()

            with autocast("cuda", enabled=True):
                # Forward pass - original
                ht, hd, hf, zt, zd, zf = self.encoder(batch_xt, batch_dx, batch_xf)

                # Forward pass - augmented
                ht_aug, hd_aug, hf_aug, zt_aug, zd_aug, zf_aug = self.encoder(
                    batch_xt_aug, batch_dx_aug, batch_xf_aug
                )
                # Explicitly delete unused tensors
                del ht_aug, hd_aug, hf_aug

                # Contrastive loss for each domain
                loss_t = info_criterion(zt, zt_aug)
                loss_d = info_criterion(zd, zd_aug)
                loss_f = info_criterion(zf, zf_aug)

                # Combined contrastive loss
                loss = self._compute_loss_by_type(
                    loss_t, loss_d, loss_f) + self._weight_regularization(self.encoder)
                
                # Supervised classification loss
                if self.config.mode != "pretrain":
                    if self.config.feature == "latent":
                        logits = self.classifier(zt, zd, zf)
                    else:
                        logits = self.classifier(ht, hd, hf)

                    loss_c = criterion(logits, batch_y.long())
                    loss = self.config.lam * loss + loss_c + \
                        self._weight_regularization(self.classifier)

                    total_loss_c += loss_c.item() * batch_xt.size(0)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Optimizer step
            if self.config.mode != "freeze":
                self.scaler.step(self.encoder_optimizer)

            if self.config.mode != "pretrain":
                self.scaler.step(self.clf_optimizer)

            self.scaler.update()

            total_loss += loss.item() * batch_xt.size(0)
            total_samples += batch_xt.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_loss_c = total_loss_c / total_samples if total_samples > 0 else 0.0

        return avg_loss, avg_loss_c

    def _train(
        self,
        train_data: TimeSeries,
        train_config=None,
        anomaly_labels: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """Core training method called by DetectorBase.train()."""
        if anomaly_labels is None:
            raise ValueError("MultiDomainClassifier requires labels for training")

        # Extract domains
        xt, dx, xf = self._extract_domains(train_data)
        labels = anomaly_labels.to_pd().values.flatten()

        print(
            f"After extraction: xt={xt.shape}, dx={dx.shape}, xf={xf.shape}, labels={labels.shape}")
        assert xt.shape[0] == dx.shape[0] == xf.shape[0] == len(labels), \
            f"Shape mismatch: xt={xt.shape[0]}, dx={dx.shape[0]}, xf={xf.shape[0]}, labels={len(labels)}"

        print(f"Training MultiDomainClassifier for {self.config.num_epochs} epochs...")
        print(f"Mode: {self.config.mode}, Loss type: {self.config.loss_type}")
        best_loss = float('inf')
        
        start_epoch = self.current_epoch
        if start_epoch != 0:
            print(f"Resuming training from last checkpoint at epoch {start_epoch+1}")
            start_epoch += 1
        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            avg_loss, avg_loss_c = self._train_epoch(xt, dx, xf, labels)
            
            if avg_loss < best_loss or (epoch + 1) % 50 == 0:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                self.save(save_config=True)

            if self.config.mode == "pretrain":
                print(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")
            else:
                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs}, "
                    f"Loss: {avg_loss:.4f}, Loss_c: {avg_loss_c:.4f}"
                )

        # Return anomaly scores on training data
        return self.get_anomaly_score(train_data)

    def get_anomaly_score(
        self, time_series: TimeSeries, time_series_prev: Optional[TimeSeries] = None
    ) -> TimeSeries:
        """Core method to compute anomaly scores."""
        self.encoder.eval()
        self.classifier.eval()

        xt, dx, xf = self._extract_domains(time_series)
        
        # xt shape is [num_timestamps, num_features, 1] - we need one score per timestamp
        batch_size = self.config.batch_size
        num_samples = xt.shape[0]  # This is number of timestamps
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                
                # Batch slice
                xt_batch = torch.FloatTensor(xt[i:end_idx]).to(self.device)
                dx_batch = torch.FloatTensor(dx[i:end_idx]).to(self.device)
                xf_batch = torch.FloatTensor(xf[i:end_idx]).to(self.device)
                
                # Forward pass
                ht, hd, hf, zt, zd, zf = self.encoder(xt_batch, dx_batch, xf_batch)
                
                if self.config.feature == "latent":
                    logits = self.classifier(zt, zd, zf)
                else:
                    logits = self.classifier(ht, hd, hf)
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Return probability of positive class
                if probs.shape[1] > 1:
                    scores = probs[:, 1]  # Shape: [batch_size]
                else:
                    scores = probs[:, 0]
                
                all_scores.append(scores.cpu().numpy())
                
                # Clean up batch tensors
                del xt_batch, dx_batch, xf_batch, ht, hd, hf, zt, zd, zf, logits, probs, scores
                
            # Clear cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all batch scores - shape: [num_timestamps]
        scores_np = np.concatenate(all_scores)
        
        score_df = pd.DataFrame(
            scores_np.reshape(-1, 1),  # Ensure 2D: [num_timestamps, 1]
            index=time_series.index,
            columns=["anom_score"]  # Single column name
        )
        
        return TimeSeries.from_pd(score_df)


