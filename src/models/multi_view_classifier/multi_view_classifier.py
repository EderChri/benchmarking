from typing import Optional
from models.multi_view_classifier.config import MultiViewClassifierConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from merlion.models.anomaly.base import DetectorBase
from merlion.utils import TimeSeries
import numpy as np
import pandas as pd
from pytorch_metric_learning import losses

from .model import Encoder, Classifier


class MultiViewClassifier(DetectorBase):
    """
    Multi-View Transformer Classifier for Merlion.

    Processes time series in three domains (time, derivative, frequency)
    using contrastive learning and supervised classification.
    """

    config_class = MultiViewClassifierConfig
    
    @property
    def require_even_sampling(self) -> bool:
        """
        Whether the model requires evenly sampled time series.
        
        Returns:
            False - model can handle irregularly sampled data
        """
        return False
    
    @property
    def require_univariate(self) -> bool:
        """
        Whether the model requires univariate time series.
        
        Returns:
            False - model supports multivariate time series across domains
        """
        return False
    

    def __init__(self, config: MultiViewClassifierConfig):
        super().__init__(config)
        self.config = config
        self.device = self._get_device()

        # Initialize encoder and classifier
        self.encoder = self._create_encoder().to(self.device)
        self.classifier = self._create_classifier().to(self.device)

        # Load pre-trained weights if specified
        if config.checkpoint_path:
            self._load_checkpoint(config.checkpoint_path)

        # Initialize optimizers
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.clf_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler("cuda")

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

    def _load_checkpoint(self, checkpoint_path: str):
        """Load pre-trained encoder weights"""
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
        xt = time_series[original_cols].values if original_cols else np.zeros((len(time_series), 1))
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

    def _contrastive_loss(self, z1, z2):
        """Compute NT-Xent contrastive loss"""
        temperature = self.config.temperature
        batch_size = z1.shape[0]

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute similarity matrix
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)

        # Create labels
        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e4)

        # Compute loss
        similarity_matrix = similarity_matrix / temperature
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

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

        # Create dataset and dataloader

        print(f"xt shape: {xt.shape}")
        print(f"dx shape: {dx.shape}")
        print(f"xf shape: {xf.shape}")
        print(f"labels shape: {labels.shape}")
        print(f"labels type: {type(labels)}")
        print(f"Batch sizes - xt: {xt.shape[0]}, dx: {dx.shape[0]}, xf: {xf.shape[0]}, labels: {len(labels)}")
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
                _, _, _, zt_aug, zd_aug, zf_aug = self.encoder(
                    batch_xt_aug, batch_dx_aug, batch_xf_aug
                )

                # Contrastive loss for each domain
                loss_t = info_criterion(zt, zt_aug)
                loss_d = info_criterion(zd, zd_aug)
                loss_f = info_criterion(zf, zf_aug)
    
                # Combined contrastive loss
                loss = self._compute_loss_by_type(self.config.loss_type, loss_t, loss_d, loss_f) + self._weight_regularization(self.encoder)
              # Supervised classification loss
                if self.config.mode != "pretrain":
                    if self.config.feature == "latent":
                        logits = self.classifier(zt, zd, zf)
                    else:
                        logits = self.classifier(ht, hd, hf)

                    loss_c = criterion(logits, batch_y.long())
                    loss = self.config.lam * loss + loss_c
                    loss = self.config.lam * loss + loss_c + self._weight_regularization(self.classifier)

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
        """
        Core training method called by DetectorBase.train().
        """
        if anomaly_labels is None:
            raise ValueError("MultiDomainClassifier requires labels for training")

        # Extract domains
        xt, dx, xf = self._extract_domains(train_data)

        labels = anomaly_labels.to_pd().values.flatten()

        print(f"After extraction: xt={xt.shape}, dx={dx.shape}, xf={xf.shape}, labels={labels.shape}")
        assert xt.shape[0] == dx.shape[0] == xf.shape[0] == len(labels), \
            f"Shape mismatch: xt={xt.shape[0]}, dx={dx.shape[0]}, xf={xf.shape[0]}, labels={len(labels)}"

        print(f"Training MultiDomainClassifier for {self.config.num_epochs} epochs...")
        print(f"Mode: {self.config.mode}, Loss type: {self.config.loss_type}")
        best_loss = float('inf')
        # Training loop
        for epoch in range(self.config.num_epochs):
            avg_loss, avg_loss_c = self._train_epoch(xt, dx, xf, labels)
            if avg_loss < best_loss or (epoch + 1) % 50 == 0:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                self.save(dirname=self.config.save_dir, save_config=True)

            if self.config.mode == "pretrain":
                print(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")
            else:
                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs}, "
                    f"Loss: {avg_loss:.4f}, Loss_c: {avg_loss_c:.4f}"
                )

        # Return anomaly scores on training data
        return self._get_anomaly_score(train_data)

    def _get_anomaly_score(
        self, time_series: TimeSeries, time_series_prev: Optional[TimeSeries] = None
    ) -> TimeSeries:
        """
        Core method to compute anomaly scores.
        Returns probability of anomaly class.
        """
        self.encoder.eval()
        self.classifier.eval()

        xt, dx, xf = self._extract_domains(time_series)

        print(xt.shape, dx.shape, xf.shape)

        xt_tensor = torch.FloatTensor(xt).to(self.device)
        dx_tensor = torch.FloatTensor(dx).to(self.device)
        xf_tensor = torch.FloatTensor(xf).to(self.device)

        print(xt_tensor.shape, dx_tensor.shape, xf_tensor.shape)

        with torch.no_grad():
            # Forward pass
            ht, hd, hf, zt, zd, zf = self.encoder(xt_tensor, dx_tensor, xf_tensor)

            if self.config.feature == "latent":
                logits = self.classifier(zt, zd, zf)
            else:
                logits = self.classifier(ht, hd, hf)

            # Get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Return probability of positive class (class 1 is anomaly)
            if probs.shape[1] > 1:
                scores = probs[:, 1]
            else:
                scores = probs[:, 0]

            scores_np = scores.squeeze().cpu().numpy()

        # Convert to TimeSeries
        score_df = pd.DataFrame(
            scores_np, index=time_series.to_pd().index, columns=["anom_score"]
        )

        return TimeSeries.from_pd(score_df)

    def save(self, dirname: str, **save_config):
        """Save model checkpoint"""
        import os

        super().save(dirname, **save_config)

        # Save PyTorch model weights
        torch.save(
            {
                "encoder_state_dict": self.encoder.state_dict(),
                "classifier_state_dict": self.classifier.state_dict(),
                "encoder_optimizer": self.encoder_optimizer.state_dict(),
                "clf_optimizer": self.clf_optimizer.state_dict(),
            },
            os.path.join(dirname, "pytorch_model.pt"),
        )

    @classmethod
    def load(cls, dirname: str, **kwargs):
        """Load model from checkpoint"""
        import os

        # Load Merlion config
        model = super().load(dirname, **kwargs)

        # Load PyTorch weights if they exist
        pytorch_path = os.path.join(dirname, "pytorch_model.pt")
        if os.path.exists(pytorch_path):
            checkpoint = torch.load(pytorch_path, map_location="cpu")

            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
            model.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
            model.clf_optimizer.load_state_dict(checkpoint["clf_optimizer"])

        return model
