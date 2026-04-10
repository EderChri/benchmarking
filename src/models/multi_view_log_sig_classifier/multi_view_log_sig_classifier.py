"""MultiViewLogSigClassifier — N-view contrastive classifier with log-signature support.

Training protocol (mirrors the existing MultiViewClassifier):

    mode="pretrain"
        Encoder trained via NTXentLoss on augmented view pairs (lam > 0 required).
        Classifier head is not updated.

    mode="finetune"
        Both encoder and classifier head trained; contrastive loss is optional
        (lam=0 → purely supervised CrossEntropy).

    mode="freeze"
        Encoder frozen (except input projection layers), only classifier head trains.

Contrastive learning is gated on lam > 0:
    lam=0  → no augmentation, single forward pass, NTXentLoss not instantiated.
    lam>0  → per-view augmentation + second encoder forward pass + NTXentLoss.

Domain extraction:
    Reads columns by marker convention (_MV_ present → domain column).
    Routes to views by suffix:
        no marker   → "xt"
        "derivative" in suffix → "dx"
        "fft" in suffix        → "xf"
        "logsig" in suffix     → "logsig"
    Only the views listed in config.views are extracted and used; other domain
    columns present in the data are silently ignored.
    logsig_dim is inferred from the data on the first call if config.logsig_dim==0.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from merlion.utils import TimeSeries

from models.classifier_base.classifier_base import SupervisedClassifierBase
from models.hash_checkpoint_model import HashCheckpointModel
from models.multi_view_log_sig_classifier.config import MultiViewLogSigClassifierConfig
from models.multi_view_log_sig_classifier.encoder import NViewEncoder
from models.multi_view_log_sig_classifier.model import NViewClassifier
from utils.config import marker

logger = logging.getLogger(__name__)

# Suffix → canonical view name mapping.
_SUFFIX_TO_VIEW = {
    "derivative": "dx",
    "fft": "xf",
    "logsig": "logsig",
}


def _col_view(col: str) -> Optional[str]:
    """Return the view name for a column, or None if it is a base (xt) column."""
    if marker not in str(col):
        return "xt"
    col_lower = str(col).lower()
    for suffix, view in _SUFFIX_TO_VIEW.items():
        if col_lower.endswith(suffix):
            return view
    return None   # unknown domain column — ignored


class MultiViewLogSigClassifier(HashCheckpointModel, SupervisedClassifierBase):

    config_class = MultiViewLogSigClassifierConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: MultiViewLogSigClassifierConfig,
        save_dir: Optional[str] = None,
    ):
        SupervisedClassifierBase.__init__(self, config)
        HashCheckpointModel.__init__(self, config, save_dir)

        if not hasattr(self, "current_epoch") or self.current_epoch is None:
            self.current_epoch = 0

        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")

        # Encoder and classifier are built lazily on first training call once
        # logsig_dim is known.  Placeholders are set here so checkpoint loading
        # works if the model was previously saved with known dims.
        self._logsig_dim: int = config.logsig_dim   # 0 means "detect from data"
        self.encoder: Optional[NViewEncoder] = None
        self.classifier: Optional[NViewClassifier] = None
        self.encoder_optimizer = None
        self.clf_optimizer = None
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        # NTXentLoss only instantiated when contrastive learning is active.
        self.info_criterion = None
        if config.lam > 0:
            from pytorch_metric_learning import losses
            info_loss = losses.NTXentLoss(temperature=config.temperature)
            self.info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)

        self.criterion = nn.CrossEntropyLoss()

        if not self._try_load_existing_checkpoint():
            if config.checkpoint_path:
                self._load_pretrained_checkpoint(config.checkpoint_path)

    # ------------------------------------------------------------------
    # Model construction (lazy — called once logsig_dim is known)
    # ------------------------------------------------------------------

    def _build_model(self, view_dims: Dict[str, int]):
        """Construct encoder + classifier and their optimizers."""
        cfg = self.config
        self.encoder = NViewEncoder(
            views=cfg.views,
            view_dims=view_dims,
            num_embedding=cfg.num_embedding,
            num_hidden=cfg.num_hidden,
            num_head=cfg.num_head,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(self.device)

        self.classifier = NViewClassifier(
            views=cfg.views,
            num_hidden=cfg.num_hidden,
            num_embedding=cfg.num_embedding,
            num_head=cfg.num_head,
            num_target=cfg.num_target,
            dropout=cfg.dropout,
            feature=cfg.feature,
        ).to(self.device)

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.clf_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        logger.info(
            f"Built NViewEncoder + NViewClassifier with views={cfg.views}, "
            f"view_dims={view_dims}"
        )

    def _view_dims(self) -> Dict[str, int]:
        """Return per-view input dimensions based on current config."""
        dims = {}
        for v in self.config.views:
            if v == "logsig":
                dims[v] = self._logsig_dim
            else:
                dims[v] = self.config.num_feature
        return dims

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _load_checkpoint_state(self, loaded_model):
        if loaded_model.encoder is not None and self.encoder is not None:
            self.encoder.load_state_dict(loaded_model.encoder.state_dict())
        if loaded_model.classifier is not None and self.classifier is not None:
            self.classifier.load_state_dict(loaded_model.classifier.state_dict())
        if self.encoder_optimizer and loaded_model.encoder_optimizer:
            self.encoder_optimizer.load_state_dict(loaded_model.encoder_optimizer.state_dict())
        if self.clf_optimizer and loaded_model.clf_optimizer:
            self.clf_optimizer.load_state_dict(loaded_model.clf_optimizer.state_dict())
        if hasattr(loaded_model, "_logsig_dim"):
            self._logsig_dim = loaded_model._logsig_dim
        loaded_epoch = getattr(loaded_model, "current_epoch", 0)
        self.current_epoch = 0 if loaded_epoch is None else int(loaded_epoch)

    def _load_pretrained_checkpoint(self, checkpoint_path: str):
        super()._load_pretrained_checkpoint(checkpoint_path)
        logger.info(f"Loaded pre-trained encoder weights from {checkpoint_path}")

    # ------------------------------------------------------------------
    # Domain extraction
    # ------------------------------------------------------------------

    def _extract_views(self, time_series) -> Dict[str, np.ndarray]:
        """Extract requested views from a samplewise DataFrame.

        Each view is returned as a float32 array shaped [n, L, D_view].
        """
        if not isinstance(time_series, pd.DataFrame):
            time_series = time_series.to_pd()

        # Bucket columns by view.
        buckets: Dict[str, List[str]] = {v: [] for v in self.config.views}
        for col in time_series.columns:
            view = _col_view(col)
            if view in buckets:
                buckets[view].append(col)

        # For xt: columns with no marker.
        n_rows = len(time_series)
        D = self.config.num_feature
        result: Dict[str, np.ndarray] = {}

        for v in self.config.views:
            cols = buckets[v]
            if not cols:
                raise ValueError(
                    f"View '{v}' is configured but no matching columns found in the data. "
                    f"Ensure the preprocessor produces the required domain columns."
                )
            arr = time_series[cols].values.astype(np.float32)  # [n, L*D_view]
            total = arr.shape[1]

            if v == "logsig":
                # Infer logsig_dim from the data if not set.
                if self._logsig_dim == 0:
                    # total = L * logsig_dim; L = total_xt_cols / D
                    xt_cols = [c for c in time_series.columns if marker not in str(c)]
                    L = len(xt_cols) // D
                    if L > 0 and total % L == 0:
                        self._logsig_dim = total // L
                    else:
                        self._logsig_dim = total  # fallback: treat whole array as one step
                    logger.info(f"Inferred logsig_dim={self._logsig_dim} from data.")
                D_view = self._logsig_dim
            else:
                D_view = D

            if total % D_view != 0:
                raise ValueError(
                    f"Column count {total} for view '{v}' not divisible by "
                    f"expected D_view={D_view}."
                )
            L = total // D_view
            result[v] = arr.reshape(n_rows, L, D_view)  # [n, L, D_view]

        return result

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment_views(
        self, view_tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply per-view augmentation as configured."""
        sigma = self.config.augmentation_strength
        augmented: Dict[str, torch.Tensor] = {}

        for v, x in view_tensors.items():
            aug_type = self.config.augmentation.get(v, "gaussian")

            if aug_type == "none":
                augmented[v] = x

            elif aug_type == "gaussian":
                augmented[v] = x + torch.normal(0.0, sigma, size=x.shape, device=x.device)

            elif aug_type == "spectral":
                perturb_ratio = 0.05
                mask_remove = (torch.rand(x.shape, device=x.device) > perturb_ratio).to(x.dtype)
                x_removed = x * mask_remove
                mask_add = (torch.rand(x.shape, device=x.device) > (1 - perturb_ratio)).to(x.dtype)
                max_amp = x.max()
                x_added = x + mask_add * (torch.rand(x.shape, device=x.device) * max_amp * 0.1)
                augmented[v] = x_removed + x_added

            else:
                raise ValueError(
                    f"Unknown augmentation type '{aug_type}' for view '{v}'. "
                    "Use 'gaussian', 'spectral', or 'none'."
                )

        return augmented

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------

    def _regularization(self) -> torch.Tensor:
        reg = torch.tensor(0.0, device=self.device)
        for module in [self.encoder, self.classifier]:
            if module is None:
                continue
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                if self.config.l1_scale > 0:
                    reg = reg + self.config.l1_scale * param.abs().sum()
                if self.config.l2_scale > 0:
                    reg = reg + self.config.l2_scale * param.pow(2).sum()
        return reg

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------

    def _set_training_mode(self):
        mode = self.config.mode
        if mode == "pretrain":
            self.encoder.train()
            for p in self.encoder.parameters():
                p.requires_grad = True

        elif mode == "finetune":
            self.encoder.train()
            for p in self.encoder.parameters():
                p.requires_grad = True
            self.classifier.train()

        elif mode == "freeze":
            self.encoder.eval()
            for name, p in self.encoder.named_parameters():
                p.requires_grad = "input_layers" in name
            self.classifier.train()

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        views_np: Dict[str, np.ndarray],
        labels: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        # Ordered tensors matching self.config.views + labels at end.
        tensors = [torch.FloatTensor(views_np[v]) for v in self.config.views]
        tensors.append(torch.LongTensor(labels))
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    def _unpack_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Unpack a DataLoader batch into view dict + label tensor."""
        view_tensors = {
            v: batch[i].to(self.device) for i, v in enumerate(self.config.views)
        }
        labels = batch[-1].to(self.device)
        return view_tensors, labels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self, view_tensors: Dict[str, torch.Tensor]
    ) -> Tuple[Dict, Dict, torch.Tensor]:
        hiddens, latents = self.encoder(view_tensors)
        logits = self.classifier(hiddens, latents)
        return hiddens, latents, logits

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self._set_training_mode()
        total_loss = total_loss_c = total_samples = 0

        for batch in loader:
            view_tensors, batch_y = self._unpack_batch(batch)

            self.encoder_optimizer.zero_grad()
            if self.config.mode != "pretrain":
                self.clf_optimizer.zero_grad()

            with autocast("cuda", enabled=self.device.type == "cuda"):
                # --- contrastive loss (only when lam > 0) ---
                if self.config.lam > 0:
                    aug_tensors = self._augment_views(view_tensors)
                    hiddens, latents, _ = self._forward(view_tensors)
                    _, latents_aug, _ = self._forward(aug_tensors)
                    contrastive_loss = torch.tensor(0.0, device=self.device)
                    for v in self.config.views:
                        contrastive_loss = contrastive_loss + self.info_criterion(
                            latents[v], latents_aug[v]
                        )
                    loss = self.config.lam * contrastive_loss + self._regularization()
                else:
                    hiddens, latents, _ = self._forward(view_tensors)
                    loss = self._regularization()

                # --- supervised loss ---
                loss_c_val = 0.0
                if self.config.mode != "pretrain":
                    logits = self.classifier(hiddens, latents)
                    loss_c = self.criterion(logits, batch_y)
                    loss = loss + loss_c + self._regularization()
                    loss_c_val = loss_c.item()

            self.scaler.scale(loss).backward()
            if self.config.mode != "freeze":
                self.scaler.step(self.encoder_optimizer)
            if self.config.mode != "pretrain":
                self.scaler.step(self.clf_optimizer)
            self.scaler.update()

            total_loss += loss.item() * batch_y.size(0)
            total_loss_c += loss_c_val * batch_y.size(0)
            total_samples += batch_y.size(0)

        if total_samples == 0:
            return 0.0, 0.0
        return total_loss / total_samples, total_loss_c / total_samples

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def _validate(self, loader: DataLoader) -> float:
        self.encoder.eval()
        self.classifier.eval()
        val_loss = total_samples = 0

        with torch.no_grad():
            for batch in loader:
                view_tensors, batch_y = self._unpack_batch(batch)
                with autocast("cuda", enabled=self.device.type == "cuda"):
                    hiddens, latents, logits = self._forward(view_tensors)
                    if self.config.mode != "pretrain":
                        loss = self.criterion(logits, batch_y)
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                val_loss += loss.item() * batch_y.size(0)
                total_samples += batch_y.size(0)

        return val_loss / total_samples if total_samples > 0 else 0.0

    def _evaluate_accuracy(self, loader: DataLoader) -> float:
        self.encoder.eval()
        self.classifier.eval()
        correct = total = 0

        with torch.no_grad():
            for batch in loader:
                view_tensors, batch_y = self._unpack_batch(batch)
                _, _, logits = self._forward(view_tensors)
                correct += (logits.argmax(dim=-1) == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train_post_process(self, train_result: TimeSeries) -> TimeSeries:
        return train_result

    def _train(self, train_data, train_config, train_labels, val_data=None, val_labels=None):
        if self.current_epoch is None:
            self.current_epoch = 0
        if self.config.num_epochs is None:
            raise ValueError("num_epochs must be set.")

        # Extract views; this also sets self._logsig_dim if needed.
        train_views = self._extract_views(train_data)
        y_train = train_labels.to_pd().values.flatten().astype(int)

        # Build model on first training call (logsig_dim now known).
        if self.encoder is None:
            self._build_model(self._view_dims())

        train_loader = self._make_loader(train_views, y_train, shuffle=True)

        val_loader = None
        if self.config.mode == "pretrain" and self.config.pretrain_validate_on_train:
            val_loader = train_loader
        elif val_data is not None and val_labels is not None:
            val_views = self._extract_views(val_data)
            y_val = val_labels.to_pd().values.flatten().astype(int)
            val_loader = self._make_loader(val_views, y_val, shuffle=False)

        monitor_accuracy = (
            str(getattr(self.config, "finetune_monitor_metric", "loss")).lower() == "accuracy"
            and self.config.mode in ["finetune", "freeze"]
        )
        scheduler_mode = "max" if monitor_accuracy else "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode=scheduler_mode, factor=0.5, patience=10
        )

        best = -float("inf") if monitor_accuracy else float("inf")
        early_stop = 0

        for epoch in range(int(self.current_epoch), int(self.config.num_epochs)):
            train_loss, train_loss_c = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader) if val_loader else train_loss
            val_acc = self._evaluate_accuracy(val_loader) if (monitor_accuracy and val_loader) else None

            monitor_val = val_acc if monitor_accuracy else val_loss
            scheduler.step(monitor_val)

            if val_acc is not None:
                logger.info(
                    f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                    f"train_loss_c={train_loss_c:.4f}, val_loss={val_loss:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                    f"train_loss_c={train_loss_c:.4f}, val_loss={val_loss:.4f}"
                )

            is_better = monitor_val > best if monitor_accuracy else monitor_val < best
            if is_better:
                best = monitor_val
                early_stop = 0
                self.current_epoch = epoch + 1
                self.save(save_config=True)
            else:
                early_stop += 1

            if early_stop >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

        return self.get_classification_score(TimeSeries.from_pd(train_data))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_batched(self, time_series: TimeSeries) -> np.ndarray:
        """Run encoder + classifier on time_series, return logits numpy array."""
        self.encoder.eval()
        self.classifier.eval()

        views_np = self._extract_views(time_series.to_pd())
        n = next(iter(views_np.values())).shape[0]
        bs = 16
        all_logits = []

        with torch.no_grad():
            for i in range(0, n, bs):
                batch_views = {
                    v: torch.FloatTensor(views_np[v][i: i + bs]).to(self.device)
                    for v in self.config.views
                }
                _, _, logits = self._forward(batch_views)
                all_logits.append(logits.cpu().numpy())
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return np.concatenate(all_logits, axis=0)

    def _get_classification_score(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        logits = self._infer_batched(time_series)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        cols = [f"class_score_{i}" for i in range(probs.shape[1])]
        df = pd.DataFrame(probs, index=time_series.to_pd().index, columns=cols)
        return TimeSeries.from_pd(df)

    def _predict(
        self,
        time_series: TimeSeries,
        time_series_prev: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        logits = self._infer_batched(time_series)
        preds = np.argmax(logits, axis=-1)
        df = pd.DataFrame(
            preds.reshape(-1, 1),
            index=time_series.to_pd().index,
            columns=["prediction"],
        )
        return TimeSeries.from_pd(df)
