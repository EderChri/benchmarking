"""MultiViewLogSigClassifier — N-view contrastive classifier with log-signature support.

Training protocol:
    pretrain  — encoder trained via NTXentLoss on augmented view pairs (lam > 0).
    finetune  — encoder + classifier; contrastive loss optional (lam=0 → pure CE).
    freeze    — encoder input-projection layers frozen; only classifier trains.

Lazy model construction:
    The encoder is not built in __init__ because its input dimensions depend on
    logsig_dim, which may be unknown until the first batch of data is seen.
    setup(stage) (a PL hook called after Trainer is attached but before fit())
    builds encoder + classifier once dimensions are known.

Domain extraction reads columns by the _MV_ marker convention; routes to view
names by suffix (derivative → dx, fft → xf, logsig → logsig, plain → xt).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from merlion.utils import TimeSeries
from pytorch_metric_learning import losses

from models.classifier_base.classifier_base import SupervisedClassifierBase
from models.multi_view_log_sig_classifier.config import MultiViewLogSigClassifierConfig
from models.multi_view_log_sig_classifier.encoder import NViewEncoder
from models.multi_view_log_sig_classifier.model import NViewClassifier
from utils.config import marker

logger = logging.getLogger(__name__)

_SUFFIX_TO_VIEW = {
    "derivative": "dx",
    "fft": "xf",
    "logsig": "logsig",
}


def _col_view(col: str) -> Optional[str]:
    if marker not in str(col):
        return "xt"
    col_lower = str(col).lower()
    for suffix, view in _SUFFIX_TO_VIEW.items():
        if col_lower.endswith(suffix):
            return view
    return None


class MultiViewLogSigClassifier(SupervisedClassifierBase):

    config_class = MultiViewLogSigClassifierConfig

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Init — lazy encoder (built in setup() once view dims are known)
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: MultiViewLogSigClassifierConfig,
        save_dir: Optional[str] = None,
    ):
        SupervisedClassifierBase.__init__(self, config, save_dir)

        self._logsig_dim: int = config.logsig_dim  # 0 = detect from data
        self.encoder: Optional[NViewEncoder] = None
        self.classifier: Optional[NViewClassifier] = None
        self.encoder_optimizer = None
        self.clf_optimizer = None
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        self.criterion = nn.CrossEntropyLoss()
        self.info_criterion = None
        if config.lam > 0:
            info_loss = losses.NTXentLoss(temperature=config.temperature)
            self.info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)

        # Stored during _train() so setup() can call _build_model
        self._view_dims_cache: Optional[Dict[str, int]] = None

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
    # Lazy model construction
    # ------------------------------------------------------------------

    def _build_model(self, view_dims: Dict[str, int]) -> None:
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
            f"Built NViewEncoder + NViewClassifier — views={cfg.views}, dims={view_dims}"
        )

    def _current_view_dims(self) -> Dict[str, int]:
        dims: Dict[str, int] = {}
        for v in self.config.views:
            dims[v] = self._logsig_dim if v == "logsig" else self.config.num_feature
        return dims

    # ------------------------------------------------------------------
    # PL setup hook — called by Trainer before fit, and on load_from_checkpoint
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        if self.encoder is not None:
            return  # already built (e.g. resumed from checkpoint)
        if self._view_dims_cache is not None:
            self._build_model(self._view_dims_cache)

    # ------------------------------------------------------------------
    # PL on_save / on_load checkpoint — persist logsig_dim + view_dims
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["logsig_dim"] = self._logsig_dim
        checkpoint["view_dims"] = self._view_dims_cache

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._logsig_dim = checkpoint.get("logsig_dim", self._logsig_dim)
        self._view_dims_cache = checkpoint.get("view_dims", None)
        # Build model before PL tries to load state_dict into it.
        if self.encoder is None and self._view_dims_cache is not None:
            self._build_model(self._view_dims_cache)

    # ------------------------------------------------------------------
    # Domain extraction
    # ------------------------------------------------------------------

    def _extract_views(self, time_series) -> Dict[str, np.ndarray]:
        if not isinstance(time_series, pd.DataFrame):
            time_series = time_series.to_pd()

        buckets: Dict[str, List[str]] = {v: [] for v in self.config.views}
        for col in time_series.columns:
            view = _col_view(col)
            if view in buckets:
                buckets[view].append(col)

        n_rows = len(time_series)
        D = self.config.num_feature
        result: Dict[str, np.ndarray] = {}

        for v in self.config.views:
            cols = buckets[v]
            if not cols:
                raise ValueError(
                    f"View '{v}' configured but no matching columns found. "
                    "Ensure the preprocessor produces the required domain columns."
                )
            arr = time_series[cols].values.astype(np.float32)
            total = arr.shape[1]

            if v == "logsig":
                if self._logsig_dim == 0:
                    xt_cols = [c for c in time_series.columns if marker not in str(c)]
                    L = len(xt_cols) // D
                    self._logsig_dim = total // L if L > 0 and total % L == 0 else total
                    logger.info(f"Inferred logsig_dim={self._logsig_dim} from data.")
                D_view = self._logsig_dim
            else:
                D_view = D

            if total % D_view != 0:
                raise ValueError(
                    f"Column count {total} for view '{v}' not divisible by D_view={D_view}."
                )
            result[v] = arr.reshape(n_rows, total // D_view, D_view)

        return result

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment_views(self, view_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
                x_added = x + mask_add * (torch.rand(x.shape, device=x.device) * x.max() * 0.1)
                augmented[v] = x_removed + x_added
            else:
                raise ValueError(f"Unknown augmentation '{aug_type}' for view '{v}'.")
        return augmented

    # ------------------------------------------------------------------
    # Regularisation
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
    # Training-mode setup
    # ------------------------------------------------------------------

    def _set_training_mode(self) -> None:
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
        tensors = [torch.FloatTensor(views_np[v]) for v in self.config.views]
        tensors.append(torch.LongTensor(labels))
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )

    def _unpack_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        view_tensors = {
            v: batch[i].to(self.device) for i, v in enumerate(self.config.views)
        }
        return view_tensors, batch[-1].to(self.device)

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
    # PL hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        view_tensors, batch_y = self._unpack_batch(batch)
        self._set_training_mode()

        self.encoder_optimizer.zero_grad()
        if self.config.mode != "pretrain":
            self.clf_optimizer.zero_grad()

        with autocast("cuda", enabled=self.device.type == "cuda"):
            hiddens, latents = self.encoder(view_tensors)

            loss = self._regularization()
            if self.config.lam > 0:
                aug_tensors = self._augment_views(view_tensors)
                _, latents_aug = self.encoder(aug_tensors)
                contrastive_loss = sum(
                    self.info_criterion(latents[v], latents_aug[v])
                    for v in self.config.views
                )
                loss = loss + self.config.lam * contrastive_loss

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

        self.log("train_loss", loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_loss_c", loss_c_val, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        view_tensors, batch_y = self._unpack_batch(batch)
        self.encoder.eval()
        self.classifier.eval()

        with torch.no_grad(), autocast("cuda", enabled=self.device.type == "cuda"):
            hiddens, latents = self.encoder(view_tensors)
            logits = self.classifier(hiddens, latents)

            if self.config.mode != "pretrain":
                loss = self.criterion(logits, batch_y)
            elif self.config.lam > 0:
                aug_tensors = self._augment_views(view_tensors)
                _, latents_aug = self.encoder(aug_tensors)
                loss = self.config.lam * sum(
                    self.info_criterion(latents[v], latents_aug[v])
                    for v in self.config.views
                ) + self._regularization()
            else:
                loss = self._regularization()

        preds = logits.argmax(dim=-1)
        acc = (preds == batch_y).float().mean()
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

    def train_post_process(self, train_result: TimeSeries) -> TimeSeries:
        return train_result

    def _train(self, train_data, train_config, train_labels, val_data=None, val_labels=None):
        train_views = self._extract_views(train_data)
        y_train = train_labels.to_pd().values.flatten().astype(int)

        # Cache view dims so setup() can build the model before Trainer.fit()
        self._view_dims_cache = self._current_view_dims()
        if self.encoder is None:
            self._build_model(self._view_dims_cache)

        train_loader = self._make_loader(train_views, y_train, shuffle=True)

        if self.config.mode == "pretrain" and self.config.pretrain_validate_on_train:
            val_loader = self._make_loader(train_views, y_train, shuffle=False)
        elif val_data is not None and val_labels is not None:
            val_views = self._extract_views(val_data)
            y_val = val_labels.to_pd().values.flatten().astype(int)
            val_loader = self._make_loader(val_views, y_val, shuffle=False)
        else:
            val_loader = None

        if self.config.checkpoint_path and not self._checkpoint_loaded:
            self._load_pretrained_checkpoint_path(self.config.checkpoint_path)

        self._fit_and_restore_best(train_loader, val_loader)
        return self.get_classification_score(TimeSeries.from_pd(train_data))

    def _load_pretrained_checkpoint_path(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("encoder_state_dict", ckpt)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pre-trained encoder weights from {checkpoint_path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_batched(self, time_series: TimeSeries) -> np.ndarray:
        self.encoder.eval()
        self.classifier.eval()
        views_np = self._extract_views(time_series.to_pd())
        n = next(iter(views_np.values())).shape[0]
        bs = self.config.batch_size
        all_logits = []

        with torch.no_grad():
            for i in range(0, n, bs):
                batch_views = {
                    v: torch.FloatTensor(views_np[v][i:i+bs]).to(self.device)
                    for v in self.config.views
                }
                _, _, logits = self._forward(batch_views)
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
