"""CustomLightningBase, ClassifierBase config, ForecasterBase — shared foundation.

Design:
  - BaseModelConfig / BaseForecasterConfig are dataclasses.  All custom model
    configs inherit from them.  Every field is overridable via
    ``overrides.model.params`` in experiments.yaml without touching Python code.
  - CustomLightningBase is a LightningModule with hash-based checkpoint path,
    _make_trainer() with EarlyStopping + ModelCheckpoint, and _ensure_ready()
    for skip_training inference loading.
  - automatic_optimization=False everywhere so models can manage multiple
    optimisers and a shared GradScaler without PL's scaler ordering constraints.
  - ClassifierBase and ForecasterBase subclass CustomLightningBase and provide
    the Protocol-facing public interface (train / predict / forecast).

Multi-GPU DDP path:
  1. Set num_devices > 1 in model YAML params.
  2. _make_trainer() automatically picks "ddp" strategy.
  3. training_step / validation_step in each model are already DDP-compatible.
"""
from __future__ import annotations

import hashlib
import logging
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging early stopping callback
# ---------------------------------------------------------------------------

class LoggingEarlyStopping(EarlyStopping):
    """EarlyStopping that emits a logger.info line when it fires."""

    def on_train_end(self, trainer, pl_module) -> None:
        if self.stopped_epoch > 0:
            logger.info(
                f"Early stopping at epoch {self.stopped_epoch}: "
                f"'{self.monitor}' did not improve for {self.patience} consecutive epochs "
                f"(best: {self.best_score:.4f})"
            )


# ---------------------------------------------------------------------------
# Shared config bases
# ---------------------------------------------------------------------------

@dataclass
class BaseModelConfig:
    """Hyperparameters shared across all custom PyTorch models.

    Every field can be overridden per-run in experiments.yaml via
    ``overrides.model.params``.
    """
    # Hardware / distribution
    use_gpu: bool = True
    num_devices: int = 1        # >1 → DDP (one run, multiple GPUs)
    num_workers: int = 4        # DataLoader worker processes per GPU
    precision: str = "32-true"  # PL precision string; "16-mixed" for AMP
    compile_model: bool = False  # torch.compile() before training (PyTorch ≥ 2.0)

    # Training loop
    batch_size: int = 32
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20
    finetune_monitor_metric: str = "loss"   # "loss" | "accuracy"

    # Kept for YAML backward-compatibility (previously forwarded to DetectorConfig)
    max_score: float = 1.0
    enable_calibrator: bool = True
    enable_threshold: bool = True


@dataclass
class BaseForecasterConfig(BaseModelConfig):
    """Additional hyperparameters shared by all custom forecasting models."""
    num_feature: int = 1
    num_out_features: int = 1
    window_size: int = 48
    forecast_horizon: int = 1
    target_seq_index: int = 0
    train_stride: int = 1
    max_train_windows: Optional[int] = None


# ---------------------------------------------------------------------------
# Config → hash
# ---------------------------------------------------------------------------

def _config_hash(config: Any) -> str:
    """16-char MD5 hex string computed from a config object.

    Handles both dataclass instances and legacy attribute-style configs.
    Lists and dicts are included so configs that differ only in ``views``
    or ``augmentation`` produce different hashes.
    """
    from dataclasses import is_dataclass
    if is_dataclass(config):
        d = asdict(config)
    else:
        d = {
            k: v
            for k, v in vars(config).items()
            if not k.startswith("_")
            and isinstance(v, (int, float, str, bool, type(None), list, dict))
        }
    return hashlib.md5(
        yaml.dump(d, sort_keys=True, default_flow_style=False).encode()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Lightning base
# ---------------------------------------------------------------------------

class CustomLightningBase(LightningModule):
    """Base LightningModule for all custom classification / forecasting models.

    Calling convention in subclass __init__:
        1.  super().__init__(config, save_dir)
        2.  Build all nn.Module objects and optimisers (using self.device).
        3.  Checkpoint loading is handled lazily:
              - Resumed training: _fit_and_restore_best() passes ckpt_path=
                to Trainer.fit(), which restores state before epoch 1.
              - Inference only: _ensure_ready() is called from predict() /
                get_classification_score() / forecast().
    """

    def __init__(self, config: BaseModelConfig, save_dir: Optional[str] = None):
        super().__init__()
        self.config = config
        self.save_dir = save_dir
        self.automatic_optimization = False
        self._checkpoint_loaded: bool = False

        # nn.Module defines `device` as a read-only property, so we store the
        # initial device under a private name and expose it via our own property
        # below.  Subclass __init__ code uses self.device before the Trainer
        # attaches; once the Trainer attaches, PL moves modules to the correct
        # device automatically via .to(device).
        use_gpu = getattr(config, "use_gpu", True)
        self._init_device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        """Return the current device.

        Before Trainer.fit() is called (i.e. during __init__), returns the
        device derived from config.use_gpu.  After attachment, PL has already
        moved all modules, so reading any parameter's device gives the same
        answer, but we keep _init_device as the stable reference.
        """
        return self._init_device

    # ------------------------------------------------------------------
    # Checkpoint location (same hash scheme as the old HashCheckpointModel)
    # ------------------------------------------------------------------

    @property
    def checkpoint_dir(self) -> Optional[Path]:
        if not self.save_dir:
            return None
        return Path(self.save_dir) / "models" / _config_hash(self.config)

    @property
    def checkpoint_path(self) -> Optional[Path]:
        """Return the best checkpoint path, falling back to last.ckpt if best was never saved."""
        d = self.checkpoint_dir
        if d is None:
            return None
        best = d / "model.ckpt"
        if best.exists():
            return best
        last = d / "last.ckpt"
        if last.exists():
            return last
        return best  # expected path, may not exist yet

    def checkpoint_exists(self) -> bool:
        d = self.checkpoint_dir
        if d is None:
            return False
        return (d / "model.ckpt").exists() or (d / "last.ckpt").exists()

    # ------------------------------------------------------------------
    # Monitor metric — override per model
    # ------------------------------------------------------------------

    def _monitor_metric(self) -> str:
        return "val_loss"

    def _monitor_mode(self) -> str:
        return "min"

    # ------------------------------------------------------------------
    # LR scheduler stepping
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self) -> None:
        """Step LR scheduler (if set) and log validation metrics.

        Called after validation runs, so val_loss is available in
        callback_metrics — required for ReduceLROnPlateau.step().
        Models register their scheduler by assigning ``self._lr_scheduler``
        (PL never touches it because automatic_optimization=False).
        """
        sched = getattr(self, "_lr_scheduler", None)
        if sched is not None:
            metric = self.trainer.callback_metrics.get(
                self._monitor_metric(), torch.tensor(float("nan"))
            )
            if not torch.isnan(metric.float()):
                sched.step(metric.item())

        metrics = {
            k: v.item() if hasattr(v, "item") else float(v)
            for k, v in self.trainer.callback_metrics.items()
        }
        if metrics:
            metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(metrics.items()))
            logger.info(f"Epoch {self.current_epoch}: {metric_str}")

    def on_train_epoch_end(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Trainer construction
    # ------------------------------------------------------------------

    def _make_trainer(self, reload_dataloaders_every_n_epochs: int = 0) -> Trainer:
        """Build a Trainer with EarlyStopping and ModelCheckpoint."""
        monitor = self._monitor_metric()
        mode = self._monitor_mode()

        callbacks: list = [
            LoggingEarlyStopping(monitor=monitor, patience=self.config.patience, mode=mode),
        ]
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.checkpoint_dir,
                    filename="model",
                    monitor=monitor,
                    mode=mode,
                    save_top_k=1,
                    save_last=True,
                )
            )

        use_gpu = getattr(self.config, "use_gpu", True)
        num_devices = int(getattr(self.config, "num_devices", 1))
        precision = getattr(self.config, "precision", "32-true")
        accelerator = "gpu" if use_gpu and torch.cuda.is_available() else "cpu"
        strategy = "ddp" if num_devices > 1 else "auto"

        return Trainer(
            max_epochs=self.config.num_epochs,
            accelerator=accelerator,
            devices=num_devices,
            strategy=strategy,
            precision=precision,
            callbacks=callbacks,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )

    # ------------------------------------------------------------------
    # Inference readiness (skip_training=True or lazy-encoder models)
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        """Load the best checkpoint for inference if not already loaded.

        Safe to call multiple times — no-op after the first successful load.
        """
        if self._checkpoint_loaded:
            return
        if not self.checkpoint_exists():
            return
        ckpt = torch.load(
            self.checkpoint_path,  # type: ignore[arg-type]
            map_location=self.device,
            weights_only=False,
        )
        self.on_load_checkpoint(ckpt)
        self.load_state_dict(ckpt["state_dict"])
        self.to(self._init_device)   # ensure model is on the expected device
        self._checkpoint_loaded = True
        logger.info(f"Loaded checkpoint ← {self.checkpoint_path}")

    # ------------------------------------------------------------------
    # Fit helper
    # ------------------------------------------------------------------

    def _fit_and_restore_best(
        self,
        train_loader=None,
        val_loader=None,
        reload_dataloaders_every_n_epochs: int = 0,
    ) -> None:
        """Run Trainer.fit() and restore the best-checkpoint weights afterwards.

        train_loader=None: PL calls model.train_dataloader() (needed for
            samplewise re-sampling where windows change each epoch).
        val_loader=None:   PL calls model.val_dataloader() if defined,
            otherwise validation is skipped.
        """
        ckpt_path = str(self.checkpoint_path) if self.checkpoint_exists() else None

        if getattr(self.config, "compile_model", False):
            torch.compile(self)

        trainer = self._make_trainer(
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs
        )
        trainer.fit(
            self,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )

        # PL does not guarantee that model weights equal the best checkpoint
        # weights after fit().  Load them explicitly.
        cb = trainer.checkpoint_callback
        best_path = getattr(cb, "best_model_path", None) or getattr(cb, "last_model_path", None)
        if best_path:
            ckpt = torch.load(best_path, map_location=self._init_device, weights_only=False)
            self.on_load_checkpoint(ckpt)
            self.load_state_dict(ckpt["state_dict"])

        # Always move to the expected device — PL moves the model to CPU after
        # fit() regardless of whether a checkpoint was loaded.
        self.to(self._init_device)
        self._checkpoint_loaded = True


# ---------------------------------------------------------------------------
# Forecaster base
# ---------------------------------------------------------------------------

class CustomForecasterBase(CustomLightningBase):
    """Base for custom forecasting models.

    Provides train() discrimination (nn.Module.train(bool) vs our training
    protocol) and the abstract _train() / forecast() interface.
    """

    def train(self, mode_or_data=True, train_config=None):
        """Discriminate nn.Module.train(bool) from task_executor's data call."""
        if isinstance(mode_or_data, bool):
            return super().train(mode_or_data)
        logger.info(f"Training {type(self).__name__}…")
        return self._train(mode_or_data, train_config)

    @abstractmethod
    def _train(self, train_data, train_config=None): ...

    @abstractmethod
    def forecast(self, time_stamps, time_series_prev=None, **kwargs): ...
