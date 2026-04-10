from typing import Dict, List, Optional

from merlion.models.anomaly.base import DetectorConfig


# Default per-view augmentation types.
# "gaussian"  — Gaussian noise with augmentation_strength std.
# "spectral"  — 5 % random removal + small amplitude addition (existing xf logic).
# "none"      — pass through unchanged.
_DEFAULT_AUGMENTATION: Dict[str, str] = {
    "xt": "gaussian",
    "dx": "gaussian",
    "xf": "spectral",
    "logsig": "none",   # log-sig coordinates have algebraic structure; noise is unsound
}


class MultiViewLogSigClassifierConfig(DetectorConfig):
    """Configuration for MultiViewLogSigClassifier.

    Key additions over the existing MultiViewClassifierConfig:

    views : list[str]
        Which domain views to use.  Valid entries: "xt", "dx", "xf", "logsig".
        Default ["xt", "dx", "xf"] reproduces the original MultiViewClassifier.

    logsig_dim : int
        Dimension of the log-signature vectors.  Set to 0 to auto-detect from
        the data at train time (recommended; requires _MV_logsig columns).

    augmentation : dict[str, str]
        Per-view augmentation type.  Supported: "gaussian", "spectral", "none".
        Any view not listed falls back to "gaussian" for standard views and
        "none" for "logsig".

    augmentation_strength : float
        Std of Gaussian noise used for "gaussian" augmentation.

    lam : float
        Weight of the contrastive (NTXentLoss) term.  Set to 0.0 (default) to
        disable contrastive learning entirely — no augmentation, no second
        forward pass.  Set > 0 to enable contrastive pretraining.
    """

    def __init__(
        self,
        # View selection
        views: Optional[List[str]] = None,
        logsig_dim: int = 0,

        # Architecture
        num_feature: int = 1,
        num_embedding: int = 128,
        num_hidden: int = 256,
        num_head: int = 8,
        num_layers: int = 2,
        num_target: int = 2,
        dropout: float = 0.1,
        feature: str = "hidden",

        # Training
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 20,
        l1_scale: float = 0.0,
        l2_scale: float = 0.01,

        # Contrastive loss
        lam: float = 0.0,
        temperature: float = 0.5,

        # Per-view augmentation
        augmentation: Optional[Dict[str, str]] = None,
        augmentation_strength: float = 0.1,

        # Training mode
        mode: str = "finetune",
        pretrain_validate_on_train: bool = False,
        finetune_monitor_metric: str = "loss",

        # Device / checkpoint
        use_gpu: bool = True,
        checkpoint_path: Optional[str] = None,

        # DetectorConfig
        max_score: float = 1.0,
        threshold=None,
        enable_calibrator: bool = True,
        enable_threshold: bool = True,
        transform=None,
        **kwargs,
    ):
        self.views = views if views is not None else ["xt", "dx", "xf"]
        self.logsig_dim = int(logsig_dim)

        self.num_feature = int(num_feature)
        self.num_embedding = int(num_embedding)
        self.num_hidden = int(num_hidden)
        self.num_head = int(num_head)
        self.num_layers = int(num_layers)
        self.num_target = int(num_target)
        self.dropout = float(dropout)
        self.feature = feature

        self.batch_size = int(batch_size)
        self.num_epochs = num_epochs
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.l1_scale = float(l1_scale)
        self.l2_scale = float(l2_scale)

        self.lam = float(lam)
        self.temperature = float(temperature)

        # Merge caller-supplied augmentation over the defaults.
        merged = dict(_DEFAULT_AUGMENTATION)
        if augmentation:
            merged.update(augmentation)
        self.augmentation = merged
        self.augmentation_strength = float(augmentation_strength)

        self.mode = mode
        self.pretrain_validate_on_train = bool(pretrain_validate_on_train)
        self.finetune_monitor_metric = finetune_monitor_metric

        self.use_gpu = bool(use_gpu)
        self.checkpoint_path = checkpoint_path

        super().__init__(
            max_score=max_score,
            threshold=threshold,
            enable_calibrator=enable_calibrator,
            enable_threshold=enable_threshold,
            transform=transform,
            **kwargs,
        )
