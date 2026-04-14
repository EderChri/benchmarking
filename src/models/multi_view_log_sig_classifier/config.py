from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.base import BaseModelConfig


# Default per-view augmentation types.
_DEFAULT_AUGMENTATION: Dict[str, str] = {
    "xt": "gaussian",
    "dx": "gaussian",
    "xf": "spectral",
    "logsig": "none",
}


@dataclass
class MultiViewLogSigClassifierConfig(BaseModelConfig):
    # View selection
    views: List[str] = field(default_factory=lambda: ["xt", "dx", "xf"])
    logsig_dim: int = 0   # 0 = auto-detect from data at train time

    # Architecture
    num_feature: int = 1
    num_embedding: int = 128
    num_hidden: int = 256
    num_head: int = 8
    num_layers: int = 2
    num_target: int = 2
    dropout: float = 0.1
    feature: str = "hidden"

    # Regularisation
    l1_scale: float = 0.0
    l2_scale: float = 0.01

    # Contrastive loss (lam=0 disables entirely)
    lam: float = 0.0
    temperature: float = 0.5

    # Per-view augmentation (merged over _DEFAULT_AUGMENTATION in __post_init__)
    augmentation: Dict[str, str] = field(default_factory=dict)
    augmentation_strength: float = 0.1

    # Training mode
    mode: str = "finetune"           # "pretrain" | "finetune" | "freeze"
    pretrain_validate_on_train: bool = False

    # For loading pretrained encoder weights from an external file
    checkpoint_path: Optional[str] = None

    # Set automatically by the factory from run.target
    target_seq_index: int = 1

    # Kept for YAML backward-compatibility
    max_forecast_steps: Optional[int] = None

    def __post_init__(self) -> None:
        # Merge caller-supplied augmentation over defaults.
        merged = dict(_DEFAULT_AUGMENTATION)
        merged.update(self.augmentation)
        self.augmentation = merged
