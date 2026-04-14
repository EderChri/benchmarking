from dataclasses import dataclass
from typing import Optional

from models.base import BaseModelConfig


@dataclass
class MultiViewClassifierConfig(BaseModelConfig):
    # Architecture
    num_feature: int = 1
    num_embedding: int = 128
    num_hidden: int = 256
    num_head: int = 8
    num_layers: int = 2
    num_target: int = 2
    dropout: float = 0.1

    # Contrastive loss
    loss_type: str = "ALL"
    feature: str = "latent"
    temperature: float = 0.5
    lam: float = 0.1
    l1_scale: float = 0.0
    l2_scale: float = 0.01

    # Training mode
    mode: str = "finetune"           # "pretrain" | "finetune" | "freeze"
    pretrain_validate_on_train: bool = False
    augmentation_strength: float = 0.1

    # For loading pretrained encoder weights from an external file
    checkpoint_path: Optional[str] = None

    # Index of the target sequence column (used by _extract_domains_matrix).
    # Set automatically by the factory from run.target.
    target_seq_index: int = 1

    # Kept for YAML backward-compatibility (used by task_executor rolling forecast)
    max_forecast_steps: Optional[int] = None
