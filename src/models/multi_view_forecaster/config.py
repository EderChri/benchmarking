from dataclasses import dataclass
from typing import Optional

from models.base import BaseForecasterConfig


@dataclass
class MultiViewForecasterConfig(BaseForecasterConfig):
    # Architecture
    num_embedding: int = 128
    num_hidden: int = 256
    num_head: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    feature: str = "latent"         # "latent" | "hidden"
    loss_type: str = "ALL"

    # Contrastive loss
    temperature: float = 0.5
    lam: float = 0.1
    l1_scale: float = 0.0
    l2_scale: float = 0.01

    # Training mode
    mode: str = "finetune"           # "pretrain" | "finetune" | "freeze"
    pretrain_validate_on_train: bool = False
    augmentation_strength: float = 0.1

    # Samplewise window sampling
    samplewise_windows_per_sample: int = 1
    samplewise_train_sampling: str = "random"
    samplewise_eval_sampling: str = "center"
    samplewise_stride: int = 1

    checkpoint_path: Optional[str] = None
    max_forecast_steps: Optional[int] = None
