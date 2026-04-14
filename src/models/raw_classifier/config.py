from dataclasses import dataclass
from typing import Optional

from models.base import BaseModelConfig


@dataclass
class RawClassifierConfig(BaseModelConfig):
    num_feature: int = 1
    num_target: int = 4
    l2_scale: float = 0.01
    input_mode: str = "raw"   # "raw" (xt only) | "multi_view" (xt + dx + xf)
    target_seq_index: int = 1
    max_forecast_steps: Optional[int] = None
