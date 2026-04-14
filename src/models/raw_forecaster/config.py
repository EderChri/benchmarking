from dataclasses import dataclass
from typing import Optional

from models.base import BaseForecasterConfig
from utils.config import marker as _default_marker


@dataclass
class RawForecasterConfig(BaseForecasterConfig):
    num_embedding: int = 128
    num_hidden: int = 256
    dropout: float = 0.1
    marker: str = _default_marker   # columns containing this string are dropped with a warning
    checkpoint_path: Optional[str] = None
    max_forecast_steps: Optional[int] = None
