from dataclasses import dataclass
from typing import Optional, Any
from merlion.utils import TimeSeries

@dataclass
class DataSplits:
    train_data: TimeSeries
    test_data: TimeSeries
    val_data: Optional[TimeSeries] = None
    train_labels: Optional[TimeSeries] = None
    test_labels: Optional[TimeSeries] = None
    val_labels: Optional[TimeSeries] = None
    has_labels: bool = False
