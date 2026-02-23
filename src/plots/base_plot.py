from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict

class BasePlot(ABC):
    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def plot(self, run_ids: List[int], artifacts: List[Dict], output_path: Path):
        pass
