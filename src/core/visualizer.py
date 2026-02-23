import pickle
import logging
import yaml
from pathlib import Path
from typing import Dict, List
from core.factory import ComponentFactory

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, results_dir: Path, config_dir: Path = Path("conf")):
        self.results_dir = results_dir
        self.plots_dir = results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.factory = ComponentFactory(config_dir)

    def run(self, vis_config_path: Path):
        with open(vis_config_path) as f:
            config = yaml.safe_load(f)
        for i, entry in enumerate(config.get("experiments", [])):
            artifacts = [self._load_run(rid) for rid in entry["runs"]]
            plot_cfg = self.factory.get_component_by_name(entry["plot_type"], "plots")
            plotter = self.factory.instantiate(plot_cfg, "plots")
            out = self.plots_dir / f"vis_{i + 1}_{entry['plot_type']}.png"
            plotter.plot(entry["runs"], artifacts, out)
            logger.info(f"Saved: {out}")

    def _load_run(self, run_id: int) -> Dict:
        run_dir = self.results_dir / str(run_id)
        with open(run_dir / "result.yaml") as f:
            result = yaml.safe_load(f)
        for fname, key in [("predictions.pkl", "predictions"), ("test_data.pkl", "test_data")]:
            p = run_dir / fname
            if p.exists():
                with open(p, "rb") as f:
                    result[key] = pickle.load(f)
        return result
