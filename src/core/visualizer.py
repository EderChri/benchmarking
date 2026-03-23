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
        outputs = []
        for i, entry in enumerate(config.get("experiments", [])):
            run_sources = []
            artifacts = []
            for rid in entry["runs"]:
                artifact, run_dir = self._load_run(rid)
                artifacts.append(artifact)
                run_sources.append(
                    {
                        "run_id": rid,
                        # This mirrors the value stored in MLflow param `results_dir`.
                        "results_dir": str(run_dir.parent),
                    }
                )
            plot_cfg = self.factory.get_component_by_name(entry["plot_type"], "plots")
            plotter = self.factory.instantiate(plot_cfg, "plots")
            run_ids_str = "_".join(str(r) for r in entry["runs"])
            out = self.plots_dir / f"runs_{run_ids_str}_{entry['plot_type']}_{i + 1}.png"
            extra = {k: v for k, v in entry.items() if k not in ("runs", "plot_type")}
            plotter.plot(entry["runs"], artifacts, out, **extra)
            logger.info(f"Saved: {out}")
            outputs.append({"path": out, "run_sources": run_sources, "plot_type": entry["plot_type"]})
        return outputs

    def _resolve_run_dir(self, run_id: int) -> Path:
        run_dir = self.results_dir / str(run_id)
        if not (run_dir / "result.yaml").exists():
            index_path = Path("src/results/index.yaml")
            if index_path.exists():
                with open(index_path) as f:
                    index = yaml.safe_load(f) or {}
                indexed_dir = index.get(run_id)
                if indexed_dir is None:
                    indexed_dir = index.get(str(run_id))
                if indexed_dir:
                    run_dir = Path(indexed_dir)
        return run_dir

    def _load_run(self, run_id: int) -> Dict:
        run_dir = self._resolve_run_dir(run_id)
        with open(run_dir / "result.yaml") as f:
            result = yaml.safe_load(f)
        for fname, key in [
            ("predictions.pkl", "predictions"),
            ("test_data.pkl", "test_data"),
            ("test_labels.pkl", "test_labels"),
        ]:
            p = run_dir / fname
            if p.exists():
                with open(p, "rb") as f:
                    result[key] = pickle.load(f)
        return result, run_dir
