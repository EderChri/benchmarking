import pickle
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)
GLOBAL_INDEX = Path("src/results/index.yaml")


class ResultsManager:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict:
        if GLOBAL_INDEX.exists():
            with open(GLOBAL_INDEX) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _update_index(self, run_id: int, run_dir: Path):
        index = self._load_index()
        index[run_id] = str(run_dir)
        GLOBAL_INDEX.parent.mkdir(parents=True, exist_ok=True)
        with open(GLOBAL_INDEX, "w") as f:
            yaml.dump(index, f, default_flow_style=False)

    def run_exists(self, run_id: int) -> bool:
        return run_id in self._load_index()

    def _build_checkpoint_path(self, save_dir: Optional[str], config_hash: Optional[str]) -> Optional[str]:
        if not save_dir:
            return None

        checkpoint_root = Path(save_dir) / "models"
        if config_hash:
            return str(checkpoint_root / config_hash)
        return str(checkpoint_root)

    def load_run(self, run_id: int, include_artifacts: bool = False) -> dict:
        index = self._load_index()
        if run_id not in index:
            raise FileNotFoundError(f"No result found for run id {run_id}")
        run_dir = Path(index[run_id])
        logger.info(
            f"Using pretrained checkpoint from run {run_id} at {run_dir}")
        with open(run_dir / "result.yaml") as f:
            result = yaml.safe_load(f)
        if include_artifacts:
            for fname, key in [("predictions.pkl", "predictions"), ("test_data.pkl", "test_data")]:
                p = run_dir / fname
                if p.exists():
                    with open(p, "rb") as f:
                        result[key] = pickle.load(f)
        return result

    def save_run(self, run_id: int, exp_name: str, results: Dict, predictions=None,
                 test_data=None, status: str = "success", error: str = None,
                 runtime: float = None, run_cfg: Dict = None,
                 save_dir: str = None, config_hash: str = None,
                 model_config: Dict = None,
                 checkpoint_path: str = None,) -> Dict:
        run_dir = self.results_dir / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved_checkpoint_path = checkpoint_path or self._build_checkpoint_path(save_dir, config_hash)

        run_result = {"id": run_id, "experiment": exp_name, "status": status,
                      "run_cfg": run_cfg, "results": results, "runtime": runtime,
                  "checkpoint_path": resolved_checkpoint_path,
                  "save_dir": save_dir,
                  "config_hash": config_hash,
                  "model_config": model_config}
        if error:
            run_result["error"] = error

        with open(run_dir / "result.yaml", "w") as f:
            yaml.dump(run_result, f, default_flow_style=False)
        if predictions is not None:
            with open(run_dir / "predictions.pkl", "wb") as f:
                pickle.dump(predictions, f)
        if test_data is not None:
            with open(run_dir / "test_data.pkl", "wb") as f:
                pickle.dump(test_data, f)
        if status == "success":
            self._update_index(run_id, run_dir)
        return run_result

    def save_summary(self, all_results: list):
        summary = {
            "total_runs": len(all_results),
            "successful": sum(1 for r in all_results if r["status"] == "success"),
            "failed": sum(1 for r in all_results if r["status"] == "failed"),
            "total_runtime": round(sum(r["runtime"] for r in all_results), 2),
            "runs": all_results,
        }
        with open(self.results_dir / "summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False)
