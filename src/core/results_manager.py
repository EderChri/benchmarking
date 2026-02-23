import pickle
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ResultsManager:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run_id: int, exp_name: str, results: Dict, predictions=None,
                 test_data=None, status: str = "success", error: str = None,
                 runtime: float = None, run_cfg: Dict = None,
                 save_dir: str = None, config_hash: str = None) -> Dict:
        run_dir = self.results_dir / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        run_result = {"id": run_id, "experiment": exp_name, "status": status,
                      "run_cfg": run_cfg, "results": results, "runtime": runtime,
                      "checkpoint_path": f"src/data/.cache/{save_dir}/models/{config_hash}"}
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
        return run_result

    def load_run(self, run_id: int) -> Dict:
        run_dir = self.results_dir / str(run_id)
        with open(run_dir / "result.yaml") as f:
            result = yaml.safe_load(f)
        for fname, key in [("predictions.pkl", "predictions"), ("test_data.pkl", "test_data")]:
            p = run_dir / fname
            if p.exists():
                with open(p, "rb") as f:
                    result[key] = pickle.load(f)
        return result

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
