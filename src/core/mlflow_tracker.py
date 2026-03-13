import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class MlflowTracker:
    def __init__(self, default_experiment: str = "merlion-benchmarking"):
        self._enabled = False
        self._mlflow = None
        self._active_run = None

        self._load_env_file()
        self._tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self._experiment = os.getenv("MLFLOW_EXPERIMENT", default_experiment)

        try:
            import mlflow

            self._mlflow = mlflow
            if self._tracking_uri:
                self._mlflow.set_tracking_uri(self._tracking_uri)
            self._mlflow.set_experiment(self._experiment)
            self._enabled = True
        except Exception as exc:
            logger.warning(f"mlflow import/init failed; tracking disabled. Reason: {exc}")

    def _load_env_file(self):
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        except Exception as exc:
            logger.debug(f"Skipping .env load (python-dotenv unavailable or failed): {exc}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _log_params(self, params: Dict[str, Any]):
        if not self._active_run:
            return

        flat_params = {}
        for key, value in params.items():
            if value is None:
                continue
            text = str(value)
            # MLflow parameter values must be strings with practical size limits.
            flat_params[key] = text[:500]

        if not flat_params:
            return

        try:
            self._mlflow.log_params(flat_params)
        except Exception as exc:
            logger.warning(f"mlflow log_params failed: {exc}")

    def start_run(self, run: Dict[str, Any], run_configs: Dict[str, Any], results_dir: Path):
        if not self._enabled:
            return

        try:
            run_name = f"run-{run.get('id', 'unknown')}-{run.get('name', 'benchmark')}"
            self._active_run = self._mlflow.start_run(run_name=run_name)

            self._log_params(
                {
                    "run_id": run.get("id"),
                    "run_name": run.get("name"),
                    "task": run.get("task"),
                    "model": run.get("model"),
                    "data": run.get("data"),
                    "preprocessor": run.get("preprocessor"),
                }
            )

            run_payload = {
                "run": {
                    "id": run.get("id"),
                    "name": run.get("name"),
                    "task": run.get("task"),
                    "model": run.get("model"),
                    "data": run.get("data"),
                    "preprocessor": run.get("preprocessor"),
                },
                "configs": run_configs,
            }
            self._mlflow.log_dict(run_payload, "run_config.json")
            self._mlflow.log_param("results_dir", str(results_dir))
            self.log({"status": 0, "stage": "init"}, step=0)
        except Exception as exc:
            logger.warning(f"Failed to start mlflow run; continuing without tracking. Reason: {exc}")
            self._active_run = None

    def watch_model(self, model: Any):
        if not self._active_run:
            return

        try:
            model_type = type(model).__name__
            model_structure = str(model)

            self._mlflow.set_tag("model_type", model_type)
            self._mlflow.log_text(model_structure, "model_structure.txt")
        except Exception as exc:
            logger.warning(f"Failed to log model structure to mlflow: {exc}")

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        if not self._active_run:
            return

        numeric = {}
        textual = {}

        for key, value in data.items():
            if isinstance(value, bool):
                numeric[key] = int(value)
            elif isinstance(value, (int, float)):
                numeric[key] = float(value)
            else:
                textual[key] = str(value)

        try:
            if numeric:
                self._mlflow.log_metrics(numeric, step=step)
            for key, value in textual.items():
                # Keep textual fields as tags for easy filtering in the UI.
                self._mlflow.set_tag(key, value[:5000])
        except Exception as exc:
            logger.warning(f"mlflow log failed: {exc}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if not self._active_run:
            return

        numeric = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                numeric[f"metric/{key}"] = int(value)
            elif isinstance(value, (int, float)):
                numeric[f"metric/{key}"] = float(value)
            else:
                try:
                    self._mlflow.set_tag(f"metric/{key}", json.dumps(value, default=str)[:5000])
                except Exception as exc:
                    logger.warning(f"mlflow set_tag failed for metric '{key}': {exc}")

        if numeric:
            try:
                self._mlflow.log_metrics(numeric, step=step)
            except Exception as exc:
                logger.warning(f"mlflow log_metrics failed: {exc}")

    def finish_run(self, status: str, runtime: float, error: Optional[str] = None):
        if not self._active_run:
            return

        try:
            payload = {"status": status, "runtime": float(runtime)}
            if error:
                payload["error"] = str(error)
            self.log(payload)

            end_status = "FINISHED" if status == "success" else "FAILED"
            self._mlflow.end_run(status=end_status)
        except Exception as exc:
            logger.warning(f"mlflow finish failed: {exc}")
        finally:
            self._active_run = None
