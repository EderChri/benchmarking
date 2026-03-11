import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class MlopTracker:
    def __init__(self, default_project: str = "merlion-benchmarking"):
        self._enabled = False
        self._mlop = None
        self._op = None
        self._load_env_file()
        self._api_key = os.getenv("MLOP_API_KEY")
        self._project = os.getenv("MLOP_PROJECT", default_project)

        if not self._api_key:
            logger.info("MLOP_API_KEY not set. mlop tracking is disabled.")
            return

        try:
            import mlop

            self._mlop = mlop
            self._enabled = True
        except Exception as exc:
            logger.warning(f"mlop import failed; tracking disabled. Reason: {exc}")

    def _load_env_file(self):
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        except Exception as exc:
            logger.debug(f"Skipping .env load (python-dotenv unavailable or failed): {exc}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_run(self, run: Dict[str, Any], run_configs: Dict[str, Any], results_dir: Path):
        if not self._enabled:
            return

        try:
            settings = self._mlop.Settings()
            settings._auth = self._api_key
            settings.project = self._project
            settings.mode = os.getenv("MLOP_MODE", "perf")

            run_name = f"run-{run.get('id', 'unknown')}-{run.get('name', 'benchmark')}"
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

            self._op = self._mlop.init(
                project=self._project,
                name=run_name,
                config=run_payload,
                settings=settings,
                dir=str(results_dir),
            )
            self.log({"status": 0, "stage": "init"}, step=0)
        except Exception as exc:
            logger.warning(f"Failed to start mlop run; continuing without tracking. Reason: {exc}")
            self._op = None

    def watch_model(self, model: Any):
        if not self._op:
            return

        try:
            model_structure = str(model)
            model_type = type(model).__name__

            self.log({"model_type": model_type})
            self.log({"model/structure": model_structure})

            for attr in ("encoder", "classifier", "detector_head", "forecaster_head"):
                module = getattr(model, attr, None)
                if module is not None:
                    try:
                        self._op.watch(module, disable_graph=True, disable_grad=True, disable_param=False)
                    except Exception as watch_exc:
                        logger.warning(f"mlop watch failed for '{attr}': {watch_exc}")
        except Exception as exc:
            logger.warning(f"Failed to log model structure to mlop: {exc}")

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        if not self._op:
            return
        try:
            self._op.log(data, step=step)
        except Exception as exc:
            logger.warning(f"mlop log failed: {exc}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        numeric = {}
        textual = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                numeric[f"metric/{key}"] = value
            else:
                textual[f"metric/{key}"] = json.dumps(value, default=str)

        if numeric:
            self.log(numeric, step=step)
        if textual:
            self.log(textual, step=step)

    def finish_run(self, status: str, runtime: float, error: Optional[str] = None):
        if not self._op:
            return

        try:
            payload = {"status": status, "runtime": float(runtime)}
            if error:
                payload["error"] = str(error)
            self.log(payload)
            self._op.finish()
        except Exception as exc:
            logger.warning(f"mlop finish failed: {exc}")
        finally:
            self._op = None
