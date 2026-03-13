import logging
import os
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import random
import numpy as np
import torch
import yaml

import loaders
import preprocessing
from core.cache import PreprocessingCache
from core.data_pipeline import DataPipeline
from core.factory import ComponentFactory
from core.metric_evaluator import MetricEvaluator
from core.mlflow_tracker import MlflowTracker
from core.results_manager import ResultsManager
from core.task_executor import TaskExecutor
from core.visualizer import Visualizer
from models.hash_checkpoint_model import HashCheckpointModel
from models.model_checkpoint import save_model, compute_model_hash, get_checkpoint_dir

warnings.filterwarnings("ignore", message="'H' is deprecated")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(self, config_dir: Path = Path("conf"), seed: int = 42, test_mode: bool = False, use_cache: bool = True):
        self.config_dir = config_dir
        self._set_seed(seed)
        cache = PreprocessingCache() if use_cache else None
        self.factory = ComponentFactory(config_dir)
        self.pipeline = DataPipeline(self.factory, cache, test_mode)
        self.executor = TaskExecutor()
        self.tracker = MlflowTracker()

    def _set_seed(self, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f)

    def _resolve_save_dir_from_result(self, prior_result: Dict[str, Any]) -> str:
        save_dir = prior_result.get("save_dir")
        if save_dir:
            return save_dir

        checkpoint_path = prior_result.get("checkpoint_path")
        if not checkpoint_path:
            return None

        cp = Path(checkpoint_path)
        if cp.name == "models":
            return str(cp.parent)
        if "models" in cp.parts:
            idx = cp.parts.index("models")
            if idx > 0:
                return str(Path(*cp.parts[:idx]))
        return str(cp)

    def _load_run_configs(self, run: Dict[str, Any]) -> Dict[str, Any]:
        data_cfg = self.factory.get_component_by_name(run["data"], "data")
        model_cfg = self.factory.get_component_by_name(run["model"], "models")
        preproc_cfg = self.factory.get_component_by_name(run["preprocessor"], "preprocessing")
        target = run.get("target")
        if target is not None:
            model_cfg["params"]["target_seq_index"] = target

        # Normalize flat metrics list into per-metric config dict
        metric_names = run.get("metrics", [])
        metric_configs = {}
        for name in metric_names:
            base_cfg = self.factory.get_component_by_name(name, "metrics")
            metric_configs[name] = base_cfg

        return {
            "data": data_cfg,
            "model": model_cfg,
            "preprocessing": preproc_cfg,
            "metrics": metric_names,          
            "metric_configs": metric_configs,  
            "target": target,
        }


    def _apply_overrides(self, configs: Dict, run: Dict) -> Dict:
        for config_type, params in run.get("overrides", {}).items():
            if config_type == "metrics":
                for metric_name, metric_overrides in params.items():
                    if metric_name in configs["metric_configs"]:
                        configs["metric_configs"][metric_name].update(metric_overrides)
                    else:
                        configs["metric_configs"][metric_name] = metric_overrides
            else:
                if config_type not in configs:
                    continue
                for key, value in params.items():
                    if isinstance(value, dict) and key in configs[config_type]:
                        configs[config_type][key].update(value)
                    else:
                        configs[config_type][key] = value

        model_num_feature = configs.get("model", {}).get("params", {}).get("num_feature")
        preproc_cfg = configs.get("preprocessing", {})
        if model_num_feature is not None and isinstance(preproc_cfg, dict) and "transforms" in preproc_cfg:
            for transform_cfg in preproc_cfg.get("transforms", []):
                params = transform_cfg.setdefault("params", {})
                if params.get("samplewise_mode"):
                    params["num_feature"] = int(model_num_feature)
        return configs



    def run_experiments(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("src/results") / timestamp
        exp_config = self._load_yaml(self.config_dir / "experiments.yaml")
        rm = ResultsManager(results_dir)
        evaluator = MetricEvaluator(self.factory)
        all_results = []

        for run in exp_config["runs"]:
            run_id = run["id"]
            logger.info(f"[Run {run_id}] {run['name']} — {run['task']}")
            if rm.run_exists(run_id):
                logger.info(f"[Run {run_id}] already exists, skipping")
                all_results.append(rm.load_run(run_id))
                continue

            start = time.time()
            tracking_started = False
            try:
                configs = self._apply_overrides(self._load_run_configs(run), run)
                self.tracker.start_run(run=run, run_configs=configs, results_dir=results_dir)
                tracking_started = True
                self.tracker.log({"status": 1, "stage": "data_loading"}, step=1)
                splits = self.pipeline.get_data(run, configs)
                logger.info(f"[Run {run_id}] Data loaded and preprocessed. Training model...")
                self.factory.current_cache_key = self.pipeline.cache.cache_key if self.pipeline.cache else None
                current_save_dir = f"src/data/.cache/{self.factory.current_cache_key}"
                pretrained_save_dir = None
                pretrained_config_path = None
                pretrained_model_config = None
    
                if pretrained_run_id := run.get("pretrained_run"):
                    prior = rm.load_run(pretrained_run_id)
                    pretrained_save_dir = self._resolve_save_dir_from_result(prior)
                    pretrained_model_config = prior.get("model_config")
                    if pretrained_model := run.get("pretrained_model"):
                        pretrained_config_path = str(self.config_dir / "models" / f"{pretrained_model}.yaml")

                model, existing = self.factory._instantiate_model(
                    configs["model"],
                    pretrained_save_dir=pretrained_save_dir,
                    pretrained_config_path=pretrained_config_path,
                    pretrained_run_id=pretrained_run_id,
                    pretrained_model_config=pretrained_model_config,
                    save_dir_override=current_save_dir,
                )
                if pretrained_run_id:
                    model.current_epoch = 0  # ensure epoch starts at 0 for pretrained runs

                if not existing:
                    self.tracker.log({"status": 2, "stage": "training"}, step=2)
                    model = self.executor.train(
                        model,
                        run["task"],
                        splits,
                        tracker=self.tracker,
                        model_config=configs.get("model"),
                    )

                # Save non-HashCheckpointModel models (HashCheckpointModel saves internally during train)
                save_dir = current_save_dir
                if not isinstance(model, HashCheckpointModel) and not existing:

                    save_model(model, save_dir, configs["model"])

                config_hash = compute_model_hash(configs["model"])
                if isinstance(model, HashCheckpointModel):
                    checkpoint_path = model._get_checkpoint_dir() if hasattr(model, "_get_checkpoint_dir") else None
                else:
                    checkpoint_path = str(get_checkpoint_dir(save_dir, configs["model"]))

                # Important: watch model only after all checkpoint serialization is done.
                self.tracker.watch_model(model)

                predictions = self.executor.predict(run["task"], model, splits.test_data)
                prediction_scores = None
                if run["task"].lower() == "classification":
                    prediction_scores = self.executor.classification_scores(model, splits.test_data)
                self.tracker.log({"status": 3, "stage": "evaluation"}, step=3)
                results = evaluator.evaluate(
                    configs["metrics"],
                    run["task"],
                    splits,
                    predictions,
                    metric_configs=configs.get("metric_configs"),
                    prediction_scores=prediction_scores,
                )
                run_result = rm.save_run(
                    run_id, run["name"], results,
                    predictions=predictions,
                    test_data=splits.test_data,
                    test_labels=splits.test_labels,
                    status="success",
                    runtime=time.time() - start,
                    run_cfg=run,
                    save_dir=save_dir,
                    config_hash=config_hash,
                    model_config=configs["model"],
                    checkpoint_path=checkpoint_path,
                )
                self.tracker.log_metrics(results, step=4)
                self.tracker.finish_run(status="success", runtime=time.time() - start)

            except Exception as e:
                logger.error(f"[Run {run_id}] failed: {e}")
                traceback.print_exc()
                if tracking_started:
                    self.tracker.finish_run(status="failed", runtime=time.time() - start, error=str(e))
                run_result = rm.save_run(run_id, run["name"], {}, status="failed",
                                          error=str(e), runtime=time.time() - start)
            all_results.append(run_result)

        rm.save_summary(all_results)
        logger.info(f"Done. Results in {results_dir}")
        return results_dir

    def visualise(self, results_dir: Path, vis_config: Path = Path("conf/visualizations.yaml")):
        Visualizer(results_dir).run(vis_config)


if __name__ == "__main__":
    runner = BenchmarkRunner(test_mode=False, use_cache=True)
    results_dir = runner.run_experiments()
    runner.visualise(results_dir)  # uncomment to visualise after run
