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
from core.results_manager import ResultsManager
from core.task_executor import TaskExecutor
from core.visualizer import Visualizer
from models.hash_checkpoint_model import HashCheckpointModel
from models.model_checkpoint import save_model

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

    def _set_seed(self, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f)

    def _load_run_configs(self, run: Dict[str, Any]) -> Dict[str, Any]:
        data_cfg = self.factory.get_component_by_name(run["data"], "data")
        model_cfg = self.factory.get_component_by_name(run["model"], "models")
        preproc_cfg = self.factory.get_component_by_name(run["preprocessor"], "preprocessing")
        target = run.get("target")
        if target is not None:
            model_cfg["params"]["target_seq_index"] = target
        return {"data": data_cfg, "model": model_cfg, "preprocessing": preproc_cfg,
                "metrics": run["metrics"], "target": target}

    def _apply_overrides(self, configs: Dict, run: Dict) -> Dict:
        for config_type, params in run.get("overrides", {}).items():
            if config_type not in configs:
                continue
            for key, value in params.items():
                if isinstance(value, dict) and key in configs[config_type]:
                    configs[config_type][key].update(value)
                else:
                    configs[config_type][key] = value
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
            try:
                configs = self._apply_overrides(self._load_run_configs(run), run)
                splits = self.pipeline.get_data(run, configs)
                logger.info(f"[Run {run_id}] Data loaded and preprocessed. Training model...")
                self.factory.current_cache_key = self.pipeline.cache.cache_key if self.pipeline.cache else None
                pretrained_save_dir = None
                pretrained_config_path = None
    
                if pretrained_run_id := run.get("pretrained_run"):
                    prior = rm.load_run(pretrained_run_id)
                    pretrained_save_dir = prior.get("save_dir")
                    if pretrained_model := run.get("pretrained_model"):
                        pretrained_config_path = str(self.config_dir / "models" / f"{pretrained_model}.yaml")

                model, existing = self.factory._instantiate_model(
                    configs["model"],
                    pretrained_save_dir=pretrained_save_dir,
                    pretrained_config_path=pretrained_config_path,
                    pretrained_run_id=pretrained_run_id
                )
                if pretrained_run_id:
                    model.current_epoch = 0  # ensure epoch starts at 0 for pretrained runs

                if not existing:
                    model = self.executor.train(model, run["task"], splits)

                # Save non-HashCheckpointModel models (HashCheckpointModel saves internally during train)
                save_dir = f"src/data/.cache/{self.factory.current_cache_key}"
                if not isinstance(model, HashCheckpointModel) and not existing:

                    save_model(model, save_dir, configs["model"])

                predictions = self.executor.predict(run["task"], model, splits.test_data)
                results = evaluator.evaluate(configs["metrics"], run["task"], splits, predictions)
                run_result = rm.save_run(
                    run_id, run["name"], results,
                    predictions=predictions,
                    test_data=splits.test_data,
                    status="success",
                    runtime=time.time() - start,
                    run_cfg=run,
                    save_dir=save_dir,
                )

            except Exception as e:
                logger.error(f"[Run {run_id}] failed: {e}")
                traceback.print_exc()
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
    #runner.visualise(results_dir)  # uncomment to visualise after run
