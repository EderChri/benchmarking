import argparse
import logging
import os
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

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
from lightning import LightningModule
from models.model_checkpoint import save_model, get_checkpoint_dir, compute_model_hash
from models.base import _config_hash

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



    def run_experiments(self, run_ids: Optional[List[int]] = None, exp_config: Optional[Dict[str, Any]] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("src/results") / timestamp
        if exp_config is None:
            exp_config = self._load_yaml(self.config_dir / "experiments.yaml")
        rm = ResultsManager(results_dir)
        evaluator = MetricEvaluator(self.factory)
        all_results = []

        runs = exp_config["runs"]
        if run_ids is not None:
            runs = [r for r in runs if r["id"] in run_ids]

        for run in runs:
            run_id = run["id"]
            logger.info(f"[Run {run_id}] {run['name']} — {run['task']}")
            if rm.run_exists(run_id):
                logger.info(f"[Run {run_id}] already exists, skipping")
                all_results.append(rm.load_run(run_id))
                continue

            self._set_seed(run.get("seed", 42))
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
                if not existing and not run.get("skip_training", False):
                    self.tracker.log({"status": 2, "stage": "training"}, step=2)
                    model = self.executor.train(
                        model,
                        run["task"],
                        splits,
                        tracker=self.tracker,
                        model_config=configs.get("model"),
                    )
                elif run.get("skip_training", False) and hasattr(model, "set_context"):
                    model.set_context(splits.train_data.to_pd())

                # LightningModule (custom) models save internally via ModelCheckpoint.
                # Only Merlion stat models need an external save call.
                save_dir = current_save_dir
                if not isinstance(model, LightningModule) and not existing:
                    save_model(model, save_dir, configs["model"])

                if isinstance(model, LightningModule):
                    checkpoint_path = str(model.checkpoint_path) if model.checkpoint_path else None
                    config_hash = _config_hash(model.config)
                else:
                    checkpoint_path = str(get_checkpoint_dir(save_dir, configs["model"]))
                    config_hash = compute_model_hash(configs["model"])

                # Important: watch model only after all checkpoint serialization is done.
                self.tracker.watch_model(model)

                predictions = self.executor.predict(run["task"], model, splits, model_config=configs.get("model"))

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
        visualizer = Visualizer(results_dir)
        outputs = visualizer.run(vis_config)

        if not self.tracker.enabled:
            return

        mlflow_run_cache = {}
        for output in outputs:
            plot_path = output["path"]
            run_sources = output.get("run_sources", [])

            for source in run_sources:
                benchmark_run_id = source["run_id"]
                source_results_dir = source["results_dir"]
                cache_key = (benchmark_run_id, source_results_dir)

                if cache_key not in mlflow_run_cache:
                    mlflow_run_cache[cache_key] = self.tracker.find_run_id(
                        results_dir=source_results_dir,
                        benchmark_run_id=benchmark_run_id,
                    )

                mlflow_run_id = mlflow_run_cache.get(cache_key)
                if not mlflow_run_id:
                    logger.warning(
                        f"No MLflow run found for benchmark run {benchmark_run_id} in results_dir '{source_results_dir}'. "
                        f"Skipping plot artifact '{plot_path}'."
                    )
                    continue

                self.tracker.log_artifact_to_run(mlflow_run_id, vis_config, artifact_path="config")
                self.tracker.log_artifact_to_run(mlflow_run_id, plot_path, artifact_path="plots")


def _build_run_chains(runs: list) -> list:
    """Group runs into independent dependency chains via BFS from each root.

    Fan-out dependencies (one pretrain → many finetunes) are all collected into
    the same chain and run sequentially on one GPU.
    """
    run_ids_in_batch = {r["id"] for r in runs}
    children: Dict[int, list] = {}
    for r in runs:
        parent = r.get("pretrained_run")
        if parent in run_ids_in_batch:
            children.setdefault(parent, []).append(r)

    dependents = {r["id"] for r in runs if r.get("pretrained_run") in run_ids_in_batch}
    chains, visited = [], set()

    for run in runs:
        rid = run["id"]
        if rid in visited or rid in dependents:
            continue
        chain, queue = [], [run]
        while queue:
            current = queue.pop(0)
            cid = current["id"]
            if cid in visited:
                continue
            chain.append(current)
            visited.add(cid)
            queue.extend(children.get(cid, []))
        chains.append(chain)

    for run in runs:
        if run["id"] not in visited:
            chains.append([run])

    return chains


def _chain_num_devices(chain: list, config_dir: Path) -> int:
    """Return the num_devices requested by the first run in a chain.

    Reads the model YAML and applies any overrides from the run config so that
    ``overrides.model.params.num_devices`` is respected.  Falls back to 1.
    """
    run = chain[0]
    try:
        model_name = run.get("model")
        if not model_name:
            return 1
        model_yaml = yaml.safe_load((config_dir / "models" / f"{model_name}.yaml").read_text())
        base = model_yaml.get("params", {}).get("num_devices", 1)
        override = (
            run.get("overrides", {}).get("model", {}).get("params", {}).get("num_devices", base)
        )
        return int(override)
    except Exception:
        return 1


def run_parallel(gpu_ids: List[int], run_ids: Optional[List[int]] = None,
                 config_path: Path = Path("conf/experiments.yaml")):
    """Dispatch dependency chains across the available GPU pool.

    Single-GPU chains (num_devices=1) are assigned one GPU from the pool
    round-robin, matching the previous behaviour.

    Multi-GPU chains (num_devices=N) consume N consecutive GPUs from the
    pool for that subprocess, which then runs Trainer with DDP across them.
    CUDA_VISIBLE_DEVICES is set to all N physical GPU IDs so that PL/NCCL
    can address them as devices 0…N-1 within the subprocess.

    If a chain requests more GPUs than remain in the pool it falls back to
    using all remaining GPUs (minimum 1).
    """
    config_dir = config_path.parent
    exp_config = yaml.safe_load(config_path.read_text())
    runs = exp_config["runs"]
    if run_ids is not None:
        runs = [r for r in runs if r.get("id") in run_ids]

    chains = _build_run_chains(runs)
    log_dir = Path("src/results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build a mutable pool of available GPU IDs (in order).
    gpu_pool = list(gpu_ids)

    # Each entry: (gpu_id_list, chain_run_ids)
    assignments: List[tuple] = []

    for chain in chains:
        n_dev = _chain_num_devices(chain, config_dir)
        n_dev = max(1, min(n_dev, len(gpu_pool)))  # clamp to what's available

        allocated = gpu_pool[:n_dev]
        # Rotate the pool so the next chain starts after the allocated block.
        gpu_pool = gpu_pool[n_dev:] + gpu_pool[:n_dev]

        assignments.append((allocated, [r["id"] for r in chain]))

    procs = []
    for allocated_gpus, ids in assignments:
        cuda_str = ",".join(str(g) for g in allocated_gpus)
        label = f"gpu{'_'.join(str(g) for g in allocated_gpus)}_runs_{'_'.join(map(str, ids))}"
        log_file = log_dir / f"{label}.log"
        logger.info(f"  GPUs {allocated_gpus}: runs {ids} → {log_file}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_str

        # Pass --gpus as the single (first) allocated GPU so that the
        # subprocess single-GPU path sets CUDA_VISIBLE_DEVICES correctly;
        # actual DDP device count is controlled by num_devices in the model config.
        f = open(log_file, "w")
        proc = subprocess.Popen(
            [
                sys.executable, __file__,
                "--config", str(config_path),
                "--runs", ",".join(map(str, ids)),
                "--gpus", str(allocated_gpus[0]),
            ],
            env=env, stdout=f, stderr=subprocess.STDOUT,
        )
        procs.append((proc, f, allocated_gpus))

    for proc, f, gpus in procs:
        proc.wait()
        f.close()
        if proc.returncode != 0:
            logger.warning(f"Subprocess for GPUs {gpus} exited with code {proc.returncode}")

    logger.info("All waves complete. Running visualisation...")
    vis_results_dir = Path("src/results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    runner = BenchmarkRunner(test_mode=False, use_cache=True)
    runner.visualise(vis_results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/experiments.yaml",
                        help="Path to experiments YAML, e.g. --config conf/experiments_test.yaml")
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated run IDs to execute, e.g. --runs 17,18")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs to use, e.g. --gpus 7,8")
    args = parser.parse_args()

    config_path = Path(args.config)
    run_ids = [int(x) for x in args.runs.split(",")] if args.runs else None
    gpu_ids = [int(x) for x in args.gpus.split(",")] if args.gpus else None

    if gpu_ids and len(gpu_ids) > 1:
        run_parallel(gpu_ids, run_ids, config_path=config_path)
    else:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        runner = BenchmarkRunner(test_mode=False, use_cache=True)
        # Load from the specified config file
        exp_config = yaml.safe_load(config_path.read_text())
        runner.factory.config_dir = config_path.parent
        results_dir = runner.run_experiments(run_ids=run_ids, exp_config=exp_config)
        if run_ids is None:
            runner.visualise(results_dir)
