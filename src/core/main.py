import trace
import traceback
from core.cache import PreprocessingCache
import test
import yaml
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from itertools import product
import os
import sys
from merlion.utils.resample import AggregationPolicy
from merlion.utils import TimeSeries
from merlion.models.base import ModelBase
from merlion.transform.base import TransformBase
from merlion.evaluate.anomaly import TSADMetric
from merlion.evaluate.forecast import ForecastMetric
import warnings
import torch
import random
import numpy as np
from zmq import has
from loaders import LOADER_REGISTRY
from datetime import datetime
import time

import loaders  # Ensure loaders are registered
import preprocessing

# Suppress specific deprecation warnings from merlion datasets
warnings.filterwarnings("ignore", message="'H' is deprecated")


class BenchmarkRunner:
    def __init__(
        self,
        config_dir: Path = Path("conf"),
        seed: int = 42,
        test_mode: bool = False,
        use_cache: bool = True,
    ):
        self.config_dir = config_dir
        self.seed = seed
        self.set_seed(seed)
        self.test_mode = test_mode
        self.cache = PreprocessingCache() if use_cache else None
        self.use_cache = use_cache

    def set_seed(self, seed: int = 42):
        """Set seeds for reproducibility across all libraries."""
        # Python
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch (if using deep learning models)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def get_component_by_name(self, name: str, component_type: str) -> Dict[str, Any]:
        """Load specific component config by name"""
        config_path = self.config_dir / component_type / f"{name}.yaml"
        return self.load_yaml(config_path)

    def instantiate(self, config: Dict[str, Any], component_type: str):
        handlers = {
            "metrics": self._instantiate_metric,
            "models": self._instantiate_model,
            "preprocessing": self._instantiate_preprocessing,
        }

        handler = handlers.get(component_type)
        if handler:
            return handler(config)

        raise ValueError(f"Unknown component_type: {component_type}")

    def _instantiate_metric(self, config: Dict[str, Any]):
        if config.get("class_path"):
            return self._import_from_path(config["class_path"])

        source = config.get("source", "default")
        if source == "merlion":
            module = importlib.import_module(
                f"merlion.evaluate.{config['module']}")
            metric_enum = getattr(module, config["class"])
            return getattr(metric_enum, config["metric_name"])
        else:
            module = importlib.import_module(f"metrics.{config['module']}")
            cls = getattr(module, config["class"])
            return cls(**config.get("params", {}))

    def _instantiate_model(self, config: Dict[str, Any]):
        source = config.get("source", "default")

        module_path = (
            f"merlion.models.{config['module']}"
            if source == "merlion"
            else config["module"]
        )
        module = importlib.import_module(module_path)

        config_cls = getattr(module, config["config_class"])
        model_config = config_cls(**config.get("params", {}))

        model_cls = getattr(module, config["class"])
        return model_cls(model_config, save_dir=f"src/data/.cache/{self.current_cache_key}")

    def _instantiate_preprocessing(self, config: Dict[str, Any]):
        source = config.get("source", "default")
        module_path = (
            f"merlion.transform.{config['module']}"
            if source == "merlion"
            else f"preprocessing.{config['module']}"
        )
        module = importlib.import_module(module_path)

        cls = getattr(module, config["class"])
        return cls(**config.get("params", {}))

    def _import_from_path(self, class_path: str):
        parts = class_path.rsplit(".", 1)
        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])

    def load_dataset(self, data_config: Dict[str, Any]):

        source = data_config.get("source", "custom")
        loader_cls = LOADER_REGISTRY[source]
        return loader_cls(
            data_config,
            data_config.get("test_split_ratio", 0.2),
            test_mode=self.test_mode,
        )

    def load_model(self, model_cfg: Dict[str, Any]):
        return self.instantiate(model_cfg, "models")

    def load_preprocessor(self, preproc_config: Dict[str, Any]):
        """Handle single or multiple transforms"""
        if "transforms" in preproc_config:  # Pipeline
            from merlion.transform.sequence import TransformSequence

            transforms = [
                self.instantiate(t, "preprocessing")
                for t in preproc_config["transforms"]
            ]
            return TransformSequence(transforms)
        else:  # Single transform
            return self.instantiate(preproc_config, "preprocessing")

    def run_experiments(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path("src/results") / timestamp
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        exp_config = self.load_yaml(self.config_dir / "experiments.yaml")
        all_results = []

        for run in exp_config["runs"]:
            print(f"\n{'='*80}")
            print(f"Experiment: {run['name']}")
            print(f"Task: {run['task']}")
            print(f"{'='*80}\n")

            try:
                configs = self._load_run_configs(run)

                data_splits = self._get_preprocessed_data(run, configs)
                self.current_cache_key = self.cache.cache_key if self.cache else None
                (
                    train_data,
                    test_data,
                    val_data,  # TODO: Unused right now
                    train_labels,
                    test_labels,
                    val_labels,  # TODO: Unused right now
                    has_labels,
                ) = self._parse_data_splits(data_splits)

                start_time = time.time()

                print("Training model...")
                model = self._train_model(
                    configs["model"], run["task"], train_data, train_labels
                )
                print("Evaluating model...")
                predictions = self.execute_task(run["task"], model, test_data)

                results = self._evaluate_metrics(
                    configs["metrics"],
                    run["task"],
                    test_data,
                    test_labels,
                    predictions,
                    has_labels,
                )

                runtime = time.time() - start_time
                print(f"Runtime: {runtime:.2f} seconds")

                run_result = self.save_run_results(
                    run["name"], results, status="success", runtime=runtime, run_cfg=run
                )
                all_results.append(run_result)

            except Exception as e:
                runtime = time.time() - start_time
                print(f"ERROR: {run['name']} failed with {str(e)}")
                traceback.print_exc()
                run_result = self.save_run_results(
                    run["name"], {}, status="failed", error=str(e), runtime=runtime
                )
                all_results.append(run_result)
        print("\nAll experiments completed.")
        self.save_summary(all_results)

    def _load_run_configs(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Load all configuration components for a run"""
        data_cfg = self.get_component_by_name(run["data"], "data")
        model_cfg = self.get_component_by_name(run["model"], "models")
        preproc_cfg = self.get_component_by_name(
            run["preprocessor"], "preprocessing")

        # Add target to model config if specified
        target_seq_index = run.get("target", None)
        if target_seq_index is not None:
            model_cfg["params"]["target_seq_index"] = target_seq_index

        metric_names = run["metrics"]

        return {
            "data": data_cfg,
            "model": model_cfg,
            "preprocessing": preproc_cfg,
            "metrics": metric_names,
            "target": target_seq_index,
        }

    def _get_preprocessed_data(
        self, run: Dict[str, Any], configs: Dict[str, Any]
    ) -> Tuple:
        """Load preprocessed data from cache or compute it"""
        if self._should_use_cache(run, configs):
            return self._load_from_cache(run, configs)
        else:
            return self._preprocess_data(run, configs)

    def _should_use_cache(self, run: Dict[str, Any], configs: Dict[str, Any]) -> bool:
        """Check if cache should be used and exists"""
        if not self.use_cache:
            return False

        return self.cache.exists(
            configs["data"], configs["preprocessing"], run["task"], configs["target"]
        )

    def _load_from_cache(self, run: Dict[str, Any], configs: Dict[str, Any]) -> Tuple:
        """Load preprocessed data from cache"""
        print("Loading preprocessed data from cache...")
        return self.cache.load(
            configs["data"], configs["preprocessing"], run["task"], configs["target"]
        )

    def _preprocess_data(self, run: Dict[str, Any], configs: Dict[str, Any]) -> Tuple:
        """Load raw data, apply preprocessing, and cache result"""
        print("Processing data (not in cache)...")

        data_loader = self.load_dataset(configs["data"])
        raw_data_splits = data_loader.load()

        (
            train_data,
            test_data,
            val_data,
            train_labels,
            test_labels,
            val_labels,
            has_labels,
        ) = self._parse_data_splits(raw_data_splits)
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        train_data, test_data, val_data = self._apply_transformations(
            configs["preprocessing"], train_data, test_data, val_data
        )

        if self.use_cache:
            self._save_to_cache(
                run,
                configs,
                train_data,
                test_data,
                val_data,
                train_labels,
                test_labels,
                val_labels,
            )

        return self._reconstruct_data_splits(
            train_data,
            test_data,
            val_data,
            train_labels,
            test_labels,
            val_labels,
            has_labels,
        )

    def _parse_data_splits(self, data_splits: Tuple):
        """
        Parse data splits into individual components.

        Returns:
            Tuple of (train_data, test_data, val_data, train_labels,
                    test_labels, val_labels, has_labels)
        """
        if isinstance(data_splits[0], tuple):  # Supervised
            (train_data, train_labels), (test_data,
                                         test_labels) = data_splits[:2]
            has_labels = True

            if len(data_splits) == 3:
                val_data, val_labels = data_splits[2]
            else:
                val_data, val_labels = None, None
        else:  # Unsupervised
            if len(data_splits) == 3:
                train_data, val_data, test_data = data_splits
            else:
                train_data, test_data = data_splits
                val_data = None

            train_labels = test_labels = val_labels = None
            has_labels = False

        return (
            train_data,
            test_data,
            val_data,
            train_labels,
            test_labels,
            val_labels,
            has_labels,
        )

    def _apply_transformations(
        self,
        preproc_cfg: Dict[str, Any],
        train_data: TimeSeries,
        test_data: TimeSeries,
        val_data: Optional[TimeSeries],
    ) -> Tuple[TimeSeries, TimeSeries, Optional[TimeSeries]]:
        """Apply preprocessing transformations to all data splits"""
        transformation = self.load_preprocessor(preproc_cfg)
        transformation.train(train_data)

        train_data = transformation(train_data)
        test_data = transformation(test_data)

        if val_data is not None:
            val_data = transformation(val_data)

        return train_data, test_data, val_data

    def _save_to_cache(
        self,
        run: Dict[str, Any],
        configs: Dict[str, Any],
        train_data: TimeSeries,
        test_data: TimeSeries,
        val_data: Optional[TimeSeries],
        train_labels: Optional[TimeSeries],
        test_labels: Optional[TimeSeries],
        val_labels: Optional[TimeSeries],
    ):
        """Save preprocessed data to cache"""
        self.cache.save(
            data_cfg=configs["data"],
            preproc_cfg=configs["preprocessing"],
            task=run["task"],
            train_data=train_data,
            test_data=test_data,
            val_data=val_data,
            train_labels=train_labels,
            test_labels=test_labels,
            val_labels=val_labels,
            target=configs["target"],
        )

    def _reconstruct_data_splits(
        self,
        train_data: TimeSeries,
        test_data: TimeSeries,
        val_data: Optional[TimeSeries],
        train_labels: Optional[TimeSeries],
        test_labels: Optional[TimeSeries],
        val_labels: Optional[TimeSeries],
        has_labels: bool,
    ) -> Tuple:
        """Reconstruct data_splits tuple in expected format"""
        if has_labels:
            if val_data is not None:
                return (
                    (train_data, train_labels),
                    (val_data, val_labels),
                    (test_data, test_labels),
                )
            else:
                return (train_data, train_labels), (test_data, test_labels)
        else:
            if val_data is not None:
                return train_data, val_data, test_data
            else:
                return train_data, test_data

    def _train_model(
        self,
        model_cfg: Dict[str, Any],
        task: str,
        train_data: TimeSeries,
        train_labels: Optional[TimeSeries],
    ) -> ModelBase:
        """Instantiate and train model"""
        model = self.instantiate(model_cfg, "models")

        if task in ["anomaly_detection", "anomaly", "classification"] and train_labels is not None:
            model.train(train_data, train_labels=train_labels)
        else:
            model.train(train_data)

        return model

    def _evaluate_metrics(
        self,
        metric_names: list,
        task: str,
        test_data: TimeSeries,
        test_labels: Optional[TimeSeries],
        predictions,
        has_labels: bool,
    ) -> Dict[str, float]:
        """Evaluate all metrics and return results"""
        results = {}

        for metric_name in metric_names:
            try:
                score = self._evaluate_single_metric(
                    metric_name, task, test_data, test_labels, predictions, has_labels
                )
                results[metric_name] = float(score)
                print(f"{metric_name.upper()}: {score:.4f}")
            except Exception as e:
                print(f"Failed to evaluate {metric_name}: {str(e)}")
                results[metric_name] = None

        return results

    def _evaluate_single_metric(
        self,
        metric_name: str,
        task: str,
        test_data: TimeSeries,
        test_labels: Optional[TimeSeries],
        predictions,
        has_labels: bool,
    ) -> float:
        """Evaluate a single metric"""
        metric_cfg = self.get_component_by_name(metric_name, "metrics")
        metric_fn = self.instantiate(metric_cfg, "metrics")

        # Determine ground truth based on task
        if task in ["anomaly_detection", "anomaly", "classification"]:
            if not has_labels:
                raise ValueError(
                    f"Metric {metric_name} requires labels for {task}"
                )
            ground_truth = test_labels
        else:  # forecasting, imputation, etc.
            ground_truth = test_data

        # Compute metric
        if hasattr(metric_fn, "value"):
            return metric_fn.value(ground_truth=ground_truth, predict=predictions)
        else:
            return metric_fn(ground_truth=ground_truth, predict=predictions)

    def execute_task(self, task_type: str, model, test_data):
        """Dispatch task execution based on task type."""
        task_methods = {
            "forecasting": self._execute_forecasting,
            "anomaly_detection": self._execute_anomaly_detection,
            "forecast": self._execute_forecasting,
            "anomaly": self._execute_anomaly_detection,
            "classification": self._execute_classification,
        }

        task_method = task_methods.get(task_type.lower())
        if not task_method:
            raise ValueError(f"Unknown task type: {task_type}")

        return task_method(model, test_data)

    def _execute_forecasting(self, model, test_data):
        """Execute forecasting task."""
        forecast_result = model.forecast(time_stamps=test_data.time_stamps)

        if isinstance(forecast_result, tuple):
            return forecast_result[0]
        return forecast_result

    def _execute_anomaly_detection(self, model, test_data):
        """Execute anomaly detection task."""
        return model.get_anomaly_score(test_data)
    
    def _execute_classification(self, model, test_data):
        """Execute classification task."""
        return model.predict(test_data)

    def save_run_results(
        self,
        exp_name: str,
        results: Dict[str, float],
        status: str = "success",
        error: str = None,
        runtime: float = None,
        run_cfg: Dict[str, Any] = None,
    ):
        run_file = self.experiment_dir / f"{exp_name}.yaml"
        run_result = {
            "experiment": exp_name,
            "status": status,
            "run_cfg": run_cfg,
            "results": results,
            "runtime": runtime,
        }
        if error:
            run_result["error"] = error

        with open(run_file, "w") as f:
            yaml.dump(run_result, f, default_flow_style=False)

        return run_result

    def save_summary(self, all_results: list):
        summary_file = self.experiment_dir / "summary.yaml"
        summary = {
            "total_runs": len(all_results),
            "successful": sum(1 for r in all_results if r["status"] == "success"),
            "failed": sum(1 for r in all_results if r["status"] == "failed"),
            "total_runtime": round(sum(r["runtime"] for r in all_results), 2),
            "runs": all_results,
        }

        with open(summary_file, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

    def is_compatible(self, task_cfg, model_cfg, metric_cfg) -> bool:
        """Check if task, model, and metric are compatible"""
        task_type = task_cfg.get("type")
        model_type = model_cfg.get("task")
        metric_type = metric_cfg.get("task")
        return task_type == model_type == metric_type

    def split_data(self, ts: TimeSeries, ratio: float):
        split_idx = int(len(ts) * ratio)
        return ts[:split_idx], ts[split_idx:]

    def run_task(self, model: ModelBase, test_data: TimeSeries, task_type: str):
        """Execute task-specific inference"""
        if task_type == "anomaly_detection":
            return model.get_anomaly_score(test_data)
        elif task_type == "forecasting":
            return model.forecast(len(test_data))
        elif task_type == "imputation":
            return model.forecast(len(test_data))
        else:
            raise ValueError(f"Unknown task: {task_type}")


if __name__ == "__main__":
    print("Working dir:", os.getcwd())

    runner = BenchmarkRunner(test_mode=True)
    runner.run_experiments()
