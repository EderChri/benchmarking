import yaml
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any
from itertools import product
import os
import sys
from merlion.utils import TimeSeries
from merlion.models.base import ModelBase
from merlion.transform.base import TransformBase
from merlion.evaluate.anomaly import TSADMetric
from merlion.evaluate.forecast import ForecastMetric
import warnings
import torch
import random
import numpy as np 
from loaders.dataset_loader import DatasetLoader

# Suppress specific deprecation warnings from merlion datasets
warnings.filterwarnings('ignore', message="'H' is deprecated")


class BenchmarkRunner:
    def __init__(self, config_dir: Path = Path("conf"), seed: int = 42):
        self.config_dir = config_dir
        self.seed = seed
        self.set_seed(seed)

    def set_seed(self, seed: int = 42):
        """Set seeds for reproducibility across all libraries."""
        # Python
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch (if using deep learning models)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_component_by_name(self, name: str, component_type: str) -> Dict[str, Any]:
        """Load specific component config by name"""
        config_path = self.config_dir / component_type / f"{name}.yaml"
        return self.load_yaml(config_path)
    
    def instantiate(self, config: Dict[str, Any], component_type: str):
        source = config.get('source', 'default')
        module_name = config.get('module')
        class_name = config.get('class')
        class_path = config.get('class_path')  # Support direct class path
        params = config.get('params', {})

        if component_type == "data":
            series_id = config.get('series_id')
            granularity = config.get('granularity')
            if series_id:
                params['series_id'] = series_id
            if granularity:
                params['subset'] = granularity  # M4 uses 'subset' parameter
        
        # Handle class_path for metrics (e.g., merlion.evaluate.forecast.smape)
        if class_path:
            parts = class_path.rsplit('.', 1)
            module_path = parts[0]
            class_name = parts[1]
            module = importlib.import_module(module_path)
        else:
            # Define module path mapping
            module_paths = {
                ('data', 'merlion'): f"ts_datasets.{module_name}",
                ('data', 'default'): module_name,
                ('preprocessing', 'merlion'): f"merlion.transform.{module_name}",
                ('preprocessing', 'default'): f"preprocessing.{module_name}",
                ('models', 'merlion'): f"merlion.models.{module_name}",
                ('models', 'default'): module_name,
                ('metrics', 'merlion'): f"merlion.evaluate.{module_name}",
                ('metrics', 'default'): f"metrics.{module_name}",
            }
            
            module_path = module_paths.get((component_type, source))
            if not module_path:
                raise ValueError(f"Unknown component_type '{component_type}' or source '{source}'")
            
            module = importlib.import_module(module_path)
        
        # Special handling for models
        if component_type == "models":
            config_class_name = config.get('config_class')
            config_cls = getattr(module, config_class_name)
            model_config = config_cls(**params)
            model_cls = getattr(module, class_name)
            return model_cls(model_config)
        
        # Standard instantiation
        cls = getattr(module, class_name)
        return cls(**params) if params else cls




    def load_dataset(self, data_config: Dict[str, Any]):
        base_dataset = self.instantiate(data_config, "data")
        test_split_ratio = data_config.get('test_split_ratio', 0.2)
        return DatasetLoader(base_dataset, test_split_ratio)

    
    def load_preprocessor(self, preproc_config: Dict[str, Any]):
        """Handle single or multiple transforms"""
        if 'transforms' in preproc_config:  # Pipeline
            from merlion.transform.sequence import TransformSequence
            transforms = [self.instantiate(t, 'preprocessing') 
                        for t in preproc_config['transforms']]
            return TransformSequence(transforms)
        else:  # Single transform
            return self.instantiate(preproc_config, 'preprocessing')

    
    def run_experiments(self):
        # Load experiment configurations
        exp_config = self.load_yaml(self.config_dir / "experiments.yaml")
        for run in exp_config['runs']:
            print(f"\n{'='*80}")
            print(f"Experiment: {run['name']}")
            print(f"Task: {run['task']}")
            print(f"{'='*80}\n")
            
            # Load component configs
            data_cfg = self.get_component_by_name(run['data'], 'data')
            model_cfg = self.get_component_by_name(run['model'], 'models')
            preproc_cfg = self.get_component_by_name(run['preprocessor'], 'preprocessing')
            metric_names = run['metrics']
            
            # Instantiate components
            data_loader = self.load_dataset(data_cfg)
            train_data, test_data = data_loader.load()
            
            transformation = self.instantiate(preproc_cfg, 'preprocessing')
            transformation.train(train_data)
            train_data = transformation(train_data)
            test_data = transformation(test_data)
            
            model = self.instantiate(model_cfg, 'models')
            model.train(train_data)
            
            predictions = self.execute_task(run['task'], model, test_data)
            
        # Evaluate metrics
        results = {}
        for metric_name in metric_names:
            metric_cfg = self.get_component_by_name(metric_name, 'metrics')
            metric_fn = self.instantiate(metric_cfg, 'metrics')
            
            # Handle both function-style (ForecastMetric enum) and class-based metrics
            if callable(metric_fn):
                if hasattr(metric_fn, 'value'):
                    # Merlion ForecastMetric enum style
                    score = metric_fn.value(ground_truth=test_data, predict=predictions)
                else:
                    # Custom metric class with evaluate method
                    score = metric_fn.evaluate(test_data, predictions)
            else:
                raise ValueError(f"Metric {metric_name} is not callable")
            
            results[metric_name] = score
            print(f"{metric_name.upper()}: {score:.4f}")
        
        self.save_results(run['name'], results)

    def execute_task(self, task_type: str, model, test_data):
        """Dispatch task execution based on task type."""
        task_methods = {
            'forecasting': self._execute_forecasting,
            'anomaly_detection': self._execute_anomaly_detection,
            'forecast': self._execute_forecasting,
            'anomaly': self._execute_anomaly_detection,
        }
        
        task_method = task_methods.get(task_type.lower())
        if not task_method:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return task_method(model, test_data)

    def _execute_forecasting(self, model, test_data):
        """Execute forecasting task."""
        return model.forecast(time_stamps=test_data.time_stamps)

    def _execute_anomaly_detection(self, model, test_data):
        """Execute anomaly detection task."""
        return model.get_anomaly_score(test_data)
    
    def save_results(self, exp_name: str, results: Dict[str, float]):
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "experiments.yaml", 'a') as f:
            yaml.dump([{"experiment": exp_name, "results": results}], f)

    
    def is_compatible(self, task_cfg, model_cfg, metric_cfg) -> bool:
        """Check if task, model, and metric are compatible"""
        task_type = task_cfg.get('type')
        model_type = model_cfg.get('task')
        metric_type = metric_cfg.get('task')
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
    
    def save_results(self, task_cfg, data_cfg, preproc_cfg, model_cfg, metric_cfg, score):
        """Save experiment results"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        result_file = results_dir / "experiments.yaml"
        result = {
            "task": task_cfg['name'],
            "dataset": data_cfg['name'],
            "preprocessing": preproc_cfg['name'],
            "model": model_cfg['name'],
            "metric": metric_cfg['name'],
            "score": float(score)
        }
        
        with open(result_file, 'a') as f:
            yaml.dump([result], f)


if __name__ == "__main__":
    print("Working dir:", os.getcwd())

    runner = BenchmarkRunner()
    runner.run_experiments()
