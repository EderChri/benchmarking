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
from loaders import LOADER_REGISTRY
import loaders # Ensure loaders are registered

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
        handlers = {
            'metrics': self._instantiate_metric,
            'models': self._instantiate_model,
            'preprocessing': self._instantiate_preprocessing,
        }
        
        handler = handlers.get(component_type)
        if handler:
            return handler(config)
        
        raise ValueError(f"Unknown component_type: {component_type}")

    def _instantiate_metric(self, config: Dict[str, Any]):
        if config.get('class_path'):
            return self._import_from_path(config['class_path'])
        
        source = config.get('source', 'default')
        if source == 'merlion':
            module = importlib.import_module(f"merlion.evaluate.{config['module']}")
            metric_enum = getattr(module, config['class'])
            return getattr(metric_enum, config['metric_name'])
        else:
            module = importlib.import_module(f"metrics.{config['module']}")
            cls = getattr(module, config['class'])
            return cls(**config.get('params', {}))

    def _instantiate_model(self, config: Dict[str, Any]):
        source = config.get('source', 'default')
        module_path = f"merlion.models.{config['module']}" if source == 'merlion' else config['module']
        module = importlib.import_module(module_path)
        
        config_cls = getattr(module, config['config_class'])
        model_config = config_cls(**config.get('params', {}))
        
        model_cls = getattr(module, config['class'])
        return model_cls(model_config)

    def _instantiate_preprocessing(self, config: Dict[str, Any]):
        source = config.get('source', 'default')
        module_path = f"merlion.transform.{config['module']}" if source == 'merlion' else f"preprocessing.{config['module']}"
        module = importlib.import_module(module_path)
        
        cls = getattr(module, config['class'])
        return cls(**config.get('params', {}))

    def _import_from_path(self, class_path: str):
        parts = class_path.rsplit('.', 1)
        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])




    def load_dataset(self, data_config: Dict[str, Any]):
        
        source = data_config.get('source', 'custom')
        loader_cls = LOADER_REGISTRY[source]
        return loader_cls(data_config, data_config.get('test_split_ratio', 0.2))

    def load_model(self, model_cfg: Dict[str, Any]):
        return self.instantiate(model_cfg, 'models') 


    
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
            
            transformation = self.load_preprocessor(preproc_cfg)
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
                
                # All Merlion metrics (ForecastMetric/TSADMetric) have .value attribute
                if hasattr(metric_fn, 'value'):
                    score = metric_fn.value(ground_truth=test_data, predict=predictions)
                else:
                    # Custom metric inheriting from MetricBase
                    score = metric_fn(ground_truth=test_data, predict=predictions)
                
                results[metric_name] = float(score) 
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
        forecast_result = model.forecast(time_stamps=test_data.time_stamps)
        
        if isinstance(forecast_result, tuple):
            return forecast_result[0] 
        return forecast_result

    def _execute_anomaly_detection(self, model, test_data):
        """Execute anomaly detection task."""
        return model.get_anomaly_score(test_data)
        
    def save_results(self, exp_name: str, results: Dict[str, float]):
        from datetime import datetime
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"{timestamp}_experiments.yaml"
        
        result = {
            "experiment": exp_name,
            "timestamp": timestamp,
            "results": results
        }
        
        with open(result_file, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)


    
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

if __name__ == "__main__":
    print("Working dir:", os.getcwd())

    runner = BenchmarkRunner()
    runner.run_experiments()
