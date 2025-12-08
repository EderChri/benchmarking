#!/usr/bin/env python3
import yaml
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, List
from itertools import product

from merlion.utils import TimeSeries
from merlion.models.base import ModelBase
from merlion.transform.base import TransformBase
from merlion.evaluate.anomaly import TSADMetric
from merlion.evaluate.forecast import ForecastMetric


class BenchmarkRunner:
    def __init__(self, config_dir: Path = Path("conf"), seed: int = 42):
        self.config_dir = config_dir
        self.src_dir = Path("src")
        self.seed = seed
        
    def load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_component_by_name(self, name: str, component_type: str) -> Dict[str, Any]:
        """Load specific component config by name"""
        config_path = self.config_dir / component_type / f"{name}.yaml"
        return self.load_yaml(config_path)
    
    def instantiate(self, config: Dict[str, Any], component_type: str):
        """Dynamically instantiate class from config"""
        module_name = config.get('module')
        class_name = config.get('class')
        params = config.get('params', {})
        
        module = importlib.import_module(f"src.{component_type}.{module_name}")
        cls = getattr(module, class_name)
        return cls(**params)
    
    def load_dataset(self, data_config: Dict[str, Any]) -> TimeSeries:
        """Load dataset based on config"""
        return self.instantiate_from_config(data_config, "data")
    
    def run_experiments(self):
        # Load experiment configurations
        exp_config = self.load_yaml(self.config_dir / "experiments.yaml")
        
        for run in exp_config['runs']:
            print(f"\n{'='*80}")
            print(f"Experiment: {run['name']}")
            print(f"Task: {run['task']}")
            print(f"{'='*80}\n")
            
            # Load component configs
            task_cfg = self.get_component_by_name(run['task'], 'tasks')
            data_cfg = self.get_component_by_name(run['data'], 'data')
            model_cfg = self.get_component_by_name(run['model'], 'models')
            preproc_cfg = self.get_component_by_name(run['preprocessor'], 'preprocessing')
            metric_names = run['metrics']
            
            # Instantiate components
            data_loader = self.instantiate(data_cfg, 'data')
            train_data, test_data = data_loader.load()
            
            preprocessor = self.instantiate(preproc_cfg, 'preprocessing')
            train_data = preprocessor.transform(train_data)
            test_data = preprocessor.transform(test_data)
            
            model = self.instantiate(model_cfg, 'models')
            model.train(train_data)
            
            # Task execution
            task_executor = self.instantiate(task_cfg, 'tasks')
            predictions = task_executor.execute(model, test_data)
            
            # Evaluate metrics
            results = {}
            for metric_name in metric_names:
                metric_cfg = self.get_component_by_name(metric_name, 'metrics')
                metric = self.instantiate(metric_cfg, 'metrics')
                score = metric.evaluate(test_data, predictions)
                results[metric_name] = score
                print(f"{metric_name.upper()}: {score:.4f}")
            
            self.save_results(run['name'], results)
    
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
    runner = BenchmarkRunner()
    runner.run_experiments()
