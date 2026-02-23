import importlib
from models.classifier_base import config
from models.model_checkpoint import load_model_if_exists
import yaml
from pathlib import Path
from typing import Dict, Any

class ComponentFactory:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.current_cache_key: str = None

    def load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f)

    def get_component_by_name(self, name: str, component_type: str) -> Dict[str, Any]:
        return self.load_yaml(self.config_dir / component_type / f"{name}.yaml")

    def instantiate(self, config: Dict[str, Any], component_type: str):
        handlers = {
            "metrics": self._instantiate_metric,
            "models": self._instantiate_model,
            "preprocessing": self._instantiate_preprocessing,
            "plots": self._instantiate_plot,          # <-- add this
        }
        handler = handlers.get(component_type)
        if not handler:
            raise ValueError(f"Unknown component_type: {component_type}")
        return handler(config)

    def load_preprocessor(self, preproc_config: Dict[str, Any]):
        if "transforms" in preproc_config:
            from merlion.transform.sequence import TransformSequence
            return TransformSequence([self.instantiate(t, "preprocessing") for t in preproc_config["transforms"]])
        return self.instantiate(preproc_config, "preprocessing")

    def _instantiate_metric(self, config: Dict[str, Any]):
        if config.get("class_path"):
            return self._import_from_path(config["class_path"])
        if config.get("source") == "merlion":
            module = importlib.import_module(f"merlion.evaluate.{config['module']}")
            return getattr(getattr(module, config["class"]), config["metric_name"])
        module = importlib.import_module(f"metrics.{config['module']}")
        return getattr(module, config["class"])(**config.get("params", {}))

    def _instantiate_model(self, config: Dict[str, Any], pretrained_save_dir=None, pretrained_config_path=None):
        module_path = f"merlion.models.{config['module']}" if config.get("source") == "merlion" else config["module"]
        module = importlib.import_module(module_path)
        model_cls = getattr(module, config["class"])
        model_config = getattr(module, config["config_class"])(**config.get("params", {}))

        save_dir = pretrained_save_dir or f"src/data/.cache/{self.current_cache_key}"

        # Try loading existing checkpoint first (works for ALL models)
        existing = load_model_if_exists(model_cls, save_dir, config)
        if existing is not None:
            print("Successfully resumed from existing checkpoint")
            return existing

        # Instantiate fresh — inject extra kwargs only for HashCheckpointModel subclasses
        import inspect
        from models.hash_checkpoint_model import HashCheckpointModel
        if issubclass(model_cls, HashCheckpointModel):
            extra = {}
            if "pretrained_config_path" in inspect.signature(model_cls.__init__).parameters:
                extra["pretrained_config_path"] = pretrained_config_path
            return model_cls(model_config, save_dir=save_dir, **extra)

        return model_cls(model_config)

    
    def _instantiate_plot(self, config: Dict[str, Any]):
        module = importlib.import_module(config["module"])
        cls = getattr(module, config["class"])
        return cls(**config.get("params", {}))
    
    def _instantiate_preprocessing(self, config: Dict[str, Any]):
        module_path = (
            f"merlion.transform.{config['module']}" if config.get("source") == "merlion"
            else f"preprocessing.{config['module']}"
        )
        module = importlib.import_module(module_path)
        return getattr(module, config["class"])(**config.get("params", {}))

    def _import_from_path(self, class_path: str):
        parts = class_path.rsplit(".", 1)
        return getattr(importlib.import_module(parts[0]), parts[1])
