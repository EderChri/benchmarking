import importlib
from models.classifier_base import config
from models.model_checkpoint import load_model_if_exists
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

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
            "plots": self._instantiate_plot,       
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

    def _instantiate_model(
        self,
        config: Dict[str, Any],
        pretrained_save_dir=None,
        pretrained_config_path=None,
        pretrained_run_id=None,
        pretrained_model_config: Dict[str, Any] = None,
        save_dir_override: str = None,
    ):
        module_path = f"merlion.models.{config['module']}" if config.get("source") == "merlion" else config["module"]
        module = importlib.import_module(module_path)
        model_cls = getattr(module, config["class"])
        model_config = getattr(module, config["config_class"])(**config.get("params", {}))
        from models.hash_checkpoint_model import HashCheckpointModel

        save_dir = save_dir_override or f"src/data/.cache/{self.current_cache_key}"

        if pretrained_run_id is not None and pretrained_save_dir:
            transferred = None

            if issubclass(model_cls, HashCheckpointModel):
                import inspect

                extra = {}
                if "pretrained_config_path" in inspect.signature(model_cls.__init__).parameters:
                    extra["pretrained_config_path"] = pretrained_config_path

                transferred = model_cls(model_config, save_dir=pretrained_save_dir, **extra)
                if not getattr(transferred, "_checkpoint_loaded", False):
                    transferred = None
            else:
                source_config = pretrained_model_config or config
                transferred = load_model_if_exists(model_cls, pretrained_save_dir, source_config)

            if transferred is not None:
                if hasattr(transferred, "config"):
                    transferred.config = model_config
                if hasattr(transferred, "save_dir"):
                    transferred.save_dir = save_dir
                if hasattr(transferred, "current_epoch"):
                    transferred.current_epoch = 0
                logger.info("Loaded transfer model from pretrained run checkpoint")
                return transferred, False

            logger.warning("Requested pretrained_run but no transferable checkpoint was found; initializing fresh model")

        # Try loading existing checkpoint first (works for ALL models)
        existing = None
        if pretrained_run_id is None:
            existing = load_model_if_exists(model_cls, save_dir, config)
            if existing is not None:
                logger.info("Successfully resumed from existing checkpoint")
                return existing, True

        # Instantiate fresh — inject extra kwargs only for HashCheckpointModel subclasses
        import inspect
        if issubclass(model_cls, HashCheckpointModel):
            extra = {}
            if "pretrained_config_path" in inspect.signature(model_cls.__init__).parameters:
                extra["pretrained_config_path"] = pretrained_config_path
            return model_cls(model_config, save_dir=save_dir, **extra), False

        return model_cls(model_config), False

    
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
