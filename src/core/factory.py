import importlib
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
        from lightning import LightningModule
        from models.model_checkpoint import load_model_if_exists

        is_merlion = config.get("source") == "merlion"
        module_path = f"merlion.models.{config['module']}" if is_merlion else config["module"]
        module = importlib.import_module(module_path)
        model_cls = getattr(module, config["class"])
        is_custom = issubclass(model_cls, LightningModule)

        save_dir = save_dir_override or f"src/data/.cache/{self.current_cache_key}"

        # --- Dataclass configs (custom models) vs keyword-arg configs (Merlion) ---
        def make_config(params_dict):
            config_cls = getattr(module, config["config_class"])
            from dataclasses import is_dataclass
            if is_dataclass(config_cls):
                return config_cls(**params_dict)
            return config_cls(**params_dict)

        model_config = make_config(config.get("params", {}))

        # --- Transfer learning from a pretrained run ---
        if pretrained_run_id is not None and pretrained_save_dir:
            transferred = None

            if is_custom:
                # Build with pretrained config+save_dir so checkpoint_dir resolves correctly.
                pretrain_params = (pretrained_model_config or config).get("params", {})
                pretrain_cfg = make_config(pretrain_params)
                transferred = model_cls(pretrain_cfg, save_dir=pretrained_save_dir)
                logger.info(f"Looking for pretrained checkpoint at: {transferred.checkpoint_path}")
                if not transferred.checkpoint_exists():
                    transferred = None
                else:
                    transferred._ensure_ready()
            else:
                source_config = pretrained_model_config or config
                transferred = load_model_if_exists(model_cls, pretrained_save_dir, source_config)

            if transferred is not None:
                # Swap config and save_dir so the model trains/saves in the new run's location.
                transferred.config = model_config
                transferred.save_dir = save_dir
                logger.info("Loaded transfer model from pretrained run checkpoint")
                return transferred, False

            logger.warning(
                "pretrained_run requested but no checkpoint found; initialising fresh model"
            )

        # --- Resume existing checkpoint (same run, re-started) ---
        if pretrained_run_id is None:
            if is_custom:
                # Instantiate fresh — checkpoint is loaded lazily in _ensure_ready() / _fit_and_restore_best()
                model = model_cls(model_config, save_dir=save_dir)
                already_done = model.checkpoint_exists()
                return model, already_done
            else:
                existing = load_model_if_exists(model_cls, save_dir, config)
                if existing is not None:
                    logger.info("Successfully resumed from existing checkpoint")
                    return existing, True

        # --- Fresh instantiation ---
        if is_custom:
            return model_cls(model_config, save_dir=save_dir), False

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
