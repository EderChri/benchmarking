import hashlib
import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from merlion.models.base import ModelBase

logger = logging.getLogger(__name__)


def compute_model_hash(config_dict: Dict[str, Any]) -> str:
    content = yaml.dump(config_dict, sort_keys=True, default_flow_style=False)
    return hashlib.md5(content.encode()).hexdigest()[:16]


def get_checkpoint_dir(save_dir: str, config_dict: Dict[str, Any]) -> Path:
    return Path(save_dir) / "models" / compute_model_hash(config_dict)


def save_model(model: ModelBase, save_dir: str, config_dict: Dict[str, Any]):
    checkpoint_dir = get_checkpoint_dir(save_dir, config_dict)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_dir), save_config=True)
    logger.info(f"Saved model checkpoint to {checkpoint_dir}")


def load_model_if_exists(model_cls, save_dir: str, config_dict: Dict[str, Any]) -> Optional[ModelBase]:
    checkpoint_dir = get_checkpoint_dir(save_dir, config_dict)
    config_json = checkpoint_dir / "config.json"
    model_pkl = checkpoint_dir / "model.pkl"
    if config_json.exists() and model_pkl.exists():
        logger.info(f"Found existing checkpoint at {checkpoint_dir}, loading...")
        return model_cls.load(str(checkpoint_dir))
    return None
