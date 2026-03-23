import os
import shutil
import hashlib
import yaml
import re
from typing import Optional, Dict, Any
import logging
from merlion.models.base import ModelBase

logger = logging.getLogger(__name__)


class HashCheckpointModel(ModelBase):
    """
    Base class for Merlion detectors with hash-based checkpoint management.

    Automatically handles:
    - Hash-based directory creation from YAML config
    - Checkpoint existence detection
    - Loading existing checkpoints
    - Saving with YAML config copying
    """

    def __init__(self, config, save_dir: Optional[str] = None, pretrained_config_path: Optional[str] = None):
        super().__init__(config)
        self.save_dir = save_dir
        self.pretrained_config_path = pretrained_config_path
        self.config_hash = None
        self._checkpoint_loaded = False
        self.current_epoch = 0

    def _compute_config_hash(self) -> str:
        """Compute hash from the live runtime config (includes experiment overrides)."""
        if self.config_hash is None:
            cfg = {
                k: v for k, v in vars(self.config).items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))
            }
            yaml_content = yaml.dump(cfg, sort_keys=True, default_flow_style=False)
            self.config_hash = hashlib.md5(yaml_content.encode()).hexdigest()[:16]
        return self.config_hash

    def _get_config_yaml_path(self) -> str:
        """Get path to the config YAML file"""
        class_name = self.__class__.__name__
        snake_case_name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        return f"conf/models/{snake_case_name}.yaml"

    def _get_checkpoint_dir(self) -> str:
        """Get the checkpoint directory based on config hash"""
        if not self.save_dir:
            raise ValueError("save_dir must be set to use checkpointing")

        config_hash = self._compute_config_hash()
        return os.path.join(self.save_dir, "models", config_hash)

    def _checkpoint_exists(self) -> bool:
        """Check if checkpoint exists"""
        try:
            checkpoint_dir = self._get_checkpoint_dir()
            config_json = os.path.join(checkpoint_dir, "config.json")
            model_pkl = os.path.join(checkpoint_dir, "model.pkl")
            return os.path.exists(config_json) and os.path.exists(model_pkl)
        except (ValueError, FileNotFoundError):
            return False

    def _try_load_existing_checkpoint(self) -> bool:
        """
        Try to load existing checkpoint if it exists.
        Subclasses should override _load_checkpoint_state() to customize loading.

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if not self.save_dir or self._checkpoint_loaded:
            return False

        if not self._checkpoint_exists():
            return False

        try:
            checkpoint_dir = self._get_checkpoint_dir()
            logger.info(f"Found existing checkpoint at {checkpoint_dir}, loading...")

            loaded_model = super(HashCheckpointModel, type(self)).load(checkpoint_dir)

            self._load_checkpoint_state(loaded_model)
            self._checkpoint_loaded = True
            logger.info("Successfully resumed from existing checkpoint")
            return True

        except Exception as e:
            print(f"Failed to load existing checkpoint: {e}")
            return False

    def _load_checkpoint_state(self, loaded_model):
        """
        Transfer state from loaded model to current instance.
        Subclasses must override this method.

        Args:
            loaded_model: The model instance loaded from checkpoint
        """
        raise NotImplementedError(
            "Subclasses must implement _load_checkpoint_state()")

    def _load_from_checkpoint(self, dirname: str, config_dict: Dict[str, Any]):
        """
        Load model from checkpoint directory.

        Args:
            dirname: Parent directory containing models/
            config_dict: Configuration dictionary

        Returns:
            Loaded model instance
        """
        return type(self).load(dirname=dirname, config_dict=config_dict)

    def save(self, **save_config):
        """
        Save checkpoint with hash-based directory and YAML config.
        Uses DetectorBase.save() for actual model serialization.
        """
        logger.info("Saving model checkpoint...")
        checkpoint_dir = self._get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Copy YAML config if not exists
        yaml_path = self._get_config_yaml_path()
        class_name = self.__class__.__name__
        snake_case_name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        yaml_dest = os.path.join(checkpoint_dir, f"{snake_case_name}.yaml")

        if not os.path.exists(yaml_dest):
            shutil.copy2(yaml_path, yaml_dest)

        # Call parent save with hash directory
        super().save(checkpoint_dir, **save_config)

    @classmethod
    def load(cls, dirname: str, config_dict: Dict[str, Any], **kwargs):
        """
        Load model from hash-based checkpoint directory.

        Args:
            dirname: Parent directory containing models/
            config_dict: Configuration dictionary (will be hashed to find checkpoint)
            **kwargs: Additional arguments passed to DetectorBase.load()

        Returns:
            Loaded model instance
        """
        # Reproduce hash from config
        yaml_content = yaml.dump(
            config_dict, sort_keys=True, default_flow_style=False)
        config_hash = hashlib.md5(yaml_content.encode()).hexdigest()[:16]
        hash_dir = os.path.join(dirname, "models", config_hash)

        # Use parent class load with hash directory
        return super(HashCheckpointModel, cls).load(hash_dir, **kwargs)
