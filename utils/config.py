"""
Configuration management for object detection framework
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration loading, validation, and inheritance
    """

    def __init__(self) -> None:
        self.configs_dir = Path(__file__).parent.parent.parent / "configs"

    def load_config(self, config_name: str = "default") -> Dict[str, Any]:
        """
        Load configuration by name

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Configuration dictionary
        """

        config_path = self.configs_dir / "framework" / f"{config_name}.yaml"

        if not config_path.exists():
            # Try environments subdirectory
            config_path = (
                self.configs_dir / "framework" / "environments" / f"{config_name}.yaml"
            )

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_name}")

        return self._load_config_file(config_path)

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load a single configuration file with inheritance support
        """

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Handle inheritance
        if "extends" in config:
            base_config_path = config_path.parent / config["extends"]
            if base_config_path.exists():
                base_config = self._load_config_file(base_config_path)
                # Merge configs (base_config is overridden by config)
                merged_config = self._deep_merge(base_config, config)
                merged_config.pop("extends", None)  # Remove extends key
                config = merged_config
            else:
                logger.warning(f"Base config not found: {base_config_path}")

        return config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries
        """

        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def save_config(self, config: Dict[str, Any], config_path: Path) -> None:
        """
        Save configuration to file
        """

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {config_path}")

    # get_available_configs method removed - unused

    def create_config_from_data_yaml(
        self, data_yaml_path: str, base_config: str = "default"
    ) -> Dict[str, Any]:
        """
        Create configuration based on data.yaml file
        Automatically detects classes and updates configuration
        """

        # Load base configuration
        config = self.load_config(base_config)

        # Load data configuration
        with open(data_yaml_path, "r") as f:
            data_config = yaml.safe_load(f)

        # Update data section
        config["data"]["yaml_path"] = data_yaml_path
        config["data"]["classes"] = data_config.get("names", [])
        config["data"]["num_classes"] = data_config.get("nc", 0)

        # Update paths
        config["data"]["train_split"] = data_config.get("train", "")
        config["data"]["val_split"] = data_config.get("val", "")
        config["data"]["test_split"] = data_config.get("test", "")

        logger.info(
            f"Configuration updated for dataset with {config['data']['num_classes']} classes: {config['data']['classes']}"
        )

        return config


# validate_config method removed - unused

# update_config_from_args method removed - unused


# Global config manager instance
config_manager = ConfigManager()


def load_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Convenience function to load configuration
    """
    return config_manager.load_config(config_name)


def create_config_from_data_yaml(
    data_yaml_path: str, base_config: str = "default"
) -> Dict[str, Any]:
    """
    Convenience function to create config from data.yaml
    """
    return config_manager.create_config_from_data_yaml(data_yaml_path, base_config)


def create_accuracy_training_config(
    model: str = "yolov8m",
    input_size: int = 224,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    box_loss: Optional[float] = None,
    cls_loss: Optional[float] = None,
    dfl_loss: Optional[float] = None,
    degrees: Optional[float] = None,
    translate: Optional[float] = None,
    scale: Optional[float] = None,
    mixup: Optional[float] = None,
    workers: int = 8,
    flat_format: bool = False,
) -> Dict[str, Any]:
    """
    Create configuration optimized for accuracy with optional hyperparameters

    Args:
        model: Model architecture (e.g., "yolov8m")
        input_size: Input image size
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay coefficient
        box_loss: Box loss weight
        cls_loss: Classification loss weight
        dfl_loss: Distribution focal loss weight
        degrees: Rotation augmentation degrees
        translate: Translation augmentation
        scale: Scale augmentation
        mixup: Mixup augmentation probability
        workers: Number of data loading workers
        flat_format: If True, return flat format for direct YOLO usage

    Returns:
        Training configuration dictionary
    """
    # Base configuration values
    config_values = {
        "model": model,
        "input_size": input_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate or 0.001,
        "weight_decay": weight_decay or 0.0005,
        "box_loss": box_loss or 7.5,
        "cls_loss": cls_loss or 0.5,
        "dfl_loss": dfl_loss or 1.5,
        "degrees": degrees if degrees is not None else 15.0,
        "translate": translate if translate is not None else 0.15,
        "scale": scale if scale is not None else 0.5,
        "shear": 5.0,
        "perspective": 0.0003,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": mixup if mixup is not None else 0.15,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "workers": workers,
    }

    if flat_format:
        # Return flat format for direct YOLO usage
        return {
            "model": config_values["model"],
            "data": "data.yaml",  # Will be set dynamically
            "epochs": config_values["epochs"],
            "imgsz": config_values["input_size"],
            "batch": config_values["batch_size"],
            "device": "auto",
            "workers": config_values["workers"],
            "lr0": config_values["learning_rate"],
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": config_values["weight_decay"],
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": config_values["box_loss"],
            "cls": config_values["cls_loss"],
            "dfl": config_values["dfl_loss"],
            "degrees": config_values["degrees"],
            "translate": config_values["translate"],
            "scale": config_values["scale"],
            "shear": config_values["shear"],
            "perspective": config_values["perspective"],
            "flipud": config_values["flipud"],
            "fliplr": config_values["fliplr"],
            "mosaic": config_values["mosaic"],
            "mixup": config_values["mixup"],
            "hsv_h": config_values["hsv_h"],
            "hsv_s": config_values["hsv_s"],
            "hsv_v": config_values["hsv_v"],
        }
    else:
        # Return nested format for framework usage
        return {
            "project": {"name": "yolo_accuracy_training"},
            "model": {
                "architecture": config_values["model"],
                "input_size": config_values["input_size"],
                "confidence_threshold": 0.5,
                "iou_threshold": 0.4,
                "max_detections": 20,
            },
            "training": {
                "epochs": config_values["epochs"],
                "batch_size": config_values["batch_size"],
                "learning_rate": config_values["learning_rate"],
                "weight_decay": config_values["weight_decay"],
                "patience": 10,
                "workers": config_values["workers"],
                "augmentation": {
                    "degrees": config_values["degrees"],
                    "translate": config_values["translate"],
                    "scale": config_values["scale"],
                    "shear": config_values["shear"],
                    "perspective": config_values["perspective"],
                    "flipud": config_values["flipud"],
                    "fliplr": config_values["fliplr"],
                    "mosaic": config_values["mosaic"],
                    "mixup": config_values["mixup"],
                    "hsv_h": config_values["hsv_h"],
                    "hsv_s": config_values["hsv_s"],
                    "hsv_v": config_values["hsv_v"],
                },
                "loss_weights": {
                    "box": config_values["box_loss"],
                    "cls": config_values["cls_loss"],
                    "dfl": config_values["dfl_loss"],
                },
            },
            "device": "cuda",
        }
