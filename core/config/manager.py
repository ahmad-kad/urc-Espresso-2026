"""
Enhanced Configuration Management
Consolidated configuration loading, validation, and management
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from utils.logger_config import get_logger

# Optional jsonschema import
try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    jsonschema = None
    HAS_JSONSCHEMA = False

logger = get_logger(__name__, debug=os.getenv("DEBUG") == "1")


@dataclass
class ConfigValidationResult:
    """Result of configuration validation"""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add validation warning"""
        self.warnings.append(warning)


class ConfigSchema:
    """Configuration schema definitions"""

    # Base configuration schema
    BASE_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "version": {"type": "string", "default": "1.0.0"},
            "model": {
                "type": "object",
                "properties": {
                    "architecture": {
                        "type": "string",
                        "enum": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                    },
                    "input_size": {"type": "integer", "minimum": 64, "maximum": 2048},
                    "confidence_threshold": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "iou_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "max_detections": {"type": "integer", "minimum": 1},
                },
                "required": ["architecture"],
            },
            "training": {
                "type": "object",
                "properties": {
                    "epochs": {"type": "integer", "minimum": 1},
                    "batch_size": {"type": "integer", "minimum": 1},
                    "learning_rate": {"type": "number", "minimum": 0.0},
                    "weight_decay": {"type": "number", "minimum": 0.0},
                    "patience": {"type": "integer", "minimum": 0},
                    "workers": {"type": "integer", "minimum": 0},
                },
            },
            "data": {
                "type": "object",
                "properties": {
                    "yaml_path": {"type": "string"},
                    "classes": {"type": "array", "items": {"type": "string"}},
                    "num_classes": {"type": "integer", "minimum": 0},
                },
            },
            "device": {"type": "string"},
        },
        "required": ["name"],
    }


class ConfigManager:
    """
    Enhanced configuration manager with validation and inheritance support
    """

    def __init__(self, configs_dir: Optional[Path] = None):
        """
        Initialize configuration manager

        Args:
            configs_dir: Base directory for configuration files
        """
        if configs_dir is None:
            configs_dir = Path(__file__).parent.parent.parent / "configs"

        self.configs_dir = configs_dir
        self._loaded_configs: Dict[str, Dict[str, Any]] = {}
        self._schema_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"ConfigManager initialized with configs directory: {self.configs_dir}"
        )

    def load_config(
        self, config_name: str = "default", validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration by name with validation and caching

        Args:
            config_name: Name of config file (without .yaml extension)
            validate: Whether to validate the configuration

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If configuration is invalid
        """
        # Check cache first
        cache_key = f"{config_name}:{validate}"
        if cache_key in self._loaded_configs:
            logger.debug(f"Loading config from cache: {config_name}")
            return self._loaded_configs[cache_key].copy()

        # Find config file
        config_path = self._find_config_file(config_name)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_name} (searched: {config_path})"
            )

        # Load configuration
        config = self._load_config_file(config_path)

        # Validate if requested
        if validate:
            validation_result = self.validate_config(config)
            if not validation_result.is_valid:
                error_msg = f"Configuration validation failed for '{config_name}': {validation_result.errors}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if validation_result.warnings:
                logger.warning(
                    f"Configuration warnings for '{config_name}': {validation_result.warnings}"
                )

        # Cache the loaded config
        self._loaded_configs[cache_key] = config.copy()

        logger.info(f"Configuration loaded successfully: {config_name}")
        return config

    def _find_config_file(self, config_name: str) -> Path:
        """
        Find configuration file with fallback paths

        Args:
            config_name: Configuration name

        Returns:
            Path to configuration file
        """
        # Try different locations
        search_paths = [
            self.configs_dir / f"{config_name}.yaml",
            self.configs_dir / f"{config_name}.yml",
            self.configs_dir / "presets" / f"{config_name}.yaml",
            self.configs_dir / "presets" / f"{config_name}.yml",
            # Legacy locations for backward compatibility
            self.configs_dir.parent / "utils" / f"{config_name}.yaml",
        ]

        for path in search_paths:
            if path.exists():
                return path

        # Return the most likely path for error message
        return self.configs_dir / f"{config_name}.yaml"

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load a single configuration file with inheritance support

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        logger.debug(f"Loading config file: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read config file {config_path}: {e}")

        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration file {config_path} must contain a dictionary at root level"
            )

        # Handle inheritance
        if "extends" in config:
            extends_name = config["extends"]
            logger.debug(f"Config {config_path.name} extends: {extends_name}")

            try:
                base_config = self.load_config(extends_name, validate=False)
                config = self._deep_merge(base_config, config)
                config.pop("extends", None)  # Remove extends key
            except Exception as e:
                logger.warning(f"Failed to load base config '{extends_name}': {e}")

        return config

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
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

    def validate_config(
        self, config: Dict[str, Any], schema_type: str = "base"
    ) -> ConfigValidationResult:
        """
        Validate configuration against schema

        Args:
            config: Configuration to validate
            schema_type: Type of schema to use ("base", "training", "evaluation")

        Returns:
            Validation result
        """
        result = ConfigValidationResult(is_valid=True)

        # Schema validation (only if jsonschema is available)
        if HAS_JSONSCHEMA:
            try:
                # Get schema
                if schema_type not in self._schema_cache:
                    schema = getattr(
                        ConfigSchema,
                        f"{schema_type.upper()}_SCHEMA",
                        ConfigSchema.BASE_SCHEMA,
                    )
                    self._schema_cache[schema_type] = schema

                schema = self._schema_cache[schema_type]
                jsonschema.validate(instance=config, schema=schema)
            except jsonschema.ValidationError as e:
                result.add_error(f"Schema validation failed: {e.message}")
                result.add_error(
                    f"Failed at path: {' -> '.join(str(p) for p in e.absolute_path)}"
                )
            except Exception as e:
                result.add_error(f"Validation error: {e}")
        else:
            result.add_warning("jsonschema not available - skipping schema validation")

        # Additional custom validations
        self._validate_custom_rules(config, result)

        return result

    def _validate_custom_rules(
        self, config: Dict[str, Any], result: ConfigValidationResult
    ) -> None:
        """
        Apply custom validation rules

        Args:
            config: Configuration to validate
            result: Validation result to update
        """
        # Check model architecture compatibility
        if "model" in config:
            model_config = config["model"]
            architecture = model_config.get("architecture", "")

            if architecture.startswith("yolov8"):
                input_size = model_config.get("input_size", 416)
                if input_size % 32 != 0:
                    result.add_warning(
                        f"Input size {input_size} is not divisible by 32, may cause issues"
                    )

            # Check confidence threshold range
            conf_threshold = model_config.get("confidence_threshold", 0.5)
            if not (0.0 <= conf_threshold <= 1.0):
                result.add_error(
                    f"Confidence threshold must be between 0.0 and 1.0, got {conf_threshold}"
                )

        # Check training configuration
        if "training" in config:
            training_config = config["training"]

            # Validate learning rate range
            lr = training_config.get("learning_rate", 0.001)
            if lr <= 0:
                result.add_error(f"Learning rate must be positive, got {lr}")

            # Check batch size
            batch_size = training_config.get("batch_size", 8)
            if batch_size < 1:
                result.add_error(f"Batch size must be at least 1, got {batch_size}")

    def save_config(
        self, config: Dict[str, Any], config_path: Union[str, Path]
    ) -> Path:
        """
        Save configuration to file

        Args:
            config: Configuration to save
            config_path: Path to save configuration

        Returns:
            Path to saved configuration file
        """
        config_path = Path(config_path)

        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Configuration saved to: {config_path}")
            return config_path

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def create_config_from_data_yaml(
        self, data_yaml_path: Union[str, Path], base_config: str = "default"
    ) -> Dict[str, Any]:
        """
        Create configuration based on data.yaml file
        Automatically detects classes and updates configuration

        Args:
            data_yaml_path: Path to data.yaml file
            base_config: Base configuration to extend

        Returns:
            Updated configuration dictionary
        """
        data_yaml_path = Path(data_yaml_path)

        # Load base configuration
        config = self.load_config(base_config, validate=False)

        # Load data configuration
        try:
            with open(data_yaml_path, "r", encoding="utf-8") as f:
                data_config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load data YAML {data_yaml_path}: {e}")
            raise

        # Initialize data section if it doesn't exist
        if "data" not in config:
            config["data"] = {}

        # Update data section
        config["data"]["yaml_path"] = str(data_yaml_path)
        config["data"]["classes"] = data_config.get("names", [])
        config["data"]["num_classes"] = data_config.get(
            "nc", len(data_config.get("names", []))
        )

        # Update paths
        config["data"]["train_split"] = data_config.get("train", "")
        config["data"]["val_split"] = data_config.get("val", "")
        config["data"]["test_split"] = data_config.get("test", "")

        logger.info(
            f"Configuration updated for dataset with {config['data']['num_classes']} classes: "
            f"{config['data']['classes']}"
        )

        return config

    def get_available_configs(self) -> List[str]:
        """
        Get list of available configuration files

        Returns:
            List of configuration names
        """
        config_names = set()

        # Search in configs directory
        if self.configs_dir.exists():
            for yaml_file in self.configs_dir.rglob("*.yaml"):
                if yaml_file.is_file():
                    config_name = yaml_file.stem
                    config_names.add(config_name)

        return sorted(list(config_names))

    def create_preset_config(
        self, preset_type: str, custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create configuration from preset

        Args:
            preset_type: Type of preset ("accuracy", "speed", "balanced")
            custom_params: Custom parameters to override

        Returns:
            Configuration dictionary
        """
        presets = {
            "accuracy": {
                "model": {
                    "architecture": "yolov8m",
                    "input_size": 640,
                    "confidence_threshold": 0.3,
                    "iou_threshold": 0.5,
                    "max_detections": 50,
                },
                "training": {
                    "epochs": 300,
                    "batch_size": 16,
                    "learning_rate": 0.0005,
                    "weight_decay": 0.0005,
                    "patience": 50,
                    "workers": 8,
                },
            },
            "speed": {
                "model": {
                    "architecture": "yolov8n",
                    "input_size": 320,
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.4,
                    "max_detections": 20,
                },
                "training": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0005,
                    "patience": 20,
                    "workers": 4,
                },
            },
            "balanced": {
                "model": {
                    "architecture": "yolov8s",
                    "input_size": 416,
                    "confidence_threshold": 0.4,
                    "iou_threshold": 0.45,
                    "max_detections": 30,
                },
                "training": {
                    "epochs": 200,
                    "batch_size": 24,
                    "learning_rate": 0.0008,
                    "weight_decay": 0.0005,
                    "patience": 30,
                    "workers": 6,
                },
            },
        }

        if preset_type not in presets:
            raise ValueError(
                f"Unknown preset type: {preset_type}. Available: {list(presets.keys())}"
            )

        config = presets[preset_type].copy()

        # Apply custom parameters
        if custom_params:
            config = self._deep_merge(config, custom_params)

        # Add metadata
        config.update(
            {
                "name": f"{preset_type}_preset",
                "version": "1.0.0",
                "preset_type": preset_type,
                "created_with_preset": True,
            }
        )

        logger.info(f"Created {preset_type} preset configuration")
        return config


# Global config manager instance
config_manager = ConfigManager()


def load_config(config_name: str = "default", validate: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load configuration

    Args:
        config_name: Configuration name
        validate: Whether to validate configuration

    Returns:
        Configuration dictionary
    """
    return config_manager.load_config(config_name, validate)


def create_config_from_data_yaml(
    data_yaml_path: Union[str, Path], base_config: str = "default"
) -> Dict[str, Any]:
    """
    Convenience function to create config from data.yaml

    Args:
        data_yaml_path: Path to data.yaml file
        base_config: Base configuration to extend

    Returns:
        Updated configuration dictionary
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
    # Use config manager to create from preset and customize
    config = config_manager.create_preset_config(
        "accuracy",
        {
            "model": {"architecture": model, "input_size": input_size},
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "workers": workers,
            },
        },
    )

    # Add custom loss weights if provided
    if any([box_loss, cls_loss, dfl_loss]):
        if "loss_weights" not in config["training"]:
            config["training"]["loss_weights"] = {}
        config["training"]["loss_weights"].update(
            {"box": box_loss, "cls": cls_loss, "dfl": dfl_loss}
        )

    # Add augmentation settings if provided
    if any([degrees, translate, scale, mixup]):
        if "augmentation" not in config["training"]:
            config["training"]["augmentation"] = {}
        config["training"]["augmentation"].update(
            {"degrees": degrees, "translate": translate, "scale": scale, "mixup": mixup}
        )

    return config
