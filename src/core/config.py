"""
Configuration management for object detection framework
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration loading, validation, and inheritance
    """

    def __init__(self):
        self.configs_dir = Path(__file__).parent.parent.parent / "configs"

    def load_config(self, config_name: str = "default") -> Dict[str, Any]:
        """
        Load configuration by name

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Configuration dictionary
        """

        config_path = self.configs_dir / f"{config_name}.yaml"

        if not config_path.exists():
            # Try environments subdirectory
            config_path = self.configs_dir / "environments" / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_name}")

        return self._load_config_file(config_path)

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load a single configuration file with inheritance support
        """

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Handle inheritance
        if 'extends' in config:
            base_config_path = config_path.parent / config['extends']
            if base_config_path.exists():
                base_config = self._load_config_file(base_config_path)
                # Merge configs (base_config is overridden by config)
                merged_config = self._deep_merge(base_config, config)
                merged_config.pop('extends', None)  # Remove extends key
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
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def save_config(self, config: Dict, config_path: Path):
        """
        Save configuration to file
        """

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {config_path}")

    def get_available_configs(self) -> list:
        """
        Get list of available configuration files
        """

        configs = []

        # Main configs
        for config_file in self.configs_dir.glob("*.yaml"):
            configs.append(config_file.stem)

        # Environment configs
        env_dir = self.configs_dir / "environments"
        if env_dir.exists():
            for config_file in env_dir.glob("*.yaml"):
                configs.append(f"environments/{config_file.stem}")

        return configs

    def create_config_from_data_yaml(self, data_yaml_path: str,
                                   base_config: str = "default") -> Dict[str, Any]:
        """
        Create configuration based on data.yaml file
        Automatically detects classes and updates configuration
        """

        # Load base configuration
        config = self.load_config(base_config)

        # Load data configuration
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # Update data section
        config['data']['yaml_path'] = data_yaml_path
        config['data']['classes'] = data_config.get('names', [])
        config['data']['num_classes'] = data_config.get('nc', 0)

        # Update paths
        config['data']['train_split'] = data_config.get('train', '')
        config['data']['val_split'] = data_config.get('val', '')
        config['data']['test_split'] = data_config.get('test', '')

        logger.info(f"Configuration updated for dataset with {config['data']['num_classes']} classes: {config['data']['classes']}")

        return config

    def validate_config(self, config: Dict) -> list:
        """
        Validate configuration and return list of issues
        """

        issues = []

        # Required sections
        required_sections = ['project', 'model', 'training', 'output']
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")

        # Model validation
        model_config = config.get('model', {})
        if 'architecture' not in model_config:
            issues.append("Model architecture not specified")

        # Training validation
        training_config = config.get('training', {})
        if 'epochs' not in training_config:
            issues.append("Training epochs not specified")

        # Output validation
        output_config = config.get('output', {})
        required_outputs = ['models', 'results', 'logs']
        for output_dir in required_outputs:
            if output_dir not in output_config:
                issues.append(f"Output directory not specified: {output_dir}")

        return issues

    def update_config_from_args(self, config: Dict, args: Dict) -> Dict:
        """
        Update configuration from command line arguments
        """

        # Model updates
        if 'architecture' in args:
            config['model']['architecture'] = args['architecture']
        if 'input_size' in args:
            config['model']['input_size'] = args['input_size']
        if 'confidence_threshold' in args:
            config['model']['confidence_threshold'] = args['confidence_threshold']

        # Training updates
        if 'epochs' in args:
            config['training']['epochs'] = args['epochs']
        if 'batch_size' in args:
            config['training']['batch_size'] = args['batch_size']
        if 'learning_rate' in args:
            config['training']['learning_rate'] = args['learning_rate']

        # Device
        if 'device' in args:
            config['device'] = args['device']

        return config


# Global config manager instance
config_manager = ConfigManager()


def load_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Convenience function to load configuration
    """
    return config_manager.load_config(config_name)


def create_config_from_data_yaml(data_yaml_path: str, base_config: str = "default") -> Dict[str, Any]:
    """
    Convenience function to create config from data.yaml
    """
    return config_manager.create_config_from_data_yaml(data_yaml_path, base_config)
