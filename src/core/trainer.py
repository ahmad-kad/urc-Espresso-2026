"""
Generic training framework for object detection models
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional

from .detector import ObjectDetector

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Generic trainer for object detection models
    """

    def __init__(self, config: Dict):
        self.config = config
        self.detector = ObjectDetector(config)

    def train(self, data_yaml: str, experiment_name: Optional[str] = None,
              project: Optional[str] = None, name: Optional[str] = None) -> Dict:
        """
        Train the model with comprehensive logging and error handling

        Args:
            data_yaml: Path to data configuration YAML
            experiment_name: Optional experiment name for output organization

        Returns:
            Training results dictionary
        """

        logger.info("Starting model training...")
        logger.info(f"Data configuration: {data_yaml}")

        try:
            # Set project name for organized outputs
            if experiment_name:
                project_name = f"{self.config.get('project', {}).get('name', 'object_detection')}_{experiment_name}"
            else:
                project_name = self.config.get('project', {}).get('name', 'object_detection')

            # Train the model
            train_kwargs = {}
            if project:
                train_kwargs['project'] = project
            if name:
                train_kwargs['name'] = name

            results = self.detector.train(
                data_yaml,
                model_name=name or project_name,
                **train_kwargs
            )

            logger.info("Training completed successfully!")
            logger.info(f"Model saved to: {results.save_dir}")

            return {
                'success': True,
                'save_dir': str(results.save_dir),
                'model_path': str(Path(results.save_dir) / 'weights' / 'best.pt'),
                'results': results
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def train_baseline(self, data_yaml: str) -> Dict:
        """Train baseline model (no attention)"""
        # Temporarily disable attention
        original_attention = self.config.get('attention', {}).get('enabled', False)
        self.config['attention'] = {'enabled': False}

        try:
            result = self.train(data_yaml, "baseline")
            return result
        finally:
            # Restore original config
            self.config['attention'] = {'enabled': original_attention}

    def train_with_attention(self, data_yaml: str, attention_type: str = "cbam") -> Dict:
        """Train model with attention modules"""
        # Enable attention
        self.config['attention'] = {
            'enabled': True,
            'type': attention_type,
            'layers': ['model.3', 'model.6', 'model.9', 'model.12']
        }

        return self.train(data_yaml, f"{attention_type}_enhanced")

    def save_config(self, output_dir: str):
        """Save the training configuration for reproducibility"""
        import yaml

        config_path = Path(output_dir) / "training_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"Training configuration saved to: {config_path}")


def run_training_pipeline(base_config: dict, data_yaml: str, train_all: bool = True):
    """
    Run complete training pipeline for all 4 model architectures

    Args:
        base_config: Base configuration dictionary
        data_yaml: Path to data configuration YAML
        train_all: Whether to train all 4 models
    """

    logger.info("="*60)
    logger.info("STARTING OBJECT DETECTION TRAINING PIPELINE")
    logger.info("="*60)
    logger.info(f"Config: loaded")
    logger.info(f"Data: {data_yaml}")
    logger.info(f"Training all models: {train_all}")
    logger.info("="*60)

    results = {}

    # Define the 4 model configurations
    model_configs = [
        ('yolov8s_baseline', {'architecture': 'yolov8s', 'attention': {'enabled': False}}),
        ('yolov8s_cbam', {'architecture': 'yolov8s', 'attention': {'enabled': True, 'type': 'cbam'}}),
        ('mobilenet_vit', {'architecture': 'mobilenet_vit', 'attention': {'enabled': False}}),
        ('efficientnet', {'architecture': 'efficientnet', 'attention': {'enabled': False}})
    ]

    logger.info(f"Training {len(model_configs)} models")

    # Train each model
    for model_name, model_overrides in model_configs:
        logger.info("="*60)
        logger.info(f"PHASE: Training {model_name.upper()}")
        logger.info("="*60)

        # Use the provided base configuration

        # Apply model-specific overrides
        import copy
        config = copy.deepcopy(base_config)
        if 'architecture' in model_overrides:
            config['model']['architecture'] = model_overrides['architecture']
        if 'attention' in model_overrides:
            config['attention'].update(model_overrides['attention'])

        # Create trainer with specific config
        trainer = ModelTrainer(config)

        # Train the model with specific output path
        result = trainer.train(data_yaml, experiment_name=model_name,
                              project='output', name=f'models/{model_name}')
        results[model_name] = result

        if result['success']:
            logger.info(f"✓ {model_name.upper()} training completed successfully")
        else:
            logger.error(f"✗ {model_name.upper()} training failed: {result.get('error', 'Unknown error')}")

    # Summary
    logger.info("="*60)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("="*60)

    successful_models = 0
    for model_name, result in results.items():
        if result['success']:
            logger.info(f"✓ {model_name.upper()}: {result['save_dir']}")
            successful_models += 1
        else:
            logger.info(f"✗ {model_name.upper()}: Failed - {result.get('error', 'Unknown error')}")

    logger.info(f"\nSuccessfully trained {successful_models}/4 models")

    return results
