#!/usr/bin/env python3
"""
Generalized object detection training script
Supports any dataset and multiple model architectures
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import load_config, create_config_from_data_yaml
from core.trainer import run_training_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train object detection models')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration to use (filename without .yaml)')
    parser.add_argument('--data_yaml', type=str, default='balanced_data/data.yaml',
                       help='Path to data configuration YAML')
    parser.add_argument('--train_all', action='store_true', default=True,
                       help='Train all 4 model architectures')

    # Override config parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--input_size', type=int, help='Input image size')
    parser.add_argument('--device', type=str, help='Training device (cpu/cuda)')

    args = parser.parse_args()

    try:
        logger.info("Starting object detection training for all 4 architectures")

        # Load or create configuration
        if Path(args.data_yaml).exists():
            config = create_config_from_data_yaml(args.data_yaml, args.config)
        else:
            print(f"Data YAML not found at {args.data_yaml}, falling back to default")
            config = load_config(args.config)

        # Apply command line overrides
        overrides = {}
        if args.epochs:
            overrides['epochs'] = args.epochs
        if args.batch_size:
            overrides['batch_size'] = args.batch_size
        if args.learning_rate:
            overrides['learning_rate'] = args.learning_rate
        if args.input_size:
            overrides['input_size'] = args.input_size
        if args.device:
            overrides['device'] = args.device

        if overrides:
            # Update training config
            config['training'].update(overrides)
            logger.info(f"Applied overrides: {overrides}")

        # Save the updated config for reference
        from core.config import ConfigManager
        config_manager = ConfigManager()
        config_manager.save_config(config, Path('output/config_used.yaml'))

        # Run training pipeline for all models with balanced dataset
        print(f"🎯 Training with balanced dataset: {args.data_yaml}")
        results = run_training_pipeline(
            base_config=config,  # Pass the loaded config object
            data_yaml=args.data_yaml,
            train_all=args.train_all
        )

        # Report results
        logger.info("Training completed!")
        successful_models = sum(1 for result in results.values() if result.get('success'))
        logger.info(f"Successfully trained {successful_models}/4 models")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
