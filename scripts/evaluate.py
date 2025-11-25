#!/usr/bin/env python3
"""
Generalized model evaluation script
Compares multiple models and generates comprehensive reports
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import load_config
from core.evaluator import ModelEvaluator
from utils.data_utils import get_image_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection models')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration to use')
    parser.add_argument('--data_yaml', type=str, default='data/data.yaml',
                       help='Path to data configuration YAML')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Paths to model files to evaluate')
    parser.add_argument('--model_names', nargs='+',
                       help='Names for the models (same order as --models)')
    parser.add_argument('--training_logs', nargs='+',
                       help='Paths to training log directories (same order as --models) for accuracy over time plots')
    parser.add_argument('--test_images', type=str,
                       help='Directory with test images for real-time testing')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for results (overrides config)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive evaluation including real-time performance')
    parser.add_argument('--compare_all', action='store_true',
                       help='Create comprehensive comparison plots and tables for all models')

    args = parser.parse_args()

    try:
        logger.info("Starting model evaluation")

        # Load configuration
        config = load_config(args.config)

        # Override output directory if specified
        if args.output_dir:
            config['output']['results'] = args.output_dir

        # Set up model names
        if args.model_names:
            if len(args.model_names) != len(args.models):
                logger.error("Number of model names must match number of models")
                return 1
            model_names = args.model_names
        else:
            model_names = [f"model_{i+1}" for i in range(len(args.models))]

        # Create model dictionary
        models_to_evaluate = dict(zip(model_names, args.models))

        # Set up training logs if provided
        training_logs = None
        if args.training_logs:
            if len(args.training_logs) != len(args.models):
                logger.error("Number of training logs must match number of models")
                return 1
            training_logs = dict(zip(model_names, args.training_logs))

        # Initialize evaluator
        evaluator = ModelEvaluator(config)

        # Get test images for real-time testing
        test_images = []
        if args.test_images and Path(args.test_images).exists():
            test_images = get_image_paths(args.test_images)
            logger.info(f"Found {len(test_images)} test images for real-time evaluation")

        if args.compare_all:
            # Run comprehensive model comparison with all visualizations
            logger.info("Running comprehensive model comparison...")
            results = evaluator.compare_models(models_to_evaluate, args.data_yaml)

            # Create comprehensive comparison plots and tables
            evaluator.create_comprehensive_model_comparison(results, training_logs)

        elif args.comprehensive and test_images:
            # Run comprehensive evaluation
            logger.info("Running comprehensive evaluation...")
            results = evaluator.generate_comprehensive_report(
                models_to_evaluate, args.data_yaml, test_images
            )
        else:
            # Run basic model comparison
            logger.info("Running model comparison...")
            results = evaluator.compare_models(models_to_evaluate, args.data_yaml)

        # Print summary
        logger.info("Evaluation completed!")
        logger.info("="*50)

        if 'model_comparison' in results:
            for model_name, result in results['model_comparison'].items():
                if 'metrics' in result:
                    logger.info(f"{model_name}:")
                    logger.info(".4f")
                    logger.info(".4f")
                else:
                    logger.info(f"{model_name}: Evaluation failed - {result.get('error', 'Unknown error')}")

        if 'realtime_performance' in results:
            logger.info("Real-time Performance:")
            for model_name, perf in results['realtime_performance'].items():
                if 'fps' in perf:
                    logger.info(".1f")
                else:
                    logger.info(f"{model_name}: Performance test failed")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
