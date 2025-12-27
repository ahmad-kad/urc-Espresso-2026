#!/usr/bin/env python3
"""
Train all models from scratch using YOLO CLI
Supports YOLOv8 variants, MobileNet, and EfficientNet at different sizes
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict
import argparse
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model configurations to train
MODEL_CONFIGS = {
    'yolov8n': ['160', '192', '224'],
    'yolov8s': ['confidence'],
    'yolov8s_cbam': ['confidence'],
    'yolov8m': ['confidence'],
    'yolov8l': ['confidence'],
    'mobilenet': ['160', '192', '224', 'confidence'],
    'efficientnet': ['confidence']
}

def run_training_command(config_path: str, model_name: str) -> bool:
    """
    Run YOLO training for a specific config

    Args:
        config_path: Path to the YAML config file
        model_name: Name of the model for logging

    Returns:
        bool: True if training completed successfully, False otherwise
    """
    logger.info(f"Starting training for {model_name} using config: {config_path}")

    try:
        # Load and parse the YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Build command with config parameters as CLI args
        cmd = ['yolo', 'train']

        # Convert config to CLI arguments
        for key, value in config.items():
            if value is not None:  # Skip null values
                if isinstance(value, bool):
                    # For boolean values, only include if True
                    if value:
                        cmd.append(f"{key}={value}")
                elif isinstance(value, str) and value.startswith('models/'):
                    # Handle model paths - ensure they exist
                    model_path = Path(value)
                    if model_path.exists():
                        cmd.append(f"{key}={value}")
                    else:
                        logger.warning(f"Model file not found: {value}, skipping {key}")
                else:
                    cmd.append(f"{key}={value}")

        logger.info(f"Running command: {' '.join(cmd)}")

        # Run the training process
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 24  # 24 hour timeout per model
        )

        if result.returncode == 0:
            logger.info(f"Successfully completed training for {model_name}")
            # Log the output summary
            if result.stdout:
                with open(f"training_output_{model_name}.log", 'w') as f:
                    f.write(result.stdout)
            return True
        else:
            logger.error(f"Training failed for {model_name}")
            logger.error(f"Return code: {result.returncode}")
            if result.stderr:
                logger.error(f"Stderr: {result.stderr}")
            if result.stdout:
                logger.error(f"Stdout: {result.stdout}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out for {model_name}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error training {model_name}: {str(e)}")
        return False

def get_config_path(model_type: str, size_or_variant: str) -> str:
    """Get the config file path for a model"""
    if model_type == 'yolov8n':
        return f"configs/yolov8n_{size_or_variant}.yaml"
    elif model_type == 'yolov8s_cbam':
        return f"configs/yolov8s_cbam_confidence.yaml"
    elif model_type in ['yolov8s', 'yolov8m', 'yolov8l']:
        return f"configs/{model_type}_confidence.yaml"
    elif model_type == 'mobilenet':
        if size_or_variant == 'confidence':
            return f"configs/mobilenet_confidence.yaml"
        else:
            return f"configs/mobilenet_{size_or_variant}.yaml"
    elif model_type == 'efficientnet':
        return f"configs/efficientnet_confidence.yaml"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_all_models(architectures: List[str] = None, sizes: List[str] = None,
                    epochs: int = 100, parallel: bool = False) -> Dict[str, bool]:
    """
    Train all specified models

    Args:
        architectures: List of architectures to train (default: all)
        sizes: List of sizes/variants to train (default: all available)
        epochs: Number of epochs (already set in configs)
        parallel: Whether to run in parallel (not implemented yet)

    Returns:
        Dict mapping model names to success status
    """
    if architectures is None:
        architectures = list(MODEL_CONFIGS.keys())

    results = {}
    total_models = 0

    # Count total models to train
    for arch in architectures:
        if arch in MODEL_CONFIGS:
            if sizes:
                total_models += len([s for s in MODEL_CONFIGS[arch] if s in sizes])
            else:
                total_models += len(MODEL_CONFIGS[arch])

    logger.info(f"Training {total_models} models total")

    model_count = 0

    for arch in architectures:
        if arch not in MODEL_CONFIGS:
            logger.warning(f"Unknown architecture: {arch}, skipping")
            continue

        available_sizes = MODEL_CONFIGS[arch]
        sizes_to_train = sizes if sizes else available_sizes

        for size in sizes_to_train:
            if size not in available_sizes:
                logger.warning(f"Size {size} not available for {arch}, skipping")
                continue

            model_count += 1
            model_name = f"{arch}_{size}" if size != 'confidence' else arch

            logger.info(f"[{model_count}/{total_models}] Training {model_name}")

            # Get config path
            config_path = get_config_path(arch, size)

            # Check if config exists
            if not Path(config_path).exists():
                logger.error(f"Config file not found: {config_path}")
                results[model_name] = False
                continue

            # Run training
            success = run_training_command(config_path, model_name)
            results[model_name] = success

            # Brief pause between models to allow system to settle
            if model_count < total_models:
                logger.info("Pausing for 30 seconds before next model...")
                time.sleep(30)

    return results

def main():
    parser = argparse.ArgumentParser(description='Train all models from scratch')
    parser.add_argument('--architectures', nargs='+',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Architectures to train (default: all)')
    parser.add_argument('--sizes', nargs='+',
                       help='Sizes/variants to train (default: all available)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run training in parallel (not implemented)')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("STARTING COMPREHENSIVE MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Architectures: {args.architectures or 'all'}")
    logger.info(f"Sizes: {args.sizes or 'all'}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Parallel: {args.parallel}")
    logger.info("="*80)

    # Run training
    start_time = time.time()
    results = train_all_models(
        architectures=args.architectures,
        sizes=args.sizes,
        epochs=args.epochs,
        parallel=args.parallel
    )
    end_time = time.time()

    # Log results
    logger.info("="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")

    successful = sum(results.values())
    total = len(results)

    logger.info(f"Results: {successful}/{total} models trained successfully")

    for model, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {model}: {status}")

    # Write summary file
    with open("output/full_training_from_scratch_summary.txt", 'w') as f:
        f.write("Model Training Summary\n")
        f.write("====================\n\n")
        f.write(f"Total models: {total}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {total - successful}\n")
        f.write(f"Total time: {end_time - start_time:.2f} seconds\n\n")

        f.write("Detailed Results:\n")
        for model, success in results.items():
            f.write(f"  {model}: {'SUCCESS' if success else 'FAILED'}\n")

    logger.info("Summary written to output/full_training_from_scratch_summary.txt")

    # Exit with appropriate code
    sys.exit(0 if successful == total else 1)

if __name__ == "__main__":
    main()
