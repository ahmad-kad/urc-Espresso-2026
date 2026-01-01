#!/usr/bin/env python3
"""
Accuracy-Focused YOLO Training Script
Complete pipeline for training YOLO models optimized for maximum accuracy
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict
import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger_config import get_logger
from core.config.manager import create_accuracy_training_config
from utils.device_utils import get_device
from scripts.prepare_training_data import create_data_yaml, validate_dataset_structure

logger = get_logger(__name__, debug=os.getenv("DEBUG") == "1")


# Configuration creation moved to utils.config.create_accuracy_training_config


def save_training_config(config: Dict, output_path: str):
    """Save training configuration to YAML file"""
    config_path = Path(output_path) / "training_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Training config saved to: {config_path}")
    return str(config_path)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Accuracy-Focused YOLO Training")
    parser.add_argument("--data-path", required=True, help="Path to dataset directory")
    parser.add_argument(
        "--classes", nargs="+", required=True, help="List of class names"
    )
    parser.add_argument(
        "--model-size",
        default="yolov8m",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLO model size",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        choices=[416, 512, 640, 832, 1024],
        help="Input image size",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--output-dir",
        default="./output/training",
        help="Output directory for training results",
    )

    args = parser.parse_args()

    # Print system information
    print("Starting Accuracy-Focused YOLO Training")
    print("=" * 60)
    device = get_device("auto")
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")

    # Validate dataset
    print("\nValidating dataset structure...")
    if not validate_dataset_structure(args.data_path):
        print("Dataset validation failed. Please prepare your data first:")
        print(
            "   python scripts/prepare_training_data.py --dataset-path",
            args.data_path,
            "--classes",
            " ".join(args.classes),
            "--create-structure --create-yaml",
        )
        sys.exit(1)

    # Create data.yaml if it doesn't exist
    data_yaml_path = Path(args.data_path) / "data.yaml"
    if not data_yaml_path.exists():
        print("\nCreating data.yaml configuration...")
        create_data_yaml(args.data_path, args.classes)

    # Create training configuration
    print("\nCreating accuracy-optimized training configuration...")
    config = create_accuracy_training_config(
        model=args.model_size,
        input_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        flat_format=True,
    )

    # Update data path in config
    config["data"] = str(data_yaml_path.resolve())

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_training_config(config, str(output_dir))

    # Print training summary
    print("\n" + "=" * 60)
    print("ACCURACY TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model:          {args.model_size}")
    print(f"Image Size:     {args.img_size}x{args.img_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Dataset:        {args.data_path}")
    print(f"Classes:        {', '.join(args.classes)}")
    print(f"Output Dir:     {output_dir}")
    print(f"Device:         {config['device']}")
    print()

    # Import ultralytics here to avoid import issues
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print("Failed to import ultralytics. Please install it:")
        print("   pip install ultralytics")
        sys.exit(1)

    # Initialize model
    print("Initializing YOLO model...")
    try:
        model = YOLO(args.model_size)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)

    # Start training
    print("Starting training...")
    print("This may take several hours depending on your dataset size and hardware.")

    try:
        results = model.train(**config)

        print("\nTraining completed successfully!")
        print(f"Results saved to: {output_dir}")

        # Print final metrics
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
            print("\nFinal Training Metrics:")
            print(f"Best mAP50:     {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"Best mAP50-95:  {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
            print(f"Best Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
            print(f"Best Recall:    {metrics.get('metrics/recall(B)', 'N/A'):.4f}")

        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("1. Evaluate model: python scripts/evaluate_per_class_accuracy.py")
        print("2. Convert to ONNX: python scripts/convert_to_onnx.py")
        print("3. Benchmark performance: python scripts/benchmark_models.py")
        print("=" * 60)

    except Exception as e:
        print(f"\nTraining failed: {e}")
        print("Check your data format and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
