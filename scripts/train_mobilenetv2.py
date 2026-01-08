#!/usr/bin/env python3
"""
Train MobileNetV2 for IMX500 compatibility on consolidated dataset
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config.manager import load_config
from core.trainer import ModelTrainer
from utils.logger_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for IMX500")
    parser.add_argument(
        "--config",
        default="mobilenetv2_classification",
        help="Configuration name (without .yaml extension)",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line args
    if "training" not in config:
        config["training"] = {}
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.lr

    logger.info("Starting MobileNetV2 training for IMX500 compatibility...")
    logger.info(f"Config: {args.config}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    # Create trainer
    trainer = ModelTrainer(config)

    # Train the model
    results = trainer.train(
        data_yaml="consolidated_dataset/data.yaml", experiment_name="mobilenetv2_imx500"
    )

    if results["success"]:
        logger.info("✅ Training completed successfully!")
        logger.info(f"Best model saved at: {results['model_path']}")

        # Instructions for next steps
        logger.info("\n" + "=" * 60)
        logger.info("NEXT STEPS:")
        logger.info("1. Test the trained model:")
        logger.info(
            f"   python scripts/test_classification_model.py --model {results['model_path']}"
        )
        logger.info("")
        logger.info("2. Quantize for IMX500:")
        logger.info(
            f"   python scripts/convert_to_imx500.py --model {results['model_path']}"
        )
        logger.info("")
        logger.info("3. Deploy to IMX500 camera")
        logger.info("=" * 60)

    else:
        logger.error("❌ Training failed!")
        logger.error(f"Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
