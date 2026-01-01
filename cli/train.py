#!/usr/bin/env python3
"""
Unified Training CLI
Provides a single entry point for all training operations
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.training.pipeline import TrainingPipeline
from core.config.manager import load_config
from utils.logger_config import get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
 """Create argument parser for training CLI"""
 parser = argparse.ArgumentParser(
 description="Unified YOLO Training CLI",
 formatter_class=argparse.RawDescriptionHelpFormatter,
 epilog="""
Examples:
 # Train with default config
 python cli/train.py

 # Train with custom config
 python cli/train.py --config accuracy

 # Resume training
 python cli/train.py --resume --model-path output/models/best.pt

 # Train with custom parameters
 python cli/train.py --epochs 100 --batch-size 16 --model yolov8s
 """,
 )

 parser.add_argument(
 "--config",
 type=str,
 default="default",
 help="Configuration to use (default: default)",
 )

 parser.add_argument(
 "--data-yaml",
 type=str,
 default="consolidated_dataset/data.yaml",
 help="Path to data YAML file",
 )

 parser.add_argument("--output-dir", type=str, help="Output directory for results")

 parser.add_argument(
 "--experiment-name", type=str, help="Name for this training experiment"
 )

 # Training parameters
 parser.add_argument(
 "--model",
 type=str,
 choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
 help="YOLO model architecture",
 )

 parser.add_argument("--epochs", type=int, help="Number of training epochs")

 parser.add_argument("--batch-size", type=int, help="Batch size for training")

 parser.add_argument("--imgsz", type=int, help="Input image size")

 parser.add_argument("--lr", type=float, help="Learning rate")

 parser.add_argument("--device", type=str, help="Device to use for training")

 # Resume options
 parser.add_argument(
 "--resume", action="store_true", help="Resume training from checkpoint"
 )

 parser.add_argument(
 "--resume-path", type=str, help="Path to checkpoint to resume from"
 )

 # Other options
 parser.add_argument(
 "--dry-run",
 action="store_true",
 help="Validate configuration without running training",
 )

 parser.add_argument(
 "--verbose", "-v", action="store_true", help="Enable verbose logging"
 )

 return parser


def main():
 """Main entry point for training CLI"""
 parser = create_parser()
 args = parser.parse_args()

 # Set up logging
 log_level = "DEBUG" if args.verbose else "INFO"
 global logger
 logger = get_logger(__name__, debug=(log_level == "DEBUG"))

 try:
    logger.info("Starting YOLO Training Pipeline")

    # Load configuration
    logger.info(f"Loading configuration: {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.model:
        config.setdefault("model", {})["architecture"] = args.model
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.imgsz:
        config.setdefault("model", {})["input_size"] = args.imgsz
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.device:
        config["device"] = args.device

    # Handle resume options
    if args.resume:
        config.setdefault("pipeline", {})["resume_training"] = True
    if args.resume_path:
        config["pipeline"]["checkpoint_path"] = args.resume_path

    # Create output directory
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path("output/training")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create training pipeline
    from pipeline.base import PipelineConfig

    pipeline_config = PipelineConfig(
        name=f"training_{args.experiment_name or 'default'}",
        output_dir=output_dir,
        dry_run=args.dry_run,
    )

    pipeline = TrainingPipeline(pipeline_config, args.data_yaml)

    # Validate configuration
    if not pipeline.validate_config():
        logger.error("Configuration validation failed")
        return 1

    if args.dry_run:
        logger.info("Dry run completed - configuration is valid")
        return 0

    # Run training
    result = pipeline.run()

    # Report results
    if result.success:
        logger.info("Training completed successfully!")
        logger.info(f"Duration: {result.duration:.2f}s")
        logger.info(f"Results saved to: {output_dir}")

        # Log key metrics
        if result.metrics:
            logger.info("Key Metrics:")
            for key, value in result.metrics.items():
                logger.info(f"  {key}: {value}")

    else:
        logger.error("Training failed!")
        for error in result.errors:
            logger.error(f"  {error}")
        return 1

 except KeyboardInterrupt:
    logger.info("Training interrupted by user")
    return 130

 except Exception as e:
    logger.error(f"Unexpected error: {e}")
    import traceback

    traceback.print_exc()
    return 1

 return 0


if __name__ == "__main__":
 sys.exit(main())
