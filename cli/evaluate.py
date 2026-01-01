#!/usr/bin/env python3
"""
Unified Evaluation CLI
Provides a single entry point for all evaluation operations
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.evaluation.pipeline import EvaluationPipeline
from core.config.manager import load_config
from utils.logger_config import get_logger, setup_logging

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation CLI"""
    parser = argparse.ArgumentParser(
        description="Unified YOLO Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model
  python cli/evaluate.py --model output/models/best.pt

  # Comprehensive evaluation with benchmarking
  python cli/evaluate.py --model output/models/best.pt --benchmark

  # Compare multiple models
  python cli/evaluate.py --models output/models/model1.pt output/models/model2.pt

  # Evaluate with custom config
  python cli/evaluate.py --model model.pt --config evaluation
  """,
    )

    parser.add_argument("--model", type=str, help="Path to PyTorch model file")

    parser.add_argument(
        "--models", nargs="+", help="Paths to multiple model files for comparison"
    )

    parser.add_argument(
        "--data-yaml",
        type=str,
        default="consolidated_dataset/data.yaml",
        help="Path to data YAML file",
    )

    parser.add_argument(
        "--config", type=str, default="default", help="Configuration to use"
    )

    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    # Benchmarking options
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comprehensive benchmarking (accuracy, speed, memory)",
    )

    parser.add_argument(
        "--accuracy-only", action="store_true", help="Run accuracy evaluation only"
    )

    parser.add_argument(
        "--speed-only", action="store_true", help="Run speed benchmarking only"
    )

    parser.add_argument(
        "--memory-only", action="store_true", help="Run memory profiling only"
    )

    # Evaluation parameters
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )

    parser.add_argument("--device", type=str, help="Device to use for evaluation")

    parser.add_argument(
        "--num-runs", type=int, default=100, help="Number of runs for benchmarking"
    )

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running evaluation",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser


def main():
    """Main entry point for evaluation CLI"""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, debug=(log_level == "DEBUG"))
    global logger
    logger = get_logger(__name__, debug=(log_level == "DEBUG"))

    # Validate arguments
    if not args.model and not args.models:
        logger.error("Must specify either --model or --models")
        return 1

    if args.model and args.models:
        logger.error("Cannot specify both --model and --models")
        return 1

    try:
        logger.info("Starting YOLO Evaluation Pipeline")

        # Determine models to evaluate
        if args.model:
            models_to_evaluate = [args.model]
        else:
            models_to_evaluate = args.models

        # Load configuration
        logger.info(f"Loading configuration: {args.config}")
        config = load_config(args.config)

        # Override config with command line arguments
        if args.device:
            config["device"] = args.device

        # Determine evaluation types
        if args.accuracy_only:
            eval_types = ["accuracy"]
        elif args.speed_only:
            eval_types = ["speed"]
        elif args.memory_only:
            eval_types = ["memory"]
        elif args.benchmark:
            eval_types = ["accuracy", "speed", "memory"]
        else:
            eval_types = ["accuracy"]  # Default to accuracy only

        # Create output directory
        output_dir = (
            Path(args.output_dir) if args.output_dir else Path("output/evaluation")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create evaluation pipeline
        from pipeline.base import PipelineConfig

        pipeline_config = PipelineConfig(
            name="evaluation_pipeline", output_dir=output_dir, dry_run=args.dry_run
        )

        pipeline = EvaluationPipeline(pipeline_config, args.data_yaml)

        # Set evaluation parameters
        pipeline.set_evaluation_types(eval_types)
        pipeline.set_models(models_to_evaluate)
        pipeline.set_batch_size(args.batch_size)
        pipeline.set_num_runs(args.num_runs)

        # Validate configuration
        if not pipeline.validate_config():
            logger.error(" Configuration validation failed")
            return 1

        if args.dry_run:
            logger.info("Dry run completed - configuration is valid")
            return 0

        # Run evaluation
        result = pipeline.run()

        # Report results
        if result.success:
            logger.info("Evaluation completed successfully!")
            logger.info(f"Duration: {result.duration:.2f}s")
            logger.info(f"Results saved to: {output_dir}")

            # Log key metrics
            if result.metrics:
                logger.info("Evaluation Summary:")
                for key, value in result.metrics.items():
                    logger.info(f" {key}: {value}")
        else:
            logger.error(" Evaluation failed!")
            for error in result.errors:
                logger.error(f" {error}")
            return 1

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 130

    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())