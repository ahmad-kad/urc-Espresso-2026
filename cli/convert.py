#!/usr/bin/env python3
"""
Unified Model Conversion CLI
Provides a single entry point for all model conversion operations
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.conversion.pipeline import ConversionPipeline
from core.config.manager import load_config
from utils.logger_config import get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for conversion CLI"""
    parser = argparse.ArgumentParser(
        description="Unified YOLO Model Conversion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single model to ONNX
  python cli/convert.py --model output/models/best.pt --format onnx

  # Convert to multiple formats
  python cli/convert.py --model model.pt --formats onnx int8

  # Batch convert all models
  python cli/convert.py --batch --input-dir output/models --output-dir output/converted

  # Convert with optimization
  python cli/convert.py --model model.pt --optimize --quantize
""",
    )

    parser.add_argument("--model", type=str, help="Path to input model file")

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch convert all models in input directory",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="output/models",
        help="Input directory for batch conversion",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/converted",
        help="Output directory for converted models",
    )

    # Conversion options
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "tensorrt", "openvino", "tflite"],
        default="onnx",
        help="Target format for conversion",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["onnx", "tensorrt", "openvino", "tflite"],
        help="Multiple target formats",
    )

    parser.add_argument(
        "--quantize", action="store_true", help="Apply quantization (INT8)"
    )

    parser.add_argument(
        "--optimize", action="store_true", help="Apply graph optimizations"
    )

    # Model parameters
    parser.add_argument("--input-size", type=int, default=416, help="Model input size")

    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="Target precision",
    )

    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running conversion",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser


def main():
    """Main entry point for conversion CLI"""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    global logger
    logger = get_logger(__name__, debug=(log_level == "DEBUG"))

    # Validate arguments
    if not args.model and not args.batch:
        logger.error(
            " Must specify either --model for single conversion or --batch for batch conversion"
        )
        return 1

    if args.model and args.batch:
        logger.error(" Cannot specify both --model and --batch")
        return 1

    # If quantization is requested, automatically set precision to int8
    if args.quantize and args.precision == "fp32":
        args.precision = "int8"

    # Determine target formats
    if args.formats:
        target_formats = args.formats
    else:
        target_formats = [args.format]

    try:
        logger.info("Starting YOLO Model Conversion Pipeline")

        # Load configuration
        config = load_config("default")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create conversion pipeline
        from pipeline.base import PipelineConfig

        pipeline_config = PipelineConfig(
            name="conversion_pipeline", output_dir=output_dir, dry_run=args.dry_run
        )

        pipeline = ConversionPipeline(pipeline_config)

        # Configure conversion options
        pipeline.set_target_formats(target_formats)
        pipeline.set_input_size(args.input_size)
        pipeline.set_precision(args.precision)
        pipeline.set_opset_version(args.opset)
        pipeline.set_quantization(args.quantize)
        pipeline.set_optimization(args.optimize)

        if args.batch:
            # Batch conversion
            input_dir = Path(args.input_dir)
            if not input_dir.exists():
                logger.error(f" Input directory not found: {input_dir}")
                return 1

            pipeline.set_batch_mode(True)
            pipeline.set_input_directory(input_dir)

        else:
            # Single model conversion
            model_path = Path(args.model)
            if not model_path.exists():
                logger.error(f" Model file not found: {model_path}")
                return 1

            pipeline.set_input_model(model_path)

        # Validate configuration
        if not pipeline.validate_config():
            logger.error(" Configuration validation failed")
            return 1

        if args.dry_run:
            logger.info("Dry run completed - configuration is valid")
            return 0

        # Run conversion
        result = pipeline.run()

        # Report results
        if result.success:
            logger.info("Model conversion completed successfully!")
            logger.info(f"Duration: {result.duration:.2f}s")
            logger.info(f"Converted models saved to: {output_dir}")

            # Log conversion results
            if result.artifacts:
                logger.info("Converted models:")
                for artifact in result.artifacts:
                    logger.info(f"  {artifact}")

            if result.metrics:
                logger.info("Conversion metrics:")
                for key, value in result.metrics.items():
                    logger.info(f"  {key}: {value}")

        else:
            logger.error(" Model conversion failed!")
            for error in result.errors:
                logger.error(f"  {error}")
            return 1

    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        return 130

    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())