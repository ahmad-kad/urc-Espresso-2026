#!/usr/bin/env python3
"""
Convert trained PyTorch models to ONNX format for deployment
Supports YOLOv8 models
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.onnx
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_yolo_to_onnx(
    model_path: str, output_path: str, input_size: int = 416, opset_version: int = 11
):
    """Convert YOLO model to ONNX format using YOLO's export with raw outputs"""
    logger.info(f"Converting YOLO model: {model_path} with raw output export")

    # Load the YOLO model
    model = YOLO(model_path)

    # Use YOLO's export method but try to get raw model outputs
    try:
        # Export using YOLO's export method with parameters to get raw outputs
        export_result = model.export(
            format="onnx",
            imgsz=input_size,
            opset=opset_version,
            simplify=False,  # Don't simplify to preserve structure
            dynamic=False,  # Static batch size
        )

        # YOLO export returns the path to the exported model
        if export_result:
            # Find the exported file (YOLO puts it in the same directory as the model)
            exported_path = Path(model_path).parent / f"{Path(model_path).stem}.onnx"
            if exported_path.exists():
                import shutil

                shutil.move(str(exported_path), output_path)
                logger.info(
                    f"YOLO model exported with raw outputs and saved to: {output_path}"
                )
            else:
                raise Exception("YOLO export completed but expected file not found")
        else:
            raise Exception("YOLO export returned None")

    except Exception as e:
        logger.warning(f"YOLO export failed ({e}), trying alternative approach")

        # Alternative: Try to export just the model backbone without detection head
        try:
            # Get the model without the detection head (just the backbone)
            backbone_model = model.model.model  # The actual neural network

            # Create dummy input
            dummy_input = torch.randn(1, 3, input_size, input_size)

            # Set to eval mode
            backbone_model.eval()

            # Export just the backbone
            torch.onnx.export(
                backbone_model,
                dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=["images"],
                output_names=["features"],
                dynamic_axes={
                    "images": {0: "batch_size"},
                    "features": {0: "batch_size"},
                },
                verbose=False,
            )

            logger.info(
                f"YOLO backbone exported (no detection head) and saved to: {output_path}"
            )

        except Exception as e2:
            logger.error(
                f"Both YOLO export methods failed. YOLO export: {e}, Backbone export: {e2}"
            )
            raise Exception(f"All export methods failed. YOLO: {e}, Backbone: {e2}")


def detect_model_type(model_path: str) -> str:
    """Detect the model type based on the path or model name"""
    path_obj = Path(model_path)
    model_name = path_obj.name.lower()

    # Check the full path for model type keywords
    full_path = str(path_obj).lower()

    # All models are YOLO models
    if "yolo" in full_path:
        return "yolo"
    else:
        return "unknown"


def convert_model(
    model_path: str,
    output_path: str = None,
    model_type: str = None,
    input_size: int = 416,
    opset_versions: list = None,
):
    """Convert a model to ONNX format, trying multiple opset versions if needed"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Auto-detect model type if not specified
    if model_type is None:
        model_type = detect_model_type(model_path)

    # Generate output path if not specified
    if output_path is None:
        model_name = Path(model_path).stem
        output_path = f"output/onnx/{model_name}.onnx"

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Try multiple opset versions if not specified
    if opset_versions is None:
        opset_versions = [11, 13, 17, 9]  # Try common versions

    logger.info(f"Converting {model_type} model from {model_path} to {output_path}")
    logger.info(f"Trying opset versions: {opset_versions}")

    success = False
    last_error = None

    for opset_version in opset_versions:
        try:
            logger.info(f"Attempting conversion with opset {opset_version}")

            # Convert based on model type
            if model_type == "yolo":
                convert_yolo_to_onnx(model_path, output_path, input_size, opset_version)
            else:
                raise ValueError(
                    f"Unsupported model type: {model_type}. Only YOLO models are supported."
                )

            # Verify the ONNX file was created and has reasonable size
            if Path(output_path).exists():
                file_size = Path(output_path).stat().st_size / (
                    1024 * 1024
                )  # Size in MB
                if file_size > 1:  # Must be at least 1MB to be valid
                    logger.info(
                        f"ONNX model converted successfully with opset {opset_version}. Size: {file_size:.2f} MB"
                    )
                    success = True
                    break
                else:
                    logger.warning(
                        f"ONNX file created but too small ({file_size:.2f} MB), trying next opset"
                    )
                    Path(output_path).unlink()  # Remove invalid file
            else:
                logger.warning(f"ONNX file not created with opset {opset_version}")

        except Exception as e:
            last_error = e
            logger.warning(f"Failed with opset {opset_version}: {e}")
            # Clean up failed file if it exists
            if Path(output_path).exists():
                Path(output_path).unlink()

    if not success:
        error_msg = f"ONNX conversion failed for all opset versions {opset_versions}"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise RuntimeError(error_msg)


def batch_convert_models(
    models_dir: str = "output/models", output_dir: str = "output/onnx"
):
    """Convert all models in the output/models directory to ONNX"""
    models_path = Path(models_dir)

    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    logger.info(f"Batch converting models from {models_dir} to {output_dir}")

    converted_count = 0

    # Find all best.pt files
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            best_model_path = model_dir / "weights" / "best.pt"
            if best_model_path.exists():
                model_name = model_dir.name
                output_path = Path(output_dir) / f"{model_name}.onnx"

                # Extract correct input size from model name
                input_size = 416  # default
                for size in [160, 192, 224, 320]:
                    if str(size) in model_name:
                        input_size = size
                        break

                logger.info(f"Converting {model_name} with input size {input_size}")

                try:
                    convert_model(
                        str(best_model_path),
                        str(output_path),
                        input_size=input_size,
                        opset_versions=[11, 13, 17, 9],
                    )
                    converted_count += 1
                except Exception as e:
                    logger.error(f"Failed to convert {model_name}: {e}")
                    continue

    logger.info(f"Successfully converted {converted_count} models to ONNX format")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to ONNX format"
    )
    parser.add_argument("--model_path", type=str, help="Path to PyTorch model file")
    parser.add_argument("--output_path", type=str, help="Path for ONNX output file")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["yolo"],
        help="Model type (auto-detected if not specified, only YOLO supported)",
    )
    parser.add_argument("--input_size", type=int, default=416, help="Model input size")
    parser.add_argument(
        "--batch_convert",
        action="store_true",
        help="Convert all models in output/models directory",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="output/models",
        help="Directory containing models for batch conversion",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/onnx",
        help="Output directory for ONNX files",
    )

    args = parser.parse_args()

    try:
        if args.batch_convert:
            batch_convert_models(args.models_dir, args.output_dir)
        elif args.model_path:
            convert_model(
                args.model_path, args.output_path, args.model_type, args.input_size
            )
        else:
            print("Usage:")
            print(
                "  Convert single model: python convert_to_onnx.py --model_path path/to/model.pt"
            )
            print(
                "  Batch convert all models: python convert_to_onnx.py --batch_convert"
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
