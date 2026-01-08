#!/usr/bin/env python3
"""
Convert trained PyTorch models to ONNX format for deployment
Supports YOLOv8 models
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.onnx
from ultralytics import YOLO  # type: ignore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_onnx_to_int8(fp32_model_path: str, output_path: str) -> str:
    """Convert ONNX FP32 model to INT8 using onnxruntime quantization"""
    logger.info(f"Converting ONNX FP32 to INT8: {fp32_model_path}")

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        logger.info("Quantizing ONNX model to INT8...")

        # Try dynamic quantization first (more compatible)
        try:
            quantize_dynamic(
                model_input=fp32_model_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
                # Use more compatible quantization settings
                per_channel=False,  # Use per-tensor quantization for compatibility
            )
            logger.info("Dynamic quantization successful")
        except Exception as dynamic_error:
            logger.warning(
                f"Dynamic quantization failed ({dynamic_error}), trying static quantization..."
            )

            # Fallback to static quantization if dynamic fails
            try:
                from onnxruntime.quantization import (
                    CalibrationDataReader,
                    quantize_static,
                )

                # Create a simple calibration data reader
                class SimpleCalibrationDataReader(CalibrationDataReader):
                    def __init__(self, model_path, num_samples=10):
                        super().__init__()
                        self.model_path = model_path
                        self.num_samples = num_samples
                        self.current_sample = 0

                    def get_next(self):
                        if self.current_sample >= self.num_samples:
                            return None

                        # Generate dummy calibration data
                        import numpy as np

                        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
                        self.current_sample += 1
                        return {"images": dummy_input}

                # Use static quantization
                calibrate_reader = SimpleCalibrationDataReader(fp32_model_path)
                quantize_static(
                    model_input=fp32_model_path,
                    model_output=output_path,
                    calibration_data_reader=calibrate_reader,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8,
                    per_channel=False,
                    reduce_range=False,
                )
                logger.info("Static quantization successful")
            except Exception as static_error:
                logger.error(
                    f"Both dynamic and static quantization failed. Dynamic: {dynamic_error}, Static: {static_error}"
                )
                raise Exception(f"Quantization failed: {static_error}")

        if Path(output_path).exists():
            int8_size = Path(output_path).stat().st_size / (1024 * 1024)
            fp32_size = Path(fp32_model_path).stat().st_size / (1024 * 1024)
            size_reduction = ((fp32_size - int8_size) / fp32_size) * 100

            logger.info(f"INT8 ONNX model saved to: {output_path} ({int8_size:.2f} MB)")
            logger.info(
                f"Size reduction: {size_reduction:.1f}% (FP32: {fp32_size:.2f} MB)"
            )
            return output_path
        else:
            raise Exception(f"INT8 quantization failed: {output_path} not created")

    except ImportError:
        logger.warning(
            "onnxruntime not available for INT8 quantization. Install with: pip install onnxruntime"
        )
        raise ImportError("onnxruntime required for INT8 quantization")
    except Exception as e:
        logger.error(f"INT8 quantization failed: {e}")
        raise


def convert_yolo_to_onnx(
    model_path: str,
    output_path: str,
    input_size: int = 416,
    opset_version: int = 15,  # Changed from 11 to 15
):
    """Convert YOLO model to ONNX format compatible with IMX500"""
    logger.info(f"Converting YOLO model for IMX500: {model_path}")

    model = YOLO(model_path)

    try:
        # IMX500 requires specific export settings
        export_result = model.export(
            format="onnx",
            imgsz=input_size,
            opset=15,  # IMX500 supports opset 15-20, use 15 for compatibility
            simplify=False,  # Don't simplify - IMX500 needs full structure
            dynamic=False,  # MUST be static (no dynamic batch/shape)
            half=False,  # Keep FP32 (MCT will quantize later)
        )

        # YOLO export returns the path to the exported model
        if export_result:
            # Check if YOLO saved directly to our desired output path
            if Path(output_path).exists():
                logger.info(
                    f"YOLO model exported with raw outputs and saved to: {output_path}"
                )
                return output_path

            # Otherwise, find the exported file (YOLO puts it in the same directory as the model)
            exported_path = Path(model_path).parent / f"{Path(model_path).stem}.onnx"
            if exported_path.exists():
                import shutil

                # Ensure destination directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(exported_path), output_path)
                logger.info(
                    f"YOLO model exported with raw outputs and saved to: {output_path}"
                )
                return output_path
            else:
                # Check if the export_result is actually the path
                if isinstance(export_result, str) and Path(export_result).exists():
                    shutil.move(export_result, output_path)
                    logger.info(
                        f"YOLO model exported with raw outputs and saved to: {output_path}"
                    )
                    return output_path
                raise Exception(
                    f"YOLO export completed but expected file not found at {exported_path} or {export_result}"
                )
        else:
            raise Exception("YOLO export returned None")

    except Exception as e:
        logger.warning(f"YOLO export failed ({e}), trying alternative approach")

        # Alternative: Try to export just the model backbone without detection head
        try:
            if model is None:
                raise Exception("Model failed to load")

            # Get the model without the detection head (just the backbone)
            if hasattr(model, "model") and hasattr(model.model, "model"):
                backbone_model = model.model.model  # The actual neural network
            elif hasattr(model, "model"):
                backbone_model = model.model  # Fallback for different model structures
            else:
                raise Exception(
                    "Model structure not compatible with backbone extraction"
                )

            # Create dummy input
            dummy_input = torch.randn(1, 3, input_size, input_size)

            # Set to eval mode
            if hasattr(backbone_model, "eval"):
                backbone_model.eval()

            # Export just the backbone
            torch.onnx.export(
                backbone_model,
                (dummy_input,),
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
    """Detect the model type based on the path, model name, or model inspection"""
    path_obj = Path(model_path)
    full_path = str(path_obj).lower()

    # Check path/name first
    if "yolo" in full_path:
        return "yolo"

    # Try to inspect the model file to detect YOLO models
    try:
        # Load the model state dict to check for YOLO-specific keys
        if model_path.endswith(".pt"):
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict):
                # Check for YOLO-specific keys in state dict
                state_keys = set(state_dict.keys())
                yolo_indicators = [
                    "model",  # YOLO models have a 'model' key
                    "epoch",
                    "best_fitness",
                    "date",  # YOLO training metadata
                ]
                if any(key in state_keys for key in yolo_indicators):
                    return "yolo"

                # Check if the model dict contains YOLO architecture markers
                if "model" in state_dict and isinstance(state_dict["model"], dict):
                    model_dict = state_dict["model"]
                    if "type" in model_dict and model_dict["type"] == "yolov8":
                        return "yolo"

            # Try loading as YOLO model directly
            try:
                from ultralytics import YOLO

                model = YOLO(model_path)
                # If we can load it with YOLO, it's a YOLO model
                return "yolo"
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Could not inspect model file: {e}")

    # Default fallback
    return "unknown"


def convert_model(
    model_path: str,
    output_path: Optional[str] = None,
    model_type: Optional[str] = None,
    input_size: int = 416,
    opset_versions: Optional[list] = None,
):
    """Convert a model to ONNX format, trying multiple opset versions if needed"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_type is None:
        model_type = detect_model_type(model_path)

    if output_path is None:
        model_name = Path(model_path).stem
        output_path = f"output/onnx/{model_name}.onnx"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if opset_versions is None:
        opset_versions = [11, 13, 17, 9]

    logger.info(f"Converting {model_type} model from {model_path} to {output_path}")

    success = False
    last_error = None

    for opset_version in opset_versions:
        try:
            logger.info(f"Attempting conversion with opset {opset_version}")

            if model_type == "yolo":
                convert_yolo_to_onnx(model_path, output_path, input_size, opset_version)
            else:
                raise ValueError(f"Unsupported model type: {model_type}.")

            if Path(output_path).exists():
                file_size = Path(output_path).stat().st_size / (1024 * 1024)
                if file_size > 1:
                    logger.info(
                        f"Converted successfully with opset {opset_version}. Size: {file_size:.2f} MB"
                    )
                    success = True
                    break
                else:
                    logger.warning(
                        f"File too small ({file_size:.2f} MB), trying next opset"
                    )
                    Path(output_path).unlink()
        except Exception as e:
            last_error = e
            logger.warning(f"Failed with opset {opset_version}: {e}")
            if Path(output_path).exists():
                Path(output_path).unlink()

    if not success:
        raise RuntimeError(f"Conversion failed. Last error: {last_error}")


def batch_convert_models(
    models_dir: str = "output/models", output_dir: str = "output/onnx"
):
    """Convert all models in the output/models directory to ONNX"""
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Directory not found: {models_dir}")

    converted_count = 0
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            best_model_path = model_dir / "weights" / "best.pt"
            if best_model_path.exists():
                model_name = model_dir.name
                output_path = Path(output_dir) / f"{model_name}.onnx"
                input_size = 416
                for size in [160, 192, 224, 320]:
                    if str(size) in model_name:
                        input_size = size
                        break

                try:
                    convert_model(
                        str(best_model_path), str(output_path), input_size=input_size
                    )
                    converted_count += 1
                except Exception as e:
                    logger.error(f"Failed to convert {model_name}: {e}")

    logger.info(f"Successfully converted {converted_count} models")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to ONNX format"
    )
    parser.add_argument("--model_path", type=str, help="Path to PyTorch model file")
    parser.add_argument("--output_path", type=str, help="Path for ONNX output file")
    parser.add_argument("--model_type", type=str, choices=["yolo"], help="Model type")
    parser.add_argument("--input_size", type=int, default=416, help="Model input size")
    parser.add_argument(
        "--batch_convert", action="store_true", help="Batch convert mode"
    )
    parser.add_argument("--models_dir", type=str, default="output/models")
    parser.add_argument("--output_dir", type=str, default="output/onnx")
    parser.add_argument(
        "--int8", action="store_true", help="Create INT8 quantized version"
    )

    args = parser.parse_args()

    try:
        if args.batch_convert:
            batch_convert_models(args.models_dir, args.output_dir)
        elif args.model_path:
            fp32_output = (
                args.output_path or f"output/onnx/{Path(args.model_path).stem}.onnx"
            )
            convert_model(
                args.model_path, fp32_output, args.model_type, args.input_size
            )

            if args.int8:
                fp32_path = Path(fp32_output)
                suffix = "_int8.onnx"
                if fp32_path.stem.endswith("_fp32"):
                    int8_output = str(
                        fp32_path.with_name(f"{fp32_path.stem[:-5]}{suffix}")
                    )
                else:
                    int8_output = str(fp32_path.with_name(f"{fp32_path.stem}{suffix}"))
                convert_onnx_to_int8(fp32_output, int8_output)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
