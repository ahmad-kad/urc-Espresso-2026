#!/usr/bin/env python3
"""
Convert trained PyTorch models to ONNX format for deployment
Supports YOLOv8, EfficientNet, and MobileNet-ViT models
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.onnx
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_yolo_to_onnx(model_path: str, output_path: str, input_size: int = 416, opset_version: int = 11):
    """Convert YOLO model to ONNX format"""
    logger.info(f"Converting YOLO model: {model_path}")

    # Load the YOLO model
    model = YOLO(model_path)

    # Create dummy input tensor
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Export to ONNX
    model.model.eval()
    torch.onnx.export(
        model.model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    logger.info(f"YOLO model converted and saved to: {output_path}")


def convert_efficientnet_to_onnx(model_path: str, output_path: str, num_classes: int = 6, input_size: int = 416):
    """Convert EfficientNet model to ONNX format"""
    logger.info(f"Converting EfficientNet model: {model_path}")

    try:
        # Add project root to path for imports
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from efficientnet import create_efficientnet_detector

        # Create model
        model = create_efficientnet_detector(num_classes=num_classes)

        # Load trained weights if provided
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint)
            logger.info(f"Loaded trained weights from: {model_path}")
        else:
            logger.warning("No trained weights provided, using randomly initialized model")

        # Set to evaluation mode
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=11,
            input_names=['input'],
            output_names=['cls_logits', 'bbox_preds'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'cls_logits': {0: 'batch_size'},
                         'bbox_preds': {0: 'batch_size'}}
        )

        logger.info(f"EfficientNet model converted and saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to convert EfficientNet model: {e}")
        raise


def convert_mobilenet_to_onnx(model_path: str, output_path: str, num_classes: int = 6, input_size: int = 416):
    """Convert MobileNet-ViT model to ONNX format"""
    logger.info(f"Converting MobileNet-ViT model: {model_path}")

    try:
        # Add project root to path for imports
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from mobilevit import create_mobilevit_detector

        # Create model
        model = create_mobilevit_detector(num_classes=num_classes)

        # Load trained weights if provided
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint)
            logger.info(f"Loaded trained weights from: {model_path}")
        else:
            logger.warning("No trained weights provided, using randomly initialized model")

        # Set to evaluation mode
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=11,
            input_names=['input'],
            output_names=['cls_logits', 'bbox_preds'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'cls_logits': {0: 'batch_size'},
                         'bbox_preds': {0: 'batch_size'}}
        )

        logger.info(f"MobileNet-ViT model converted and saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to convert MobileNet-ViT model: {e}")
        raise


def detect_model_type(model_path: str) -> str:
    """Detect the model type based on the path or model name"""
    path_obj = Path(model_path)
    model_name = path_obj.name.lower()

    # Check the full path for model type keywords
    full_path = str(path_obj).lower()

    # All models are YOLO models since they use YOLO's training framework
    if 'yolo' in full_path or 'efficientnet' in full_path or 'mobilenet' in full_path:
        return 'yolo'
    else:
        return 'unknown'


def convert_model(model_path: str, output_path: str = None, model_type: str = None, input_size: int = 416):
    """Convert a model to ONNX format"""
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

    logger.info(f"Converting {model_type} model from {model_path} to {output_path}")

    # Convert based on model type
    if model_type == 'yolo':
        convert_yolo_to_onnx(model_path, output_path, input_size)
    elif model_type == 'efficientnet':
        convert_efficientnet_to_onnx(model_path, output_path, input_size=input_size)
    elif model_type == 'mobilenet':
        convert_mobilenet_to_onnx(model_path, output_path, input_size=input_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Verify the ONNX file was created
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"ONNX model converted successfully. Size: {file_size:.2f} MB")
    else:
        raise RuntimeError("ONNX conversion failed - output file not found")


def batch_convert_models(models_dir: str = "output/models", output_dir: str = "output/onnx"):
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
                    convert_model(str(best_model_path), str(output_path), input_size=input_size)
                    converted_count += 1
                except Exception as e:
                    logger.error(f"Failed to convert {model_name}: {e}")
                    continue

    logger.info(f"Successfully converted {converted_count} models to ONNX format")


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX format')
    parser.add_argument('--model_path', type=str, help='Path to PyTorch model file')
    parser.add_argument('--output_path', type=str, help='Path for ONNX output file')
    parser.add_argument('--model_type', type=str, choices=['yolo', 'efficientnet', 'mobilenet'],
                       help='Model type (auto-detected if not specified)')
    parser.add_argument('--input_size', type=int, default=416, help='Model input size')
    parser.add_argument('--batch_convert', action='store_true',
                       help='Convert all models in output/models directory')
    parser.add_argument('--models_dir', type=str, default='output/models',
                       help='Directory containing models for batch conversion')
    parser.add_argument('--output_dir', type=str, default='output/onnx',
                       help='Output directory for ONNX files')

    args = parser.parse_args()

    try:
        if args.batch_convert:
            batch_convert_models(args.models_dir, args.output_dir)
        elif args.model_path:
            convert_model(args.model_path, args.output_path, args.model_type, args.input_size)
        else:
            print("Usage:")
            print("  Convert single model: python convert_to_onnx.py --model_path path/to/model.pt")
            print("  Batch convert all models: python convert_to_onnx.py --batch_convert")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
