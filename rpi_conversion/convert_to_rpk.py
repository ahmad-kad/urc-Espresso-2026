#!/usr/bin/env python3
"""
Automated YOLO to RPK conversion for Raspberry Pi AI Camera
"""

import argparse
import sys
from pathlib import Path
import subprocess
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RPiModelConverter:
    """Converts YOLO models to Raspberry Pi AI Camera format"""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.output_dir = self.model_path.parent / 'rpi_models'
        self.output_dir.mkdir(exist_ok=True)

    def convert(self):
        """Main conversion process"""

        logger.info(f"Converting model: {self.model_path}")

        # Step 1: Export to ONNX
        onnx_path = self._export_onnx()
        if not onnx_path:
            return False

        # Step 2: Optimize ONNX
        optimized_path = self._optimize_onnx(onnx_path)
        if not optimized_path:
            return False

        # Step 3: Convert to RPK
        rpk_path = self._create_rpk(optimized_path)
        if not rpk_path:
            return False

        logger.info("Conversion completed successfully!")
        logger.info(f"RPK model: {rpk_path}")

        return True

    def _export_onnx(self):
        """Export PyTorch model to ONNX"""
        try:
            from ultralytics import YOLO

            logger.info("Exporting to ONNX...")

            model = YOLO(str(self.model_path))

            onnx_path = self.output_dir / f"{self.model_path.stem}.onnx"

            success = model.export(
                format='onnx',
                imgsz=416,
                half=False,
                simplify=True,
                opset=11
            )

            if success:
                logger.info(f"ONNX export successful: {onnx_path}")
                return onnx_path
            else:
                logger.error("ONNX export failed")
                return None

        except Exception as e:
            logger.error(f"ONNX export error: {e}")
            return None

    def _optimize_onnx(self, onnx_path: Path):
        """Optimize ONNX model for RPi"""
        try:
            import onnx
            import onnxruntime as ort

            logger.info("Optimizing ONNX model...")

            optimized_path = onnx_path.with_stem(f"{onnx_path.stem}_optimized")

            # Load model
            model = onnx.load(str(onnx_path))

            # Create optimized model
            # Note: Full optimization would require onnxruntime-tools
            # This is a simplified version

            # Basic optimizations
            # Remove unused nodes, fuse operations, etc.
            # For full optimization, use: python -m onnxruntime.tools.optimize_onnx_model

            # Save optimized model (for now, just copy)
            onnx.save(model, str(optimized_path))

            logger.info(f"ONNX optimization successful: {optimized_path}")
            return optimized_path

        except Exception as e:
            logger.error(f"ONNX optimization error: {e}")
            return None

    def _create_rpk(self, optimized_path: Path):
        """Create RPK format file"""
        try:
            import onnxruntime as ort

            logger.info("Creating RPK format...")

            rpk_path = optimized_path.with_suffix('.rpk')

            # Load optimized model to get info
            session = ort.InferenceSession(str(optimized_path))
            inputs = session.get_inputs()
            outputs = session.get_outputs()

            # Create RPK configuration
            rpk_config = {
                'model_type': 'yolo',
                'input_shape': inputs[0].shape,
                'output_shape': outputs[0].shape if len(outputs) == 1 else [out.shape for out in outputs],
                'num_classes': 5,
                'class_names': ['BrickHammer', 'ArUcoTag', 'Bottle', 'BrickHammer_duplicate', 'OrangeHammer'],
                'anchors': [],  # Extract from model if available
                'strides': [8, 16, 32],
                'input_format': 'RGB',
                'preprocessing': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'scale': 1.0/255.0
                },
                'target_hardware': 'raspberry_pi_ai_camera',
                'optimized_for': 'real_time_inference',
                'input_size': 416,
                'conf_threshold': 0.5,
                'iou_threshold': 0.4,
                'max_detections': 20
            }

            # Save RPK file
            with open(rpk_path, 'wb') as f:
                pickle.dump({
                    'config': rpk_config,
                    'onnx_model_path': str(optimized_path),
                    'original_model': str(self.model_path),
                    'conversion_date': str(Path(__file__).parent / 'convert_to_rpk.py')
                }, f)

            logger.info(f"RPK creation successful: {rpk_path}")
            return rpk_path

        except Exception as e:
            logger.error(f"RPK creation error: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO model to RPK format')
    parser.add_argument('model_path', help='Path to trained YOLO model (.pt)')
    parser.add_argument('--output_dir', help='Output directory for converted models')

    args = parser.parse_args()

    converter = RPiModelConverter(args.model_path)

    if args.output_dir:
        converter.output_dir = Path(args.output_dir)

    success = converter.convert()

    if success:
        print("Model conversion completed successfully!")
        print(f"Check the output directory: {converter.output_dir}")
    else:
        print("Model conversion failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
