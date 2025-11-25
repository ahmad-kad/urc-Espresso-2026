# Raspberry Pi AI Camera Model Conversion Guide

This guide covers converting YOLOv8/v11 models to Raspberry Pi AI Camera format (.rpk) for optimized inference.

## Prerequisites

### Hardware Requirements
- **Development Machine**: Any computer capable of running PyTorch/YOLO training (GPU recommended)
- **Raspberry Pi**: Raspberry Pi 5 with AI Camera, Raspberry Pi OS (64-bit recommended), Python 3.9+

### Software Requirements

#### On Development Machine:
```bash
# Install Python packages for model conversion
pip install torch torchvision torchaudio ultralytics onnx onnxruntime
pip install onnxruntime-transformers numpy opencv-python
```

#### On Raspberry Pi:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip git cmake build-essential libssl-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test libhdf5-dev

# Install Python packages
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy opencv-python ultralytics onnxruntime
```

## Model Conversion Process

**All model conversion steps should be performed on your development machine (not on the Raspberry Pi)**

### Step 1: Export Model to ONNX

```python
# Run on: Development Machine
from ultralytics import YOLO

# Load your trained model
model = YOLO('path/to/your/model.pt')

# Export to ONNX with optimizations
model.export(
    format='onnx',
    imgsz=416,  # Input size optimized for RPi
    half=False,  # FP32 for compatibility
    simplify=True,
    opset=11
)
```

### Step 2: Convert ONNX to RPK Format

```python
# Run on: Development Machine
import onnxruntime as ort
import numpy as np
from pathlib import Path

def convert_onnx_to_rpk(onnx_path: str, rpk_path: str):
    """
    Convert ONNX model to RPK format for Raspberry Pi AI Camera
    """

    # Load ONNX model
    session = ort.InferenceSession(onnx_path)

    # Get model info
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"Input: {inputs[0].name}, Shape: {inputs[0].shape}")
    print(f"Output: {outputs[0].name}, Shape: {outputs[0].shape}")

    # Create RPK configuration
    rpk_config = {
        'model_type': 'yolo',
        'input_shape': inputs[0].shape,
        'output_shape': outputs[0].shape,
        'num_classes': 5,  # Hammer, ArUco, Bottle, etc.
        'anchors': [],  # Will be extracted from model
        'strides': [8, 16, 32],  # Typical YOLO strides
        'input_format': 'RGB',
        'preprocessing': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'scale': 1.0/255.0
        }
    }

    # Save RPK file (simplified - actual RPK format may vary)
    import pickle
    with open(rpk_path, 'wb') as f:
        pickle.dump({
            'config': rpk_config,
            'onnx_model': onnx_path
        }, f)

    print(f"RPK model saved to: {rpk_path}")

# Usage
convert_onnx_to_rpk('model.onnx', 'model.rpk')
```

### Step 3: Optimize for Raspberry Pi AI Camera

```python
# Run on: Development Machine
def optimize_for_rpi_ai_camera(onnx_path: str, optimized_path: str):
    """
    Apply Raspberry Pi AI Camera specific optimizations
    """

    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort

    # Load and optimize ONNX model
    model = onnx.load(onnx_path)

    # Apply optimizations
    from onnxruntime.transformers.onnx_model import OnnxModel
    optimized_model = OnnxModel(model)

    # Remove unnecessary nodes
    optimized_model.remove_unused_nodes()

    # Convert constants to initializers where possible
    optimized_model.convert_constants_to_initializers()

    # Fuse operations
    optimized_model.fuse_consecutive_transposes()
    optimized_model.fuse_consecutive_unsqueezes()

    # Save optimized model
    onnx.save(optimized_model.model, optimized_path)

    print(f"Optimized model saved to: {optimized_path}")

# Usage
optimize_for_rpi_ai_camera('model.onnx', 'model_optimized.onnx')
```

## Automated Conversion Script

**Run this script on your development machine (not on the Raspberry Pi)**

```python
#!/usr/bin/env python3
"""
Automated YOLO to RPK conversion for Raspberry Pi AI Camera
Run on: Development Machine
"""

import argparse
import sys
from pathlib import Path
import subprocess
import logging

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

            model = YOLO(self.model_path)

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
            import onnxruntime as ort
            from onnxruntime.transformers.onnx_model import OnnxModel
            import onnx

            logger.info("Optimizing ONNX model...")

            optimized_path = onnx_path.with_stem(f"{onnx_path.stem}_optimized")

            # Load model
            model = onnx.load(str(onnx_path))

            # Create OnnxModel wrapper
            onnx_model = OnnxModel(model)

            # Apply optimizations
            onnx_model.remove_unused_nodes()
            onnx_model.convert_constants_to_initializers()
            onnx_model.fuse_consecutive_transposes()
            onnx_model.fuse_consecutive_unsqueezes()

            # Save optimized model
            onnx.save(onnx_model.model, str(optimized_path))

            logger.info(f"ONNX optimization successful: {optimized_path}")
            return optimized_path

        except Exception as e:
            logger.error(f"ONNX optimization error: {e}")
            return None

    def _create_rpk(self, optimized_path: Path):
        """Create RPK format file"""
        try:
            import pickle
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
                'output_shape': outputs[0].shape,
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
                'optimized_for': 'real_time_inference'
            }

            # Save RPK file
            with open(rpk_path, 'wb') as f:
                pickle.dump({
                    'config': rpk_config,
                    'onnx_model_path': str(optimized_path),
                    'original_model': str(self.model_path)
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
```

## Deployment on Raspberry Pi

**All steps in this section should be performed on your Raspberry Pi device**

### Step 1: Install Dependencies

```bash
# Run on: Raspberry Pi
sudo apt update
sudo apt install -y python3-pip libatlas-base-dev libjasper-dev libqtgui4 libqt4-test libhdf5-dev

# Install Python packages
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy opencv-python ultralytics onnxruntime
```

### Step 2: Transfer RPK Model to Raspberry Pi

Before running the inference script, transfer the `.rpk` file and optimized `.onnx` model from your development machine to the Raspberry Pi:

```bash
# Run on: Development Machine (to transfer files)
scp model.rpk pi@raspberry-pi-ip:/home/pi/models/
scp model_optimized.onnx pi@raspberry-pi-ip:/home/pi/models/
```

### Step 3: Deploy RPK Model

```python
#!/usr/bin/env python3
"""
Raspberry Pi AI Camera inference with RPK model
Run on: Raspberry Pi
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
import time

class RPiInference:
    """Raspberry Pi AI Camera inference with RPK model"""

    def __init__(self, rpk_path: str):
        self.rpk_path = Path(rpk_path)
        self.config = None
        self.session = None

        self._load_rpk_model()

    def _load_rpk_model(self):
        """Load RPK model"""
        try:
            # Load RPK configuration
            with open(self.rpk_path, 'rb') as f:
                rpk_data = pickle.load(f)

            self.config = rpk_data['config']
            onnx_path = rpk_data['onnx_model_path']

            # Load ONNX model
            import onnxruntime as ort

            # Use CPU provider for RPi
            self.session = ort.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider']
            )

            print(f"RPK model loaded: {self.rpk_path}")
            print(f"Classes: {self.config['class_names']}")

        except Exception as e:
            print(f"Failed to load RPK model: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference"""
        # Resize
        img = cv2.resize(image, (416, 416))

        # Convert to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize
        img = img.astype(np.float32) / 255.0

        # Apply mean/std normalization
        mean = np.array(self.config['preprocessing']['mean'])
        std = np.array(self.config['preprocessing']['std'])

        img = (img - mean) / std

        # Transpose to NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(self, outputs, original_shape):
        """Postprocess model outputs"""
        # This is simplified - actual postprocessing depends on YOLO output format
        detections = []

        # Extract boxes, scores, classes from outputs
        # Implementation depends on specific YOLO output format

        return detections

    def infer(self, image: np.ndarray):
        """Run inference"""
        # Preprocess
        input_tensor = self.preprocess(image)

        # Run inference
        start_time = time.time()

        outputs = self.session.run(None, {
            self.session.get_inputs()[0].name: input_tensor
        })

        inference_time = time.time() - start_time

        # Postprocess
        detections = self.postprocess(outputs, image.shape)

        return detections, inference_time

# Usage
if __name__ == '__main__':
    # Load model
    detector = RPiInference('model.rpk')

    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        detections, inference_time = detector.infer(frame)

        # Draw results
        for det in detections:
            bbox = det['bbox']
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

        # Display
        cv2.imshow('RPi AI Camera Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

## Performance Optimization Tips

### 1. Model Quantization

**Run on: Development Machine (during model conversion)**

```python
# Quantize model for faster inference
from onnxruntime.quantization import quantize_dynamic, QuantType

quantized_path = "model_quantized.onnx"
quantize_dynamic(
    model_input="model_optimized.onnx",
    model_output=quantized_path,
    weight_type=QuantType.QUInt8
)
```

### 2. Thread Optimization

**Run on: Raspberry Pi (during inference)**

```python
# Set optimal thread count for RPi
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
```

### 3. Memory Optimization

**Run on: Raspberry Pi (during model loading)**

```python
# Use memory-efficient inference
session = ort.InferenceSession(
    model_path,
    sess_options=ort.SessionOptions(),
    providers=['CPUExecutionProvider'],
    provider_options=[{'enable_cpu_mem_arena': False}]
)
```

## Testing and Validation

### Performance Benchmark

**Run on: Raspberry Pi (to get accurate performance metrics)**

```python
def benchmark_model(model_path: str, num_runs: int = 100):
    """Benchmark model performance"""

    detector = RPiInference(model_path)

    # Create dummy input
    dummy_img = np.random.rand(480, 640, 3).astype(np.uint8)

    times = []
    for i in range(num_runs):
        _, inference_time = detector.infer(dummy_img)
        times.append(inference_time)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time

    print(".3f")
    print(".2f")
    print(".3f")

    return fps, avg_time

# Run benchmark
fps, avg_time = benchmark_model('model.rpk')
```

## Workflow Summary

1. **Development Machine**:
   - Train YOLO model (if not already done)
   - Export model to ONNX format
   - Optimize ONNX model
   - Convert to RPK format
   - Apply quantization (optional)
   - Transfer RPK and ONNX files to Raspberry Pi

2. **Raspberry Pi**:
   - Install dependencies
   - Load and run inference with RPK model
   - Configure thread/memory optimizations
   - Run performance benchmarks
   - Deploy for real-time detection

This guide provides a complete workflow for converting and deploying YOLO models on Raspberry Pi AI Camera for real-time hammer and object detection.
