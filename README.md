# YOLO AI Camera System

**Production-Ready Object Detection for Raspberry Pi AI Camera**

High-performance YOLO object detection optimized for edge deployment. Real-time inference, intelligent alerts, and automated benchmarking for robotics applications.

[![Accuracy](https://img.shields.io/badge/Accuracy-87%25_F1-green)](https://)
[![Speed](https://img.shields.io/badge/Speed-200+_FPS-blue)](https://)
[![Platform](https://img.shields.io/badge/Platform-Raspberry_Pi-orange)](https://)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Key Features

- **High-Performance YOLO Detection**: 87% F1-Score across robotics objects
- **Real-Time Inference**: 200+ FPS on Raspberry Pi with ONNX optimization
- **Production-Ready**: Complete deployment package with ROS2 integration
- **Model Optimization**: Automatic ONNX conversion and INT8 quantization
- **Advanced Analytics**: Per-class performance analysis and benchmarking

## Performance

| Class | F1-Score | Status |
|-------|----------|---------|
| **OrangeHammer** | **97.5%** | ⭐ Excellent |
| **Bottle** | **86.3%** | Very Good |
| **BrickHammer** | **82.5%** | Good |
| **ArUcoTag** | **64.2%** | Needs Improvement but alright |

- **ONNX Optimization** with INT8 quantization support
- **ROS2 Integration** for robotics applications 

## Quick Start

```bash
# Install dependencies
pip install -e .

# Train model
python cli/train.py --data-yaml consolidated_dataset/data.yaml

# Evaluate performance
python cli/evaluate.py --model output/models/best.pt --benchmark

# Convert to ONNX
python cli/convert.py --model output/models/best.pt --format onnx --quantize
```

## Dataset

Consolidated URC robotics dataset with 1,806 images across 6 object classes. Pre-processed and ready for training.

## License

MIT License - see LICENSE file for details.
