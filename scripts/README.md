# YOLO AI Camera Scripts

Scripts for training, benchmarking, and deploying YOLO object detection models for AI camera applications.

## Directory Structure

```
scripts/
├── __init__.py                    # Package initialization
├── training/                      # Model training scripts
│   ├── __init__.py
│   └── basic_training.py          # Basic training script foundation
├── config/                        # Configuration utilities
│   └── training_config_generator.py
├── benchmark_models.py            # Model performance benchmarking
├── convert_to_onnx.py             # ONNX conversion utilities
├── dataset_utils.py               # Dataset management utilities
├── evaluate_per_class_accuracy.py # Per-class accuracy evaluation (fixed)
├── prepare_training_data.py       # Data preparation utility
├── train_accuracy_focused.py      # Complete accuracy-focused training pipeline
├── validation_utils.py            # Validation and visualization utilities
└── README.md                      # This file
```

## Quick Start

### Accuracy-Focused Training
```bash
# 1. Prepare your dataset
python scripts/prepare_training_data.py --dataset-path ./data --classes cup plate fork --create-structure --create-yaml

# 2. Train with accuracy optimization
python scripts/train_accuracy_focused.py --data-path ./data --classes cup plate fork --model-size yolov8m --img-size 640 --epochs 200

# 3. Evaluate results
python scripts/evaluate_per_class_accuracy.py
```

### Basic Training (Foundation)
```bash
python scripts/training/basic_training.py --data data/data.yaml --epochs 50
```

### Benchmark Model Performance
```bash
python scripts/benchmark_models.py
```

### Convert Model to ONNX
```bash
python scripts/convert_to_onnx.py --model output/models/best.pt --output output/onnx/model.onnx
```

### Evaluate Per-Class Accuracy
```bash
python scripts/evaluate_per_class_accuracy.py
```

### Prepare Training Data
```bash
# Create dataset structure and configuration
python scripts/prepare_training_data.py --dataset-path ./data --classes cup plate fork --create-structure --create-yaml

# Validate existing dataset
python scripts/prepare_training_data.py --dataset-path ./data --validate
```

##  Script Categories

### Accuracy-Focused Training
- **`train_accuracy_focused.py`**: Complete training pipeline optimized for maximum accuracy
  - YOLOv8m/l/x models for better accuracy than nano/tiny variants
  - Higher resolution (640x640) for finer details
  - Comprehensive data augmentation (rotation, scale, mosaic, mixup)
  - Longer training (200 epochs) with early stopping
  - Optimized hyperparameters for accuracy vs speed trade-off

### Data Preparation
- **`prepare_training_data.py`**: Dataset setup and validation utilities
  - Creates proper YOLO directory structure
  - Generates data.yaml configuration files
  - Validates dataset integrity and completeness

### Training (`training/`)
- **basic_training.py**: Foundation training script with configuration system

### Quantization (`quantization/`)
- **quantize_models.py**: Convert PyTorch models to ONNX and quantize to INT8 for efficient deployment

### Benchmarking (`benchmarking/`)
- **benchmark_models.py**: Comprehensive benchmarking of accuracy, speed, and memory usage
- **analyze_memory.py**: Detailed memory footprint analysis for deployment planning

### Utils (`utils/`)
- **monitor_training.py**: Real-time monitoring of training progress and status
- **analyze_tradeoffs.py**: Analysis of input size vs performance trade-offs

##  Key Features

-  **Modular Design**: Each script has a single, clear purpose
-  **Comprehensive**: Covers training → quantization → benchmarking pipeline
-  **Memory-Aware**: All scripts include memory usage analysis
-  **Production-Ready**: Optimized for embedded/rover deployment
-  **Well-Documented**: Clear docstrings and usage examples

##  Workflow

### Accuracy-First Training Pipeline

1. **Data Preparation**: `prepare_training_data.py`
   - Create YOLO dataset structure (images/train, images/val, labels/train, labels/val)
   - Generate data.yaml configuration file
   - Validate dataset completeness

2. **Accuracy Training**: `train_accuracy_focused.py`
   - YOLOv8m model (better accuracy than yolov8n/s)
   - 640x640 resolution for detail preservation
   - 200 epochs with comprehensive augmentation
   - Optimized for mAP50-95 accuracy metric

3. **Evaluation**: `evaluate_per_class_accuracy.py`
   - Per-class precision, recall, F1 scores
   - IoU-based matching for accurate assessment
   - ONNX model evaluation support

4. **Optimization**: `convert_to_onnx.py` → `benchmark_models.py`
   - Convert to ONNX for deployment
   - Benchmark accuracy vs speed trade-offs
   - Quantization for production deployment

### Legacy Training (Basic)
- **basic_training.py**: Foundation training script with configuration system

2. **Quantization**: `quantization/quantize_models.py`
   - ONNX export with real dataset calibration
   - INT8 quantization for 60-75% size reduction
   - Maintains accuracy while reducing memory footprint

3. **Benchmarking**: `benchmarking/benchmark_models.py`
   - Accuracy evaluation on real espresso dataset
   - Speed benchmarking (FPS, latency)
   - Memory usage analysis
   - Efficiency scoring for deployment ranking

4. **Monitoring**: `utils/monitor_training.py`
   - Real-time training progress tracking
   - Resource usage monitoring
   - Early problem detection

##  Memory Considerations

All scripts are designed for systems with limited memory:
- **Training**: Requires ~8GB GPU memory
- **Quantization**: CPU-based, ~4GB RAM
- **Benchmarking**: Minimal memory requirements
- **Deployment**: INT8 models use ~14MB runtime memory

##  Performance Targets

- **Accuracy**: >85% mAP50 on espresso detection
- **Speed**: >150 FPS for real-time operation
- **Memory**: <50 MB runtime memory footprint
- **Size**: <5 MB quantized model size

##  Development

Scripts follow these quality standards:
- **Black**: Consistent formatting
- **Flake8**: Code quality linting
- **Mypy**: Type checking
- **Modular**: Single responsibility principle

##  Usage Examples

### Python API
```python
from scripts.training import run_rover_optimization_pipeline
from scripts.quantization import quantize_all_models
from scripts.benchmarking import benchmark_rover_models

# Run complete pipeline
results = run_rover_optimization_pipeline(config, "data.yaml")

# Quantize best models
quantized = quantize_all_models(model_list)

# Benchmark performance
benchmarks = benchmark_rover_models(model_configs, "data.yaml")
```

### Command Line
```bash
# Complete optimization
python scripts/training/optimize_rover_models.py

# Monitor progress
python scripts/utils/monitor_training.py

# Memory analysis
python scripts/benchmarking/analyze_memory.py
```









