# Project Structure and Organization

This document explains the reorganized project structure and how to use the generalized object detection framework.

## Directory Structure

```
urc-espresso-2026/
├── configs/                       # All configuration files (flattened)
│   ├── default.yaml               # Default framework configuration
│   ├── embedded.yaml              # Embedded device settings
│   ├── robotics.yaml              # Robotics-optimized settings
│   ├── yolov8s_confidence.yaml    # YOLOv8s training config
│   └── ...                        # Other training configurations
├── consolidated_dataset/          # Dataset for training/benchmarking
│   ├── data.yaml                  # Dataset configuration
│   ├── train/                     # Training data
│   ├── val/                       # Validation data
│   └── test/                      # Test data
├── docs/                          # Documentation
├── models/                        # Pretrained model weights
├── scripts/                       # All executable scripts (flattened)
│   ├── run_rover_optimization.py  # Main optimization pipeline
│   ├── benchmark_models.py        # Model benchmarking
│   ├── quantize_models.py         # Model quantization
│   ├── analyze_memory.py          # Memory analysis
│   └── ...                        # Other utility scripts
├── output/                        # Generated outputs (auto-created)
│   ├── models/                    # Trained models
│   ├── results/                   # Test results and reports
│   ├── logs/                      # Training logs
│   └── quantized/                 # Quantized models
├── config.py                      # Configuration management
├── data_utils.py                  # Data processing utilities
├── detector.py                    # Generic YOLO detector class
├── efficientnet.py                # EfficientNet model architecture
├── evaluator.py                   # Model evaluation tools
├── metrics.py                     # Performance metrics
├── mobilevit.py                   # MobileViT model architecture
├── trainer.py                     # Training framework
├── visualization.py               # Plotting and visualization
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Key Features

### 1. **Flattened Directory Structure**
- All core modules moved to root level for easier navigation
- Scripts consolidated in single `scripts/` directory
- Configuration files flattened into `configs/`
- Removed unnecessary nesting (src/core/ → core/, etc.)

### 2. **ML Training, Benchmarking, and Conversion Focus**
- Removed all deployment/inference code
- Focused on model training workflows
- Comprehensive benchmarking capabilities
- Model quantization for deployment preparation

### 3. **Modular Architecture**
- **Core modules**: trainer.py, detector.py, evaluator.py at root level
- **Model architectures**: efficientnet.py, mobilevit.py
- **Utilities**: config.py, data_utils.py, metrics.py, visualization.py
- **Scripts**: Organized by function in single directory

### 4. **Configuration-Driven**
- YAML-based configuration system
- Environment-specific settings
- Easy to customize for different use cases

## Usage Examples

### Basic Training

```bash
# Train baseline model
python scripts/train.py --data_yaml data/data.yaml --train_baseline --epochs 50

# Train attention-enhanced model
python scripts/train.py --data_yaml data/data.yaml --train_attention --attention_type cbam --epochs 50

# Use custom configuration
python scripts/train.py --config configs/framework/environments/embedded.yaml --data_yaml data/data.yaml
```

### Model Evaluation

```bash
# Compare multiple models
python scripts/evaluate.py \
    --models output/models/baseline/weights/best.pt output/models/cbam_enhanced/weights/best.pt \
    --data_yaml data/data.yaml \
    --comprehensive
```

### Real-time Demo

```bash
# Webcam detection demo
python scripts/webcam_demo.py \
    --model output/models/cbam_enhanced/weights/best.pt \
    --config configs/framework/environments/robotics.yaml
```

### ROS2 Integration

```bash
# Build ROS2 package
cd ros2_ws
colcon build
source install/setup.bash

# Launch detector
ros2 run object_detection camera_detector \
    --ros-args -p model_path:=output/models/cbam_enhanced/weights/best.pt
```

## Configuration System

The framework uses YAML-based configuration for flexibility:

### Default Configuration (`configs/framework/default.yaml`)
- Baseline settings for general use
- Can be extended for specific environments

### Environment Configurations
- **robotics.yaml**: Optimized for robotics applications
- **embedded.yaml**: Optimized for Raspberry Pi/embedded devices

### Configuration Inheritance
```yaml
# Example: robotics.yaml extends default.yaml
extends: ../default.yaml

model:
  input_size: 416  # Higher resolution for robotics

ros2:
  enabled: true
```

## Output Organization

All outputs are now centralized in the `output/` directory:

- **models/**: Trained model weights and configurations
- **results/**: Evaluation reports, metrics, confusion matrices
- **logs/**: Training logs and tensorboard data
- **visualizations/**: Performance plots, detection examples

## Migration Guide

### From Old Structure

1. **Scripts**: Move from root level to `scripts/`
   - `train_and_test.py` → `scripts/train.py`
   - `test_deployment.py` → `scripts/evaluate.py`

2. **Configuration**: Create `data/data.yaml` for your dataset
   ```yaml
   train: data/train
   val: data/valid
   test: data/test
   nc: 3
   names: ['class1', 'class2', 'class3']
   ```

3. **ROS2 Package**: Updated from `hammer_detection` to `object_detection`
   - Update launch files and documentation

4. **Outputs**: Now saved to `output/` instead of `runs/` or scattered locations

## Customization

### Adding New Attention Mechanisms

1. Implement in `src/models/attention_modules.py`
2. Update configuration options
3. Add to trainer logic in `src/core/trainer.py`

### Adding New Model Architectures

1. Create new detector class in `src/core/detector.py`
2. Add configuration options
3. Update training and evaluation scripts

### Custom Metrics

1. Add to `src/utils/metrics.py`
2. Update `src/core/evaluator.py` to use new metrics
3. Add visualization support in `src/utils/visualization.py`

## Best Practices

1. **Configuration**: Use environment-specific configs for different deployment scenarios
2. **Outputs**: Always use the `output/` directory structure
3. **Modularity**: Extend core classes rather than modifying them
4. **Documentation**: Update this document when making structural changes

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes `src/` directory
2. **Configuration Not Found**: Check file paths in `configs/framework/` directory
3. **ROS2 Build Failures**: Ensure package name updated in all files
4. **Output Paths**: Use absolute paths or ensure `output/` directory exists

### Getting Help

- Check the main README.md for usage examples
- Review configuration files for available options
- Examine existing scripts for implementation patterns
