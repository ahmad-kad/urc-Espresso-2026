# Project Structure and Organization

This document explains the reorganized project structure and how to use the generalized object detection framework.

## Directory Structure

```
robotics_objdetection/
├── configs/                       # Configuration files
│   ├── default.yaml               # Default configuration
│   └── environments/              # Environment-specific configs
│       ├── robotics.yaml          # Robotics-optimized settings
│       └── embedded.yaml          # Embedded device settings
├── output/                        # Centralized outputs (auto-created)
│   ├── models/                    # Trained models
│   ├── results/                   # Test results and reports
│   ├── logs/                      # Training logs
│   └── visualizations/            # Plots and images
├── src/                           # Core source code
│   ├── core/                      # Core detection framework
│   │   ├── detector.py            # Generic YOLO detector class
│   │   ├── trainer.py             # Training framework
│   │   ├── evaluator.py           # Model evaluation tools
│   │   └── config.py              # Configuration management
│   ├── models/                    # Model architectures
│   │   └── attention_modules.py   # Attention mechanisms (CBAM, SE)
│   └── utils/                     # Utilities
│       ├── data_utils.py          # Data processing utilities
│       ├── visualization.py       # Plotting and visualization
│       └── metrics.py             # Performance metrics
├── scripts/                       # Executable scripts
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   ├── webcam_demo.py             # Real-time demo
│   ├── setup_webcam_test.py       # Webcam setup utility
│   ├── test_temporal_filter.py    # Temporal filter testing
├── ros2_ws/                       # ROS2 workspace
│   └── src/object_detection/      # ROS2 package
├── data/                          # Dataset (configurable)
├── docs/                          # Documentation
└── requirements.txt               # Python dependencies
```

## Key Changes from Previous Version

### 1. **Generalized Framework**
- Removed hardcoded references to "hammer", "bottle", "ArUco"
- Support for any object classes via configuration
- Flexible dataset support (any YOLO-compatible format)

### 2. **Organized Structure**
- **configs/**: All configuration files centralized
- **output/**: All outputs go to organized subdirectories
- **src/**: Refactored into core/, models/, utils/
- **scripts/**: All executable scripts in one place

### 3. **Modular Architecture**
- **core/**: Core framework classes (Detector, Trainer, Evaluator)
- **models/**: Model architectures and attention modules
- **utils/**: Data processing, visualization, metrics

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
python scripts/train.py --config configs/environments/embedded.yaml --data_yaml data/data.yaml
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
    --config configs/environments/robotics.yaml
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

### Default Configuration (`configs/default.yaml`)
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
2. **Configuration Not Found**: Check file paths in `configs/` directory
3. **ROS2 Build Failures**: Ensure package name updated in all files
4. **Output Paths**: Use absolute paths or ensure `output/` directory exists

### Getting Help

- Check the main README.md for usage examples
- Review configuration files for available options
- Examine existing scripts for implementation patterns
