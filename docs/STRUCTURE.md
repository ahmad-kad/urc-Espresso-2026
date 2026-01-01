# Project Structure and Organization

**YOLO AI Camera Production System** - Complete deployment-ready object detection framework for Raspberry Pi.

## Directory Structure

```
urc-espresso-2026/
├──  deployment_package/         # PRODUCTION deployment system
│   ├──  scripts/                # Production inference & ROS2 scripts
│   │   ├── detector_service.py   # Main production service
│   │   ├── alert_manager.py      # ROS2 alert system
│   │   ├── listen_detections.py  # Remote testing listener
│   │   └── ros2_listener.py     # Flexible ROS2 listener
│   ├──  config/                 # Service & alert configuration
│   ├──  models/                 # (ONNX model copied here)
│   ├── install.sh                # Installation script
│   ├── raspberry_pi_setup.sh     # Initial Pi setup
│   ├── yolo-detector.service     # Systemd service
│   └── requirements_pi.txt       # Pi dependencies
│
├──  configs/                    # Model training configurations
│   ├── default.yaml              # Default framework configuration
│   ├── embedded.yaml             # Embedded device settings
│   ├── robotics.yaml             # Robotics-optimized settings
│   └── yolov8n_224.yaml          # Production model training config
│
├──  scripts/                    # Development & maintenance utilities
│   ├── evaluation/              # Model evaluation scripts
│   │   ├── model_evaluator.py     # Core evaluation framework
│   │   ├── compare_fp32_int8_accuracy.py
│   │   ├── run_comparison.py     # Benchmark comparisons
│   │   ├── evaluator.py          # Evaluation utilities
│   │   └── webcam_testing.py     # Live webcam testing
│   ├── training/                # Training scripts
│   │   ├── train_and_deploy.py   # Complete training pipeline
│   │   ├── run_training_pipeline.py
│   │   └── retraining_manager.py # Model retraining utilities
│   ├── models/                  # Model architecture definitions (if needed)
│   ├── config/                  # Configuration utilities
│   │   └── training_config_generator.py
│   ├── benchmark_models.py       # Performance benchmarking
│   ├── convert_to_onnx.py       # ONNX conversion
│   ├── dataset_utils.py         # Dataset management
│   ├── evaluate_per_class_accuracy.py
│   ├── training_utils.py         # Training helpers
│   └── validation_utils.py      # Validation utilities
│
├──  output/                     # Training outputs & results
│   ├── models/                   # Trained model weights
│   ├── evaluation/               # Evaluation results & metrics
│   ├── visualization/            # Performance plots
│   ├── testing/                  # Test coverage reports
│   └── benchmarking/             # Benchmark results
│
├──  consolidated_dataset/       # Training dataset (6 classes)
│   ├── data.yaml                 # Dataset configuration
│   ├── train/                    # Training images & labels
│   ├── val/                      # Validation images & labels
│   └── test/                     # Test images & labels
│
├──  docs/                       # Documentation
│   └── STRUCTURE.md              # This file
│
├──  utils/                      # Utility modules
│   ├── config.py                 # Configuration utilities
│   ├── data_utils.py             # Data processing utilities
│   ├── detection_utils.py        # Detection post-processing
│   ├── device_utils.py           # Device management
│   ├── logger_config.py          # Logging configuration
│   ├── metrics.py                # Performance metrics
│   ├── output_utils.py           # Output management
│   └── visualization.py          # Plotting and visualization
│
├──  tests/                      # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
│
├──  Core Framework Files        # Main application modules
│   ├── detector.py               # ObjectDetector class
│   ├── trainer.py                # ModelTrainer class
│   └── setup_dev.py              # Development setup
│
├──  Configuration Files         # Project configuration
│   ├── pyproject.toml            # Python project config
│   ├── pytest.ini               # Test configuration
│   ├── pyrightconfig.json        # Type checking config
│   └── Makefile                  # Build automation
│
└──  Documentation               # Project documentation
    ├── README.md                 # Project overview
    ├── MAINTENANCE.md            # Maintenance procedures
    ├── TESTING.md                # Testing guide
    └── CODE_QUALITY.md           # Code quality standards
```

## Key Features

### 1. **Production-Ready Deployment System**
- **deployment_package/**: Complete turnkey Raspberry Pi deployment
- **ROS2 Integration**: Native ROS2 alerts with confidence-based routing
- **Automated Startup**: Systemd service with auto-restart
- **Remote Testing**: Listener scripts for remote monitoring

### 2. **Clean Organization**
- **Root Directory**: Only core modules and essential config files
- **Scripts Organized**: Evaluation, training, and model scripts in subdirectories
- **Output Centralized**: All results in `output/` directory
- **Clear Separation**: Development vs production code

### 3. **Maintainability**
- **Modular Design**: Easy to extend and customize
- **Comprehensive Tests**: Unit, integration, and E2E test coverage
- **Documentation-Driven**: Guides for all procedures
- **LTS Ready**: Clean structure for long-term maintenance

## Usage Examples

### Production Deployment

```bash
# Copy deployment package to Pi
scp -r deployment_package pi@raspberrypi.local:~

# On Pi, install
cd ~/deployment_package
chmod +x install.sh && ./install.sh

# Start service
sudo systemctl enable yolo-detector
sudo systemctl start yolo-detector
```

### Model Training

```bash
# Run complete training pipeline
python scripts/training/run_training_pipeline.py

# Or use individual components
python scripts/training/train_and_deploy.py
```

### Model Evaluation

```bash
# Compare model performance
python scripts/evaluation/run_comparison.py

# Evaluate per-class accuracy
python scripts/evaluate_per_class_accuracy.py

# Compare FP32 vs INT8
python scripts/evaluation/compare_fp32_int8_accuracy.py
```

### Remote Testing

```bash
# On remote machine with ROS2
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
python deployment_package/scripts/listen_detections.py
```

## Configuration System

### Training Configurations (`configs/`)
- **yolov8n_224.yaml**: Production model training settings
- **default.yaml**: Baseline framework configuration
- **embedded.yaml**: Optimized for Raspberry Pi deployment
- **robotics.yaml**: Settings for robotics applications

### Deployment Configurations (`deployment_package/config/`)
- **service_config.json**: Production service settings
- **alert_config.json**: ROS2 alert configuration with confidence levels

### Utility Configuration (`utils/config.py`)
- Centralized configuration management
- Environment variable support
- YAML configuration loading

## Output Organization

### Training Outputs (`output/`)
- **models/**: Trained model weights and configurations
- **evaluation/**: Accuracy metrics and comparison results
- **visualization/**: Performance plots and detection examples
- **testing/**: Coverage reports and test results
- **benchmarking/**: Performance benchmark data

### Deployment Package (`deployment_package/`)
- **scripts/**: Production inference and monitoring scripts
- **config/**: Service and alert configurations
- **models/**: Production ONNX model (optimized for 252 FPS)

## Development Workflow

### Adding New Features

1. **Core Modules**: Add to root-level files (`detector.py`, `trainer.py`)
2. **Utilities**: Add to `utils/` directory
3. **Scripts**: Organize by purpose in `scripts/` subdirectories
4. **Tests**: Add corresponding tests in `tests/` directory
5. **Documentation**: Update relevant docs

### Code Organization Principles

- **Root Directory**: Only core framework files and essential configs
- **Scripts**: Organized by function (evaluation, training, models)
- **Output**: All generated files in `output/` directory
- **Utils**: Reusable utility functions
- **Tests**: Mirror source structure

## Maintenance

### Regular Tasks

1. **Model Updates**: Use `scripts/training/` for retraining
2. **Performance Monitoring**: Run evaluation scripts regularly
3. **Deployment Updates**: Update `deployment_package/` as needed
4. **Documentation**: Keep docs current with code changes

### Best Practices

- **Version Control**: Use Git for all changes
- **Testing**: Run test suite before commits
- **Documentation**: Update docs with code changes
- **Clean Commits**: Use .gitignore to avoid committing unnecessary files

## Troubleshooting

### Import Errors

If imports fail after reorganization:
- Ensure `sys.path` includes project root
- Check relative imports in moved files
- Verify `__init__.py` files exist in packages

### Deployment Issues

- Check `deployment_package/README.md` for deployment-specific guidance
- Verify ROS2 installation on Pi
- Check service logs: `sudo journalctl -u yolo-detector -f`

### Performance Issues

- Run benchmarks: `python scripts/benchmark_models.py`
- Check evaluation results in `output/evaluation/`
- Review visualization outputs in `output/visualization/`
