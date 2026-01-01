# Rover Model Optimization Scripts

Modular scripts for training, quantizing, and benchmarking object detection models for rover deployment.

##  Directory Structure

```
scripts/
├── __init__.py                 # Package initialization
├── training/                   # Model training scripts
│   ├── __init__.py
│   └── optimize_rover_models.py    # Main optimization pipeline
├── quantization/               # Model quantization scripts
│   ├── __init__.py
│   └── quantize_models.py      # INT8 quantization for all models
├── benchmarking/               # Performance evaluation scripts
│   ├── __init__.py
│   ├── benchmark_models.py     # Accuracy/speed/memory benchmarking
│   └── analyze_memory.py       # Memory usage analysis
└── utils/                      # Utility and helper scripts
    ├── __init__.py
    ├── monitor_training.py     # Training progress monitoring
    └── analyze_tradeoffs.py    # Input size trade-off analysis
```

##  Quick Start

### Complete Rover Optimization Pipeline
```bash
cd scripts
python training/optimize_rover_models.py
```

### Quantize Optimized Models
```bash
python quantization/quantize_models.py
```

### Benchmark Model Performance
```bash
python benchmarking/benchmark_models.py
```

### Monitor Training Progress
```bash
python utils/monitor_training.py
```

##  Script Categories

### Training (`training/`)
- **optimize_rover_models.py**: Complete pipeline training multiple architectures at different sizes with early stopping

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

1. **Training**: `training/optimize_rover_models.py`
   - Trains YOLOv8s models at multiple sizes
   - Early stopping with patience=10
   - Automatic best model selection

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










arking/analyze_memory.py
```












