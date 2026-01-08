# Setup Guide

Complete environment setup for MobileNetV2 Training & IMX500 Deployment

## Quick Setup (3 Options)

### Option 1: Modern Setup (Recommended)

```bash
# Install everything with pyproject.toml
pip install -e .[gpu,ros2,dev]

# Or install step-by-step
pip install -e .           # Core ML dependencies
pip install -e .[gpu]      # Add GPU support
pip install -e .[ros2]     # Add ROS2 integration
pip install -e .[dev]      # Add development tools
```

### Option 2: Traditional Setup

```bash
# Use requirements.txt
pip install -r requirements.txt

# Add ROS2 separately (requires system ROS2 installation)
pip install rclpy sensor-msgs cv-bridge std-msgs
```

### Option 3: Legacy Setup Script

```bash
# Run the automated setup (may be outdated)
bash setup_env.sh
```

## What's Included

### Core Dependencies (Always Installed)
- **PyTorch 2.9.1** + TorchVision 0.24.1
- **MobileNetV2/MobileNetV3/ResNet** architectures
- **OpenCV 4.12** + scikit-image
- **Sony IMX500 Converter** + MCT 2.5.0
- **ONNX Runtime** + ONNX tools
- **NumPy, Pandas, Matplotlib, Seaborn**
- **Tqdm, Pillow, Albumentations**

### Optional Dependencies
- **`[gpu]`**: CUDA 12.8, cuDNN, NVIDIA libraries
- **`[ros2]`**: Complete ROS2 Humble/Jazzy stack
- **`[dev]`**: Black, isort, mypy, pytest, sphinx
- **`[imx500]`**: TensorBoard, additional Sony tools

## What's Installed

### ✅ ML Dependencies (All Installed)
- **PyTorch**: 2.9.1+cu128 (with CUDA support)
- **OpenCV**: 4.12.0
- **ONNX Runtime**: 1.23.2
- **Ultralytics**: YOLO framework
- All other ML/data science libraries

### ⚠️ ROS2 Dependencies (Requires System Installation)
- `rclpy` - ROS2 Python client library
- `sensor_msgs` - Sensor message types
- `cv_bridge` - OpenCV-ROS bridge
- `std_msgs` - Standard messages
- `vision_msgs` - Vision messages

## ROS2 Installation

### For Ubuntu/Debian (ROS2 Jazzy)

```bash
# Install ROS2 Jazzy
sudo apt update
sudo apt install ros-jazzy-desktop

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Or install individual packages
sudo apt install ros-jazzy-rclpy \
                 ros-jazzy-sensor-msgs \
                 ros-jazzy-cv-bridge \
                 ros-jazzy-std-msgs \
                 ros-jazzy-vision-msgs
```

### Alternative: Use ROS2 in Docker

If you prefer containerized ROS2:

```bash
docker pull osrf/ros:jazzy-desktop
docker run -it --rm osrf/ros:jazzy-desktop bash
```

## Verify Installation

### Test ML Components

```bash
source venv/bin/activate
python -c "from core.models.detector import ObjectDetector; print('✓ PyTorch detector OK')"
python -c "from core.models.onnx_detector import ONNXDetector; print('✓ ONNX detector OK')"
```

### Test ROS2 Components

```bash
source venv/bin/activate
source /opt/ros/jazzy/setup.bash  # If ROS2 installed
python -c "import rclpy; print('✓ ROS2 OK')"
python -c "from cv_bridge import CvBridge; print('✓ cv_bridge OK')"
```

### Full Test

```bash
source activate_env.sh
python ros_nodes/test_setup.py --model output/models/best.pt
```

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Virtual Environment | ✅ Ready | Located in `venv/` |
| PyTorch Models | ✅ Ready | Can load `.pt` models |
| ONNX Models | ✅ Ready | Can load `.onnx` models (FP32 & INT8) |
| ROS2 Node Code | ✅ Ready | Code is complete |
| ROS2 Runtime | ⚠️ Needs Install | Requires system ROS2 |

## What You Can Do Now

### Without ROS2:
1. ✅ Load and test PyTorch models
2. ✅ Load and test ONNX models
3. ✅ Run inference on images
4. ✅ Test detector classes

### With ROS2 (after installation):
1. ✅ Run camera detector node
2. ✅ Publish/subscribe to ROS topics
3. ✅ View annotated images in rqt
4. ✅ Full ROS2 integration

## Troubleshooting

### Virtual Environment Issues

If `venv` doesn't activate:
```bash
python3 -m venv venv --clear
source venv/bin/activate
bash setup_env.sh
```

### ROS2 Not Found

ROS2 Python packages (`rclpy`, `sensor_msgs`, etc.) are typically installed as system packages, not via pip. You need:

1. **System ROS2 installation** (recommended):
   ```bash
   sudo apt install ros-jazzy-desktop
   source /opt/ros/jazzy/setup.bash
   ```

2. **Or use ROS2 workspace**:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   ```

### Import Errors

If you get import errors:
```bash
# Make sure you're in project root
cd /media/durian/AI/AI/urc-Espresso-2026

# Activate environment
source venv/bin/activate

# Add to PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Model Not Found

Check available models:
```bash
find output/models -name "*.pt" | head -5
find output/onnx -name "*.onnx" | head -5
```

## Environment Structure

```
urc-Espresso-2026/
├── venv/                    # Virtual environment
├── setup_env.sh             # Setup script
├── activate_env.sh          # Activation script
├── core/                    # Core ML code
├── ros_nodes/               # ROS2 nodes
└── output/                  # Models and results
```

## Daily Usage

1. **Activate environment:**
   ```bash
   source activate_env.sh
   ```

2. **Run detector:**
   ```bash
   python ros_nodes/camera_detector_node.py --model output/models/best.pt
   ```

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

## Dependencies Summary

### Installed via pip:
- PyTorch, Torchvision
- Ultralytics (YOLO)
- OpenCV
- ONNX Runtime
- NumPy, Pandas, Matplotlib
- Other ML/data science libraries

### Requires system installation:
- ROS2 (rclpy, sensor_msgs, cv_bridge, std_msgs, vision_msgs)

## Next Steps

1. **Test ML components:**
   ```bash
   source activate_env.sh
   python ros_nodes/test_setup.py --model output/models/best.pt
   ```

2. **Install ROS2 (if needed):**
   ```bash
   sudo apt install ros-jazzy-desktop
   source /opt/ros/jazzy/setup.bash
   ```

3. **Run full test:**
   ```bash
   # Terminal 1: Camera publisher
   python ros_nodes/camera_publisher.py --source 0
   
   # Terminal 2: Detector node
   python ros_nodes/camera_detector_node.py --model output/models/best.pt
   
   # Terminal 3: View results
   ros2 run rqt_image_view rqt_image_view /object_detector/annotated_image
   ```

## Environment Location

- **Virtual Environment**: `/media/durian/AI/AI/urc-Espresso-2026/venv/`
- **Python**: `venv/bin/python3`
- **Activation**: `source activate_env.sh`

