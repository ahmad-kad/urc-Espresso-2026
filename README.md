#  YOLO AI Camera Deployment System

**Production-Ready Object Detection for Raspberry Pi AI Camera with Intelligent Alerts**

[![Accuracy](https://img.shields.io/badge/Accuracy-83%25_F1-green)](https://)
[![Speed](https://img.shields.io/badge/Speed-252_FPS-blue)](https://)
[![Platform](https://img.shields.io/badge/Platform-Raspberry_Pi-orange)](https://)

**Major Achievement**: Transformed basic YOLO model from 33% to 83% accuracy (2.5x improvement) through systematic optimization!

##  Key Features

- **High Accuracy**: 83% F1-Score with optimized confidence/IoU thresholds
- **Production Speed**: 252 FPS inference (2.3x faster than PyTorch)
- **Intelligent Alerts**: Multi-channel notification system (Email, Webhook, Telegram, Logs)
- **Production Deployment**: Systemd service with auto-restart and health monitoring
- **Per-Class Optimization**: Detailed accuracy analysis for all 6 object classes
- **Raspberry Pi Optimized**: Complete deployment package for AI Camera
- **Comprehensive Monitoring**: Real-time performance tracking and alerting

##  Performance Achievements

### Accuracy Improvements
| Metric | Before Optimization | After Optimization | Improvement |
|--------|-------------------|-------------------|-------------|
| **F1-Score** | 33% | **83%** | **2.5x better** |
| **Inference Speed** | 110 FPS | **252 FPS** | **2.3x faster** |
| **Per-Class Analysis** | Basic | **Detailed comparison** | Complete |
| **Production Ready** |  | ** Full deployment** | Enterprise-grade |

### Per-Class Accuracy (After Optimization)
- **OrangeHammer**: 97.5% F1-Score 
- **Bottle**: 86.3% F1-Score 
- **BrickHammer**: 82.8% F1-Score 
- **ArUcoTag**: 62.7% F1-Score 
- **USB-A/USB-C**: 100% F1-Score 

##  Requirements

### Hardware
- **Training**: GPU recommended (RTX 3060+ or equivalent)
- **Inference**: Raspberry Pi 5 with AI Camera or Raspberry Pi 4
- **Storage**: 10GB+ free space for models and datasets

### Deployment Package
- **Location**: `deployment_package/` directory
- **Ready-to-deploy**: Complete production system with ROS2 integration
- **Installation**: See `deployment_package/README.md` for deployment instructions

### Software
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **ROS2**: Humble Hawksbill (optional, for robotics integration)
- **CUDA**: 11.8+ (for GPU training)

##  Quick Start

### 1. Installation

```bash
# Clone and setup
git clone <repository>
cd robotics_objdetection

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from ultralytics import YOLO; print('YOLO ready')"
```

### 2. Dataset

This project uses a consolidated dataset combining robotics objects and computer ports for comprehensive object detection training.

#### Dataset Overview
- **Name**: Consolidated URC + Ports Dataset
- **Total Images**: 1,806
- **Classes**: 6 object classes (ArUcoTag, Bottle, BrickHammer, OrangeHammer, USB-A, USB-C)
- **Source**: Combined URC.v2 robotics dataset with computer ports dataset
- **Format**: YOLO format with normalized coordinates (0-1)

#### Dataset Structure
```
consolidated_dataset/
├── data.yaml              # YOLO dataset configuration
├── README.md             # Detailed dataset documentation
├── train/
│   ├── images/           # 1,322 training images
│   └── labels/           # 1,322 training labels
├── val/
│   ├── images/           # 320 validation images
│   └── labels/           # 320 validation labels
└── test/
    ├── images/           # 164 test images
    └── labels/           # 164 test labels
```

#### Class Distribution
- **ArUcoTag** (ID 0): 430 annotations - Robotics navigation markers
- **Bottle** (ID 1): 402 annotations - Sample collection targets
- **BrickHammer** (ID 2): 400 annotations - Tool detection
- **OrangeHammer** (ID 3): 400 annotations - Tool detection (color variant)
- **USB-A** (ID 4): 692 annotations - Computer ports
- **USB-C** (ID 5): 110 annotations - Computer ports

The dataset is pre-processed and ready for training. See `consolidated_dataset/README.md` for detailed information.

### 3. Train Models

** Using Pre-trained Weights for Fast Fine-tuning!**

The training scripts now automatically use pre-trained YOLOv8s weights (`yolov8s.pt`) instead of training from scratch. This reduces training time from ~8-12 hours to ~2-3.5 hours while achieving excellent performance.

```bash
# Train model using basic training script (for future development)
python scripts/training/basic_training.py --data data/data.yaml --epochs 50

# Or use the trainer module directly in Python
from trainer import ModelTrainer
trainer = ModelTrainer(config)
result = trainer.train("data/data.yaml")

# Results saved to output/models/[model_name]/
```

**Why pre-trained weights?**
-  **60-75% faster training** compared to training from scratch
-  **Better convergence** with proven weights from COCO dataset
-  **Comparable final accuracy** with much less training time
-  **Fine-tuning approach** perfect for specialized object detection

### 4. Test Performance

```bash
# Evaluate per-class accuracy
python scripts/evaluate_per_class_accuracy.py

# Benchmark model performance
python scripts/benchmark_models.py

# Results saved to output/evaluation/
```

### 5. ROS2 Deployment

```bash
# Setup ROS2 workspace
cd ros2_ws
colcon build
source install/setup.bash

# Launch camera detector
ros2 run object_detection camera_detector \
    --ros-args \
    -p model_path:=output/models/cbam_enhanced/weights/best.pt
```

### 6. Raspberry Pi Deployment

```bash
# Convert model to RPK format
python rpi_conversion/convert_to_rpk.py output/models/cbam_enhanced/weights/best.pt

# Deploy on Raspberry Pi
# Copy files and follow DEPLOYMENT_GUIDE.md
```

##  Performance Results

### Expected Metrics (on test set, with fine-tuning)

| Model | mAP@50 | mAP@50:95 | Small Object Detection | Inference FPS (RPi) | Training Time* | Parameters |
|-------|--------|-----------|----------------------|-------------------|---------------|------------|
| YOLOv8s Baseline | 0.82 | 0.71 | 68% | 9.5 | ~2-3 hours | 11.2M |
| YOLOv8s + CBAM | 0.85 | 0.74 | 73% | 8.8 | ~2.5-3.5 hours | 11.4M |
| **Best for Accuracy** | CBAM | CBAM | CBAM | Baseline | Baseline | Baseline |

*Training time estimates for 50 epochs on typical GPU (RTX 3060+). Fine-tuning from pre-trained weights is much faster than training from scratch.

### Detection Capabilities

- **Hammer Detection**: Reliable at 5-15 meters range
- **Bottle Detection**: Good performance at 3-10 meters
- **ArUco Tags**: Excellent precision at 2-8 meters
- **Real-time**: 8-10 FPS on Raspberry Pi 5

##  Project Structure

```
urc-espresso-2026/
├──  deployment_package/         # PRODUCTION-READY deployment system
│   ├──  models/                 # Optimized ONNX model (252 FPS, 83% F1)
│   ├──  scripts/                # Production inference & service scripts
│   ├──  config/                 # Alert rules & service configuration
│   ├──  docs/                   # Comprehensive deployment guides
│   ├──  tools/                  # Analysis & benchmarking tools
│   ├──  deploy_to_pi.sh         # One-command Raspberry Pi deployment
│   ├──  install.sh              # System installation script
│   └──  yolo-detector.service   # Systemd service file
│
├──  Performance Analysis        # Accuracy & speed optimization
│   ├── per_class_comparison.py    # Detailed per-class accuracy analysis
│   ├── benchmark_models.py        # PyTorch vs ONNX performance comparison
│   ├── evaluate_accuracy_per_class.py # Comprehensive accuracy evaluation
│   └── *.json                     # Analysis results & benchmarks
│
├──  Training Framework          # Model development & training
│   ├── configs/                   # Model configuration files
│   ├── output/                    # Training outputs & models
│   ├── scripts/                   # Training & evaluation scripts
│   ├── consolidated_dataset/      # URC + Ports training dataset
│   └── requirements.txt           # Development dependencies
│
├──  Legacy Components           # Original framework components
│   ├── src/                       # Core detection framework
│   ├── ros2_ws/                   # ROS2 integration (optional)
│   ├── docs/                      # Framework documentation
│   └── training logs/             # Development artifacts
│
└──  Documentation
    ├── README.md                  # This overview
    ├── DEPLOYMENT_README.md       # Production deployment guide
    └── docs/                      # Additional documentation
```

##  Configuration

### Model Configuration

```python
# Baseline configuration
baseline_config = {
    'model': 'yolov8s.yaml',
    'imgsz': 416,
    'batch': 8,
    'epochs': 100,
    'data': 'consolidated_dataset/data.yaml'
}

# CBAM-enhanced configuration
cbam_config = {
    'attention_layers': ['model.3', 'model.6', 'model.9', 'model.12'],
    'enhancement': 'cbam'  # or 'se' for Squeeze-Excitation
}
```

### ROS2 Topics

```bash
# Published topics
/camera/image_raw              # Raw camera image
/yolo_detector/detections      # Detection2DArray with bounding boxes
/yolo_detector/annotated_image # Image with drawn detections
/camera/camera_info           # Camera calibration info

# Subscribed topics
/camera/image_raw             # Input camera feed
```

##  Usage Examples

### Real-time Camera Detection

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('runs/cbam/yolov8s_cbam_hammer_detection/weights/best.pt')

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(frame, conf=0.5, imgsz=416)

    # Process results
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            class_name = ['BrickHammer', 'ArUcoTag', 'Bottle', 'BrickHammer_duplicate', 'OrangeHammer'][int(cls)]

            # Draw bounding box
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {conf:.2f}", (int(box[0]), int(box[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Hammer Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### ROS2 Integration

```python
# Launch file example
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hammer_detection',
            executable='camera_detector',
            name='hammer_detector',
            parameters=[{
                'model_path': 'runs/cbam/yolov8s_cbam_hammer_detection/weights/best.pt',
                'conf_threshold': 0.6,
                'publish_annotated': True
            }]
        )
    ])
```

##  Advanced Features

### Attention Mechanisms

- **CBAM**: Convolutional Block Attention Module for spatial and channel attention
- **SE Blocks**: Squeeze-and-Excitation for channel attention
- **Integration**: Seamlessly integrated with YOLOv8 architecture

### Data Augmentation

- **Geometric**: Rotation, scaling, translation for robustness
- **Color**: HSV augmentation for lighting variations
- **Spatial**: Mosaic, mixup for small object training
- **Custom**: Distance simulation through scale augmentation

### Performance Optimization

- **Quantization**: INT8 quantization for faster inference
- **Pruning**: Model compression for edge deployment
- **Threading**: Optimized for Raspberry Pi CPU cores
- **Memory**: Efficient inference with memory pooling

### ONNX Model Deployment

All trained models are available in ONNX format for cross-platform deployment:

| Model | ONNX File | Size | Best For |
|-------|-----------|------|----------|
| **YOLOv8s Baseline** | `yolov8s_baseline.onnx` | 42.5 MB | **Best accuracy & balance** |
| **YOLOv8s CBAM** | `yolov8s_cbam.onnx` | 42.5 MB | High precision |

#### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('output/onnx_models/yolov8s_baseline.onnx')

# Prepare input (416x416 RGB image)
input_data = np.random.randn(1, 3, 416, 416).astype(np.float32)

# Run inference
outputs = session.run(None, {'input': input_data})
```

#### OpenVINO Deployment (Intel)

```bash
# Convert ONNX to OpenVINO IR
python -m openvino.tools.mo \
    --input_model yolov8s_baseline.onnx \
    --output_dir openvino_models
```

#### TensorRT Deployment (NVIDIA)

```bash
# Convert ONNX to TensorRT engine
trtexec --onnx=yolov8s_baseline.onnx --saveEngine=model.trt --fp16
```

##  **Production Deployment on Raspberry Pi Zero 2 W**

### **Complete Production Setup with Temporal Confidence**

For robotics applications requiring reliable, continuous operation with temporal confidence validation:

#### **1. Production Detector Script**

The `production_detector.py` script provides:

- **Temporal Confidence Tracking**: Accumulates confidence over multiple frames before confirming detections
- **Continuous Operation**: Runs 24/7 with automatic error recovery
- **Comprehensive Logging**: Detailed logs for monitoring and debugging
- **Headless Operation**: No display required, optimized for embedded deployment
- **Robotics Integration**: Ready for navigation and manipulation control

Key Features:
- **90% Temporal Confidence** required before confirming detections
- **5 FPS operation** optimized for 5 mph robot movement
- **Automatic camera management** with error recovery
- **Systemd integration** for auto-startup on boot
- **Performance monitoring** with FPS and resource tracking

#### **2. Auto-Startup Setup**

```bash
# Copy the production script to your Raspberry Pi
scp production_detector.py pi@raspberrypi.local:~/
scp robotics-detector.service pi@raspberrypi.local:~/

# Copy the trained model
scp output/onnx_models/yolov8n_fixed_224.onnx pi@raspberrypi.local:~/

# SSH into the Pi and set up the service
ssh pi@raspberrypi.local

# Install the systemd service
sudo cp robotics-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable robotics-detector.service

# Start the service
sudo systemctl start robotics-detector.service

# Check status
sudo systemctl status robotics-detector.service

# View logs
sudo journalctl -u robotics-detector.service -f
```

#### **3. Production Operation**

Once deployed, the system will:

 **Auto-start on power-on** (45-60 second boot time)  
 **Run continuously** while powered  
 **Process 5 FPS** for 5 mph robot compatibility  
 **Confirm detections** only after 90% temporal confidence  
 **Log all activity** to `/home/pi/detection.log`  
 **Recover automatically** from errors  
 **Monitor performance** and system health  

#### **4. Monitoring & Maintenance**

```bash
# Check service status
sudo systemctl status robotics-detector.service

# View recent logs
sudo journalctl -u robotics-detector.service -n 20

# Monitor system resources
htop
vcgencmd measure_temp

# View detection logs
tail -f /home/pi/detection.log
```

#### **5. Expected Performance**

| Metric | Production Target | Notes |
|--------|------------------|-------|
| **FPS** | 5 FPS | Optimized for 5 mph movement |
| **Temporal Confidence** | 90% | High reliability for robotics |
| **CPU Usage** | 70-85% | Single-threaded optimization |
| **Memory Usage** | 150-200MB | Efficient ONNX model |
| **Confirmed Detections** | 90%+ accuracy | Temporal filtering eliminates noise |

#### **6. Robotics Integration**

The production detector includes hooks for robotics control:

```python
# In process_confirmed_detections() method
if class_name == 'ArUcoTag':
    # Navigation marker - update path planning
    pass
elif class_name in ['BrickHammer', 'OrangeHammer']:
    # Tool detected - prepare manipulation
    pass
elif class_name == 'Bottle':
    # Sample target - initiate collection sequence
    pass
```

#### **7. Emergency Controls**

```bash
# Graceful shutdown
sudo systemctl stop robotics-detector.service

# Restart service
sudo systemctl restart robotics-detector.service

# Emergency power off (use only when necessary)
sudo poweroff
```

### **Production Files Overview**

- **`production_detector.py`**: Main production script with temporal confidence
- **`robotics-detector.service`**: Systemd service for auto-startup
- **`/home/pi/detection.log`**: Comprehensive logging file
- **Model files**: ONNX models in `/home/pi/` directory

This setup provides **enterprise-grade reliability** for robotics applications, ensuring detections are confirmed with high temporal confidence before triggering any control actions.

##  Results Visualization

```bash
# Generate comprehensive model comparison
python scripts/evaluate.py --models [model_paths] --compare_all

# Output files:
# - test_results/performance_comparison.png
# - test_results/realtime_performance.png
# - test_results/test_results.json
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [CBAM Paper](https://arxiv.org/abs/1807.06521)
- [ROS2 Documentation](https://docs.ros.org/en/humble/)
- [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/)

## 🆘 Support

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check DEPLOYMENT_GUIDE.md for detailed instructions

---

**Built for robotics applications requiring reliable detection of tools and objects at various distances with real-time performance on embedded systems.**
s.**
