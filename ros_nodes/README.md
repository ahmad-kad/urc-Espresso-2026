# ROS2 Multi-Camera Streaming System

Organized ROS2 camera streaming setup for multiple Raspberry Pi Zero 2 W devices with PC coordination.

## System Overview

- **3x Pi Zero 2 W with v2.1 cameras**: Camera publishers only
- **1x Pi Zero 2 W with AI camera**: Camera publisher + ML inference
- **1x PC**: Multi-camera subscriber and display

## Directory Structure

```
ros_nodes/
├── cameras/                    # Camera-specific nodes and configs
│   ├── pi_zero_2_w/           # Pi Zero 2 W configurations
│   │   ├── camera_publisher/   # For v2.1 cameras (3 units)
│   │   │   ├── camera_publisher.py
│   │   │   ├── start_camera.sh
│   │   │   └── camera-publisher.service.template
│   │   └── ai_camera_publisher/ # For AI camera with inference
│   │       ├── camera_detector_node.py
│   │       ├── start_ai_camera.sh
│   │       ├── ai-camera-detector.service
│   │       └── components/     # Inference components
│   └── pc/                     # PC subscriber
│       └── camera_subscriber/
│           ├── camera_subscriber.py
│           └── multi_camera_subscriber.launch.py
├── configs/                    # Configuration files
│   ├── pi_zero_v21/           # Configs for v2.1 camera Pis
│   │   ├── camera_config_pi1.yaml
│   │   ├── camera_config_pi2.yaml
│   │   └── camera_config_pi3.yaml
│   ├── pi_zero_ai/            # Config for AI camera Pi
│   │   └── camera_config_ai.yaml
│   ├── pc/                     # Config for PC subscriber
│   │   └── camera_subscriber_config.yaml
│   └── templates/              # Configuration templates
│       └── camera_config_template.yaml
├── zenoh/                      # Zenoh middleware and benchmarks
│   ├── benchmarks/            # Performance benchmarking tools
│   └── setup/                  # Zenoh setup scripts
├── launch/                     # ROS2 launch files
├── utils/                      # Utility scripts
│   └── setup_pi.sh            # Pi setup automation
└── README.md                   # This file
```

## Quick Setup

### 1. Pi Zero 2 W with v2.1 Camera (Repeat for 3 Pis)

```bash
# On each Pi Zero 2 W with v2.1 camera
cd ros_nodes/utils
./setup_pi.sh v21 1  # For Pi #1
./setup_pi.sh v21 2  # For Pi #2
./setup_pi.sh v21 3  # For Pi #3
```

### 2. Pi Zero 2 W with AI Camera

```bash
# On the Pi Zero 2 W with AI camera
cd ros_nodes/utils
./setup_pi.sh ai
```

### 3. PC Subscriber

```bash
# On your PC (Ubuntu/Linux with ROS2)
cd ros_nodes/cameras/pc/camera_subscriber
python3 camera_subscriber.py --config ../../configs/pc/camera_subscriber_config.yaml
```

## Configuration

### Adjusting Settings

All configuration is done through YAML files. Copy the template and customize:

```bash
# Copy template for each Pi
cp configs/templates/camera_config_template.yaml configs/pi_zero_v21/camera_config_pi1.yaml
cp configs/templates/camera_config_template.yaml configs/pi_zero_ai/camera_config_ai.yaml

# Edit settings (resolution, compression, framerate, etc.)
nano configs/pi_zero_v21/camera_config_pi1.yaml
```

### Quality vs Performance Options

**High Quality** (more CPU/bandwidth):
```yaml
fps: 30
resolution: "full"
jpeg_quality: 90
target_frame_size_kb: 100
```

**Balanced** (recommended):
```yaml
fps: 15
resolution: "half"
jpeg_quality: 75
target_frame_size_kb: 50
```

**Low Bandwidth** (for slow networks):
```yaml
fps: 10
resolution: "quarter"
jpeg_quality: 60
target_frame_size_kb: 25
```

## Manual Setup (Alternative)

If you prefer manual setup instead of the automated script:

### Pi Zero 2 W Setup

1. **Install ROS2 and dependencies:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install ros-jazzy-ros-base ros-jazzy-rclpy ros-jazzy-sensor-msgs ros-jazzy-cv-bridge
sudo apt install python3-opencv v4l-utils
```

2. **Configure systemd service:**
```bash
# For v2.1 camera Pi
sudo cp cameras/pi_zero_2_w/camera_publisher/camera-publisher.service.template /etc/systemd/system/camera-publisher-pi1.service
sudo sed -i 's/%i/1/g' /etc/systemd/system/camera-publisher-pi1.service
sudo systemctl enable camera-publisher-pi1.service
sudo systemctl start camera-publisher-pi1.service

# For AI camera Pi
sudo cp cameras/pi_zero_2_w/ai_camera_publisher/ai-camera-detector.service /etc/systemd/system/
sudo systemctl enable ai-camera-detector.service
sudo systemctl start ai-camera-detector.service
```

3. **Set environment variables:**
```bash
echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
source ~/.bashrc
```

## Monitoring and Debugging

### Check Service Status

```bash
# On each Pi
sudo systemctl status camera-publisher-pi1  # For v2.1 cameras
sudo systemctl status ai-camera-detector     # For AI camera

# View logs
journalctl -u camera-publisher-pi1 -f
```

### Monitor Camera Streams

```bash
# On PC, check topics
ros2 topic list

# Monitor frame rates
ros2 topic hz /camera/pi1/image_raw
ros2 topic hz /camera/pi2/image_raw
ros2 topic hz /camera/pi3/image_raw
ros2 topic hz /camera/pi4/image_raw

# View detections
ros2 topic echo /object_detector/detections
```

### Test Individual Components

```bash
# Test camera publisher manually
cd cameras/pi_zero_2_w/camera_publisher
./start_camera.sh 1 ../configs/pi_zero_v21/camera_config_pi1.yaml

# Test AI camera manually
cd cameras/pi_zero_2_w/ai_camera_publisher
./start_ai_camera.sh ../configs/pi_zero_ai/camera_config_ai.yaml

# Test PC subscriber
cd cameras/pc/camera_subscriber
python3 camera_subscriber.py --config ../../configs/pc/camera_subscriber_config.yaml
```

## Troubleshooting

### Camera Not Detected

```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera access
v4l2-ctl -d /dev/video0 --all
```

### ROS2 Network Issues

```bash
# Check ROS2 discovery
ros2 node list
ros2 topic list

# Test network connectivity
ping pi-camera-1.local  # Adjust hostname
```

### Performance Issues

- **High CPU usage**: Reduce FPS or resolution
- **Network lag**: Lower JPEG quality or use target frame size
- **Memory issues**: Disable unnecessary services in config

### Common Fixes

1. **Restart services:**
```bash
sudo systemctl restart camera-publisher-pi1
```

2. **Reboot Pis:**
```bash
sudo reboot
```

3. **Check logs:**
```bash
journalctl -u camera-publisher-pi1 --since "1 hour ago"
```

## Benchmarking

Use the Zenoh benchmarking tools to optimize performance:

```bash
cd zenoh/benchmarks
./run_middleware_benchmark.sh 0 30  # Compare Zenoh vs CycloneDDS
./benchmark_compression.sh 0 30 cyclonedds  # Test compression settings
```

## Advanced Configuration

### Custom Resolutions

Add custom resolutions to the camera publisher:

```python
# In camera_publisher.py, add to resolution_map
resolution_map = {
    "full": (640, 480),
    "half": (320, 240),
    "quarter": (160, 120),
    "custom": (800, 600)  # Add custom resolution
}
```

### Multiple Cameras per Pi

To use multiple cameras on one Pi, modify the config:

```yaml
camera:
  source: 0  # First camera
# Add secondary camera config if needed
camera2:
  source: 1  # Second camera
  topic: "/camera/pi1/camera2/image_raw"
```

### Custom Inference Models

Update the AI camera config for different models:

```yaml
inference:
  model: "output/models/yolov8n.pt"  # Different model
  config: "configs/yolov8.yaml"     # Different config
  input_size: 640                   # Different input size
```

## Contributing

When adding new features:

1. Follow the directory structure
2. Update configurations in `configs/templates/`
3. Add documentation to this README
4. Test on actual Pi Zero 2 W hardware