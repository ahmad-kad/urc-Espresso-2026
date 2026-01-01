# YOLO Detection Service - Raspberry Pi Deployment

Minimal deployment package for Raspberry Pi 5 or Pi Zero 2 W running Raspberry Pi OS.

## Quick Start

1. **Copy to Pi:**
   ```bash
   scp -r deployment_package pi@raspberrypi.local:~
   ```

2. **On Pi, install:**
   ```bash
   cd ~/deployment_package
   chmod +x install.sh
   ./install.sh
   ```

3. **Copy model:**
   ```bash
   scp best.onnx pi@raspberrypi.local:/home/pi/models/
   ```

4. **Start service:**
   ```bash
   sudo systemctl enable yolo-detector
   sudo systemctl start yolo-detector
   ```

## Requirements

- Raspberry Pi OS (64-bit recommended)
- ROS2 Humble (for alerts)
- Pi Camera or compatible camera
- ONNX model file (`best.onnx`)

## Service Management

```bash
# Start service
sudo systemctl start yolo-detector

# Stop service
sudo systemctl stop yolo-detector

# Check status
sudo systemctl status yolo-detector

# View logs
sudo journalctl -u yolo-detector -f
```

## ROS2 Topics

- `/yolo_detector/detections` - All detections
- `/yolo_detector/alerts/low_confidence` - Low confidence (≥0.3)
- `/yolo_detector/alerts/medium_confidence` - Medium confidence (≥0.5)
- `/yolo_detector/alerts/high_confidence` - High confidence (≥0.7)
- `/yolo_detector/alerts/critical` - Critical confidence (≥0.9)

## Remote Testing

Two listener scripts are available for remote testing:

### Option 1: Listen to All Topics (Recommended)
Listens to all detection and alert topics simultaneously:
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0  # Match Pi's domain ID
python3 listen_detections.py
```

### Option 2: Listen to Specific Topic
Listen to a single topic (more flexible):
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
python3 ros2_listener.py /yolo_detector/detections
# Or any other topic:
# python3 ros2_listener.py /yolo_detector/alerts/high_confidence
```

**Note:** Both scripts can be copied to a remote machine for testing:
```bash
scp listen_detections.py user@remote:/path/
scp ros2_listener.py user@remote:/path/
```

## Configuration

Edit `config/alert_config.json` to adjust confidence thresholds.
Edit `config/service_config.json` to adjust model and camera settings.

