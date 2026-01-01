#!/bin/bash
# Installation script for YOLO Detection Service
# Run this from the deployment_package directory after copying to Pi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo " Installing YOLO Detection Service with ROS2 Alerts"
echo "==================================================="

# Check if running as root for systemd operations
if [[ $EUID -eq 0 ]]; then
   echo " Do not run as root. Run as regular user (pi)."
   exit 1
fi

# Create directories
echo " Creating directories..."
mkdir -p /home/pi/yolo-detector
mkdir -p /home/pi/yolo-detector/logs
mkdir -p /home/pi/yolo-detector/snapshots
mkdir -p /home/pi/models

# Create virtual environment
echo " Setting up Python virtual environment..."
cd /home/pi/yolo-detector
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo " Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements_pi.txt"

# Check for ROS2
echo " Checking for ROS2 installation..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo " ROS2 Humble found"
else
    echo " WARNING: ROS2 Humble not found"
    echo " Alerts will not work without ROS2"
    echo " Install ROS2: https://docs.ros.org/en/humble/Installation.html"
fi

# Copy files from deployment package
echo " Copying service files..."
cp "$SCRIPT_DIR/scripts/detector_service.py" ./
cp "$SCRIPT_DIR/scripts/alert_manager.py" ./
cp "$SCRIPT_DIR/scripts/listen_detections.py" ./
cp "$SCRIPT_DIR/scripts/ros2_listener.py" ./
cp "$SCRIPT_DIR/config/service_config.json" ./
cp "$SCRIPT_DIR/config/alert_config.json" ./

# Copy model if it exists in deployment package
if [ -f "$SCRIPT_DIR/models/best.onnx" ]; then
    echo " Copying model from deployment package..."
    cp "$SCRIPT_DIR/models/best.onnx" /home/pi/models/
elif [ -f "/home/pi/models/best.onnx" ]; then
    echo " Model found at /home/pi/models/best.onnx"
else
    echo "  WARNING: Model not found. Please copy your ONNX model to /home/pi/models/best.onnx"
fi

# Set permissions
echo " Setting permissions..."
chmod +x detector_service.py
chmod +x alert_manager.py
chmod +x listen_detections.py
chmod +x ros2_listener.py
chown -R pi:pi /home/pi/yolo-detector

# Install systemd service
echo " Installing systemd service..."
sudo cp "$SCRIPT_DIR/yolo-detector.service" /etc/systemd/system/
sudo systemctl daemon-reload

# Enable camera
echo " Enabling camera..."
sudo raspi-config nonint do_camera 0

# Test camera access
echo " Testing camera access..."
python3 -c "
from picamera2 import Picamera2
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={'size': (224, 224), 'format': 'RGB888'})
    picam2.configure(config)
    picam2.start()
    frame = picam2.capture_array()
    picam2.stop()
    print(' Camera test successful')
except Exception as e:
    print(f' Camera test failed: {e}')
    exit(1)
"

echo ""
echo " Installation complete!"
echo ""
echo "Next steps:"
echo "1. Configure alert settings in alert_config.json"
echo "2. Test the service: python3 detector_service.py"
echo "3. Enable auto-start: sudo systemctl enable yolo-detector"
echo "4. Start service: sudo systemctl start yolo-detector"
echo "5. Check logs: sudo journalctl -u yolo-detector -f"
echo ""
echo "Configuration files:"
echo "- Service config: service_config.json"
echo "- Alert config: alert_config.json"
echo "- Logs: logs/detector.log"
echo "- Snapshots: snapshots/"
echo ""
echo "Service management:"
echo "- Start: sudo systemctl start yolo-detector"
echo "- Stop: sudo systemctl stop yolo-detector"
echo "- Status: sudo systemctl status yolo-detector"
echo "- Restart: sudo systemctl restart yolo-detector"
echo "- Logs: sudo journalctl -u yolo-detector -f"
 -f"
