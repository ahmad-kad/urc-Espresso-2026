#!/bin/bash
# Raspberry Pi AI Camera Setup Script
# This script sets up Raspberry Pi 5 with AI Camera for YOLO inference using ONNX Runtime

set -e

echo " Raspberry Pi AI Camera Setup"
echo "================================="

# Update system
echo " Updating system packages..."
sudo apt update
sudo apt full-upgrade -y

# Install Python and pip
echo " Installing Python dependencies..."
sudo apt install -y python3-pip python3-dev python3-opencv

# Install camera libraries
echo " Installing camera libraries..."
sudo apt install -y python3-picamera2 python3-libcamera

# Install ONNX Runtime (CPU version for Raspberry Pi)
echo " Installing ONNX Runtime..."
pip3 install onnxruntime
pip3 install numpy opencv-python

# Install additional utilities
echo " Installing additional utilities..."
sudo apt install -y htop iotop vim git

# ROS2 installation note
echo ""
echo " ROS2 Installation Required"
echo "============================="
echo "ROS2 Humble must be installed separately for alert functionality."
echo "Follow the official guide: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html"
echo ""
echo "After installing ROS2, source it in your shell:"
echo "  source /opt/ros/humble/setup.bash"
echo ""

# Create project directory
echo " Setting up project directory..."
mkdir -p ~/yolo_ai_camera
cd ~/yolo_ai_camera

# Create models directory
mkdir -p models

echo ""
echo " Raspberry Pi setup complete!"
echo ""
echo "Next steps:"
echo "1. Install ROS2 Humble (if not already installed)"
echo "2. Copy deployment_package to Pi: scp -r deployment_package pi@raspberrypi.local:~/"
echo "3. On Pi, run: cd ~/deployment_package && chmod +x install.sh && ./install.sh"
echo "4. Copy ONNX model to /home/pi/models/best.onnx"
echo "5. Enable and start service: sudo systemctl enable yolo-detector && sudo systemctl start yolo-detector"
