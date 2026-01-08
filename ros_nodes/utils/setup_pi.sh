#!/bin/bash

# Ubuntu 24.04 setup for Raspberry Pi Zero 2W camera nodes
# Supports: Pi Camera v2.0, v2.1, and AI camera modules
# Includes: DNS fix, camera packages, ROS2 workspace, systemd service
# Usage: ./setup_pi.sh [camera_type] [pi_number]
# Example: ./setup_pi.sh v20 1
# Example: ./setup_pi.sh v21 2
# Example: ./setup_pi.sh ai 1

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

CAMERA_TYPE=${1:-"v21"}
PI_NUMBER=${2:-"1"}

# Validate camera type
if [[ ! "$CAMERA_TYPE" =~ ^(v20|v21|ai)$ ]]; then
    echo -e "${RED}ERROR: Invalid camera type '$CAMERA_TYPE'${NC}"
    echo "Supported types: v20, v21, ai"
    exit 1
fi

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}Pi Zero 2W Camera Setup${NC}"
echo "Ubuntu: 24.04 | ROS2: Jazzy"
echo "Camera: $CAMERA_TYPE | Pi #$PI_NUMBER"
echo -e "${GREEN}==========================================${NC}"

# Configure reliable DNS (fix for resolution issues)
echo "[1/9] Configuring DNS..."
sudo tee /etc/resolv.conf > /dev/null <<EOF
nameserver 8.8.8.8
nameserver 8.8.4.4
options edns0 trust-ad
EOF
echo -e "${GREEN}✓ DNS configured with Google DNS${NC}"

# Update system
echo "[2/9] Updating system..."
sudo apt update && sudo apt upgrade -y
echo -e "${GREEN}✓ System updated${NC}"

# Add ROS2 Jazzy repository
echo "[3/9] Adding ROS2 Jazzy repository..."
sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
echo -e "${GREEN}✓ ROS2 repository added${NC}"

# Install ROS2 Jazzy
echo "[4/9] Installing ROS2 Jazzy..."
sudo apt install -y \
    ros-jazzy-ros-base \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-rmw-cyclonedds-cpp \
    python3-colcon-common-extensions
echo -e "${GREEN}✓ ROS2 Jazzy installed${NC}"

# Install camera support
echo "[5/9] Installing camera support..."
sudo apt install -y \
    libcamera-dev \
    libcamera-tools \
    python3-libcamera \
    python3-opencv \
    python3-yaml \
    python3-numpy \
    python3-pip \
    v4l-utils

# Install picamera2 and dependencies
sudo apt install -y libcap-dev python3-prctl
pip3 install --break-system-packages picamera2
echo -e "${GREEN}✓ Camera packages installed${NC}"

# Configure camera permissions
echo "[6/10] Configuring camera permissions..."
sudo usermod -aG video $USER
echo -e "${GREEN}✓ Camera permissions configured${NC}"

# Test camera detection
echo "[7/10] Testing camera detection..."
CAMERA_FOUND=false
if command -v libcamera-vid &> /dev/null; then
    if timeout 3s libcamera-vid --list-cameras 2>/dev/null | grep -q "Camera"; then
        echo -e "${GREEN}✓ Camera detected${NC}"
        CAMERA_FOUND=true
    fi
fi

if [ "$CAMERA_FOUND" = false ]; then
    echo -e "${YELLOW}⚠ Camera not detected - will check after reboot${NC}"
fi

# Install AI dependencies if needed
if [ "$CAMERA_TYPE" = "ai" ]; then
    echo "Installing AI dependencies..."
    pip3 install --break-system-packages torch torchvision ultralytics
    echo -e "${GREEN}✓ AI dependencies installed${NC}"
fi

# Create workspace
echo "[8/10] Setting up workspace..."
mkdir -p $HOME/ros2_camera_ws/src

# Copy project files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we're running from extracted tar (ros_nodes directory)
if [[ "$SCRIPT_DIR" == *"/ros_nodes/utils" ]]; then
    # Extracted from tar: ros_nodes is at the same level as script
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    if [ -d "$PROJECT_ROOT/cameras" ] && [ -d "$PROJECT_ROOT/utils" ]; then
        echo "[INFO] Detected extracted tar structure"
        # Create full directory structure first
        mkdir -p $HOME/ros2_camera_ws/src/urc-Espresso-2026
        rm -rf $HOME/ros2_camera_ws/src/urc-Espresso-2026/*
        cp -r "$PROJECT_ROOT"/* $HOME/ros2_camera_ws/src/urc-Espresso-2026/
        echo -e "${GREEN}✓ Project copied from tar structure${NC}"
    else
        echo -e "${RED}✗ Invalid tar structure at: $PROJECT_ROOT${NC}"
        echo "Contents of PROJECT_ROOT:"
        ls -la "$PROJECT_ROOT"
        exit 1
    fi
else
    # Normal project structure: script is in project/ros_nodes/utils/
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    if [ -d "$PROJECT_ROOT/ros_nodes" ] && [ -d "$PROJECT_ROOT/core" ]; then
        rm -rf $HOME/ros2_camera_ws/src/urc-Espresso-2026
        mkdir -p $HOME/ros2_camera_ws/src
        cp -r "$PROJECT_ROOT" $HOME/ros2_camera_ws/src/urc-Espresso-2026
        echo -e "${GREEN}✓ Project copied from full project structure${NC}"
    else
        echo -e "${RED}✗ Full project structure not found at: $PROJECT_ROOT${NC}"
        echo "Expected: $PROJECT_ROOT/ros_nodes and $PROJECT_ROOT/core"
        echo "Contents of PROJECT_ROOT:"
        ls -la "$PROJECT_ROOT" 2>/dev/null || echo "Directory not accessible"
        exit 1
    fi
fi

# Build workspace
echo "Building ROS2 workspace..."
cd $HOME/ros2_camera_ws
source /opt/ros/jazzy/setup.bash
if colcon build --symlink-install; then
    echo -e "${GREEN}✓ Workspace built successfully${NC}"
else
    echo -e "${RED}✗ Workspace build failed${NC}"
    echo "You may need to run: cd ~/ros2_camera_ws && source /opt/ros/jazzy/setup.bash && colcon build --symlink-install"
    exit 1
fi

# Configure CycloneDDS
echo "[9/10] Configuring CycloneDDS..."
sudo mkdir -p /etc/cyclonedds
sudo tee /etc/cyclonedds/config.xml > /dev/null <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="42">
    <General>
      <Interfaces>
        <NetworkInterface name="auto"/>
      </Interfaces>
      <MaxMessageSize>65536B</MaxMessageSize>
    </General>
  </Domain>
</CycloneDDS>
EOF
echo -e "${GREEN}✓ CycloneDDS configured${NC}"

# Create auto-start service
echo "[10/10] Creating auto-start service..."

if [ "$CAMERA_TYPE" = "ai" ]; then
    SERVICE_NAME="ai-camera-pi$PI_NUMBER"
    NODE_SCRIPT="ai_camera_detector_node.py"
    NODE_DIR="ai_camera_publisher"
else
    SERVICE_NAME="camera-pi$PI_NUMBER"
    NODE_SCRIPT="camera_publisher.py"
    NODE_DIR="camera_publisher"
fi

# Create service file with corrected paths
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=ROS2 Camera $CAMERA_TYPE Pi $PI_NUMBER
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/ros2_camera_ws
Environment="ROS_DOMAIN_ID=42"
Environment="RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
Environment="CYCLONEDDS_URI=file:///etc/cyclonedds/config.xml"
ExecStart=/bin/bash -c 'source /opt/ros/jazzy/setup.bash && source ~/ros2_camera_ws/install/setup.bash && python3 ~/ros2_camera_ws/src/urc-Espresso-2026/ros_nodes/cameras/pi_zero_2_w/$NODE_DIR/$NODE_SCRIPT --pi-number $PI_NUMBER --camera-type $CAMERA_TYPE'
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable service
sudo systemctl daemon-reload
if sudo systemctl enable $SERVICE_NAME.service 2>/dev/null; then
    echo -e "${GREEN}✓ Service $SERVICE_NAME enabled for startup${NC}"
else
    echo -e "${YELLOW}⚠ Service enable failed, but service file created${NC}"
fi

# Start service
if sudo systemctl start $SERVICE_NAME.service 2>/dev/null; then
    echo -e "${GREEN}✓ Service $SERVICE_NAME started${NC}"
    sleep 2
    if systemctl is-active --quiet $SERVICE_NAME.service; then
        echo -e "${GREEN}✓ Service is running successfully${NC}"
    else
        echo -e "${YELLOW}⚠ Service started but may have issues - check logs${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Service start failed - check configuration${NC}"
fi

# Setup environment
cat >> $HOME/.bashrc <<'EOF'

# ROS2 Camera Setup
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///etc/cyclonedds/config.xml
EOF

# Set hostname
sudo hostnamectl set-hostname "pi-$CAMERA_TYPE-$PI_NUMBER"

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}==========================================${NC}"
echo "Hostname: pi-$CAMERA_TYPE-$PI_NUMBER"
echo "Service: $SERVICE_NAME"
echo ""

# Check current service status
echo "Current Service Status:"
systemctl status $SERVICE_NAME.service --no-pager -l | head -5 || echo "Service status check failed"

echo ""
echo "Next steps:"
echo "  Check logs: journalctl -u $SERVICE_NAME.service -f"
echo "  View topics: ros2 topic list"
echo "  Test camera: ros2 topic hz /camera/image_raw"
echo ""
echo -e "${GREEN}Environment variables loaded automatically on login${NC}"
echo -e "${YELLOW}Note: Service should start on next boot${NC}"
echo -e "${GREEN}==========================================${NC}"