#!/bin/bash
# Activation script for ROS2 Camera Detector environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found. Run setup_env.sh first."
    return 1
fi

# Check if ROS2 is available
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
    echo "✓ ROS2 Jazzy sourced"
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "✓ ROS2 Humble sourced"
elif [ -f "/opt/ros/foxy/setup.bash" ]; then
    source /opt/ros/foxy/setup.bash
    echo "✓ ROS2 Foxy sourced"
elif [ -f "/opt/ros/galactic/setup.bash" ]; then
    source /opt/ros/galactic/setup.bash
    echo "✓ ROS2 Galactic sourced"
else
    echo "⚠ ROS2 not found in standard location"
    echo "  If ROS2 is installed elsewhere, source it manually:"
    echo "  source /path/to/ros2/install/setup.bash"
fi

# Add current directory to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "✓ Added project to PYTHONPATH"

echo ""
echo "Environment ready!"
echo "Current directory: $SCRIPT_DIR"
echo ""
echo "Quick test commands:"
echo "  python ros_nodes/test_setup.py --model output/models/best.pt"
echo "  python ros_nodes/camera_detector_node.py --model output/models/best.pt"
echo ""

