#!/bin/bash
# Run live camera inference with ROS2

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Live Camera Inference Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "ros_nodes/camera_detector_node.py" ]; then
    echo "Error: Must run from project root"
    exit 1
fi

MODEL="${1:-output/models/best.pt}"

if [ ! -f "$MODEL" ]; then
    echo "⚠ Model not found: $MODEL"
    echo "Available models:"
    find output/models -name "*.pt" 2>/dev/null | head -3
    find output/onnx -name "*.onnx" 2>/dev/null | head -3
    echo ""
    echo "Usage: $0 [model_path]"
    exit 1
fi

echo "Model: $MODEL"
echo ""

# Check for cameras
echo "Checking for cameras..."
CAMERAS=$(ls /dev/video* 2>/dev/null | wc -l)
if [ "$CAMERAS" -gt 0 ]; then
    echo "✓ Found $CAMERAS video device(s)"
    v4l2-ctl --list-devices 2>/dev/null | head -5 || true
else
    echo "⚠ No cameras found, will use test pattern"
fi

echo ""
echo "=========================================="
echo "Starting Live Inference"
echo "=========================================="
echo ""
echo "Opening 3 terminals..."
echo ""
echo "Terminal 1: Camera Publisher"
echo "Terminal 2: Detector Node"
echo "Terminal 3: Image Viewer"
echo ""
echo "Press Ctrl+C in each terminal to stop"
echo ""

# Function to run in new terminal
run_terminal() {
    local title="$1"
    local cmd="$2"
    gnome-terminal --title="$title" -- bash -c "$cmd; exec bash" 2>/dev/null || \
    xterm -title "$title" -e bash -c "$cmd; exec bash" 2>/dev/null || \
    echo "Please run manually: $cmd"
}

# Activate environment command
ACTIVATE="cd '$(pwd)' && source venv/bin/activate && source /opt/ros/jazzy/setup.bash && export PYTHONPATH='$(pwd):\$PYTHONPATH'"

# Terminal 1: Camera Publisher
echo "Starting camera publisher..."
run_terminal "Camera Publisher" "$ACTIVATE && python ros_nodes/camera_publisher.py --source 0 --test-mode"

sleep 2

# Terminal 2: Detector Node
echo "Starting detector node..."
run_terminal "Detector Node" "$ACTIVATE && python ros_nodes/camera_detector_node.py --model '$MODEL'"

sleep 2

# Terminal 3: Image Viewer
echo "Starting image viewer..."
run_terminal "Image Viewer" "source /opt/ros/jazzy/setup.bash && ros2 run rqt_image_view rqt_image_view /object_detector/annotated_image"

echo ""
echo "=========================================="
echo "All terminals started!"
echo "=========================================="
echo ""
echo "If terminals didn't open automatically, run these commands manually:"
echo ""
echo "Terminal 1:"
echo "  source activate_env.sh"
echo "  python ros_nodes/camera_publisher.py --source 0 --test-mode"
echo ""
echo "Terminal 2:"
echo "  source activate_env.sh"
echo "  python ros_nodes/camera_detector_node.py --model $MODEL"
echo ""
echo "Terminal 3:"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  ros2 run rqt_image_view rqt_image_view /object_detector/annotated_image"
echo ""



