#!/bin/bash
# Verify deployment is working correctly

cd "$(dirname "$0")"
source venv/bin/activate
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "=========================================="
echo "Deployment Verification"
echo "=========================================="
echo ""

# Check 1: RealSense topic
echo "1. Checking RealSense camera topic..."
if ros2 topic list | grep -q "/camera/camera/color/image_raw"; then
    echo "   ✓ RealSense topic exists"
    ros2 topic hz /camera/camera/color/image_raw --window 5 2>&1 | head -3 || echo "   ⚠ No frames yet"
else
    echo "   ✗ RealSense topic not found"
fi

echo ""

# Check 2: Detector topic
echo "2. Checking detector output topic..."
if ros2 topic list | grep -q "/object_detector/annotated_image"; then
    echo "   ✓ Annotated image topic exists"
    ros2 topic hz /object_detector/annotated_image --window 5 2>&1 | head -3 || echo "   ⚠ No frames yet"
else
    echo "   ✗ Annotated image topic not found (detector may not be running)"
fi

echo ""

# Check 3: Model file
echo "3. Checking model file..."
MODEL="output/models/best.pt"
if [ -f "$MODEL" ]; then
    echo "   ✓ Model file exists: $MODEL"
    ls -lh "$MODEL" | awk '{print "     Size: " $5}'
else
    echo "   ✗ Model file not found: $MODEL"
fi

echo ""

# Check 4: Config
echo "4. Checking configuration..."
python3 -c "from core.config.manager import load_config; c = load_config('robotics'); print('   ✓ Config loads'); print(f'     Input topic: {c[\"ros2\"][\"input_topic\"]}')" 2>&1 | grep -E "(✓|Input topic)" || echo "   ⚠ Config issue"

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "If all checks pass, your deployment is working!"
echo ""
echo "To view annotated images:"
echo "  ros2 run rqt_image_view rqt_image_view /object_detector/annotated_image"
echo ""



