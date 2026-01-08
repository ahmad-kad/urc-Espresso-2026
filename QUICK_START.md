# Quick Start Guide

Get your MobileNetV2 model trained and deployed to IMX500 in minutes!

## What You'll Accomplish

1. **Train MobileNetV2** to 99.17% accuracy on your robotics dataset
2. **Quantize with Sony MCT** for IMX500 compatibility
3. **Deploy to Raspberry Pi** AI Camera
4. **Run live inference** with ROS2

## Prerequisites

- Python 3.10+
- 4GB+ RAM
- GPU recommended (but not required)

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
# Option A: Complete installation (recommended)
pip install -e .[gpu,ros2]

# Option B: Minimal installation
pip install -e .

# Option C: From requirements file
pip install -r requirements.txt
```

### Step 2: Train MobileNetV2 (2 minutes)
```bash
# Train on your consolidated dataset
python scripts/train_mobilenetv2.py --epochs 25 --batch-size 32

# Watch the progress bars - you'll see ~99% accuracy!
```

### Step 3: Test Your Model (1 minute)
```bash
# Test the trained model
python scripts/test_classification_model.py \
  --model output/models/mobilenetv2_imx500/weights/best.pth

# Expected: 99.17% accuracy with per-class breakdown
```

### Step 4: Deploy to IMX500 (2 minutes)
```bash
# Sony MCT quantization + IMX500 conversion
python scripts/convert_to_imx500.py \
  --model output/models/mobilenetv2_imx500/weights/best.pth

# Result: output/imx500/packerOut.zip ready for Raspberry Pi!
```

## Automated Script

Use the automated launcher:
```bash
./run_live_inference.sh [model_path]
```

This will automatically open 3 terminals for you.

## What You'll See

- **Terminal 1**: Camera publisher running (test pattern or real camera)
- **Terminal 2**: Detector node processing frames and showing inference stats
- **Terminal 3**: Live annotated video stream with bounding boxes around detected objects

## Camera Options

### Webcam / USB Camera

```bash
# Try different camera indices
python ros_nodes/camera_publisher.py --source 0
python ros_nodes/camera_publisher.py --source 1
python ros_nodes/camera_publisher.py --source 2
```

### RealSense Camera

If you have a RealSense camera:

1. **Install RealSense SDK (if needed):**
   ```bash
   pip install pyrealsense2
   ```

2. **Check if RealSense is detected:**
   ```bash
   python -c "import pyrealsense2 as rs; ctx = rs.context(); print(f'Found {len(ctx.query_devices())} device(s)')"
   ```

3. **Use RealSense directly:**
   ```bash
   python ros_nodes/camera_publisher.py --source 0
   ```

4. **Or use RealSense ROS2 node:**
   ```bash
   # Terminal 1: Start RealSense ROS2 node
   ros2 run realsense2_camera realsense2_camera_node
   
   # Terminal 2: Start detector (subscribes to /camera/camera/color/image_raw)
   source activate_env.sh
   python ros_nodes/camera_detector_node.py --model output/models/best.pt
   
   # Terminal 3: View results
   ros2 run rqt_image_view rqt_image_view /object_detector/annotated_image
   ```

5. **Or use RealSense bridge:**
   ```bash
   # Terminal 1: RealSense bridge
   python ros_nodes/realsense_bridge.py
   
   # Terminal 2: Detector
   python ros_nodes/camera_detector_node.py --model output/models/best.pt
   ```

### Video File

```bash
python ros_nodes/camera_publisher.py --source /path/to/video.mp4
```

## Model Options

```bash
# PyTorch model
python ros_nodes/camera_detector_node.py --model output/models/best.pt

# ONNX FP32 (faster)
python ros_nodes/camera_detector_node.py --model output/onnx/best_fp32.onnx

# ONNX INT8 (fastest)
python ros_nodes/camera_detector_node.py --model output/onnx/best_int8.onnx
```

## Troubleshooting

### No Camera Found

Use `--test-mode` flag to test without camera:
```bash
python ros_nodes/camera_publisher.py --source 0 --test-mode
```

### Camera Permissions

```bash
# Check if you're in the video group
groups | grep video

# If not, add yourself (requires logout/login)
sudo usermod -a -G video $USER
```

### Check Available Cameras

```bash
# List video devices
ls -la /dev/video*

# Check camera info
v4l2-ctl --list-devices

# Check USB devices (for RealSense)
lsusb | grep -i intel
```

### Camera Being Used by Another Process

```bash
# Check what's using the camera
lsof /dev/video*

# Kill processes if needed
killall <process_name>
```

### RealSense Specific Issues

1. **Check USB connection:**
   ```bash
   lsusb | grep -i intel
   ```

2. **Try RealSense viewer:**
   ```bash
   realsense-viewer
   ```

3. **Check RealSense topics (if using ROS2 node):**
   ```bash
   ros2 topic list | grep camera
   ros2 topic echo /camera/camera/color/image_raw --once
   ```

### View Topics

```bash
# List all topics
ros2 topic list

# Check camera topic
ros2 topic echo /camera/image_raw --once

# Check detector output
ros2 topic echo /object_detector/annotated_image --once
```

### No Images Received

1. **Check if camera publisher is running:**
   ```bash
   ros2 topic list
   ros2 topic echo /camera/image_raw --once
   ```

2. **Check detector is subscribed:**
   ```bash
   ros2 topic info /camera/image_raw
   ```

3. **Verify topic names match:**
   - Publisher: `/camera/image_raw` (default)
   - Detector: Configured in `configs/robotics.yaml`

## Performance Tips

- Use ONNX INT8 model for fastest inference
- Reduce FPS if CPU is overloaded: `--fps 15`
- Adjust confidence threshold in config file for more/fewer detections
- Use compression for network streaming (see `ros_nodes/BENCHMARK_GUIDE.md`)

## Configuration

Edit `configs/robotics.yaml` to configure:
- Input topic (default: `/camera/image_raw`)
- Confidence threshold
- Input size
- ROS2 settings

## Expected Output

When everything is working:

**Camera Publisher:**
```
[INFO] [camera_publisher]: RealSense camera initialized successfully
```
OR
```
[INFO] [camera_publisher]: ✓ Camera 0 working with V4L2 backend
```

**Detector Node:**
```
[INFO] [camera_detector_node]: Processed 30 frames
[INFO] [camera_detector_node]: Inference time: 15.2ms
```

**Viewer:**
Live annotated video stream with bounding boxes around detected objects!

## Next Steps

- See [SETUP.md](SETUP.md) for detailed environment setup
- See [ros_nodes/README.md](ros_nodes/README.md) for ROS2 node documentation
- See [ros_nodes/BENCHMARK_GUIDE.md](ros_nodes/BENCHMARK_GUIDE.md) for performance benchmarking
- See [RETRAINING_GUIDE.md](RETRAINING_GUIDE.md) for model retraining

