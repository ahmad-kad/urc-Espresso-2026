# Webcam Testing for Hammer Detection Models

Real-time webcam testing script to evaluate model performance with live camera feed.

## Features

- **Real-time Detection**: Live object detection with bounding boxes and labels
- **Performance Metrics**: FPS, inference time, and detection counts
- **Flexible Model Loading**: Supports trained models or falls back to YOLOv8s
- **Interactive Controls**: Keyboard shortcuts for testing different scenarios
- **Class-specific Thresholds**: Optimized confidence thresholds for different object types

## Quick Start

### Basic Usage (with default YOLOv8s model)

```bash
python webcam_test.py
```

### Test with Trained Model

```bash
# Test baseline model
python webcam_test.py --model runs/baseline/yolov8s_baseline_hammer_detection/weights/best.pt

# Test CBAM-enhanced model
python webcam_test.py --model runs/cbam/yolov8s_cbam_hammer_detection/weights/best.pt
```

### Custom Configuration

```bash
# Use different camera and confidence threshold
python webcam_test.py --camera 1 --conf 0.3 --size 640

# Full command with all options
python webcam_test.py \
    --model runs/cbam/yolov8s_cbam_hammer_detection/weights/best.pt \
    --camera 0 \
    --conf 0.5 \
    --size 416
```

## Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | None | Path to trained YOLO model (.pt file) |
| `--camera` | `-c` | 0 | Camera device ID |
| `--conf` | - | 0.5 | Confidence threshold for detections |
| `--size` | - | 416 | Model input image size |

## Interactive Controls

During testing, use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| `q` / `Q` | Quit testing |
| `f` / `F` | Toggle FPS display |
| `t` / `T` | Toggle inference time display |
| `d` / `D` | Toggle detections display |
| `c` / `C` | Clear performance history |
| `p` / `P` | Print performance summary |
| `h` / `H` | Show/hide help |

## Object Classes

The script is configured for hammer detection with these classes:

1. **BrickHammer** - Green bounding boxes
2. **ArUcoTag** - Blue bounding boxes
3. **Bottle** - Red bounding boxes
4. **BrickHammer_duplicate** - Green bounding boxes
5. **OrangeHammer** - Green bounding boxes

## Class-Specific Thresholds

Different confidence thresholds are applied for safety-critical objects:

- **BrickHammer**: 0.6 (higher for safety)
- **ArUcoTag**: 0.7 (highest for precise detection)
- **Bottle**: 0.5 (standard)
- **OrangeHammer**: 0.6 (higher for safety)
- **BrickHammer_duplicate**: 0.6 (higher for safety)

## Performance Metrics

The script tracks and displays:

- **FPS**: Current frames per second
- **Inference Time**: Time taken for model prediction (ms)
- **Detection Count**: Number of objects detected in current frame
- **Rolling Statistics**: Average performance over last 100 frames

## Requirements

```bash
pip install opencv-python ultralytics torch numpy
```

## Troubleshooting

### Camera Issues
- Try different camera IDs: `--camera 1`, `--camera 2`, etc.
- Check camera permissions on macOS/Linux
- Ensure no other applications are using the camera

### Model Loading Issues
- Verify model path exists and is accessible
- Check if model was trained with compatible YOLO version
- Falls back to YOLOv8s if trained model not found

### Performance Issues
- Lower `--size` for faster inference (416 → 320)
- Increase `--conf` threshold to reduce false positives
- Close other applications using CPU/GPU resources

## Integration with Training Pipeline

After training models with `train_baseline.py` or `train_cbam.py`, test them immediately:

```bash
# Train baseline model
python src/train_baseline.py

# Test the trained model
python webcam_test.py --model runs/baseline/yolov8s_baseline_hammer_detection/weights/best.pt

# Compare with CBAM model
python src/train_cbam.py
python webcam_test.py --model runs/cbam/yolov8s_cbam_hammer_detection/weights/best.pt
```

## Expected Performance

On typical hardware:
- **CPU**: 5-15 FPS depending on model size and hardware
- **GPU**: 30-60+ FPS with CUDA acceleration
- **Raspberry Pi 4**: 2-5 FPS with optimized settings

Performance varies based on:
- Model complexity (CBAM models may be slower but more accurate)
- Input image size
- Hardware capabilities
- Number of concurrent detections

## Next Steps

1. **Train Models**: Use `train_baseline.py` or `train_cbam.py` to train on your dataset
2. **Test Performance**: Run webcam tests to evaluate real-world performance
3. **Optimize**: Adjust confidence thresholds and model parameters
4. **Deploy**: Use results for deployment decisions (see DEPLOYMENT_GUIDE.md)

## Related Files

- `src/train_baseline.py` - Train baseline YOLO model
- `src/train_cbam.py` - Train CBAM-enhanced model
- `test_deployment.py` - Comprehensive model comparison
- `ros2_ws/src/hammer_detection/` - ROS2 integration
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
