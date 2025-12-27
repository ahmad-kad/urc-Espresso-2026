# Raspberry Pi 5 Object Detection Stream

Real-time object detection script for streaming camera feed and highlighting detected objects.

## Installation

### 1. Install Dependencies

```bash
pip install -r scripts/requirements_pi5.txt
```

Or install manually:
```bash
pip install opencv-python onnxruntime numpy
```

### 2. For Pi Camera (Optional)

If using the Raspberry Pi camera module, you may need:
```bash
sudo apt-get install python3-picamera2
```

## Usage

### Basic Usage

```bash
python scripts/stream_detection_pi5.py --model output/deployment_models/mobilenet_224_224.onnx
```

### With Custom Settings

```bash
python scripts/stream_detection_pi5.py \
    --model output/deployment_models/yolov8s_confidence_224.onnx \
    --input-size 224 \
    --conf 0.3 \
    --iou 0.45 \
    --camera 0 \
    --width 640 \
    --height 480 \
    --fps-display
```

### Arguments

- `--model`: Path to ONNX model file (required)
- `--input-size`: Model input size (default: 224)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--camera`: Camera index (default: 0)
- `--width`: Camera width (default: 640)
- `--height`: Camera height (default: 480)
- `--fps-display`: Display FPS counter

## Controls

- **'q'**: Quit the application
- **'s'**: Save current frame as image

## Recommended Models for Pi 5

### For Best Performance (Speed)
```bash
python scripts/stream_detection_pi5.py \
    --model output/deployment_models/mobilenet_224_224.onnx \
    --input-size 224 \
    --fps-display
```
- Expected: ~250 FPS
- Accuracy: 0.789 mAP50

### For Best Accuracy
```bash
python scripts/stream_detection_pi5.py \
    --model output/deployment_models/yolov8s_cbam_confidence_cbam_224.onnx \
    --input-size 224 \
    --fps-display
```
- Expected: ~124 FPS
- Accuracy: 0.883 mAP50

### Balanced Option
```bash
python scripts/stream_detection_pi5.py \
    --model output/deployment_models/yolov8s_confidence_224.onnx \
    --input-size 224 \
    --fps-display
```
- Expected: ~125 FPS
- Accuracy: 0.880 mAP50

## Performance Tips

1. **Lower Resolution**: Use lower camera resolution (e.g., 320x240) for better FPS
2. **Adjust Confidence**: Lower confidence threshold (--conf 0.2) for more detections
3. **Model Selection**: Use MobileNet models for maximum speed
4. **Input Size**: Smaller input sizes (160 or 192) will be faster but less accurate

## Troubleshooting

### Camera Not Found
- Try different camera indices: `--camera 0`, `--camera 1`, etc.
- Check camera permissions: `sudo usermod -a -G video $USER`
- For Pi Camera: May need to use `libcamera` instead

### Low FPS
- Reduce camera resolution
- Use MobileNet models instead of YOLOv8
- Lower input size (e.g., 160 instead of 224)
- Close other applications

### Model Loading Errors
- Ensure ONNX model file exists and is valid
- Check that input-size matches the model's expected input
- Verify onnxruntime is installed correctly

## Example Output

The script will:
- Display live camera feed with bounding boxes
- Show detected objects with class names and confidence
- Display FPS and inference time (if --fps-display is used)
- Color-code detections by class:
  - Green: ArUcoTag
  - Blue: Bottle
  - Red: BrickHammer
  - Orange: OrangeHammer
  - Magenta: USB-A
  - Cyan: USB-C

## Notes

- First run may be slower due to model loading
- FPS will vary based on model complexity and Pi 5 load
- For IMX500 camera, you may need additional setup for the AI camera module
