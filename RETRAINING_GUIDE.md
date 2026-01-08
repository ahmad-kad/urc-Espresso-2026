# MobileNetV2 Retraining & Fine-Tuning Guide

**When to Retrain Your Classification Model for Better Robotics Performance**

## Current Status: 99.17% Accuracy Achieved!

Your MobileNetV2 model is performing **exceptionally well** at 99.17% validation accuracy. This guide helps you decide when additional training might be beneficial.

## When Do You Need to Retrain?

### You DON'T Need to Retrain If:
- Current model performs well (99.17% accuracy is excellent!)
- You want faster inference (use quantization instead)
- You need ROS2 integration (already working)
- You want IMX500 deployment (already compatible)

### You SHOULD Retrain If:
- You have new object classes to add
- Model struggles with specific lighting conditions
- You want to experiment with different architectures
- You have additional training data available
- Model performance drops in production environment

## What Works WITHOUT Retraining

These improvements work immediately with your current `best.pt` model:

### ✅ Inference-Time Improvements (No Retraining Needed)
1. **Image Preprocessing** (lighting robustness)
   - Adaptive brightness normalization
   - Works immediately, improves detection in varying lighting

2. **Temporal Smoothing**
   - Stabilizes detections across frames
   - Works immediately, reduces flickering

3. **Optimized Configuration**
   - Better confidence thresholds
   - Edge device optimizations
   - Works immediately

### ❌ Model-Level Improvements (Requires Retraining)

These require retraining because they change what the model learns:

1. **Perspective Robustness** (`perspective: 0.0005`)
   - Model needs to see perspective-transformed examples during training
   - Current model wasn't trained with this

2. **Occlusion Handling** (`mosaic: 1.0`, `mixup: 0.15`, `copy_paste: 0.3`)
   - Model needs to learn from occluded examples
   - Current model wasn't trained with occlusion simulation

3. **Multi-Scale Training** (`multi_scale: true`)
   - Model learns to handle objects at different scales
   - Current model was trained at fixed size

4. **Enhanced Geometric Augmentations**
   - Increased rotation (`degrees: 20.0`)
   - Shear transformations (`shear: 10.0`)
   - Larger scale variation (`scale: 0.9`)
   - Current model has limited geometric robustness

## How to Decide

### Test Your Current Model First

1. **Test with current model + inference improvements:**
   ```bash
   python ros_nodes/camera_detector_node.py --model output/models/best.pt
   ```

2. **Check if it handles:**
   - ✅ Different lighting conditions (should work with preprocessing)
   - ❓ Different perspectives/viewpoints (may need retraining)
   - ❓ Partially occluded objects (may need retraining)
   - ❓ Objects at very different distances (may need retraining)

3. **If it fails on perspective/occlusion/distance → Retrain**

## How to Retrain

### Option 1: Quick Retrain (Recommended First)

Retrain with the new robust augmentations:

```bash
# Using the robotics config with all new augmentations
python cli/train.py \
  --config robotics \
  --data-yaml consolidated_dataset/data.yaml \
  --epochs 100 \
  --model yolov8s \
  --imgsz 416
```

This will:
- Use all the new augmentations (perspective, occlusion, multi-scale)
- Train for 100 epochs
- Save to `output/models/robotics_object_detection_training/weights/best.pt`

### Option 2: Resume from Current Model (Faster)

Fine-tune your existing model with new augmentations:

```bash
python cli/train.py \
  --config robotics \
  --data-yaml consolidated_dataset/data.yaml \
  --resume \
  --resume-path output/models/best.pt \
  --epochs 50  # Fewer epochs since starting from trained model
```

### Option 3: Edge-Optimized Training

For edge devices, you might want a smaller model:

```bash
# Train yolov8n (nano) instead of yolov8s (small)
python cli/train.py \
  --config robotics \
  --data-yaml consolidated_dataset/data.yaml \
  --model yolov8n \
  --imgsz 320 \
  --epochs 100
```

## Training Time Estimates

- **Full retrain (100 epochs)**: ~4-8 hours (depends on GPU/CPU)
- **Resume training (50 epochs)**: ~2-4 hours
- **Edge model (yolov8n, 320px)**: ~2-4 hours

## After Retraining

1. **Test the new model:**
   ```bash
   python ros_nodes/camera_detector_node.py --model output/models/[new_model_path]/weights/best.pt
   ```

2. **Convert to ONNX for edge deployment:**
   ```bash
   python scripts/convert_to_onnx.py \
     --model output/models/[new_model_path]/weights/best.pt \
     --int8  # For edge devices
   ```

3. **Compare performance:**
   - Test on challenging scenarios (occlusion, perspective, distance)
   - Measure accuracy improvement
   - Check inference speed (should be similar)

## Recommendation

**Start with inference improvements** (already done):
- Test current model with new preprocessing and temporal smoothing
- See if it meets your needs

**If not sufficient, retrain:**
- Use `--resume` to fine-tune existing model (faster)
- Or full retrain if you want maximum robustness

**For edge devices:**
- Consider `yolov8n` + `320px` input size
- Use INT8 quantized ONNX model
- Trade some accuracy for speed/memory


