# MobileNetV2 Training & IMX500 Deployment Guide

This guide shows how to train MobileNetV2 classification models that are **compatible with Sony's IMX500 MCT quantization** using your consolidated dataset.

## 🎯 Overview

Unlike YOLOv8 which is incompatible with IMX500's MCT toolkit, torchvision models like MobileNetV2 work perfectly with Sony's quantization pipeline.

## 📋 Supported Architectures

- ✅ **MobileNetV2** (recommended for IMX500)
- ✅ **MobileNetV3**
- ✅ **ResNet18/ResNet34**
- ✅ **EfficientNet-Lite** (approximated)
- ❌ **YOLOv8/v11** (incompatible with MCT)

## 🚀 Quick Start

### 1. Train MobileNetV2

```bash
# Train MobileNetV2 on consolidated dataset
python scripts/train_mobilenetv2.py --epochs 25 --batch-size 32

# Or use custom config
python scripts/train_mobilenetv2.py --config configs/mobilenetv2_classification.yaml
```

### 2. Test the Trained Model

```bash
# Test on validation set
python scripts/test_classification_model.py --model output/models/mobilenetv2_imx500/weights/best.pth
```

### 3. Quantize for IMX500

```bash
# MCT quantization + IMX500 conversion
python scripts/convert_to_imx500.py --model output/models/mobilenetv2_imx500/weights/best.pth
```

### 4. Deploy to IMX500

The final output will be in `output/imx500/packerOut.zip` ready for Raspberry Pi deployment.

## 📁 Project Structure

```
├── configs/
│   └── mobilenetv2_classification.yaml    # Training config
├── core/
│   ├── data/
│   │   └── classification_dataset.py      # Dataset loader
│   ├── models/
│   │   └── detector.py                    # Model loading (updated)
│   ├── trainer/
│   │   └── classification_trainer.py     # Classification training
│   └── trainer.py                        # Unified trainer
├── scripts/
│   ├── train_mobilenetv2.py              # Training script
│   ├── test_classification_model.py      # Testing script
│   └── convert_to_imx500.py              # MCT quantization
└── consolidated_dataset/                  # Your dataset
    ├── data.yaml
    ├── train/
    ├── val/
    └── test/
```

## ⚙️ Configuration

### Training Config (`configs/mobilenetv2_classification.yaml`)

```yaml
project:
  name: "mobilenetv2_imx500"

model:
  architecture: "mobilenetv2"    # Change this for other models
  input_size: 224
  num_classes: 3                 # Matches your dataset

data:
  path: "consolidated_dataset"

training:
  epochs: 25
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 1e-4
  num_workers: 4
```

### Supported Architectures

Change the `architecture` field to use different models:

```yaml
model:
  architecture: "mobilenetv2"     # MobileNetV2
  # architecture: "mobilenetv3"   # MobileNetV3-Large
  # architecture: "resnet18"      # ResNet18
  # architecture: "resnet34"      # ResNet34
  # architecture: "efficientnet_lite0"  # EfficientNet-Lite
```

## 🧪 Testing Your Model

After training, test accuracy on the validation set:

```bash
python scripts/test_classification_model.py \
  --model output/models/mobilenetv2_imx500/weights/best.pth \
  --architecture mobilenetv2
```

Expected output:
```
Testing model: output/models/mobilenetv2_imx500/weights/best.pth
Architecture: mobilenetv2
✓ Model loaded successfully
✓ Test data loader created
Test Accuracy: 95.23%

Per-class accuracy:
Bottle: 94.12%
BrickHammer: 96.45%
OrangeHammer: 95.12%

🎯 Final Result: 95.23% test accuracy
🎉 Excellent performance!
```

## 🔧 IMX500 Conversion Pipeline

The conversion pipeline now works seamlessly with torchvision models:

1. **MCT Quantization**: Uses Sony's Model Compression Toolkit
2. **Calibration**: Uses your validation images for quantization
3. **IMX500 Export**: Produces hardware-compatible models

```bash
# The conversion script automatically detects torchvision models
python scripts/convert_to_imx500.py --model your_trained_model.pth
```

## 🐛 Troubleshooting

### Issue: "Unsupported framework onnx"
**Solution**: Use the PyTorch model directly (`.pth`), not ONNX exports.

### Issue: Low accuracy after quantization
**Solutions**:
- Use more calibration images (increase from 50 to 100+)
- Ensure calibration images represent your use case
- Try different MCT quantization settings

### Issue: Model doesn't fit IMX500 memory
**Solutions**:
- Use MobileNetV2 instead of larger models
- Reduce input size from 224 to 192
- Use ResNet18 instead of ResNet34

## 📊 Performance Expectations

| Model | Size | Accuracy | IMX500 Compatible |
|-------|------|----------|-------------------|
| MobileNetV2 | ~14MB | 90-95% | ✅ Yes |
| MobileNetV3 | ~11MB | 88-93% | ✅ Yes |
| ResNet18 | ~45MB | 92-97% | ✅ Yes |
| ResNet34 | ~84MB | 93-98% | ⚠️ May be tight |
| YOLOv8 | ~50MB | 85-95% | ❌ No |

## 🎯 Next Steps

1. **Train your model** with the consolidated dataset
2. **Fine-tune** hyperparameters for your specific use case
3. **Deploy to IMX500** camera for real-time inference
4. **Optimize** for edge performance (quantization, pruning)

The key advantage: **MobileNetV2 + IMX500 gives you excellent accuracy with ultra-low latency and power consumption!**

