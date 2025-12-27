@echo off
echo Starting comprehensive model training...
echo.

REM Train YOLO models first (most reliable)
echo Training YOLOv8 Nano models...
python scripts/train_all_models_from_scratch.py --architectures yolov8n --sizes 160 192 224 320 --epochs 100

echo Training YOLOv8 Small models...
python scripts/train_all_models_from_scratch.py --architectures yolov8s --sizes 160 192 224 320 --epochs 100

echo Training YOLOv8 Medium models...
python scripts/train_all_models_from_scratch.py --architectures yolov8m --sizes 160 192 224 320 --epochs 100

echo Training YOLOv8 Large models...
python scripts/train_all_models_from_scratch.py --architectures yolov8l --sizes 160 192 224 320 --epochs 100

REM Train custom models (if working)
echo Training EfficientNet models...
python scripts/train_all_models_from_scratch.py --architectures efficientnet --sizes 160 192 224 320 --epochs 100

echo Training MobileNet-ViT models...
python scripts/train_all_models_from_scratch.py --architectures mobilenet_vit --sizes 160 192 224 320 --epochs 100

echo.
echo All training completed!
echo Check output/full_training_from_scratch_summary.txt for results
pause



