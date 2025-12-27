#!/bin/bash
# CPU-based retraining script for confidence-optimized models

echo "Starting confidence-focused retraining on CPU..."
echo "This will take significant time - monitor progress in output/confidence/"

# Function to train a model
train_model() {
    local config_file=$1
    local model_name=$(basename "$config_file" .yaml)

    echo "=========================================="
    echo "Training $model_name"
    echo "=========================================="

    # Load weights if this is a fine-tuned model
    if [[ $model_name == *"cbam"* ]]; then
        echo "Loading CBAM weights..."
        yolo train cfg="$config_file" model="output/models/yolov8s_cbam/weights/best.pt"
    elif [[ $model_name == *"efficientnet"* ]]; then
        echo "Loading EfficientNet weights..."
        yolo train cfg="$config_file" model="output/models/efficientnet/weights/best.pt"
    else
        # Standard training
        yolo train cfg="$config_file"
    fi

    echo "$model_name training completed"
    echo ""
}

# Train all models sequentially
echo "Training YOLOv8s (confidence-optimized)..."
train_model "configs/training/yolov8s_confidence.yaml"

echo "Training YOLOv8m (confidence-optimized)..."
train_model "configs/training/yolov8m_confidence.yaml"

echo "Training YOLOv8l (confidence-optimized)..."
train_model "configs/training/yolov8l_confidence.yaml"

echo "Training YOLOv8s-CBAM (confidence-optimized)..."
train_model "configs/training/yolov8s_cbam_confidence.yaml"

echo "Training EfficientNet (confidence-optimized)..."
train_model "configs/training/efficientnet_confidence.yaml"

echo "=========================================="
echo "All confidence training completed!"
echo "Check output/confidence/ for results"
echo "=========================================="
