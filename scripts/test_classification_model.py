#!/usr/bin/env python3
"""
Test trained classification model on consolidated dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torchvision.models as models

from core.data.classification_dataset import create_classification_data_loaders


def load_model(model_path: str, architecture: str = "mobilenetv2"):
    """Load the trained model"""
    # Create base model
    if architecture == "mobilenetv2":
        model = models.mobilenet_v2()
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Load trained weights
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model


def test_model(model, test_loader, device="cpu"):
    """Test the model on test dataset"""
    model = model.to(device)
    correct = 0
    total = 0
    class_correct = [0] * 3
    class_total = [0] * 3

    class_names = ["Bottle", "BrickHammer", "OrangeHammer"]

    # Use test_loader if it has samples, otherwise skip
    loader_to_use = test_loader if len(test_loader.dataset) > 0 else None

    if loader_to_use is None:
        print("No test samples available, cannot test model")
        return 0.0

    with torch.no_grad():
        for images, labels in loader_to_use:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    # Overall accuracy
    if total == 0:
        print("No samples were processed")
        return 0.0

    accuracy = 100 * correct / total
    print(".2f")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(3):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(".2f")
        else:
            print(f"{class_names[i]}: No samples")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Test classification model")
    parser.add_argument(
        "--model", required=True, help="Path to trained model (.pth file)"
    )
    parser.add_argument(
        "--architecture", default="mobilenetv2", help="Model architecture"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for testing"
    )

    args = parser.parse_args()

    print(f"Testing model: {args.model}")
    print(f"Architecture: {args.architecture}")

    # Load model
    model = load_model(args.model, args.architecture)
    print("✓ Model loaded successfully")

    # Create data loaders
    train_loader, val_loader, test_loader = create_classification_data_loaders(
        data_dir="consolidated_dataset",
        batch_size=args.batch_size,
        target_size=(224, 224),
    )
    print("✓ Data loaders created")

    # Test the model (use validation set if test set is empty)
    test_data_loader = test_loader if len(test_loader.dataset) > 0 else val_loader
    test_set_name = "test" if len(test_loader.dataset) > 0 else "validation"

    print(
        f"Testing on {test_set_name} set with {len(test_data_loader.dataset)} samples"
    )

    # Test the model
    accuracy = test_model(model, test_data_loader)

    print(f"\n🎯 Final Result: {accuracy:.2f}% test accuracy")

    if accuracy > 90:
        print("🎉 Excellent performance!")
    elif accuracy > 80:
        print("👍 Good performance!")
    else:
        print("⚠️  Model may need more training or fine-tuning")


if __name__ == "__main__":
    main()
