#!/usr/bin/env python3
"""
Data preparation script for accuracy-focused YOLO training
Creates proper directory structure and data.yaml configuration
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List
import yaml


def create_data_yaml(
    dataset_path: str,
    classes: List[str],
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
) -> str:
    """
    Create data.yaml file for YOLO training

    Args:
        dataset_path: Path to dataset directory
        classes: List of class names
        train_split: Training data proportion
        val_split: Validation data proportion
        test_split: Test data proportion

    Returns:
        Path to created data.yaml file
    """
    data_yaml = {
        "path": str(Path(dataset_path).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: cls for i, cls in enumerate(classes)},
        "nc": len(classes),
    }

    yaml_path = Path(dataset_path) / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Created data.yaml at: {yaml_path}")
    return str(yaml_path)


def create_directory_structure(base_path: str, classes: List[str]):
    """
    Create YOLO dataset directory structure

    Args:
        base_path: Base dataset directory path
        classes: List of class names
    """
    base_path = Path(base_path)

    # Create main directories
    dirs = [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]

    for dir_path in dirs:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)

    print(f"Created directory structure in: {base_path}")
    print("Expected structure:")
    print(f"  {base_path}/")
    print("    images/")
    print("      train/  <- Training images")
    print("      val/    <- Validation images")
    print("      test/   <- Test images")
    print("    labels/")
    print("      train/  <- Training labels (.txt)")
    print("      val/    <- Validation labels (.txt)")
    print("      test/   <- Test labels (.txt)")
    print("    data.yaml <- Dataset configuration")


def validate_dataset_structure(dataset_path: str) -> bool:
    """
    Validate that dataset has proper YOLO structure

    Args:
        dataset_path: Path to dataset directory

    Returns:
        True if valid, False otherwise
    """
    base_path = Path(dataset_path)

    required_dirs = ["train/images", "train/labels", "val/images", "val/labels"]

    missing_dirs = []
    for dir_path in required_dirs:
        if not (base_path / dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print("Missing directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False

    # Check for data.yaml
    if not (base_path / "data.yaml").exists():
        print("Missing data.yaml file")
        return False

    # Check for images and labels
    train_images = list((base_path / "images/train").glob("*"))
    train_labels = list((base_path / "labels/train").glob("*.txt"))

    if len(train_images) == 0:
        print("Warning: No training images found")
    else:
        print(f"Found {len(train_images)} training images")

    if len(train_labels) == 0:
        print("Warning: No training labels found")
    else:
        print(f"Found {len(train_labels)} training labels")

    return True


def main():
    """Main data preparation function"""
    parser = argparse.ArgumentParser(description="Prepare YOLO Training Data")
    parser.add_argument(
        "--dataset-path", required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="List of class names (e.g., --classes cat dog person)",
    )
    parser.add_argument(
        "--create-structure", action="store_true", help="Create directory structure"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate existing dataset structure"
    )
    parser.add_argument(
        "--create-yaml", action="store_true", help="Create data.yaml file"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if args.create_structure:
        print("Creating dataset directory structure...")
        create_directory_structure(str(dataset_path), args.classes)

    if args.create_yaml:
        print("Creating data.yaml configuration...")
        create_data_yaml(str(dataset_path), args.classes)

    if args.validate:
        print("Validating dataset structure...")
        is_valid = validate_dataset_structure(str(dataset_path))

        if is_valid:
            print("Dataset structure is valid!")
        else:
            print("Dataset structure has issues. Please fix them before training.")

    if not any([args.create_structure, args.validate, args.create_yaml]):
        print("No action specified. Use --help for options.")
        print("\nExample usage:")
        print("  # Create structure and config")
        print(
            "  python scripts/prepare_training_data.py --dataset-path ./data --classes cup plate fork --create-structure --create-yaml"
        )
        print()
        print("  # Validate existing dataset")
        print(
            "  python scripts/prepare_training_data.py --dataset-path ./data --validate"
        )


if __name__ == "__main__":
    main()
