"""
Data processing and loading utilities
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_data_config(data_yaml_path: str) -> Dict:
    """
    Load and validate data configuration from YAML
    """

    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ["train", "val", "nc", "names"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in data config: {field}")

    logger.info(f"Loaded data config: {config['nc']} classes, {config['names']}")

    return config


def get_image_paths(data_split: str, data_root: str = ".") -> List[str]:
    """
    Get all image paths for a data split
    """

    if isinstance(data_split, list):
        # Multiple directories
        image_paths = []
        for split_dir in data_split:
            split_path = Path(data_root) / split_dir
            if split_path.exists():
                image_paths.extend(get_image_paths_from_dir(split_path))
        return image_paths
    else:
        # Single directory
        split_path = Path(data_root) / data_split
        return get_image_paths_from_dir(split_path)


def get_image_paths_from_dir(directory: Path) -> List[str]:
    """
    Get all image paths from a directory
    """

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = []

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return image_paths

    for file_path in directory.rglob("*"):
        if file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))

    logger.info(f"Found {len(image_paths)} images in {directory}")

    return sorted(image_paths)


def validate_dataset(data_config: Dict, data_root: str = ".") -> Dict:
    """
    Validate dataset integrity and return statistics
    """

    stats = {
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "classes": data_config["names"],
        "num_classes": data_config["nc"],
        "issues": [],
    }

    # Check training images
    try:
        train_paths = get_image_paths(data_config["train"], data_root)
        stats["train_images"] = len(train_paths)
        stats["total_images"] += len(train_paths)
    except Exception as e:
        stats["issues"].append(f"Error loading train images: {e}")

    # Check validation images
    try:
        val_paths = get_image_paths(data_config["val"], data_root)
        stats["val_images"] = len(val_paths)
        stats["total_images"] += len(val_paths)
    except Exception as e:
        stats["issues"].append(f"Error loading val images: {e}")

    # Check test images if available
    if "test" in data_config:
        try:
            test_paths = get_image_paths(data_config["test"], data_root)
            stats["test_images"] = len(test_paths)
            stats["total_images"] += len(test_paths)
        except Exception as e:
            stats["issues"].append(f"Error loading test images: {e}")

    # Check class balance (rough estimate)
    if stats["issues"]:
        logger.warning("Dataset validation issues found:")
        for issue in stats["issues"]:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Dataset validation passed")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(
            f"  Train/Val/Test: {stats['train_images']}/{stats['val_images']}/{stats['test_images']}"
        )
        logger.info(
            f"  Classes: {stats['num_classes']} ({', '.join(stats['classes'])})"
        )

    return stats


def create_data_yaml(
    output_path: str,
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    class_names: List[str] = None,
) -> str:
    """
    Create a data.yaml file for YOLO training
    """

    if class_names is None:
        # Try to infer from directory structure or use generic names
        class_names = (
            [f"class_{i}" for i in range(len(class_names))]
            if class_names
            else ["object"]
        )

    data_config = {
        "train": train_dir,
        "val": val_dir,
        "nc": len(class_names),
        "names": class_names,
    }

    if test_dir:
        data_config["test"] = test_dir

    # Write to file
    with open(output_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created data configuration: {output_path}")

    return output_path


def analyze_image_sizes(image_paths: List[str], sample_size: int = 1000) -> Dict:
    """
    Analyze image dimensions in the dataset
    """

    sizes = []
    sampled_paths = (
        image_paths[:sample_size] if len(image_paths) > sample_size else image_paths
    )

    for img_path in sampled_paths:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                sizes.append((w, h))
        except Exception as e:
            logger.warning(f"Could not read image {img_path}: {e}")

    if not sizes:
        return {"error": "No valid images found"}

    widths, heights = zip(*sizes)

    return {
        "count": len(sizes),
        "avg_width": int(np.mean(widths)),
        "avg_height": int(np.mean(heights)),
        "min_width": int(np.min(widths)),
        "min_height": int(np.min(heights)),
        "max_width": int(np.max(widths)),
        "max_height": int(np.max(heights)),
        "recommended_size": _recommend_input_size(widths, heights),
    }


def _recommend_input_size(widths: List[int], heights: List[int]) -> int:
    """
    Recommend optimal input size based on image dimensions
    """

    avg_size = (np.mean(widths) + np.mean(heights)) / 2

    # Common YOLO input sizes
    common_sizes = [320, 416, 512, 640, 768, 896, 1024, 1280, 1408, 1536]

    # Find closest common size
    recommended = min(common_sizes, key=lambda x: abs(x - avg_size))

    # Prefer larger sizes for better accuracy, smaller for speed
    if avg_size > 640:
        recommended = max(recommended, 640)  # Minimum for good accuracy
    elif avg_size < 416:
        recommended = min(recommended, 416)  # Maximum for speed

    return recommended


def load_yolo_labels(label_path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Load YOLO format labels from a text file
    Returns: boxes (list of [x,y,w,h] in normalized coords), classes (list of class ids)
    """
    boxes = []
    classes = []

    try:
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])

                    classes.append(class_id)
                    boxes.append([x, y, w, h])

    except FileNotFoundError:
        pass  # Return empty lists if file doesn't exist

    return boxes, classes


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Split dataset into train/val/test sets
    """

    import shutil

    from sklearn.model_selection import train_test_split

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = []

    for ext in image_extensions:
        all_images.extend(list(source_path.glob(f"**/*{ext}")))
        all_images.extend(list(source_path.glob(f"**/*{ext.upper()}")))

    if not all_images:
        raise ValueError(f"No image files found in {source_dir}")

    # Shuffle and split
    all_images = [str(p) for p in all_images]
    np.random.seed(seed)
    np.random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_images = all_images[:n_train]
    val_images = all_images[n_train : n_train + n_val]
    test_images = all_images[n_train + n_val :]

    # Create output directories
    splits = {
        "train": (train_images, output_path / "train"),
        "val": (val_images, output_path / "valid"),
        "test": (test_images, output_path / "test"),
    }

    for split_name, (images, split_dir) in splits.items():
        if not images:
            continue

        split_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            img_name = Path(img_path).name
            shutil.copy2(img_path, split_dir / img_name)

        logger.info(f"Created {split_name} split: {len(images)} images in {split_dir}")

    return {
        "train": str(splits["train"][1]),
        "val": str(splits["val"][1]),
        "test": str(splits["test"][1]) if test_images else None,
    }
