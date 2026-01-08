"""
Data processing and loading utilities
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_data_config(data_yaml_path: str) -> Dict[str, Any]:
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


def get_image_paths(data_split: Union[str, List[str]], data_root: str = ".") -> List[str]:
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
    image_paths: List[str] = []

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return image_paths

    for file_path in directory.rglob("*"):
        if file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))

    logger.info(f"Found {len(image_paths)} images in {directory}")

    return sorted(image_paths)


def create_data_yaml(
    output_path: str,
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Create a data.yaml file for YOLO training
    """

    if class_names is None:
        # Use generic names if not provided
        class_names = ["object"]

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


def analyze_image_sizes(image_paths: List[str], sample_size: int = 1000) -> Dict[str, Any]:
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
        "recommended_size": _recommend_input_size(list(widths), list(heights)),
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
