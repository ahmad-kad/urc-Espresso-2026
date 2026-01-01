#!/usr/bin/env python3
"""
Consolidated dataset utilities for data loading, augmentation, preprocessing, and analysis.
Combines functionality from data_analysis.py, enhanced_data_augmentation.py,
fix_class_imbalance.py, generate_synthetic_aruco.py, merge_hammer_classes.py,
and simple_aruco_generator.py
"""

import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split


class YOLODataBalancer:
    """Data balancing and splitting utilities"""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"

        # Output directories
        self.output_dir = Path("balanced_data")
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"

        # Create output directories
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            (dir_path / "images").mkdir(parents=True, exist_ok=True)
            (dir_path / "labels").mkdir(parents=True, exist_ok=True)

    def analyze_class_distribution(self):
        """Analyze the current class distribution in the dataset"""
        print("Analyzing class distribution...")

        class_counts = defaultdict(int)
        image_class_map = defaultdict(list)

        label_files = list(self.labels_dir.glob("*.txt"))

        for label_file in label_files:
            image_name = label_file.stem.replace(".txt", "")
            classes_in_image = set()

            try:
                with open(label_file, "r") as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                                classes_in_image.add(class_id)
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
                continue

            # Map image to its classes
            for class_id in classes_in_image:
                image_class_map[class_id].append(image_name)

        return class_counts, image_class_map

    def create_balanced_split(
        self, class_counts, image_class_map, target_samples_per_class=200
    ):
        """Create balanced train/val/test splits"""
        print("Creating balanced splits...")

        # Calculate split ratios
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

        train_images = set()
        val_images = set()
        test_images = set()

        for class_id, images in image_class_map.items():
            if len(images) < 10:  # Skip classes with too few samples
                continue

            # Use stratification for better balance
            class_train, temp = train_test_split(
                images, train_size=train_ratio, random_state=42
            )
            class_val, class_test = train_test_split(
                temp, test_size=test_ratio / (test_ratio + val_ratio), random_state=42
            )

            train_images.update(class_train)
            val_images.update(class_val)
            test_images.update(class_test)

        return list(train_images), list(val_images), list(test_images)

    def copy_files_to_split(self, images, split_dir):
        """Copy image and label files to split directory"""
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        for image_name in images:
            # Copy image
            for ext in [".jpg", ".jpeg", ".png"]:
                src_image = self.images_dir / f"{image_name}{ext}"
                if src_image.exists():
                    shutil.copy2(src_image, images_dir / f"{image_name}{ext}")
                    break

            # Copy label
            src_label = self.labels_dir / f"{image_name}.txt"
            if src_label.exists():
                shutil.copy2(src_label, labels_dir / f"{image_name}.txt")

    def create_data_yaml(self):
        """Create data.yaml for the balanced dataset"""
        data_yaml = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {0: "ArUcoTag", 1: "Bottle", 2: "BrickHammer", 3: "OrangeHammer"},
        }

        with open(self.output_dir / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"Created data.yaml at {self.output_dir / 'data.yaml'}")

    def balance_dataset(self):
        """Main method to balance the dataset"""
        print("Starting dataset balancing...")

        # Analyze current distribution
        class_counts, image_class_map = self.analyze_class_distribution()

        print("Current class distribution:")
        for class_id, count in class_counts.items():
            print(f"  Class {class_id}: {count} samples")

        # Create balanced splits
        train_images, val_images, test_images = self.create_balanced_split(
            class_counts, image_class_map
        )

        print(
            f"Balanced split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test"
        )

        # Copy files
        self.copy_files_to_split(train_images, self.train_dir)
        self.copy_files_to_split(val_images, self.val_dir)
        self.copy_files_to_split(test_images, self.test_dir)

        # Create data.yaml
        self.create_data_yaml()

        print(f"Balanced dataset created in {self.output_dir}")


class EnhancedDataAugmenter:
    """Enhanced data augmentation for robotics applications"""

    def __init__(self, data_dir="consolidated_dataset"):
        self.data_dir = Path(data_dir)
        # Use train directory for source data
        self.source_images_dir = self.data_dir / "train" / "images"
        self.source_labels_dir = self.data_dir / "train" / "labels"
        self.output_dir = Path("enhanced_dataset")

        # Create output directories
        for split in ["train", "val", "test"]:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # get_class_distribution method removed - unused

    def create_confidence_focused_augmentations(self):
        """Create confidence-focused augmentations for distance, occlusion, and partial visibility"""
        return A.Compose(
            [
                # === DISTANCE SIMULATION (Critical for confidence at varying ranges) ===
                A.RandomScale(
                    scale_limit=0.9, p=0.8
                ),  # Extreme scale changes for far/close objects
                A.Resize(width=640, height=640, p=1.0),  # Consistent output size
                # === OCCLUSION SIMULATION ===
                A.CoarseDropout(
                    max_holes=8, max_height=50, max_width=50, p=0.3
                ),  # Random occlusions
                A.GridDropout(ratio=0.3, p=0.2),  # Structured occlusions
                # === ROBOTICS-SPECIFIC AUGMENTATIONS ===
                A.GaussNoise(var_limit=(10, 50), p=0.3),  # Camera noise
                A.ISONoise(
                    color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3
                ),  # Sensor noise
                A.ImageCompression(
                    quality_lower=70, quality_upper=100, p=0.3
                ),  # Compression artifacts
                # === LIGHTING VARIATIONS ===
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3
                ),
                # === GEOMETRIC TRANSFORMATIONS ===
                A.Rotate(limit=15, p=0.5),
                A.Perspective(scale=(0.05, 0.1), p=0.3),  # Slight perspective changes
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    def apply_augmentation_safe(
        self,
        image_path,
        label_path,
        output_image_path,
        output_label_path,
        augmentations,
    ):
        """Safely apply augmentations with error handling"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return False

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read YOLO labels
            bboxes = []
            class_labels = []

            with open(label_path, "r") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])

                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)

            if not bboxes:
                # No objects to augment, just copy
                cv2.imwrite(
                    str(output_image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                )
                shutil.copy2(label_path, output_label_path)
                return True

            # Apply augmentations
            augmented = augmentations(
                image=image, bboxes=bboxes, class_labels=class_labels
            )
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["class_labels"]

            # Save augmented image
            cv2.imwrite(
                str(output_image_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            )

            # Save augmented labels
            with open(output_label_path, "w") as f:
                for label, bbox in zip(aug_labels, aug_bboxes):
                    x_center, y_center, width, height = bbox
                    f.write(
                        f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )

            return True

        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")
            return False

    def augment_dataset(self, augmentation_factor=3):
        """Apply augmentations to create enhanced dataset"""
        print("Starting enhanced data augmentation...")

        # Get source files
        image_files = list(self.source_images_dir.glob("*.jpg")) + list(
            self.source_images_dir.glob("*.png")
        )
        augmentations = self.create_confidence_focused_augmentations()

        total_augmented = 0

        for image_path in image_files:
            label_path = self.source_labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            # Copy original
            for split in ["train", "val", "test"]:
                output_image_dir = self.output_dir / split / "images"
                output_label_dir = self.output_dir / split / "labels"

                shutil.copy2(image_path, output_image_dir / image_path.name)
                shutil.copy2(label_path, output_label_dir / label_path.name)

            # Create augmentations
            for i in range(augmentation_factor):
                output_image_path = (
                    self.output_dir
                    / "train"
                    / "images"
                    / f"{image_path.stem}_aug_{i}{image_path.suffix}"
                )
                output_label_path = (
                    self.output_dir
                    / "train"
                    / "labels"
                    / f"{image_path.stem}_aug_{i}.txt"
                )

                if self.apply_augmentation_safe(
                    image_path,
                    label_path,
                    output_image_path,
                    output_label_path,
                    augmentations,
                ):
                    total_augmented += 1

        print(f"Created {total_augmented} augmented samples")


class OrangeHammerRelabeler:
    """Identify and relabel orange hammers that were incorrectly labeled"""

    def __init__(self, data_path="balanced_data"):
        self.data_path = Path(data_path)
        self.train_images = self.data_path / "train" / "images"
        self.train_labels = self.data_path / "train" / "labels"
        self.val_images = self.data_path / "val" / "images"
        self.val_labels = self.data_path / "val" / "labels"
        self.test_images = self.data_path / "test" / "images"
        self.test_labels = self.data_path / "test" / "labels"

    def is_orange_hammer(self, image_path, bbox):
        """Determine if a hammer in the image appears orange"""

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return False

        height, width = image.shape[:2]

        # Extract bounding box region
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1 * width))
        y1 = max(0, int(y1 * height))
        x2 = min(width, int(x2 * width))
        y2 = min(height, int(y2 * height))

        if x2 <= x1 or y2 <= y1:
            return False

        # Extract hammer region
        hammer_region = image[y1:y2, x1:x2]

        if hammer_region.size == 0:
            return False

        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(hammer_region, cv2.COLOR_BGR2HSV)

        # Define orange color range in HSV
        lower_orange = np.array([5, 50, 50])  # Hue 5-15, Sat 50+, Val 50+
        upper_orange = np.array([25, 255, 255])

        # Create mask for orange pixels
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Calculate percentage of orange pixels in the hammer region
        orange_pixels = cv2.countNonZero(orange_mask)
        total_pixels = hammer_region.shape[0] * hammer_region.shape[1]

        orange_percentage = (orange_pixels / total_pixels) * 100

        # Consider it an orange hammer if more than 20% of pixels are orange
        return orange_percentage > 20

    def relabel_orange_hammers(self):
        """Scan dataset and relabel hammers that appear orange"""
        print("Scanning for orange hammers to relabel...")

        splits = [
            ("train", self.train_images, self.train_labels),
            ("val", self.val_images, self.val_labels),
            ("test", self.test_images, self.test_labels),
        ]

        total_relabeled = 0

        for split_name, images_dir, labels_dir in splits:
            print(f"Processing {split_name} set...")

            relabeled_in_split = 0
            for label_file in labels_dir.glob("*.txt"):
                image_file = images_dir / f"{label_file.stem}.jpg"
                if not image_file.exists():
                    continue

                try:
                    with open(label_file, "r") as f:
                        lines = f.readlines()

                    modified_lines = []
                    changed = False

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            # Only check hammer classes (0, 2, 3)
                            if class_id in [0, 2, 3]:
                                bbox = [
                                    float(parts[1]),
                                    float(parts[2]),
                                    float(parts[3]),
                                    float(parts[4]),
                                ]
                                # Convert YOLO bbox to corner format for analysis
                                x_center, y_center, w, h = bbox
                                bbox_corners = [
                                    x_center - w / 2,
                                    y_center - h / 2,
                                    x_center + w / 2,
                                    y_center + h / 2,
                                ]

                                if self.is_orange_hammer(image_file, bbox_corners):
                                    parts[0] = "3"  # Change to OrangeHammer class
                                    changed = True
                                    relabeled_in_split += 1

                        modified_lines.append(" ".join(parts))

                    if changed:
                        with open(label_file, "w") as f:
                            f.write("\n".join(modified_lines) + "\n")

                except Exception as e:
                    print(f"Error processing {label_file}: {e}")

            print(f"  Relabeled {relabeled_in_split} hammers in {split_name}")
            total_relabeled += relabeled_in_split

        print(f"Total hammers relabeled: {total_relabeled}")
        return total_relabeled


class OrangeHammerGenerator:
    """Generate synthetic OrangeHammer samples to fix class imbalance"""

    def __init__(self, data_dir="balanced_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("enhanced_data")

    def generate_synthetic_hammers(self, num_samples=200):
        """Generate synthetic orange hammer samples"""
        print(f"Generating {num_samples} synthetic OrangeHammer samples...")

        # Get existing hammer images
        train_images_dir = self.data_dir / "train" / "images"
        hammer_images = []

        for img_file in train_images_dir.glob("*.jpg"):
            if "hammer" in img_file.name.lower():
                hammer_images.append(img_file)

        if not hammer_images:
            print("No hammer images found for synthesis")
            return

        # Create output directories
        output_images_dir = self.output_dir / "train" / "images"
        output_labels_dir = self.output_dir / "train" / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        for i in range(num_samples):
            # Pick random hammer image
            source_image = random.choice(hammer_images)
            source_label = (
                self.data_dir / "train" / "labels" / f"{source_image.stem}.txt"
            )

            if not source_label.exists():
                continue

            # Modify color to orange
            image = cv2.imread(str(source_image))
            if image is None:
                continue

            # Convert to HSV and shift hue toward orange
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + 15) % 180  # Shift hue toward orange
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)  # Increase saturation
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Add some noise and variations
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)

            # Save modified image
            output_name = f"synthetic_orange_hammer_{i:04d}"
            output_image_path = output_images_dir / f"{output_name}.jpg"
            cv2.imwrite(str(output_image_path), image)

            # Modify label to OrangeHammer class (assuming class 3)
            with open(source_label, "r") as f:
                labels = f.readlines()

            with open(output_labels_dir / f"{output_name}.txt", "w") as f:
                for label in labels:
                    parts = label.strip().split()
                    if parts:
                        parts[0] = "3"  # OrangeHammer class
                        f.write(" ".join(parts) + "\n")

            generated += 1

        print(f"Generated {generated} synthetic OrangeHammer samples")


def generate_aruco_marker(marker_id, size=200):
    """Generate an ArUco marker image"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    return marker_image


def create_industrial_background(width=640, height=640):
    """Create a realistic industrial background"""
    # Create base gray background
    background = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)

    # Add some texture
    noise = np.random.normal(0, 10, (height, width, 3)).astype(np.uint8)
    background = cv2.add(background, noise)

    # Add some geometric shapes
    for _ in range(5):
        # Random rectangles
        pt1 = (random.randint(0, width // 2), random.randint(0, height // 2))
        pt2 = (random.randint(pt1[0] + 20, width), random.randint(pt1[1] + 20, height))
        cv2.rectangle(background, pt1, pt2, (80, 80, 80), -1)

    return background


def apply_robotics_augmentations(image, bbox):
    """Apply robotics-specific augmentations"""
    augmented = A.Compose(
        [
            A.Rotate(limit=10, p=0.5),
            A.GaussNoise(var_limit=(5, 20), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.Blur(blur_limit=3, p=0.2),
        ]
    )(image=image, bboxes=[bbox], class_labels=[0])

    return augmented["image"], augmented["bboxes"][0]


def generate_synthetic_aruco_sample(marker_id, sample_id, output_dir="synthetic_aruco"):
    """Generate a single synthetic ArUco sample"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate marker
    marker = generate_aruco_marker(marker_id)

    # Create background
    background = create_industrial_background()

    # Random position and size
    marker_size = random.randint(50, 150)
    marker_resized = cv2.resize(marker, (marker_size, marker_size))

    # Random position
    max_x = background.shape[1] - marker_size
    max_y = background.shape[0] - marker_size
    x = random.randint(20, max_x)
    y = random.randint(20, max_y)

    # Overlay marker on background
    roi = background[y : y + marker_size, x : x + marker_size]
    mask = (marker_resized > 128).astype(np.uint8) * 255
    background[y : y + marker_size, x : x + marker_size] = np.where(
        mask, marker_resized, roi
    )

    # Apply augmentations
    bbox = [x, y, marker_size, marker_size]
    aug_image, aug_bbox = apply_robotics_augmentations(background, bbox)

    # Save image
    image_path = output_dir / f"aruco_{marker_id}_{sample_id:04d}.jpg"
    cv2.imwrite(str(image_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

    # Save YOLO label (class 0 for ArUco)
    label_path = output_dir / f"aruco_{marker_id}_{sample_id:04d}.txt"
    x_center = (aug_bbox[0] + aug_bbox[2] / 2) / background.shape[1]
    y_center = (aug_bbox[1] + aug_bbox[3] / 2) / background.shape[0]
    width = aug_bbox[2] / background.shape[1]
    height = aug_bbox[3] / background.shape[0]

    with open(label_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    return image_path, label_path


def generate_aruco_dataset(num_samples=500, output_dir="synthetic_aruco"):
    """Generate a complete synthetic ArUco dataset"""
    print(f"Generating {num_samples} synthetic ArUco samples...")

    output_dir = Path(output_dir)
    generated = 0

    for i in range(num_samples):
        marker_id = random.randint(0, 49)  # 4x4_50 dictionary has IDs 0-49
        try:
            generate_synthetic_aruco_sample(marker_id, i, output_dir)
            generated += 1
        except Exception as e:
            print(f"Error generating sample {i}: {e}")

    print(f"Generated {generated} synthetic ArUco samples in {output_dir}")


def merge_hammer_classes():
    """Merge BrickHammer and OrangeHammer classes into a single Hammer class"""
    print("Merging hammer classes...")

    data_dirs = ["balanced_data", "consolidated_dataset", "enhanced_dataset"]

    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            continue

        print(f"Processing {data_dir}...")

        # Process each split
        for split in ["train", "val", "test"]:
            labels_dir = data_path / split / "labels"
            if not labels_dir.exists():
                continue

            modified = 0
            for label_file in labels_dir.glob("*.txt"):
                try:
                    with open(label_file, "r") as f:
                        lines = f.readlines()

                    modified_lines = []
                    changed = False
                    for line in lines:
                        parts = line.strip().split()
                        if parts and parts[0] in [
                            "2",
                            "3",
                        ]:  # BrickHammer (2) or OrangeHammer (3)
                            parts[0] = "1"  # Merge to Hammer class (1)
                            changed = True
                        modified_lines.append(" ".join(parts))

                    if changed:
                        with open(label_file, "w") as f:
                            f.write("\n".join(modified_lines) + "\n")
                        modified += 1

                except Exception as e:
                    print(f"Error processing {label_file}: {e}")

            print(f"  Modified {modified} files in {split}")

        # Update data.yaml
        yaml_path = data_path / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)

                if "names" in data:
                    data["names"] = {
                        0: "ArUcoTag",
                        1: "Hammer",  # Merged class
                        2: "Bottle",
                    }

                with open(yaml_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)

                print(f"  Updated {yaml_path}")

            except Exception as e:
                print(f"Error updating {yaml_path}: {e}")


def main():
    """Main function for dataset utilities"""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument(
        "--action",
        choices=[
            "balance",
            "augment",
            "generate_hammers",
            "generate_aruco",
            "merge_hammers",
            "relabel_orange",
        ],
        required=True,
        help="Action to perform",
    )
    parser.add_argument("--data_dir", default="data", help="Input data directory")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument(
        "--num_samples", type=int, default=500, help="Number of samples to generate"
    )

    args = parser.parse_args()

    if args.action == "balance":
        balancer = YOLODataBalancer(args.data_dir)
        balancer.balance_dataset()

    elif args.action == "augment":
        augmenter = EnhancedDataAugmenter(args.data_dir)
        augmenter.augment_dataset()

    elif args.action == "generate_hammers":
        generator = OrangeHammerGenerator(args.data_dir)
        generator.generate_synthetic_hammers(args.num_samples)

    elif args.action == "generate_aruco":
        generate_aruco_dataset(args.num_samples, args.output_dir or "synthetic_aruco")

    elif args.action == "merge_hammers":
        merge_hammer_classes()

    elif args.action == "relabel_orange":
        relabeler = OrangeHammerRelabeler(args.data_dir)
        relabeler.relabel_orange_hammers()


if __name__ == "__main__":
    main()
