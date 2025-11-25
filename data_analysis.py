#!/usr/bin/env python3
"""
Data Analysis and Balanced Splitting Script for YOLOv8 Object Detection
Analyzes class distribution and creates balanced train/val/test splits
"""

import os
import shutil
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import yaml
from sklearn.model_selection import train_test_split
import albumentations as A
from PIL import Image
import cv2

class YOLODataBalancer:
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
        print("🔍 Analyzing class distribution...")

        class_counts = defaultdict(int)
        image_class_map = defaultdict(list)

        label_files = list(self.labels_dir.glob("*.txt"))

        for label_file in label_files:
            image_name = label_file.stem.replace('.txt', '')
            classes_in_image = set()

            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            classes_in_image.add(class_id)
                            class_counts[class_id] += 1

                # Map image to its classes
                for class_id in classes_in_image:
                    image_class_map[class_id].append(image_name)

            except Exception as e:
                print(f"Error reading {label_file}: {e}")

        print("📊 Class Distribution:")
        class_names = ['BrickHammer', 'ArUcoTag', 'Bottle', 'BrickHammer_duplicate', 'OrangeHammer']
        for class_id, count in sorted(class_counts.items()):
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            print(".2f")

        print(f"\n📈 Total images: {len(label_files)}")
        print(f"📈 Total annotations: {sum(class_counts.values())}")

        return class_counts, image_class_map

    def balance_classes(self, class_counts, image_class_map):
        """Ensure stratified splitting by class - no SMOTE duplication"""
        print("⚖️ Preparing for stratified splitting...")

        # Just return all unique images - we'll do stratified splitting later
        all_images = set()
        for images in image_class_map.values():
            all_images.update(images)

        print(f"📊 Total unique images: {len(all_images)}")
        return list(all_images), image_class_map

    def apply_augmentation(self, image_path, label_path, output_image_path, output_label_path, augmentations):
        """Apply augmentations to an image and its labels"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return False

        height, width = image.shape[:2]

        # Read labels in YOLO format (normalized 0-1)
        bboxes = []
        class_ids = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.split()
                        class_id = int(parts[0])
                        # YOLO format: class x_center y_center width height (normalized)
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        # Convert to COCO format for albumentations
                        x_min = (x_center - w/2) * width
                        y_min = (y_center - h/2) * height
                        x_max = (x_center + w/2) * width
                        y_max = (y_center + h/2) * height

                        # Ensure bbox is within image bounds
                        x_min = max(0, min(x_min, width))
                        y_min = max(0, min(y_min, height))
                        x_max = max(0, min(x_max, width))
                        y_max = max(0, min(y_max, height))

                        # Skip invalid bboxes
                        if x_max <= x_min or y_max <= y_min:
                            continue

                        bboxes.append([x_min, y_min, x_max, y_max])
                        class_ids.append(class_id)
        except Exception as e:
            print(f"Error reading labels from {label_path}: {e}")
            return False

        # Apply augmentation
        if bboxes:
            try:
                transformed = augmentations(image=image, bboxes=bboxes, class_labels=class_ids)
                aug_image = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_class_ids = transformed['class_labels']
            except Exception as e:
                print(f"Augmentation failed for {image_path}: {e}")
                # Save original if augmentation fails
                cv2.imwrite(str(output_image_path), image)
                shutil.copy2(label_path, output_label_path)
                return True
        else:
            aug_image = augmentations(image=image)['image']
            aug_bboxes = bboxes
            aug_class_ids = class_ids

        # Save augmented image
        cv2.imwrite(str(output_image_path), aug_image)

        # Save augmented labels
        with open(output_label_path, 'w') as f:
            for bbox, class_id in zip(aug_bboxes, aug_class_ids):
                # Convert back to YOLO format (normalized)
                x_min, y_min, x_max, y_max = bbox
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                w = (x_max - x_min) / width
                h = (y_max - y_min) / height

                # Ensure values are in valid range
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        return True

    def create_balanced_splits(self, balanced_images, image_class_map, val_split=0.2, test_split=0.1):
        """Create stratified train/val/test splits to maintain class balance"""
        print("✂️ Creating stratified splits...")

        train_images = []
        val_images = []
        test_images = []

        # For each class, split images proportionally
        for class_id, images in image_class_map.items():
            if not images:
                continue

            # Shuffle images for this class
            class_images = list(set(images))  # Remove duplicates
            random.shuffle(class_images)

            # Calculate split sizes for this class
            n_class = len(class_images)
            n_test_class = max(1, int(n_class * test_split))
            n_val_class = max(1, int(n_class * val_split))
            n_train_class = n_class - n_test_class - n_val_class

            # Ensure at least 1 image per split if possible
            if n_class >= 3:
                train_class = class_images[:n_train_class]
                val_class = class_images[n_train_class:n_train_class+n_val_class]
                test_class = class_images[n_train_class+n_val_class:]
            else:
                # For small classes, put at least 1 in each split if possible
                train_class = class_images[:max(1, len(class_images)//3)]
                remaining = class_images[len(train_class):]
                val_class = remaining[:max(1, len(remaining)//2)] if remaining else []
                test_class = remaining[len(val_class):] if remaining else []

            train_images.extend(train_class)
            val_images.extend(val_class)
            test_images.extend(test_class)

            print(f"  Class {class_id}: {len(train_class)} train, {len(val_class)} val, {len(test_class)} test")

        # Remove any duplicates across splits
        train_images = list(set(train_images))
        val_images = list(set(val_images))
        test_images = list(set(test_images))

        print(f"📊 Final split sizes: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

        return train_images, val_images, test_images

    def copy_files(self, image_list, split_name):
        """Copy images and labels to the appropriate split directory"""
        print(f"📋 Copying {len(image_list)} files to {split_name}...")

        output_images_dir = self.output_dir / split_name / "images"
        output_labels_dir = self.output_dir / split_name / "labels"

        for image_name in image_list:
            # Copy image
            src_image = self.images_dir / f"{image_name}.jpg"
            dst_image = output_images_dir / f"{image_name}.jpg"
            if src_image.exists():
                shutil.copy2(src_image, dst_image)

            # Copy label
            src_label = self.labels_dir / f"{image_name}.txt"
            dst_label = output_labels_dir / f"{image_name}.txt"
            if src_label.exists():
                shutil.copy2(src_label, dst_label)

    def augment_data(self, image_list, split_name, augmentation_factor=2):
        """Apply data augmentation to increase dataset size"""
        print(f"🎨 Applying augmentations to {split_name} data...")

        output_images_dir = self.output_dir / split_name / "images"
        output_labels_dir = self.output_dir / split_name / "labels"

        # Define augmentations
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.4),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

        augmented_images = []

        for image_name in image_list:
            original_image_path = self.images_dir / f"{image_name}.jpg"
            original_label_path = self.labels_dir / f"{image_name}.txt"

            if not original_image_path.exists() or not original_label_path.exists():
                continue

            # Copy original
            shutil.copy2(original_image_path, output_images_dir / f"{image_name}.jpg")
            shutil.copy2(original_label_path, output_labels_dir / f"{image_name}.txt")
            augmented_images.append(image_name)

            # Create augmented versions
            for i in range(augmentation_factor):
                aug_image_name = f"{image_name}_aug_{i}"
                aug_image_path = output_images_dir / f"{aug_image_name}.jpg"
                aug_label_path = output_labels_dir / f"{aug_image_name}.txt"

                success = self.apply_augmentation(
                    original_image_path, original_label_path,
                    aug_image_path, aug_label_path,
                    augmentations
                )

                if success:
                    augmented_images.append(aug_image_name)

        return augmented_images

    def create_data_yaml(self):
        """Create data.yaml file for the balanced dataset"""
        data_yaml = {
            'train': str(self.train_dir),
            'val': str(self.val_dir),
            'test': str(self.test_dir),
            'nc': 5,
            'names': ['BrickHammer', 'ArUcoTag', 'Bottle', 'BrickHammer_duplicate', 'OrangeHammer']
        }

        with open(self.output_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"📄 Created data.yaml at {self.output_dir / 'data.yaml'}")

    def process_dataset(self, val_split=0.2, test_split=0.1, augmentation_factor=2):
        """Main processing pipeline"""
        print("🚀 Starting dataset balancing and augmentation pipeline...")

        # Step 1: Analyze current distribution
        class_counts, image_class_map = self.analyze_class_distribution()

        # Step 2: Balance classes
        balanced_images, balanced_class_map = self.balance_classes(class_counts, image_class_map)

        # Step 3: Create balanced splits
        train_images, val_images, test_images = self.create_balanced_splits(
            balanced_images, balanced_class_map, val_split, test_split
        )

        # Step 4: Apply augmentation and copy files
        print("📸 Processing training data with augmentation...")
        final_train_images = self.augment_data(train_images, "train", augmentation_factor)

        print("📸 Processing validation data...")
        final_val_images = self.augment_data(val_images, "val", 0)  # No augmentation for val

        print("📸 Processing test data...")
        final_test_images = self.augment_data(test_images, "test", 0)  # No augmentation for test

        # Step 5: Create data.yaml
        self.create_data_yaml()

        print("✅ Dataset processing complete!")
        print(f"📊 Final sizes: Train={len(final_train_images)}, Val={len(final_val_images)}, Test={len(final_test_images)}")

        return {
            'train_count': len(final_train_images),
            'val_count': len(final_val_images),
            'test_count': len(final_test_images),
            'total_images': len(final_train_images) + len(final_val_images) + len(final_test_images)
        }

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create balancer and process dataset
    balancer = YOLODataBalancer()

    # Process with default parameters
    results = balancer.process_dataset(
        val_split=0.2,      # 20% validation
        test_split=0.1,     # 10% test
        augmentation_factor=2  # Create 2 augmented versions per image
    )

    print("\n🎉 Dataset balancing complete!")
    print(f"📂 Balanced data saved to: {balancer.output_dir}")
    print(f"📊 Training images: {results['train_count']}")
    print(f"📊 Validation images: {results['val_count']}")
    print(f"📊 Test images: {results['test_count']}")
    print(f"📊 Total images: {results['total_images']}")

if __name__ == "__main__":
    main()
