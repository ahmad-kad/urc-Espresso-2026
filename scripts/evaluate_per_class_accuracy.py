#!/usr/bin/env python3
"""
Per-class accuracy evaluation utilities
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import onnxruntime as ort

# Import cv2 with fallback
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available. ONNX evaluation will not work.")
    cv2 = None


def load_ground_truth_labels(
    data_yaml: str, image_paths: List[str]
) -> Dict[str, List[Dict]]:
    """
    Load ground truth labels for evaluation

    Args:
        data_yaml: Path to data configuration
        image_paths: List of image paths to evaluate

    Returns:
        Dictionary mapping image paths to ground truth annotations
    """
    ground_truth = {}

    for img_path in image_paths:
        img_stem = Path(img_path).stem
        label_path = Path(img_path).parent.parent / "labels" / f"{img_stem}.txt"

        if label_path.exists():
            boxes = []
            classes = []

            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = list(
                            map(float, parts[1:5])
                        )  # x_center, y_center, width, height
                        boxes.append(bbox)
                        classes.append(class_id)

            ground_truth[img_path] = {"boxes": boxes, "classes": classes}
        else:
            ground_truth[img_path] = {"boxes": [], "classes": []}

    return ground_truth


def evaluate_onnx_model(
    onnx_path: str,
    image_paths: List[str],
    input_size: int = 416,
    conf_threshold: float = 0.25,
) -> Dict[str, Dict]:
    """
    Evaluate ONNX model predictions

    Args:
        onnx_path: Path to ONNX model
        image_paths: List of image paths
        input_size: Model input size
        conf_threshold: Confidence threshold

    Returns:
        Dictionary mapping image paths to predictions
    """
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    predictions = {}

    for img_path in image_paths:
        try:
            # Load and preprocess image
            if cv2 is None:
                predictions[img_path] = {"boxes": [], "classes": [], "scores": []}
                continue

            img = cv2.imread(img_path)
            if img is None:
                predictions[img_path] = {"boxes": [], "classes": [], "scores": []}
                continue

            original_height, original_width = img.shape[:2]

            # Preprocess
            img_resized = cv2.resize(img, (input_size, input_size))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            img_batch = np.expand_dims(img_transposed, axis=0)

            # Run inference
            outputs = session.run(None, {input_name: img_batch})

            # Parse YOLOv8 ONNX outputs (simplified)
            # YOLOv8 typically outputs a single tensor with shape [1, 84, 8400] for COCO
            # or [1, 30, 8400] for custom models (6 classes * 5 + 0 = 30)
            if len(outputs) == 0:
                predictions[img_path] = {"boxes": [], "classes": [], "scores": []}
                continue

            # Convert to numpy array if needed
            output_tensor = outputs[0]
            try:
                # Try to convert to numpy - works for most tensor types
                if hasattr(output_tensor, "numpy"):
                    output = output_tensor.numpy()
                else:
                    output = np.array(output_tensor)
            except Exception:
                # Fallback: assume it's already a numpy array
                output = np.asarray(output_tensor)

            output = output[0]  # Remove batch dimension

            boxes = []
            classes = []
            scores = []

            # Parse detections (simplified - assumes YOLO format)
            for detection in output.T:  # Transpose to iterate over detections
                if len(detection) >= 6:  # x, y, w, h, conf, class_scores...
                    x, y, w, h, conf = detection[:5]

                    if conf > conf_threshold:
                        # Get class with highest score
                        class_scores = detection[5:]
                        class_id = np.argmax(class_scores)
                        class_score = class_scores[class_id]

                        if class_score > conf_threshold:
                            # Convert from center format to corner format
                            x1 = max(0, x - w / 2)
                            y1 = max(0, y - h / 2)
                            x2 = min(1, x + w / 2)
                            y2 = min(1, y + h / 2)

                            # Scale to original image size
                            x1 *= original_width
                            y1 *= original_height
                            x2 *= original_width
                            y2 *= original_height

                            boxes.append([x1, y1, x2, y2])
                            classes.append(int(class_id))
                            scores.append(float(class_score))

            predictions[img_path] = {
                "boxes": boxes,
                "classes": classes,
                "scores": scores,
            }

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            predictions[img_path] = {"boxes": [], "classes": [], "scores": []}

    return predictions


def calculate_per_class_metrics(
    predictions: Dict[str, Dict[str, List]],
    ground_truth: Dict[str, Dict[str, List]],
    class_names: List[str],
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Calculate per-class precision, recall, and F1 scores

    Args:
        predictions: Model predictions
        ground_truth: Ground truth annotations
        class_names: List of class names
        iou_threshold: IoU threshold for matching

    Returns:
        DataFrame with per-class metrics
    """
    results = []

    for class_idx, class_name in enumerate(class_names):
        tp = fp = fn = 0

        for img_path in predictions.keys():
            pred = predictions[img_path]
            gt = ground_truth[img_path]

            # Get predictions and ground truth for this class
            pred_boxes = [
                box
                for box, cls in zip(pred.get("boxes", []), pred.get("classes", []))
                if cls == class_idx
            ]
            gt_boxes = [
                box
                for box, cls in zip(gt.get("boxes", []), gt.get("classes", []))
                if cls == class_idx
            ]

            # Calculate matches (simplified)
            matched_gt = set()

            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue

                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            # False negatives
            fn += len(gt_boxes) - len(matched_gt)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    return pd.DataFrame(results)


def print_comparison_results(results_df: pd.DataFrame, title: str = "Model Comparison"):
    """
    Print formatted comparison results

    Args:
        results_df: DataFrame with comparison results
        title: Title for the comparison
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    if results_df.empty:
        print("No results to display")
        return

    # Print summary
    print(f"Average Precision: {results_df['precision'].mean():.3f}")
    print(f"Average Recall: {results_df['recall'].mean():.3f}")
    print(f"Average F1: {results_df['f1_score'].mean():.3f}")
    print()

    # Print per-class results
    print("Per-Class Results:")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(
            f"{row['class']:<12} {row['precision']:.3f} {row['recall']:.3f} {row['f1_score']:.3f}"
        )

    print(f"{'='*80}")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        box1, box2: Bounding boxes in [x, y, w, h] format

    Returns:
        IoU value
    """
    # Convert to [x1, y1, x2, y2] format
    x1_1, y1_1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x2_1, y2_1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

    x1_2, y1_2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_2, y2_2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
