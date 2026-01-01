"""
Performance metrics and evaluation utilities
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """

    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_precision_recall(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    gt_classes: List[int],
    pred_classes: List[int],
    iou_threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Calculate precision and recall for object detection

    Args:
        gt_boxes: Ground truth bounding boxes [[x1,y1,x2,y2], ...]
        pred_boxes: Predicted bounding boxes [[x1,y1,x2,y2], ...]
        gt_classes: Ground truth class labels
        pred_classes: Predicted class labels
        iou_threshold: IoU threshold for true positive

    Returns:
        Tuple of (precision, recall)
    """

    if not pred_boxes:
        return 0.0, 0.0

    # Sort predictions by confidence (assuming last element is confidence)
    # For now, assume predictions are already sorted or we don't have confidence
    pred_boxes = np.array(pred_boxes)
    pred_classes = np.array(pred_classes)

    true_positives = 0
    false_positives = 0

    gt_matched = set()

    for i, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
        # Find best matching ground truth box of same class
        best_iou = 0
        best_gt_idx = -1

        for j, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
            if j in gt_matched or gt_class != pred_class:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            gt_matched.add(best_gt_idx)
        else:
            false_positives += 1

    false_negatives = len(gt_boxes) - len(gt_matched)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    return precision, recall


def calculate_map(
    gt_boxes_list: List[List[List[float]]],
    pred_boxes_list: List[List[List[float]]],
    gt_classes_list: List[List[int]],
    pred_classes_list: List[List[int]],
    num_classes: int,
    iou_thresholds: List[float] = [0.5],
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP)

    Args:
        gt_boxes_list: List of ground truth boxes for each image
        pred_boxes_list: List of predicted boxes for each image
        gt_classes_list: List of ground truth classes for each image
        pred_classes_list: List of predicted classes for each image
        num_classes: Number of classes
        iou_thresholds: IoU thresholds to evaluate at

    Returns:
        Dictionary with mAP metrics
    """

    results = {}

    for iou_thresh in iou_thresholds:
        ap_scores = []

        for class_id in range(num_classes):
            # Collect all predictions and ground truths for this class
            class_gt_boxes = []
            class_pred_boxes = []
            class_pred_confs = []

            for _, (gt_boxes, pred_boxes, gt_classes, pred_classes) in enumerate(
                zip(gt_boxes_list, pred_boxes_list, gt_classes_list, pred_classes_list)
            ):

                # Filter by class
                gt_class_mask = np.array(gt_classes) == class_id
                pred_class_mask = np.array(pred_classes) == class_id

                class_gt_boxes.extend(np.array(gt_boxes)[gt_class_mask].tolist())

                # For predictions, we need confidence scores
                # For now, assume equal confidence or add confidence to input
                pred_boxes_class = np.array(pred_boxes)[pred_class_mask].tolist()
                class_pred_boxes.extend(pred_boxes_class)

                # Placeholder confidence scores (should be passed in)
                class_pred_confs.extend(
                    [0.5] * len(pred_boxes_class)
                )  # Default confidence

            if not class_gt_boxes:
                ap_scores.append(0.0)
                continue

            # Calculate AP for this class
            ap = calculate_ap(
                class_gt_boxes, class_pred_boxes, class_pred_confs, iou_thresh
            )
            ap_scores.append(ap)

        # Calculate mean AP
        map_score = np.mean(ap_scores)
        results[f"mAP@{iou_thresh}"] = map_score

    return results


def calculate_ap(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    pred_confs: List[float],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate Average Precision for a single class

    Args:
        gt_boxes: Ground truth boxes
        pred_boxes: Predicted boxes
        pred_confs: Prediction confidence scores
        iou_threshold: IoU threshold

    Returns:
        Average Precision score
    """

    if not pred_boxes or not gt_boxes:
        return 0.0

    # Sort predictions by confidence
    pred_indices = np.argsort(pred_confs)[::-1]  # Descending order
    pred_boxes = np.array(pred_boxes)[pred_indices].tolist()
    pred_confs = np.array(pred_confs)[pred_indices].tolist()

    gt_matched = [False] * len(gt_boxes)
    tp = []
    fp = []

    for pred_box in pred_boxes:
        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1

        for i, (gt_box, matched) in enumerate(zip(gt_boxes, gt_matched)):
            if matched:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp.append(1)
            fp.append(0)
            gt_matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    # Calculate precision and recall
    tp = np.array(tp)
    fp = np.array(fp)

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_boxes)

    # Calculate AP using 11-point interpolation (VOC style)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11

    return ap


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall
    """

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def analyze_detection_errors(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    gt_classes: List[int],
    pred_classes: List[int],
    iou_threshold: float = 0.5,
) -> Dict[str, int]:
    """
    Analyze types of detection errors

    Args:
        gt_boxes: Ground truth boxes
        pred_boxes: Predicted boxes
        gt_classes: Ground truth classes
        pred_classes: Predicted classes
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with error counts
    """

    errors = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "class_errors": 0,  # Correct location, wrong class
        "localization_errors": 0,  # Poor IoU but correct class
        "background_errors": 0,  # False positives on background
    }

    gt_matched = set()
    pred_matched = set()

    # Match predictions to ground truth
    for i, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou = 0
        best_gt_idx = -1
        best_gt_class = -1

        for j, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
            if j in gt_matched:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
                best_gt_class = gt_class

        if best_iou >= iou_threshold:
            if best_gt_class == pred_class:
                errors["true_positives"] += 1
                gt_matched.add(best_gt_idx)
                pred_matched.add(i)
            else:
                errors["class_errors"] += 1
        elif best_iou >= 0.1:  # Some overlap but below threshold
            errors["localization_errors"] += 1
        else:
            errors["background_errors"] += 1

    # Count unmatched predictions as false positives
    errors["false_positives"] += len(pred_boxes) - len(pred_matched)

    # Count unmatched ground truth as false negatives
    errors["false_negatives"] += len(gt_boxes) - len(gt_matched)

    return errors


def calculate_small_object_metrics(
    detections: List[Dict], min_area: int = 32 * 32
) -> Dict[str, float]:
    """
    Calculate metrics specifically for small objects

    Args:
        detections: List of detection dictionaries with 'bbox' and 'confidence' keys
        min_area: Minimum bounding box area for "small" objects

    Returns:
        Dictionary with small object metrics
    """

    small_objects = []
    all_objects = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        area = (x2 - x1) * (y2 - y1)

        obj_info = {
            "area": area,
            "confidence": det["confidence"],
            "is_small": area < min_area,
        }

        all_objects.append(obj_info)
        if obj_info["is_small"]:
            small_objects.append(obj_info)

    if not small_objects:
        return {
            "small_object_count": 0,
            "small_object_ratio": 0.0,
            "avg_small_confidence": 0.0,
            "small_object_recall": 0.0,
        }

    total_objects = len(all_objects)
    small_count = len(small_objects)

    return {
        "small_object_count": small_count,
        "small_object_ratio": small_count / total_objects if total_objects > 0 else 0,
        "avg_small_confidence": np.mean([obj["confidence"] for obj in small_objects]),
        "small_object_recall": len(
            [obj for obj in small_objects if obj["confidence"] > 0.5]
        )
        / small_count,
    }
