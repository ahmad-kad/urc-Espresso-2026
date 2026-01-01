"""
Detection utilities for YOLO object detection
Consolidated functions for bounding box operations and post-processing
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from einops import rearrange

    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False

from utils.logger_config import get_logger

logger = get_logger(__name__, debug=os.getenv("DEBUG") == "1")


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2] format

    Args:
        boxes: Array of shape (N, 4) with [x_center, y_center, width, height]

    Returns:
        Array of shape (N, 4) with [x1, y1, x2, y2]
    """
    # Use numpy for simplicity and reliability
    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4
) -> List[int]:
    """
    Apply Non-Maximum Suppression to filter overlapping bounding boxes

    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2]
        scores: Array of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        List of indices to keep after NMS
    """
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=0.0,  # Already filtered by confidence
        nms_threshold=iou_threshold,
    )

    if indices is not None and len(indices) > 0:
        # NMSBoxes returns numpy array or list, convert to list
        if isinstance(indices, np.ndarray):
            return indices.flatten().tolist()
        elif isinstance(indices, (list, tuple)):
            return [int(i) for i in indices]
        else:
            # Single value case
            try:
                return [int(indices)]
            except (TypeError, ValueError):
                return []
    return []


def post_process_yolov8(
    output: np.ndarray,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.4,
    input_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Post-process YOLOv8 output for inference

    Args:
        output: Raw YOLOv8 output tensor of shape (1, num_boxes, 4 + num_classes + 1)
        conf_threshold: Confidence threshold for filtering detections
        iou_threshold: IoU threshold for NMS
        input_size: Input image size for scaling

    Returns:
        Tuple of (boxes, confidences, class_ids) after post-processing
    """
    # Remove batch dimension: (num_boxes, 4 + num_classes + 1)
    output = output[0]

    # Transpose to (num_boxes, features) - use einops if available
    if EINOPS_AVAILABLE:
        import torch

        output_t = torch.from_numpy(output)
        output = rearrange(output_t, "features boxes -> boxes features").numpy()
    else:
        output = output.transpose(1, 0)

    # Extract boxes and scores
    boxes = output[:, :4]  # (num_boxes, 4) - [x_center, y_center, w, h]
    class_scores = output[:, 4:]  # (num_boxes, num_classes)

    # Get best class for each detection
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Filter by confidence
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert to xyxy format and scale to input size
    boxes = xywh2xyxy(boxes)
    boxes[:, [0, 2]] *= input_size  # x coordinates
    boxes[:, [1, 3]] *= input_size  # y coordinates

    # Apply NMS
    indices = non_max_suppression(boxes, confidences, iou_threshold)

    return boxes[indices], confidences[indices], class_ids[indices]


def scale_boxes_to_original(
    boxes: np.ndarray, original_size: Tuple[int, int], input_size: int = 224
) -> np.ndarray:
    """
    Scale bounding boxes from input size back to original image size

    Args:
        boxes: Bounding boxes in xyxy format
        original_size: (height, width) of original image
        input_size: Size the model was trained on

    Returns:
        Scaled bounding boxes
    """
    if len(boxes) == 0:
        return boxes

    original_h, original_w = original_size

    # Scale factors
    scale_x = original_w / input_size
    scale_y = original_h / input_size

    scaled_boxes = boxes.astype(np.float64).copy()
    scaled_boxes[:, [0, 2]] *= scale_x  # x coordinates
    scaled_boxes[:, [1, 3]] *= scale_y  # y coordinates

    return scaled_boxes


def filter_boxes_by_class(
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    target_classes: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter detections by class IDs

    Args:
        boxes: Bounding boxes
        confidences: Confidence scores
        class_ids: Class IDs for each detection
        target_classes: List of class IDs to keep (None keeps all)

    Returns:
        Filtered detections
    """
    if target_classes is None or len(boxes) == 0:
        return boxes, confidences, class_ids

    mask = np.isin(class_ids, target_classes)
    return boxes[mask], confidences[mask], class_ids[mask]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: List[str],
    colors: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image

    Args:
        image: Input image
        boxes: Bounding boxes in xyxy format
        confidences: Confidence scores
        class_ids: Class IDs
        class_names: List of class names
        colors: Optional list of colors for each class

    Returns:
        Image with detections drawn
    """
    if colors is None:
        # Generate random colors
        np.random.seed(42)
        colors_list = np.random.randint(0, 255, (len(class_names), 3)).tolist()
    else:
        colors_list = colors  # Use provided colors

    result = image.copy()

    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        class_name = class_names[int(class_id)]
        color = colors_list[int(class_id) % len(colors_list)]

        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw label background
        cv2.rectangle(result, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)

        # Draw label text
        cv2.putText(
            result,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return result
