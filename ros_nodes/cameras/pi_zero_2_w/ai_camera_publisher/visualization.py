"""
Visualization Component
Handles drawing detections on images
"""

import time
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.logger_config import get_logger

logger = get_logger(__name__)


class VisualizationComponent:
    """
    Component for visualizing detections on images
    Handles bounding box drawing, labels, and persisted detections
    """

    def __init__(self, config: Dict[str, Any], class_names: List[str]):
        """
        Initialize visualization component

        Args:
            config: Configuration dictionary
            class_names: List of class names
        """
        self.config = config
        self.class_names = class_names

        realtime_config = config.get("realtime", {})
        self.persistence_enabled = realtime_config.get("detection_persistence", False)
        self.persistence_duration = realtime_config.get("persistence_duration", 1.0)
        self.persisted_detections: Dict[int, Dict] = {}
        self._detection_id_counter = 0

        # Generate colors for classes
        num_classes = len(class_names) if class_names else 10
        self.colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

    def draw_detections(
        self,
        image: np.ndarray,
        results: Any,
        smoothed_confidences: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            image: Input image
            results: Detection results (YOLO format)
            smoothed_confidences: Optional dictionary of smoothed confidences

        Returns:
            Annotated image
        """
        annotated = image.copy()
        conf_threshold = self.config.get("model", {}).get("confidence_threshold", 0.5)

        if hasattr(results, "__iter__") and len(results) > 0:
            result = results[0]

            if hasattr(result, "boxes"):
                boxes = result.boxes

                for i in range(len(boxes.xyxy)):
                    if isinstance(boxes.xyxy, np.ndarray):
                        box = boxes.xyxy[i]
                        conf = float(boxes.conf[i])
                        cls = int(boxes.cls[i])
                    else:
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())

                    # Use temporally smoothed confidence if available
                    if smoothed_confidences and i in smoothed_confidences:
                        conf = smoothed_confidences[i]

                    # Filter by confidence threshold
                    if conf < conf_threshold:
                        continue

                    x1, y1, x2, y2 = map(int, box)

                    # Get color
                    color_idx = cls % len(self.colors)
                    color = tuple(int(c * 255) for c in self.colors[color_idx][:3])
                    color = (int(color[2]), int(color[1]), int(color[0]))  # BGR

                    # Draw rectangle
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    class_name = (
                        self.class_names[cls]
                        if cls < len(self.class_names)
                        else f"class_{cls}"
                    )
                    label = f"{class_name}: {conf:.2f}"

                    # Text background
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - text_height - 5),
                        (x1 + text_width, y1),
                        color,
                        -1,
                    )

                    # Text
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

        # Draw persisted detections (faded if old)
        if self.persistence_enabled:
            self._draw_persisted_detections(annotated)

        return annotated

    def _draw_persisted_detections(self, image: np.ndarray) -> None:
        """Draw persisted detections with fade effect"""
        if len(self.persisted_detections) == 0:
            return

        current_time = time.time()

        for det_id, det in list(self.persisted_detections.items()):
            age = current_time - det["timestamp"]
            if age > self.persistence_duration:
                del self.persisted_detections[det_id]
                continue

            # Calculate fade: 1.0 (current) to 0.5 (old)
            fade_factor = 1.0 - (age / self.persistence_duration) * 0.5
            fade_factor = max(0.5, fade_factor)

            box = det["box"]
            conf = det["conf"]
            cls = det["cls"]
            x1, y1, x2, y2 = map(int, box)

            # Get color (faded)
            color_idx = cls % len(self.colors)
            color = tuple(
                int(c * 255 * fade_factor) for c in self.colors[color_idx][:3]
            )
            color = (int(color[2]), int(color[1]), int(color[0]))  # BGR

            # Draw with thinner line for persisted (older) detections
            line_thickness = 1 if age > 2.0 else 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

            # Draw label (faded)
            class_name = (
                self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
            )
            label = f"{class_name}: {conf:.2f}"

            # Text background (faded)
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_color = tuple(int(c * fade_factor) for c in (255, 255, 255))
            cv2.rectangle(
                image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1
            )

            # Text
            cv2.putText(
                image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
            )

    def update_persisted_detections(self, results: Any, current_time: float) -> None:
        """
        Update persisted detections with current frame detections

        Args:
            results: Current frame detection results
            current_time: Current timestamp
        """
        if not self.persistence_enabled:
            return

        # Extract current detections
        current_detections = []
        if hasattr(results, "__iter__") and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes.xyxy)):
                    try:
                        if isinstance(boxes.xyxy, np.ndarray):
                            box = boxes.xyxy[i].copy()
                            conf = float(boxes.conf[i])
                            cls = int(boxes.cls[i])
                        else:
                            box = boxes.xyxy[i].cpu().numpy().copy()
                            conf = float(boxes.conf[i].cpu().numpy())
                            cls = int(boxes.cls[i].cpu().numpy())

                        current_detections.append(
                            {"box": box, "conf": conf, "cls": cls}
                        )
                    except:
                        continue

        # Match current detections to persisted ones (by IoU and class)
        matched_ids = set()
        for det in current_detections:
            best_match_id = None
            best_iou = 0.3  # Minimum IoU threshold

            for det_id, persisted in self.persisted_detections.items():
                if det_id in matched_ids:
                    continue
                if persisted["cls"] == det["cls"]:
                    iou = self._calculate_iou(det["box"], persisted["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = det_id

            if best_match_id is not None:
                # Update existing persisted detection
                self.persisted_detections[best_match_id].update(
                    {"box": det["box"], "conf": det["conf"], "timestamp": current_time}
                )
                matched_ids.add(best_match_id)
            else:
                # New detection - add to persisted
                self._detection_id_counter += 1
                self.persisted_detections[self._detection_id_counter] = {
                    "box": det["box"],
                    "conf": det["conf"],
                    "cls": det["cls"],
                    "timestamp": current_time,
                }

        # Remove expired detections
        expired_ids = [
            det_id
            for det_id, det in self.persisted_detections.items()
            if current_time - det["timestamp"] > self.persistence_duration
        ]
        for det_id in expired_ids:
            del self.persisted_detections[det_id]

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area
