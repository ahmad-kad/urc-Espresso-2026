"""
Temporal Smoothing Component
Applies temporal smoothing to stabilize detections across frames
"""

from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger_config import get_logger

logger = get_logger(__name__)


class TemporalSmoother:
    """
    Component for temporal smoothing of detections
    Boosts confidence for consistently detected objects
    """

    def __init__(self, config: Dict[str, Any], aruco_class_id: Optional[int] = None):
        """
        Initialize temporal smoother

        Args:
            config: Configuration dictionary
            aruco_class_id: Class ID for ArUco tags (to skip smoothing)
        """
        self.config = config
        self.aruco_class_id = aruco_class_id

        realtime_config = config.get("realtime", {})
        edge_config = config.get("edge", {})

        self.enabled = realtime_config.get("temporal_filter", False)
        self.alpha = realtime_config.get("temporal_alpha", 0.4)
        self.min_frames = realtime_config.get("temporal_min_frames", 5)
        max_history = realtime_config.get("temporal_max_history", 10)

        if edge_config.get("optimize_memory", False):
            max_history = min(max_history, 5)

        self.max_history_size = max_history
        self.history: List[List[Dict]] = []

    def smooth(self, results: Any) -> Dict[int, float]:
        """
        Apply temporal smoothing to detection results

        Args:
            results: Current frame detection results

        Returns:
            Dictionary mapping detection index to boosted confidence
        """
        if not self.enabled:
            return {}

        smoothed_confidences: Dict[str, float] = {}

        if not hasattr(results, "__iter__") or len(results) == 0:
            return smoothed_confidences

        result = results[0]
        if not hasattr(result, "boxes") or len(result.boxes) == 0:
            return smoothed_confidences

        # Extract current detections
        current_detections = self._extract_detections(result)

        # Add to history
        self.history.append(current_detections)
        if len(self.history) > self.max_history_size:
            self.history.pop(0)

        # Need at least min_frames for smoothing
        if len(self.history) < self.min_frames:
            return smoothed_confidences

        # Apply smoothing
        for det in current_detections:
            # Skip ArUco tags - they're already reliable
            if self.aruco_class_id is not None and det["cls"] == self.aruco_class_id:
                continue

            match_count = self._count_matches(det)

            # Boost confidence if detection is consistent
            original_conf = det["conf"]
            boost_factor = 0.1 if original_conf > 0.95 else 0.2

            if match_count >= self.min_frames - 1:
                boosted_conf = min(
                    1.0, original_conf + (1.0 - original_conf) * boost_factor
                )
                smoothed_confidences[det["index"]] = boosted_conf
            elif match_count >= self.min_frames // 2:
                boosted_conf = min(
                    1.0, original_conf + (1.0 - original_conf) * (boost_factor * 0.5)
                )
                smoothed_confidences[det["index"]] = boosted_conf

        return smoothed_confidences

    def _extract_detections(self, result: Any) -> List[Dict]:
        """Extract detection information from result"""
        detections = []
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

                detections.append({"index": i, "box": box, "conf": conf, "cls": cls})
            except:
                continue

        return detections

    def _count_matches(self, det: Dict) -> int:
        """Count how many times similar detection appeared in history"""
        match_count = 0

        for hist_frame in self.history[-self.min_frames :]:
            for hist_det in hist_frame:
                if hist_det["cls"] == det["cls"]:
                    iou = self._calculate_iou(det["box"], hist_det["box"])
                    if iou > 0.3:  # Overlapping
                        match_count += 1
                        break

        return match_count

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

        return float(inter_area / union_area)
