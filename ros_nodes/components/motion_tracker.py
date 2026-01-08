"""
Motion Tracker Component
Handles Kalman filtering, position validation, and consensus checking for moving ArUco markers
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger_config import get_logger

logger = get_logger(__name__)


class MotionTracker:
    """
    Component for tracking moving ArUco markers
    Handles Kalman filtering, position validation, consensus checking, and confidence decay
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize motion tracker

        Args:
            config: Configuration dictionary
        """
        aruco_config = config.get("aruco", {})

        self.kalman_tracking = aruco_config.get("kalman_tracking", True)
        self.consensus_frames = aruco_config.get("consensus_frames", 1)
        self.confidence_decay_rate = aruco_config.get("confidence_decay_rate", 0.1)
        self.motion_adaptive = aruco_config.get("motion_adaptive", True)

        # Kalman filters for marker tracking
        self.marker_trackers: Dict[int, cv2.KalmanFilter] = {}

        # Multi-frame consensus tracking
        self.marker_detection_history: Dict[int, List[Dict]] = defaultdict(list)
        self.marker_last_seen: Dict[int, float] = {}

        # Motion velocity estimation (pixels per frame)
        self.marker_velocities: Dict[int, float] = {}

    def init_tracker(self, marker_id: int, bbox: List[float]) -> None:
        """
        Initialize Kalman filter for a marker

        Args:
            marker_id: Marker ID
            bbox: Bounding box [x_min, y_min, x_max, y_max]
        """
        kf = cv2.KalmanFilter(4, 2)  # 4 state (x, y, vx, vy), 2 measurement (x, y)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        kf.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        kf.statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32)
        kf.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32)

        self.marker_trackers[marker_id] = kf

    def predict_position(self, marker_id: int) -> Optional[Tuple[float, float]]:
        """
        Predict next marker position using Kalman filter

        Args:
            marker_id: Marker ID

        Returns:
            Predicted (x, y) position or None if no tracker
        """
        if not self.kalman_tracking or marker_id not in self.marker_trackers:
            return None

        kf = self.marker_trackers[marker_id]
        prediction = kf.predict()
        return (float(prediction[0]), float(prediction[1]))

    def update_tracker(self, marker_id: int, bbox: List[float]) -> None:
        """
        Update Kalman filter with new measurement

        Args:
            marker_id: Marker ID
            bbox: Bounding box [x_min, y_min, x_max, y_max]
        """
        if not self.kalman_tracking:
            return

        if marker_id not in self.marker_trackers:
            self.init_tracker(marker_id, bbox)
            return

        kf = self.marker_trackers[marker_id]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        measurement: np.ndarray = np.array([[center_x], [center_y]], dtype=np.float32)
        kf.correct(measurement)

        # Update velocity estimate
        if marker_id in self.marker_last_seen:
            prev_center = self.marker_trackers[marker_id].statePost[:2]
            dt = time.time() - self.marker_last_seen[marker_id]
            if dt > 0:
                velocity = (
                    np.sqrt(
                        (center_x - prev_center[0]) ** 2
                        + (center_y - prev_center[1]) ** 2
                    )
                    / dt
                )
                self.marker_velocities[marker_id] = velocity

        self.marker_last_seen[marker_id] = time.time()

    def validate_position(self, marker_id: int, bbox: List[float]) -> bool:
        """
        Validate detection position against prediction

        Args:
            marker_id: Marker ID
            bbox: Bounding box [x_min, y_min, x_max, y_max]

        Returns:
            True if position is valid (within reasonable distance of prediction)
        """
        if not self.kalman_tracking:
            return True

        predicted = self.predict_position(marker_id)
        if predicted is None:
            return True  # New marker, accept

        if marker_id not in self.marker_last_seen:
            return True  # New marker, accept

        tracking_duration = time.time() - self.marker_last_seen.get(marker_id, 0)
        if tracking_duration < 0.1:
            return True  # Too new, accept without validation

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        distance = np.sqrt(
            (center_x - predicted[0]) ** 2 + (center_y - predicted[1]) ** 2
        )

        velocity = self.marker_velocities.get(marker_id, 0)
        max_distance = 200 if velocity > 2.0 else 150

        return bool(distance < max_distance)

    def check_consensus(
        self, marker_id: int, bbox: List[float], confidence: float
    ) -> bool:
        """
        Check if marker has been detected consistently across frames

        Args:
            marker_id: Marker ID
            bbox: Bounding box [x_min, y_min, x_max, y_max]
            confidence: Detection confidence

        Returns:
            True if marker has consensus or is new detection
        """
        if self.consensus_frames <= 1:
            return True

        current_time = time.time()

        # Add current detection
        self.marker_detection_history[marker_id].append(
            {"bbox": bbox, "confidence": confidence, "timestamp": current_time}
        )

        # Remove old detections (>1 second)
        self.marker_detection_history[marker_id] = [
            d
            for d in self.marker_detection_history[marker_id]
            if current_time - d["timestamp"] < 1.0
        ]

        history_size = len(self.marker_detection_history[marker_id])

        # Allow first detection (new markers)
        if history_size == 1:
            return True

        # Require consensus across frames for existing markers
        if history_size >= self.consensus_frames:
            recent = self.marker_detection_history[marker_id][-self.consensus_frames :]
            centers_x = [(d["bbox"][0] + d["bbox"][2]) / 2 for d in recent]
            centers_y = [(d["bbox"][1] + d["bbox"][3]) / 2 for d in recent]

            position_tolerance = 100
            if (
                max(centers_x) - min(centers_x) < position_tolerance
                and max(centers_y) - min(centers_y) < position_tolerance
            ):
                return True

        # Allow high confidence detections even without full consensus
        if history_size >= 1 and confidence > 0.9:
            return True

        return False

    def apply_confidence_decay(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply confidence decay for markers not seen recently

        Args:
            detections: List of detection dictionaries

        Returns:
            Detections with decayed confidence
        """
        if self.confidence_decay_rate <= 0:
            return detections

        current_time = time.time()

        for det in detections:
            marker_id = det.get("marker_id")
            if marker_id is not None:
                if marker_id in self.marker_last_seen:
                    time_since_seen = current_time - self.marker_last_seen[marker_id]
                    if time_since_seen > 0.1:
                        decay = self.confidence_decay_rate * time_since_seen
                        det["confidence"] = max(0.5, det["confidence"] * (1.0 - decay))
                else:
                    self.marker_last_seen[marker_id] = current_time

        return detections

    def get_velocity(self, marker_id: int) -> float:
        """Get velocity estimate for marker"""
        return self.marker_velocities.get(marker_id, 0.0)
