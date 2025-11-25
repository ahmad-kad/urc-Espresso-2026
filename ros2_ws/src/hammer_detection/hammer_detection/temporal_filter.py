#!/usr/bin/env python3
"""
Temporal Confidence Filter for YOLO object detection
Optimized for desert/low-feature environments with EMA filtering
"""

import numpy as np
from collections import defaultdict, deque
import time
import logging


class TemporalConfidenceFilter:
    """
    Exponential Moving Average filter for temporal confidence boosting
    Optimized for challenging environments with sparse features
    """

    def __init__(self, alpha=0.4, min_frames=5, max_history=15, confidence_boost_factor=0.25,
                 stability_threshold=0.7, cleanup_age=60, desert_mode=True,
                 dust_tolerance=0.3, lighting_adaptation=True):
        """
        Initialize temporal confidence filter

        Args:
            alpha: EMA smoothing factor (higher = more weight to recent detections)
            min_frames: Minimum frames needed before applying temporal boost
            max_history: Maximum frames to keep in history per object
            confidence_boost_factor: Maximum confidence boost multiplier
            stability_threshold: Confidence threshold for considering detection stable
            cleanup_age: Remove objects not seen for this many frames
            desert_mode: Enable desert-specific optimizations (larger tracking grid, dust tolerance)
            dust_tolerance: Maximum confidence variance tolerated before reducing boost (0.0-1.0)
            lighting_adaptation: Adapt to extreme lighting conditions with slower confidence decay
        """
        self.alpha = alpha
        self.min_frames = min_frames
        self.max_history = max_history
        self.confidence_boost_factor = confidence_boost_factor
        self.stability_threshold = stability_threshold
        self.cleanup_age = cleanup_age
        self.desert_mode = desert_mode
        self.dust_tolerance = dust_tolerance
        self.lighting_adaptation = lighting_adaptation

        # Desert-specific optimizations
        if self.desert_mode:
            # Larger tracking grid for sparse features and camera shake
            self.tracking_grid_size = 80  # pixels (larger than default 60)
            # More conservative stability requirements
            self.position_stability_weight = 0.4  # Higher weight for position consistency
            # Slower cleanup for very sparse environments
            self.cleanup_age = max(cleanup_age, 90)  # Keep objects longer
        else:
            self.tracking_grid_size = 40  # Standard indoor tracking
            self.position_stability_weight = 0.3

        # Lighting adaptation tracking
        self.extreme_lighting_detected = False
        self.lighting_confidence_history = []

        # Object tracking: object_id -> history dict
        self.object_history = {}

        # Frame counter for cleanup
        self.frame_count = 0

        # Logger
        self.logger = logging.getLogger(__name__)

    def update_confidence(self, detections, timestamp=None):
        """
        Update confidence scores with temporal filtering

        Args:
            detections: List of detection dicts with 'bbox', 'confidence', 'class_name', etc.
            timestamp: Optional timestamp for tracking

        Returns:
            Filtered detections with boosted confidence scores
        """
        self.frame_count += 1
        filtered_detections = []

        # Process each detection
        for det in detections:
            obj_id = self._get_object_id(det)

            # Initialize object history if new
            if obj_id not in self.object_history:
                self.object_history[obj_id] = {
                    'confidences': deque(maxlen=self.max_history),
                    'bbox_centers': deque(maxlen=self.max_history),
                    'classes': deque(maxlen=self.max_history),
                    'timestamps': deque(maxlen=self.max_history),
                    'ema_confidence': None,
                    'stability_score': 0.0,
                    'last_seen': self.frame_count,
                    'consecutive_frames': 0
                }

            history = self.object_history[obj_id]
            history['last_seen'] = self.frame_count

            # Add current detection to history
            center_x = (det['bbox'][0] + det['bbox'][2]) / 2
            center_y = (det['bbox'][1] + det['bbox'][3]) / 2

            history['confidences'].append(det['confidence'])
            history['bbox_centers'].append((center_x, center_y))
            history['classes'].append(det['class_name'])
            history['timestamps'].append(timestamp or time.time())

            # Update consecutive frames counter
            if det['confidence'] >= self.stability_threshold:
                history['consecutive_frames'] += 1
            else:
                history['consecutive_frames'] = max(0, history['consecutive_frames'] - 1)

            # Calculate EMA confidence if we have enough history
            if len(history['confidences']) >= self.min_frames:
                ema_conf = self._calculate_ema(history['confidences'])

                # Store EMA for next iteration
                if history['ema_confidence'] is None:
                    history['ema_confidence'] = ema_conf
                else:
                    history['ema_confidence'] = self.alpha * ema_conf + (1 - self.alpha) * history['ema_confidence']

                # Calculate stability score based on consistency
                stability = self._calculate_stability_score(history)
                history['stability_score'] = stability

                # Desert-specific adjustments
                adjusted_boost_factor = self._calculate_desert_boost_factor(
                    history, stability, ema_conf
                )

                boosted_confidence = min(1.0, history['ema_confidence'] * (1.0 + adjusted_boost_factor))

                # Only keep detections that meet minimum boosted confidence
                if boosted_confidence >= 0.3:  # Lower threshold for challenging environments
                    det_copy = det.copy()
                    det_copy['confidence'] = boosted_confidence
                    det_copy['temporal_boost'] = adjusted_boost_factor
                    det_copy['stability_score'] = stability
                    filtered_detections.append(det_copy)
            else:
                # Not enough history yet, keep original confidence
                det_copy = det.copy()
                det_copy['temporal_boost'] = 0.0
                det_copy['stability_score'] = 0.0
                filtered_detections.append(det_copy)

        # Periodic cleanup of old objects
        if self.frame_count % 30 == 0:  # Clean up every 30 frames
            self._cleanup_old_objects()

        return filtered_detections

    def _calculate_ema(self, confidence_history):
        """Calculate exponential moving average of confidence values"""
        if not confidence_history:
            return 0.0

        ema = confidence_history[0]
        for conf in list(confidence_history)[1:]:
            ema = self.alpha * conf + (1 - self.alpha) * ema
        return ema

    def _calculate_stability_score(self, history):
        """
        Calculate stability score based on confidence variance and position consistency
        """
        if len(history['confidences']) < self.min_frames:
            return 0.0

        # Confidence stability (lower variance = higher stability)
        conf_array = np.array(history['confidences'])
        conf_std = np.std(conf_array)
        conf_stability = max(0.0, 1.0 - conf_std * 2)  # Scale variance to 0-1

        # Position stability (lower movement = higher stability)
        centers = list(history['bbox_centers'])
        if len(centers) >= 3:
            # Calculate average movement between frames
            movements = []
            for i in range(1, len(centers)):
                dx = centers[i][0] - centers[i-1][0]
                dy = centers[i][1] - centers[i-1][1]
                movement = np.sqrt(dx**2 + dy**2)
                movements.append(movement)

            avg_movement = np.mean(movements)
            # Normalize movement (using desert-optimized threshold)
            movement_threshold = 70.0 if self.desert_mode else 50.0  # More tolerant in desert
            pos_stability = max(0.0, 1.0 - avg_movement / movement_threshold)
        else:
            pos_stability = 0.5  # Neutral stability for short history

        # Combine confidence and position stability with desert weighting
        conf_weight = 0.6 if self.desert_mode else 0.7
        pos_weight = self.position_stability_weight
        stability_score = (conf_stability * conf_weight + pos_stability * pos_weight)
        return stability_score

    def _calculate_desert_boost_factor(self, history, stability, ema_conf):
        """
        Calculate confidence boost factor with desert-specific optimizations
        """
        base_boost = min(self.confidence_boost_factor,
                        stability * history['consecutive_frames'] / 20.0)

        if not self.desert_mode:
            return base_boost

        # Dust tolerance check - reduce boost if confidence fluctuates too much
        conf_array = np.array(list(history['confidences']))
        conf_variance = np.var(conf_array)

        if conf_variance > self.dust_tolerance:
            # High variance suggests dust/sand interference - reduce boost
            dust_penalty = min(0.5, conf_variance / self.dust_tolerance)
            base_boost *= (1.0 - dust_penalty * 0.3)  # Max 30% reduction

        # Lighting adaptation - detect extreme lighting conditions
        if self.lighting_adaptation:
            self._update_lighting_detection(history)

            if self.extreme_lighting_detected:
                # In extreme lighting, be more conservative with boosting
                # and rely more on temporal consistency
                temporal_weight = min(1.0, history['consecutive_frames'] / 15.0)
                base_boost = base_boost * 0.8 + temporal_weight * self.confidence_boost_factor * 0.2

        return base_boost

    def _update_lighting_detection(self, history):
        """
        Detect extreme lighting conditions based on confidence patterns
        """
        recent_confs = list(history['confidences'])[-10:]  # Last 10 frames
        if len(recent_confs) >= 5:
            # Extreme lighting often causes very high or very low confidence spikes
            conf_mean = np.mean(recent_confs)
            conf_std = np.std(recent_confs)

            # High variance or extreme values suggest lighting issues
            extreme_lighting = (conf_std > 0.25 or
                              max(recent_confs) > 0.9 or
                              min(recent_confs) < 0.2)

            self.extreme_lighting_detected = extreme_lighting

    def _get_object_id(self, detection):
        """
        Generate consistent object ID for tracking
        Uses class name and quantized bounding box center
        """
        center_x = (detection['bbox'][0] + detection['bbox'][2]) / 2
        center_y = (detection['bbox'][1] + detection['bbox'][3]) / 2

        # Quantize position using desert-optimized grid size
        grid_x = int(center_x // self.tracking_grid_size)
        grid_y = int(center_y // self.tracking_grid_size)

        # Include class to separate different object types in same area
        obj_id = f"{detection['class_name']}_{grid_x}_{grid_y}"

        return obj_id

    def _cleanup_old_objects(self):
        """Remove objects that haven't been seen recently"""
        current_objects = len(self.object_history)
        to_remove = []

        for obj_id, history in self.object_history.items():
            frames_since_seen = self.frame_count - history['last_seen']
            if frames_since_seen > self.cleanup_age:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.object_history[obj_id]

        if to_remove:
            self.logger.debug(f"Cleaned up {len(to_remove)} old objects. "
                            f"Active objects: {len(self.object_history)}")

    def get_object_stats(self):
        """Get statistics about tracked objects for debugging"""
        stats = {
            'total_objects': len(self.object_history),
            'objects_with_history': sum(1 for h in self.object_history.values()
                                      if len(h['confidences']) >= self.min_frames),
            'average_stability': np.mean([h['stability_score'] for h in self.object_history.values()
                                        if h['stability_score'] > 0])
        }
        return stats

    def reset(self):
        """Reset filter state"""
        self.object_history.clear()
        self.frame_count = 0
        self.logger.info("Temporal confidence filter reset")
