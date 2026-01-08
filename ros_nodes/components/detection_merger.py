"""
Detection Merger Component
Merges ML and ArUco detections into unified format
"""

from typing import Any, Dict, List

import numpy as np

from utils.logger_config import get_logger

logger = get_logger(__name__)


class DetectionMerger:
    """
    Component for merging ML and ArUco detection results
    Handles format conversion and merging logic
    """

    def __init__(self, aruco_class_id: int):
        """
        Initialize detection merger

        Args:
            aruco_class_id: Class ID for ArUco tags
        """
        self.aruco_class_id = aruco_class_id

    def merge(self, ml_results: Any, aruco_detections: List[Dict]) -> Any:
        """
        Merge ML and ArUco detection results

        Args:
            ml_results: ML model detection results (YOLO format)
            aruco_detections: List of ArUco detection dictionaries

        Returns:
            Merged results in YOLO format
        """
        if not aruco_detections:
            return ml_results

        aruco_result = self._aruco_to_yolo_format(aruco_detections)
        if aruco_result is None:
            return ml_results

        return self._merge_results(ml_results, aruco_result)

    def _aruco_to_yolo_format(self, aruco_detections: List[Dict]) -> Any:
        """Convert ArUco detections to YOLO-like format"""
        if not aruco_detections:
            return None

        class MockBoxes:
            def __init__(self, detections):
                self.xyxy: List[np.ndarray] = []
                self.conf = []
                self.cls = []
                for det in detections:
                    self.xyxy.append(np.array(det["bbox"], dtype=np.float32))
                    self.conf.append(det["confidence"])
                    self.cls.append(det["class_id"])
                if self.xyxy:
                    self.xyxy = np.array(self.xyxy)
                    self.conf = np.array(self.conf)
                    self.cls = np.array(self.cls)
                else:
                    self.xyxy = np.array([])
                    self.conf = np.array([])
                    self.cls = np.array([])

            def __len__(self):
                return len(self.xyxy)

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        boxes = MockBoxes(aruco_detections)
        return MockResult(boxes)

    def _merge_results(self, ml_results: Any, aruco_result: Any) -> Any:
        """Merge ML and ArUco results"""
        # Extract ML detections
        ml_boxes = []
        ml_confs = []
        ml_clses = []

        if ml_results and hasattr(ml_results, "__iter__") and len(ml_results) > 0:
            ml_result = ml_results[0]
            if hasattr(ml_result, "boxes") and len(ml_result.boxes.xyxy) > 0:
                boxes = ml_result.boxes
                for i in range(len(boxes.xyxy)):
                    try:
                        if isinstance(boxes.xyxy, np.ndarray):
                            ml_boxes.append(boxes.xyxy[i].copy())
                            ml_confs.append(float(boxes.conf[i]))
                            ml_clses.append(int(boxes.cls[i]))
                        else:
                            ml_boxes.append(boxes.xyxy[i].cpu().numpy().copy())
                            ml_confs.append(float(boxes.conf[i].cpu().numpy()))
                            ml_clses.append(int(boxes.cls[i].cpu().numpy()))
                    except Exception as e:
                        logger.debug(f"Error extracting ML detection {i}: {e}")
                        continue

        # Add ArUco detections
        aruco_boxes = aruco_result.boxes.xyxy
        aruco_confs = aruco_result.boxes.conf
        aruco_clses = aruco_result.boxes.cls

        # Combine all detections
        all_boxes = ml_boxes + [aruco_boxes[i] for i in range(len(aruco_boxes))]
        all_confs = ml_confs + [aruco_confs[i] for i in range(len(aruco_confs))]
        all_clses = ml_clses + [aruco_clses[i] for i in range(len(aruco_clses))]

        # Create merged result
        if not all_boxes:
            return self._create_empty_result()

        return self._create_result(all_boxes, all_confs, all_clses)

    def _create_empty_result(self) -> Any:
        """Create empty result in YOLO format"""

        class MockBoxes:
            def __init__(self):
                self.xyxy = np.array([])
                self.conf = np.array([])
                self.cls = np.array([])

            def __len__(self):
                return 0

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        return [MockResult(MockBoxes())]

    def _create_result(self, boxes: List, confs: List, clses: List) -> Any:
        """Create result object from detection lists"""

        class MockBoxes:
            def __init__(self, boxes, confs, clses):
                self.xyxy = np.array(boxes) if boxes else np.array([])
                self.conf = np.array(confs) if confs else np.array([])
                self.cls = np.array(clses) if clses else np.array([])

            def __len__(self):
                return len(self.xyxy)

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        merged_boxes = MockBoxes(boxes, confs, clses)
        return [MockResult(merged_boxes)]
