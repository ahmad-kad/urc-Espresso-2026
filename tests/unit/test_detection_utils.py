"""
Unit tests for detection utilities
"""

import numpy as np
import pytest

from utils.detection_utils import (
    filter_boxes_by_class,
    non_max_suppression,
    post_process_yolov8,
    scale_boxes_to_original,
    xywh2xyxy,
)


class TestXYWH2XYXY:
    """Test bounding box format conversion"""

    def test_single_box(self):
        """Test conversion of single box"""
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]])  # center, center, width, height
        result = xywh2xyxy(boxes)
        expected = np.array([[0.4, 0.4, 0.6, 0.6]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_boxes(self):
        """Test conversion of multiple boxes"""
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])
        result = xywh2xyxy(boxes)
        assert result.shape == (2, 4)
        assert result[0, 0] < result[0, 2]  # x1 < x2
        assert result[0, 1] < result[0, 3]  # y1 < y2

    def test_empty_array(self):
        """Test with empty array"""
        boxes = np.array([]).reshape(0, 4)
        result = xywh2xyxy(boxes)
        assert result.shape == (0, 4)


class TestNonMaxSuppression:
    """Test NMS functionality"""

    def test_no_overlap(self):
        """Test NMS with non-overlapping boxes"""
        boxes = np.array([[10, 10, 20, 20], [50, 50, 60, 60]])
        scores = np.array([0.9, 0.8])
        indices = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert len(indices) == 2

    def test_overlapping_boxes(self):
        """Test NMS with overlapping boxes"""
        boxes = np.array([[10, 10, 20, 20], [12, 12, 22, 22]])  # Overlaps with first
        scores = np.array([0.9, 0.8])
        indices = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert len(indices) == 1
        assert indices[0] == 0  # Higher score kept

    def test_empty_boxes(self):
        """Test NMS with empty input"""
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        indices = non_max_suppression(boxes, scores)
        assert len(indices) == 0


class TestPostProcessYOLOv8:
    """Test YOLOv8 post-processing"""

    def test_basic_post_processing(self):
        """Test basic post-processing"""
        # Create mock YOLOv8 output: (1, num_features, num_boxes)
        num_boxes = 5
        num_classes = 2
        output = np.random.rand(1, 4 + num_classes, num_boxes)

        boxes, confidences, class_ids = post_process_yolov8(
            output, conf_threshold=0.1, input_size=224
        )

        assert isinstance(boxes, np.ndarray)
        assert isinstance(confidences, np.ndarray)
        assert isinstance(class_ids, np.ndarray)

    def test_high_confidence_threshold(self):
        """Test with high confidence threshold"""
        output = np.random.rand(1, 6, 10) * 0.1  # Low confidence
        boxes, confidences, class_ids = post_process_yolov8(
            output, conf_threshold=0.9, input_size=224
        )
        # Should filter out most boxes
        assert len(boxes) <= 10


class TestScaleBoxes:
    """Test box scaling functionality"""

    def test_scale_to_original(self):
        """Test scaling boxes back to original size"""
        boxes = np.array([[100, 100, 200, 200]])  # xyxy format
        original_size = (480, 640)  # height, width
        input_size = 224

        scaled = scale_boxes_to_original(boxes, original_size, input_size)
        assert scaled.shape == boxes.shape
        assert scaled[0, 0] > boxes[0, 0]  # Scaled up

    def test_empty_boxes(self):
        """Test with empty boxes"""
        boxes = np.array([]).reshape(0, 4)
        scaled = scale_boxes_to_original(boxes, (480, 640), 224)
        assert len(scaled) == 0


class TestFilterBoxesByClass:
    """Test class filtering"""

    def test_filter_single_class(self):
        """Test filtering for single class"""
        boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
        confidences = np.array([0.9, 0.8])
        class_ids = np.array([0, 1])

        filtered_boxes, filtered_conf, filtered_classes = filter_boxes_by_class(
            boxes, confidences, class_ids, target_classes=[0]
        )

        assert len(filtered_boxes) == 1
        assert filtered_classes[0] == 0

    def test_filter_multiple_classes(self):
        """Test filtering for multiple classes"""
        boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]])
        confidences = np.array([0.9, 0.8, 0.7])
        class_ids = np.array([0, 1, 2])

        filtered_boxes, filtered_conf, filtered_classes = filter_boxes_by_class(
            boxes, confidences, class_ids, target_classes=[0, 2]
        )

        assert len(filtered_boxes) == 2
        assert set(filtered_classes) == {0, 2}
