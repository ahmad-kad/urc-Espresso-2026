"""
Unit tests for metrics calculations
"""

import numpy as np
import pytest

from utils.metrics import calculate_f1_score, calculate_iou, calculate_precision_recall


class TestCalculateIOU:
    """Test IoU calculation"""

    def test_perfect_overlap(self):
        """Test IoU with perfect overlap"""
        box1 = [10, 10, 20, 20]
        box2 = [10, 10, 20, 20]
        iou = calculate_iou(box1, box2)
        assert iou == 1.0

    def test_no_overlap(self):
        """Test IoU with no overlap"""
        box1 = [10, 10, 20, 20]
        box2 = [30, 30, 40, 40]
        iou = calculate_iou(box1, box2)
        assert iou == 0.0

    def test_partial_overlap(self):
        """Test IoU with partial overlap"""
        box1 = [10, 10, 20, 20]
        box2 = [15, 15, 25, 25]
        iou = calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0

    def test_contained_box(self):
        """Test IoU with one box contained in another"""
        box1 = [10, 10, 30, 30]
        box2 = [15, 15, 25, 25]
        iou = calculate_iou(box1, box2)
        assert iou > 0.0


class TestCalculatePrecisionRecall:
    """Test precision and recall calculation"""

    def test_perfect_detection(self):
        """Test with perfect detection"""
        gt_boxes = [[10, 10, 20, 20], [30, 30, 40, 40]]
        pred_boxes = [[10, 10, 20, 20], [30, 30, 40, 40]]
        gt_classes = [0, 1]
        pred_classes = [0, 1]

        precision, recall = calculate_precision_recall(
            gt_boxes, pred_boxes, gt_classes, pred_classes, iou_threshold=0.5
        )

        assert precision == 1.0
        assert recall == 1.0

    def test_false_positives(self):
        """Test with false positives"""
        gt_boxes = [[10, 10, 20, 20]]
        pred_boxes = [[10, 10, 20, 20], [50, 50, 60, 60]]  # Extra detection
        gt_classes = [0]
        pred_classes = [0, 1]

        precision, recall = calculate_precision_recall(
            gt_boxes, pred_boxes, gt_classes, pred_classes, iou_threshold=0.5
        )

        assert precision < 1.0
        assert recall == 1.0

    def test_false_negatives(self):
        """Test with false negatives"""
        gt_boxes = [[10, 10, 20, 20], [30, 30, 40, 40]]
        pred_boxes = [[10, 10, 20, 20]]  # Missing one
        gt_classes = [0, 1]
        pred_classes = [0]

        precision, recall = calculate_precision_recall(
            gt_boxes, pred_boxes, gt_classes, pred_classes, iou_threshold=0.5
        )

        assert precision == 1.0
        assert recall < 1.0


class TestCalculateF1Score:
    """Test F1 score calculation"""

    def test_perfect_f1(self):
        """Test F1 with perfect precision and recall"""
        f1 = calculate_f1_score(1.0, 1.0)
        assert f1 == 1.0

    def test_zero_f1(self):
        """Test F1 with zero precision or recall"""
        f1 = calculate_f1_score(0.0, 0.5)
        assert f1 == 0.0

    def test_balanced_f1(self):
        """Test F1 with balanced precision and recall"""
        f1 = calculate_f1_score(0.8, 0.8)
        assert abs(f1 - 0.8) < 1e-10

    def test_imbalanced_f1(self):
        """Test F1 with imbalanced precision and recall"""
        f1 = calculate_f1_score(1.0, 0.5)
        assert 0.0 < f1 < 1.0
