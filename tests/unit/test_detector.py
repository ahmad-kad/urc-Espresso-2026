"""
Unit tests for ObjectDetector
"""

from unittest.mock import Mock, patch

import pytest

from core.models import ObjectDetector


class TestObjectDetector:
    """Test ObjectDetector class"""

    @patch("core.models.detector.YOLO")
    def test_init_with_yolov8(self, mock_yolo, sample_config):
        """Test detector initialization with YOLOv8"""
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(sample_config)

        assert detector.model is not None
        mock_yolo.assert_called_once()

    @patch("core.models.detector.YOLO")
    def test_predict_yolo(self, mock_yolo, sample_config):
        """Test prediction with YOLO model"""
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
        mock_model.predict.return_value = [Mock(boxes=Mock())]
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(sample_config)
        result = detector.predict("test_image.jpg")

        assert result is not None
        # predict is called twice: once for validation, once for actual prediction
        assert mock_model.predict.call_count == 2

    @patch("core.models.detector.YOLO")
    def test_train_yolo(self, mock_yolo, sample_config, temp_dir, mock_data_yaml):
        """Test training with YOLO model"""
        mock_model = Mock()
        mock_model.train.return_value = Mock(save_dir=str(temp_dir))
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(sample_config)
        result = detector.train(mock_data_yaml, model_name="test")

        assert result is not None
        mock_model.train.assert_called_once()

    @pytest.mark.debug
    @patch("core.models.detector.YOLO")
    def test_debug_mode(self, mock_yolo, sample_config, debug_mode):
        """Test debug mode functionality"""
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
        mock_yolo.return_value = mock_model

        detector = ObjectDetector(sample_config)
        # Debug mode should enable verbose logging
        assert detector.config == sample_config
