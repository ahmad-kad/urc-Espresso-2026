"""
Unit tests for ModelTrainer
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from core.trainer import ModelTrainer


class TestModelTrainer:
    """Test ModelTrainer class"""

    @patch("core.trainer.ObjectDetector")
    def test_init(self, mock_detector, sample_config):
        """Test trainer initialization"""
        trainer = ModelTrainer(sample_config)

        assert trainer.config == sample_config
        mock_detector.assert_called_once_with(sample_config)

    @patch("core.trainer.ObjectDetector")
    def test_train_success(self, mock_detector, sample_config, temp_dir):
        """Test successful training"""
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance

        # Mock training result
        mock_result = Mock()
        mock_result.save_dir = temp_dir / "models" / "test"
        mock_result.save_dir.mkdir(parents=True)
        mock_detector_instance.train.return_value = mock_result

        trainer = ModelTrainer(sample_config)
        result = trainer.train("data.yaml", experiment_name="test")

        assert result["success"] is True
        assert "model_path" in result

    @patch("core.trainer.ObjectDetector")
    def test_train_failure(self, mock_detector, sample_config):
        """Test training failure handling"""
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance
        mock_detector_instance.train.side_effect = Exception("Training failed")

        trainer = ModelTrainer(sample_config)
        result = trainer.train("data.yaml")

        assert result["success"] is False
        assert "error" in result

    @patch("core.trainer.ObjectDetector")
    def test_evaluate_model_performance(
        self, mock_detector, sample_config, mock_model_path
    ):
        """Test model performance evaluation"""
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance

        trainer = ModelTrainer(sample_config)

        with patch("ultralytics.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_results = Mock()
            mock_results.box.map50 = 0.85
            mock_results.box.map = 0.75
            mock_results.box.mp = 0.90
            mock_results.box.mr = 0.80
            mock_results.box.f1 = Mock(mean=lambda: 0.85)
            mock_model.val.return_value = mock_results
            mock_yolo.return_value = mock_model

            # Method removed - skip test
            pytest.skip("evaluate_model_performance method removed from ModelTrainer")

    @patch("core.trainer.ObjectDetector")
    def test_measure_inference_speed(
        self, mock_detector, sample_config, mock_model_path
    ):
        """Test inference speed measurement - method removed"""
        # Method removed - skip test
        pytest.skip("measure_inference_speed method removed from ModelTrainer")

    @patch("core.trainer.ObjectDetector")
    def test_get_model_size_mb(self, mock_detector, sample_config, temp_dir):
        """Test model size calculation"""
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance

        # Create a test file
        test_file = temp_dir / "test_model.pt"
        test_file.write_bytes(b"x" * (5 * 1024 * 1024))  # 5 MB

        # Method removed - skip test
        pytest.skip("get_model_size_mb method removed from ModelTrainer")

    @patch("core.trainer.ObjectDetector")
    def test_extract_save_dir(self, mock_detector, sample_config):
        """Test save directory extraction"""
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance

        trainer = ModelTrainer(sample_config)

        # Test with object result
        mock_result = Mock()
        mock_result.save_dir = Path("output/models/test")
        save_dir = trainer._extract_save_dir(mock_result, None, None, "test")
        assert save_dir == Path("output/models/test")

        # Test with dict result
        dict_result = {"save_dir": "output/models/test2"}
        save_dir = trainer._extract_save_dir(dict_result, None, None, "test2")
        assert save_dir == Path("output/models/test2")
