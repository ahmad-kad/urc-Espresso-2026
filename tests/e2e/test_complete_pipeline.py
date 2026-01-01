"""
End-to-end tests for complete detection pipeline
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.mark.e2e
@pytest.mark.slow
class TestCompletePipeline:
    """Test complete detection pipeline from config to inference"""

    @patch("detector.YOLO")
    def test_config_to_inference_pipeline(
        self, mock_detector_yolo, sample_config, temp_dir
    ):
        """Test complete pipeline: config -> training -> inference"""
        # Setup mocks
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
        mock_model.predict.return_value = [Mock(boxes=Mock())]
        mock_result = Mock()
        mock_result.save_dir = temp_dir / "models" / "test"
        mock_result.save_dir.mkdir(parents=True, exist_ok=True)
        (mock_result.save_dir / "weights").mkdir()
        (mock_result.save_dir / "weights" / "best.pt").touch()
        mock_model.train.return_value = mock_result
        mock_detector_yolo.return_value = mock_model

        # Import here to use mocked YOLO
        from core.models import ObjectDetector
        from core.trainer import ModelTrainer

        # Step 1: Train model
        trainer = ModelTrainer(sample_config)
        train_result = trainer.train("data.yaml", experiment_name="e2e_test")
        assert train_result["success"] is True

        # Step 2: Create detector
        detector = ObjectDetector(sample_config)

        # Step 3: Run inference
        result = detector.predict("test_image.jpg")
        assert result is not None

    @pytest.mark.requires_model
    def test_evaluation_pipeline(self, sample_config):
        """Test complete evaluation pipeline"""
        # This test requires actual models - skip if not available
        pytest.skip("Requires trained models - run manually with --requires-model flag")
