"""
Integration tests for device consistency in workflows
"""

from unittest.mock import Mock, patch

import pytest
import torch

from detector import ObjectDetector
from trainer import ModelTrainer


@pytest.mark.integration
class TestDeviceWorkflow:
    """Test device consistency in complete workflows"""

    def test_training_workflow_device_consistency(self, sample_config, monkeypatch):
        """Test device consistency throughout training workflow"""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        config = sample_config.copy()
        config["device"] = "auto"  # Should resolve to CPU

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_result = Mock()
            mock_result.save_dir = "/tmp/test"
            mock_model.train.return_value = mock_result
            mock_yolo.return_value = mock_model

            trainer = ModelTrainer(config)

            # Check device resolution
            training_kwargs = trainer.detector._build_training_kwargs("test")
            assert training_kwargs["device"] == "cpu"

            # Training should use same device
            with patch("trainer.Path") as mock_path:
                mock_path.return_value.exists.return_value = True
                result = trainer.train("data.yaml")
                assert result["success"] is True

    def test_evaluation_workflow_device_consistency(self, sample_config, temp_dir):
        """Test device consistency in evaluation workflow"""
        config = sample_config.copy()
        config["device"] = "cpu"

        with patch("detector.YOLO") as mock_yolo, patch(
            "ultralytics.YOLO"
        ) as mock_eval_yolo:

            # Setup detector model
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_yolo.return_value = mock_model

            # Setup evaluation model
            mock_eval_model = Mock()
            mock_eval_results = Mock()
            mock_eval_results.box.map50 = 0.85
            mock_eval_results.box.map = 0.75
            mock_eval_results.box.mp = 0.90
            mock_eval_results.box.mr = 0.80
            mock_eval_results.box.f1 = Mock(mean=lambda: 0.85)
            mock_eval_model.val.return_value = mock_eval_results
            mock_eval_yolo.return_value = mock_eval_model

            trainer = ModelTrainer(config)

            # Evaluation should work with consistent device
            model_path = str(temp_dir / "test_model.pt")
            metrics = trainer.evaluate_model_performance(model_path, "data.yaml", 224)

            assert "mAP50" in metrics

    def test_no_device_switching_during_inference(self, sample_config):
        """Test that device doesn't switch during inference operations"""
        config = sample_config.copy()
        config["device"] = "cpu"

        device_checks = []

        def track_check():
            device_checks.append(1)
            return False

        with patch("detector.YOLO") as mock_yolo, patch(
            "torch.cuda.is_available", side_effect=track_check
        ):
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_model.predict.return_value = [Mock(boxes=Mock())]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)

            # Multiple predictions should not trigger device checks
            detector.predict("test1.jpg")
            detector.predict("test2.jpg")
            detector.predict("test3.jpg")

            # With explicit device, should not check CUDA during inference
            assert (
                len(device_checks) == 0
            ), "Device should not be checked during inference with explicit device"
