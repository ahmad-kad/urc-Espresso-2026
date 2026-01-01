"""
Integration tests for training workflow
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from detector import ObjectDetector
from trainer import ModelTrainer


@pytest.mark.integration
class TestTrainingWorkflow:
    """Test complete training workflow"""

    @patch("detector.YOLO")
    def test_training_to_evaluation_workflow(self, mock_yolo, sample_config, temp_dir):
        """Test training followed by evaluation"""
        # Setup mocks
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
        mock_result = Mock()
        mock_result.save_dir = temp_dir / "models" / "test"
        mock_result.save_dir.mkdir(parents=True)
        (mock_result.save_dir / "weights").mkdir()
        (mock_result.save_dir / "weights" / "best.pt").touch()

        mock_model.train.return_value = mock_result
        mock_model.val.return_value = Mock(
            box=Mock(map50=0.85, map=0.75, mp=0.90, mr=0.80, f1=Mock(mean=lambda: 0.85))
        )
        mock_yolo.return_value = mock_model

        # Train model
        trainer = ModelTrainer(sample_config)
        train_result = trainer.train("data.yaml", experiment_name="test")

        assert train_result["success"] is True

        # Evaluate model - need to mock YOLO for evaluation
        with patch("ultralytics.YOLO") as mock_eval_yolo:
            mock_eval_model = Mock()
            mock_eval_results = Mock()
            mock_eval_results.box.map50 = 0.85
            mock_eval_results.box.map = 0.75
            mock_eval_results.box.mp = 0.90
            mock_eval_results.box.mr = 0.80
            mock_eval_results.box.f1 = Mock(mean=lambda: 0.85)
            mock_eval_model.val.return_value = mock_eval_results
            mock_eval_yolo.return_value = mock_eval_model

            metrics = trainer.evaluate_model_performance(
                train_result["model_path"], "data.yaml", input_size=224
            )

            assert "mAP50" in metrics
            assert metrics["mAP50"] > 0

    @patch("detector.YOLO")
    def test_model_conversion_workflow(self, mock_yolo, sample_config, temp_dir):
        """Test model training to ONNX conversion"""
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
        mock_result = Mock()
        mock_result.save_dir = temp_dir / "models" / "test"
        mock_result.save_dir.mkdir(parents=True)
        (mock_result.save_dir / "weights").mkdir()
        model_file = mock_result.save_dir / "weights" / "best.pt"
        model_file.touch()

        mock_model.train.return_value = mock_result
        mock_model.export.return_value = None
        mock_yolo.return_value = mock_model

        trainer = ModelTrainer(sample_config)
        train_result = trainer.train("data.yaml", experiment_name="test")

        # Convert to ONNX - need to mock YOLO for conversion
        with patch("ultralytics.YOLO") as mock_onnx_yolo:
            mock_onnx_model = Mock()
            mock_onnx_model.export.return_value = None
            mock_onnx_yolo.return_value = mock_onnx_model

            # Create a valid model file for loading
            model_file = Path(train_result["model_path"])
            model_file.parent.mkdir(parents=True, exist_ok=True)
            # Write minimal valid PyTorch checkpoint
            import torch

            torch.save({"model": "dummy"}, model_file)

            onnx_path = trainer.convert_to_onnx(
                train_result["model_path"],
                input_size=224,
                output_dir=str(temp_dir / "onnx"),
            )

            assert mock_onnx_model.export.called
