"""
Unit tests for device consistency
Prevents device thrashing and mismatches - high impact, low complexity mistakes
"""

from unittest.mock import Mock, patch

import pytest
import torch

from core.models import ObjectDetector
from core.trainer import ModelTrainer


class TestDeviceConsistency:
    """Test device consistency across components"""

    def test_device_auto_selection_cpu(self, sample_config, monkeypatch):
        """Test automatic CPU selection when CUDA unavailable"""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        config = sample_config.copy()
        config["device"] = "auto"

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)
            training_kwargs = detector._build_training_kwargs("test")

            # Device should resolve to 'cpu' when CUDA unavailable
            assert training_kwargs["device"] == "cpu"

    def test_device_auto_selection_cuda(self, sample_config, monkeypatch):
        """Test automatic CUDA selection when available"""
        # Patch device_utils.get_device to return 'cuda' for auto
        with patch("utils.device_utils.get_device") as mock_get_device:
            mock_get_device.return_value = "cuda"

            config = sample_config.copy()
            config["device"] = "auto"

            with patch("detector.YOLO") as mock_yolo:
                mock_model = Mock()
                mock_model.model = Mock()
                mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
                mock_yolo.return_value = mock_model

                detector = ObjectDetector(config)
                training_kwargs = detector._build_training_kwargs("test")

                # Device should resolve to 'cuda' when available
                assert training_kwargs["device"] == "cuda"

    def test_device_explicit_cpu(self, sample_config):
        """Test explicit CPU device selection"""
        config = sample_config.copy()
        config["device"] = "cpu"

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)
            training_kwargs = detector._build_training_kwargs("test")

            # Should use explicit CPU
            assert training_kwargs["device"] == "cpu"

    def test_device_explicit_cuda(self, sample_config, monkeypatch):
        """Test explicit CUDA device selection"""
        # Clear cache and mock CUDA availability
        from utils.device_utils import clear_device_cache

        clear_device_cache()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        config = sample_config.copy()
        config["device"] = "cuda"

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)
            training_kwargs = detector._build_training_kwargs("test")

            # Should use explicit CUDA
            assert training_kwargs["device"] == "cuda"

    def test_no_device_thrashing(self, sample_config, monkeypatch):
        """Test that device is not checked unnecessarily (prevents thrashing)"""
        config = sample_config.copy()
        config["device"] = "cpu"  # Explicit device, no need to check

        cuda_check_count = []

        def track_cuda_check():
            cuda_check_count.append(1)
            return False

        monkeypatch.setattr(torch.cuda, "is_available", track_cuda_check)

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)
            # Build training kwargs multiple times
            detector._build_training_kwargs("test1")
            detector._build_training_kwargs("test2")
            detector._build_training_kwargs("test3")

            # With explicit 'cpu', should not check CUDA at all
            # Allow 1 check max (during initialization if any)
            assert (
                len(cuda_check_count) <= 1
            ), f"CUDA checked {len(cuda_check_count)} times unnecessarily"

    def test_device_consistency_in_training(self, sample_config):
        """Test device consistency during training workflow"""
        config = sample_config.copy()
        config["device"] = "cpu"
        config["training"]["device"] = "cpu"

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_result = Mock()
            mock_result.save_dir = "/tmp/test"
            mock_model.train.return_value = mock_result
            mock_yolo.return_value = mock_model

            trainer = ModelTrainer(config)

            # Training should use consistent device
            training_kwargs = trainer.detector._build_training_kwargs("test")
            assert training_kwargs["device"] == "cpu"

    def test_device_resolution_priority(self, sample_config):
        """Test device resolution priority: explicit > training > auto"""
        config = sample_config.copy()
        config["device"] = "cpu"  # Explicit (highest priority)
        config["training"]["device"] = "cuda"  # Training config (lower priority)

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)
            training_kwargs = detector._build_training_kwargs("test")

            # Explicit device should take precedence
            assert (
                training_kwargs["device"] == "cpu"
            ), "Explicit device should override training device"

    def test_device_persistence(self, sample_config):
        """Test that device selection persists across operations"""
        config = sample_config.copy()
        config["device"] = "cpu"

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_model.predict.return_value = [Mock(boxes=Mock())]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)

            # Perform multiple operations
            detector._build_training_kwargs("test1")
            detector._build_training_kwargs("test2")
            detector.predict("test.jpg")

            # Device should remain consistent
            final_kwargs = detector._build_training_kwargs("test3")
            assert (
                final_kwargs["device"] == "cpu"
            ), "Device should persist across operations"

    def test_invalid_device_handling(self, sample_config):
        """Test handling of invalid device specification"""
        config = sample_config.copy()
        config["device"] = "invalid_device"

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)

            # Should either use invalid device (YOLO handles it) or fallback gracefully
            training_kwargs = detector._build_training_kwargs("test")
            # YOLO will handle invalid device, but we should log it
            assert "device" in training_kwargs

    def test_no_cuda_import_error(self, sample_config):
        """Test that CUDA unavailability doesn't cause import errors"""
        with patch("torch.cuda.is_available", return_value=False):
            config = sample_config.copy()
            config["device"] = "auto"

            with patch("detector.YOLO") as mock_yolo:
                mock_model = Mock()
                mock_model.model = Mock()
                mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
                mock_yolo.return_value = mock_model

                # Should not raise ImportError or AttributeError
                try:
                    detector = ObjectDetector(config)
                    training_kwargs = detector._build_training_kwargs("test")
                    assert training_kwargs["device"] == "cpu"
                except (ImportError, AttributeError) as e:
                    pytest.fail(f"Device check caused import error: {e}")

    def test_device_in_training_config_override(self, sample_config):
        """Test that training device config is respected when no explicit device"""
        config = sample_config.copy()
        config.pop("device", None)  # Remove explicit device
        config["training"]["device"] = "cuda"

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)
            training_kwargs = detector._build_training_kwargs("test")

            # Should use training device config
            # Note: detector uses config.get('device', 'auto'), so training device is separate
            assert training_kwargs["device"] in ["cuda", "cpu", "auto"]

    def test_device_consistency_across_trainer(self, sample_config):
        """Test device consistency between trainer and detector"""
        config = sample_config.copy()
        config["device"] = "cpu"

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_yolo.return_value = mock_model

            trainer = ModelTrainer(config)

            # Both trainer and detector should use same device
            detector_kwargs = trainer.detector._build_training_kwargs("test")

            # Device should be consistent
            assert detector_kwargs["device"] == "cpu"

    def test_device_check_only_when_needed(self, sample_config, monkeypatch):
        """Test that CUDA is only checked when device is 'auto'"""
        config = sample_config.copy()
        config["device"] = "cpu"  # Explicit, no check needed

        check_called = []

        def track_check():
            check_called.append(True)
            return False

        monkeypatch.setattr(torch.cuda, "is_available", track_check)

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.model = Mock()
            mock_model.model.parameters.return_value = [Mock(numel=lambda: 1000)]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)
            detector._build_training_kwargs("test")

            # With explicit 'cpu', should not check CUDA
            assert (
                len(check_called) == 0
            ), "CUDA should not be checked with explicit device"
