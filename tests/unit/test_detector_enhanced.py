"""
Enhanced unit tests for the ObjectDetector class
Tests performance, caching, error handling, and monitoring features
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from core.models import (
    ObjectDetector,
    ModelLoadError,
    InferenceError,
    TrainingError,
    ModelCache,
    get_model_hash,
    model_cache,
)


class TestModelCache:
    """Test the ModelCache functionality"""

    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = ModelCache(max_cache_size=2)
        assert len(cache._cache) == 0
        assert cache._max_size == 2

    def test_cache_put_and_get(self):
        """Test basic cache operations"""
        cache = ModelCache(max_cache_size=2)

        # Put item in cache
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test non-existent key
        assert cache.get("nonexistent") is None

    def test_cache_eviction(self):
        """Test LRU cache eviction"""
        cache = ModelCache(max_cache_size=2)

        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 to make it most recently used
        cache.get("key1")

        # Add third item - should evict key2 (least recently used)
        cache.put("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"

    def test_thread_safety(self):
        """Test thread-safe operations"""
        cache = ModelCache(max_cache_size=5)

        # Test concurrent access (basic smoke test)
        import threading
        import time

        results = []

        def worker(worker_id):
            for i in range(10):
                cache.put(f"key_{worker_id}_{i}", f"value_{worker_id}_{i}")
                time.sleep(0.001)  # Small delay to encourage race conditions
                result = cache.get(f"key_{worker_id}_{i}")
                results.append(result)

        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All operations should have succeeded
        assert len(results) == 30
        assert all(r is not None for r in results)


class TestGetModelHash:
    """Test model hash generation"""

    def test_hash_generation(self):
        """Test hash generation for model config"""
        config1 = {"architecture": "yolov8s", "input_size": 224}
        config2 = {"architecture": "yolov8s", "input_size": 224}
        config3 = {"architecture": "yolov8m", "input_size": 224}

        hash1 = get_model_hash(config1)
        hash2 = get_model_hash(config2)
        hash3 = get_model_hash(config3)

        # Same config should produce same hash
        assert hash1 == hash2
        # Different config should produce different hash
        assert hash1 != hash3

    def test_hash_length(self):
        """Test hash length"""
        config = {"model": {"architecture": "yolov8s"}}
        hash_value = get_model_hash(config)
        assert len(hash_value) == 8  # Should be 8 characters


class TestObjectDetector:
    """Test the ObjectDetector class"""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear model cache before each test"""
        model_cache._cache.clear()
        model_cache._access_times.clear()
        yield

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            "model": {
                "architecture": "yolov8s",
                "input_size": 224,
                "confidence_threshold": 0.5,
                "iou_threshold": 0.4,
                "max_detections": 20,
            },
            "data": {"classes": ["class1", "class2", "class3"]},
            "training": {"epochs": 10, "batch_size": 8, "learning_rate": 0.001},
        }

    def test_initialization(self, sample_config):
        """Test ObjectDetector initialization"""
        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(sample_config)

            assert detector.config == sample_config
            assert detector.class_names == ["class1", "class2", "class3"]
            assert isinstance(detector.model_hash, str)
            assert len(detector.model_hash) == 8
            assert detector._performance_stats["cache_misses"] == 1

    def test_model_caching(self, sample_config):
        """Test model caching functionality"""
        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model

            # First detector - should cache the model
            detector1 = ObjectDetector(sample_config)
            assert detector1._performance_stats["cache_misses"] == 1

            # Second detector with same config - should hit cache
            detector2 = ObjectDetector(sample_config)
            assert detector2._performance_stats["cache_hits"] == 1

            # Verify cache contains the model
            cache_key = detector1.model_hash
            assert model_cache.get(cache_key) is not None

    def test_model_validation(self, sample_config):
        """Test model validation during initialization"""
        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            # Mock the predict method for validation
            mock_model.predict.return_value = []
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(sample_config)

            # Should not raise an exception
            assert detector.model is not None

    def test_model_validation_failure(self):
        """Test model validation failure"""
        # Use different config to avoid cache
        config = {
            "model": {"architecture": "yolov8n", "input_size": 224},
            "data": {"classes": ["class1"]},
        }

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            # Make predict fail to cause validation failure
            mock_model.predict.side_effect = Exception("Validation failed")
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)

            # Health check should detect unhealthy model
            health = detector.health_check()
            assert health["status"] == "unhealthy"
            assert "error" in health

    @patch("detector.torch.randn")
    def test_predict_success(self, mock_randn):
        """Test successful prediction"""
        mock_randn.return_value = torch.randn(1, 3, 224, 224)

        # Use unique config to avoid cache
        config = {
            "model": {"architecture": "yolov8m", "input_size": 224},
            "data": {"classes": ["test"]},
        }

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_results = Mock()
            mock_model.predict.return_value = mock_results
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)

            # Test prediction
            result = detector.predict("dummy_image.jpg")

            assert result == mock_results
            assert detector._performance_stats["inference_count"] == 1
            assert detector._performance_stats["total_inference_time"] > 0

    def test_predict_failure(self):
        """Test prediction failure handling"""
        # Use unique config to avoid cache
        config = {
            "model": {"architecture": "yolov8l", "input_size": 224},
            "data": {"classes": ["test"]},
        }

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("Prediction failed")
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)

            # Prediction should fail with InferenceError
            with pytest.raises(InferenceError):
                detector.predict("dummy_image.jpg")

    def test_performance_stats_calculation(self, sample_config):
        """Test performance statistics calculation"""
        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.predict.return_value = []
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(sample_config)

            # Simulate some inference calls
            detector._performance_stats["inference_count"] = 5
            detector._performance_stats["total_inference_time"] = 2.5
            detector._performance_stats["cache_hits"] = 3
            detector._performance_stats["cache_misses"] = 2

            stats = detector.get_performance_stats()

            assert stats["avg_inference_time"] == 0.5  # 2.5 / 5
            assert stats["cache_hit_rate"] == 0.6  # 3 / (3 + 2)

    def test_health_check_healthy(self):
        """Test health check for healthy model"""
        # Use unique config to avoid cache
        config = {
            "model": {"architecture": "yolov8x", "input_size": 224},
            "data": {"classes": ["test"]},
        }

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.predict.return_value = []
            # Mock model structure
            mock_model.model = Mock()
            mock_model.ckpt_path = "/tmp/test_model.pt"
            mock_parameters = [Mock()]
            mock_parameters[0].device = torch.device("cuda:0")
            mock_model.model.parameters = Mock(return_value=iter(mock_parameters))
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)

            with patch("pathlib.Path.exists", return_value=True), patch(
                "pathlib.Path.stat"
            ) as mock_stat:

                mock_stat.return_value.st_size = 25 * 1024 * 1024  # 25MB

                health = detector.health_check()

                assert health["status"] == "healthy"
                assert health["model_loaded"] is True
                # Device detection may not work with mocks, just check it's present
                assert "device" in health
                assert health["model_size_mb"] == 25.0

    def test_health_check_unhealthy(self, sample_config):
        """Test health check for unhealthy model"""
        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("Model broken")
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(sample_config)

            health = detector.health_check()

            assert health["status"] == "unhealthy"
            assert "error" in health

    def test_warmup(self):
        """Test model warmup functionality"""
        # Use unique config to avoid cache
        config = {
            "model": {"architecture": "yolov8-tiny", "input_size": 224},
            "data": {"classes": ["test"]},
        }

        with patch("detector.YOLO") as mock_yolo, patch(
            "detector.torch.randn"
        ) as mock_randn:

            mock_model = Mock()
            mock_model.predict.return_value = []
            mock_yolo.return_value = mock_model
            mock_randn.return_value = torch.randn(1, 3, 224, 224)

            detector = ObjectDetector(config)

            # Should not raise exception
            detector.warmup(num_runs=3)

            # Should have called predict 4 times (3 warmup + 1 validation)
            assert mock_model.predict.call_count == 4

    def test_cache_clearing(self, sample_config):
        """Test cache clearing functionality"""
        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(sample_config)

            # Add something to cache
            model_cache.put("test_key", "test_value")
            assert model_cache.get("test_key") == "test_value"

            # Clear cache
            detector.clear_cache()

            assert model_cache.get("test_key") is None
            assert len(model_cache._cache) == 0

    def test_string_representation(self, sample_config):
        """Test string representation"""
        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(sample_config)

            repr_str = str(detector)
            assert "ObjectDetector" in repr_str
            assert "yolov8s" in repr_str
            assert detector.model_hash in repr_str

    @patch("detector.Path.exists")
    def test_train_with_missing_data_yaml(self, mock_exists, sample_config):
        """Test training with missing data YAML"""
        mock_exists.return_value = False

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(sample_config)

            with pytest.raises(TrainingError):
                detector.train("nonexistent.yaml")

    def test_save_success(self, tmp_path):
        """Test successful model saving"""
        save_path = tmp_path / "test_model.pt"

        # Use unique config to avoid cache
        config = {
            "model": {"architecture": "yolov8s-custom", "input_size": 224},
            "data": {"classes": ["test"]},
        }

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)

            # Should not raise exception
            detector.save(str(save_path))

            # Should have called model.save
            mock_model.save.assert_called_once_with(str(save_path))

    def test_save_failure(self):
        """Test model saving failure"""
        # Use unique config to avoid cache
        config = {
            "model": {"architecture": "yolov8-fail", "input_size": 224},
            "data": {"classes": ["test"]},
        }

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.save.side_effect = Exception("Save failed")
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)

            with pytest.raises(Exception):
                detector.save("/invalid/path/model.pt")

    def test_thread_safety(self, sample_config):
        """Test thread-safe operations"""
        with patch("detector.YOLO") as mock_yolo, patch(
            "detector.torch.randn"
        ) as mock_randn:

            mock_model = Mock()
            mock_model.predict.return_value = []
            mock_yolo.return_value = mock_model
            mock_randn.return_value = torch.randn(1, 3, 224, 224)

            detector = ObjectDetector(sample_config)

            results = []
            errors = []

            def worker(thread_id):
                """Worker function for thread testing"""
                try:
                    for i in range(10):
                        result = detector.predict(
                            f"image_{thread_id}_{i}.jpg", verbose=False
                        )
                        results.append((thread_id, i, result))
                except Exception as e:
                    errors.append((thread_id, e))

            # Run multiple threads
            import threading

            threads = []
            for i in range(3):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # All operations should have succeeded
            assert len(results) == 30  # 3 threads * 10 calls each
            assert len(errors) == 0

            # Performance stats should reflect all calls
            assert detector._performance_stats["inference_count"] == 30


class TestExceptionClasses:
    """Test custom exception classes"""

    def test_model_load_error(self):
        """Test ModelLoadError exception"""
        error = ModelLoadError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_inference_error(self):
        """Test InferenceError exception"""
        error = InferenceError("Inference failed")
        assert str(error) == "Inference failed"
        assert isinstance(error, Exception)

    def test_training_error(self):
        """Test TrainingError exception"""
        error = TrainingError("Training failed")
        assert str(error) == "Training failed"
        assert isinstance(error, Exception)


class TestIntegrationScenarios:
    """Test integration scenarios"""

    def test_full_workflow_simulation(self, tmp_path):
        """Test a simulated full workflow"""
        # Use unique config to avoid cache
        config = {
            "model": {"architecture": "yolov8-workflow", "input_size": 224},
            "data": {"classes": ["test"]},
        }

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            mock_model.predict.return_value = [Mock()]
            mock_model.val.return_value = Mock(box=Mock(map50=0.85, map=0.75))
            # Mock model structure for health check
            mock_model.model = Mock()
            mock_model.ckpt_path = str(tmp_path / "test.pt")
            mock_parameters = [Mock()]
            mock_parameters[0].device = torch.device("cpu")
            mock_model.model.parameters.return_value = iter(mock_parameters)
            mock_yolo.return_value = mock_model

            # Initialize detector
            detector = ObjectDetector(config)

            # Test health check
            health = detector.health_check()
            assert health["status"] == "healthy"

            # Test prediction
            results = detector.predict("test_image.jpg")
            assert results is not None

            # Test validation
            val_results = detector.val("dummy_data.yaml")
            assert val_results is not None

            # Test performance stats
            stats = detector.get_performance_stats()
            assert "inference_count" in stats
            assert "avg_inference_time" in stats

            # Test cache operations
            detector.clear_cache()
            assert len(model_cache._cache) == 0

    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        # Use unique config to avoid cache
        config = {
            "model": {"architecture": "yolov8-recovery", "input_size": 224},
            "data": {"classes": ["test"]},
        }

        with patch("detector.YOLO") as mock_yolo:
            mock_model = Mock()
            # Make predict succeed for validation, then fail twice, then succeed
            mock_model.predict.side_effect = [
                [Mock()],  # Validation succeeds
                Exception("First failure"),
                Exception("Second failure"),
                [Mock()],  # Final success
            ]
            mock_yolo.return_value = mock_model

            detector = ObjectDetector(config)

            # First two calls should fail
            with pytest.raises(InferenceError):
                detector.predict("test.jpg")

            with pytest.raises(InferenceError):
                detector.predict("test.jpg")

            # Third call should succeed
            result = detector.predict("test.jpg")
            assert result is not None

            # Error count should be 2
            assert detector._performance_stats["errors"] == 2
            assert (
                detector._performance_stats["inference_count"] == 1
            )  # Only successful call


if __name__ == "__main__":
    pytest.main([__file__])
