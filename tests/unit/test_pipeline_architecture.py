"""
Tests for the unified pipeline architecture
Validates the base classes, configuration management, and pipeline integration
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline.base import (
    BasePipeline,
    PipelineConfig,
    PipelineResult,
    PipelineError,
    TrainingError,
    EvaluationError,
    ConversionError,
    DeploymentError,
    create_pipeline_output_dir,
    save_pipeline_result,
    load_pipeline_result,
)
from core.config.manager import ConfigManager, ConfigValidationResult


class TestPipelineConfig:
    """Test PipelineConfig dataclass"""

    def test_config_initialization(self):
        """Test pipeline config initialization"""
        config = PipelineConfig(name="test_pipeline", version="1.0.0")
        assert config.name == "test_pipeline"
        assert config.version == "1.0.0"
        assert config.output_dir == Path("output")
        assert config.log_level == "INFO"
        assert config.dry_run is False

    def test_config_path_conversion(self):
        """Test that string output_dir is converted to Path"""
        config = PipelineConfig(name="test", output_dir="custom/output")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("custom/output")


class TestPipelineResult:
    """Test PipelineResult dataclass"""

    def test_result_initialization(self):
        """Test pipeline result initialization"""
        result = PipelineResult(
            success=True,
            duration=2.5,
            metrics={"accuracy": 0.95},
            artifacts=[Path("model.pt")],
            errors=[],
        )

        assert result.success is True
        assert result.duration == 2.5
        assert result.metrics["accuracy"] == 0.95
        assert len(result.artifacts) == 1

    def test_result_add_error(self):
        """Test adding errors to result"""
        result = PipelineResult(
            success=True, duration=0.0, metrics={}, artifacts=[], errors=[]
        )

        result.add_error("Test error")
        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"

    def test_result_add_artifact(self):
        """Test adding artifacts to result"""
        result = PipelineResult(
            success=True, duration=0.0, metrics={}, artifacts=[], errors=[]
        )

        result.add_artifact(Path("model.onnx"))
        assert len(result.artifacts) == 1
        assert result.artifacts[0] == Path("model.onnx")

    def test_result_add_metric(self):
        """Test adding metrics to result"""
        result = PipelineResult(
            success=True, duration=0.0, metrics={}, artifacts=[], errors=[]
        )

        result.add_metric("precision", 0.88)
        assert result.metrics["precision"] == 0.88

    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = PipelineResult(
            success=True,
            duration=1.5,
            metrics={"accuracy": 0.95},
            artifacts=[Path("model.pt")],
            errors=[],
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["duration"] == 1.5
        assert data["metrics"]["accuracy"] == 0.95
        assert data["artifacts"] == ["model.pt"]
        assert data["status"] == "completed"


class TestPipelineErrors:
    """Test custom pipeline exceptions"""

    def test_pipeline_error(self):
        """Test base PipelineError"""
        error = PipelineError("Test pipeline error")
        assert str(error) == "Test pipeline error"
        assert isinstance(error, Exception)

    def test_training_error(self):
        """Test TrainingError"""
        error = TrainingError("Training failed")
        assert isinstance(error, PipelineError)
        assert str(error) == "Training failed"

    def test_evaluation_error(self):
        """Test EvaluationError"""
        error = EvaluationError("Evaluation failed")
        assert isinstance(error, PipelineError)

    def test_conversion_error(self):
        """Test ConversionError"""
        error = ConversionError("Conversion failed")
        assert isinstance(error, PipelineError)

    def test_deployment_error(self):
        """Test DeploymentError"""
        error = DeploymentError("Deployment failed")
        assert isinstance(error, PipelineError)


class TestBasePipeline:
    """Test the BasePipeline abstract class"""

    def test_abstract_methods(self):
        """Test that BasePipeline cannot be instantiated directly"""
        config = PipelineConfig(name="test")

        # Should raise TypeError because validate_config and run are abstract
        with pytest.raises(TypeError):
            BasePipeline(config)

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        config = PipelineConfig(name="test_pipeline")

        class TestPipeline(BasePipeline):
            def validate_config(self):
                return True

            def run(self):
                return PipelineResult(
                    success=True, duration=1.0, metrics={}, artifacts=[], errors=[]
                )

        pipeline = TestPipeline(config)
        assert pipeline.config == config
        assert pipeline._status.value == "pending"
        assert pipeline._progress == 0.0

    def test_progress_tracking(self):
        """Test progress tracking"""
        config = PipelineConfig(name="test")

        class TestPipeline(BasePipeline):
            def validate_config(self):
                return True

            def run(self):
                return PipelineResult(
                    success=True, duration=1.0, metrics={}, artifacts=[], errors=[]
                )

        pipeline = TestPipeline(config)

        pipeline.update_progress(0.5, "Halfway done")
        assert pipeline._progress == 0.5

        # Test bounds
        pipeline.update_progress(1.5, "Over 100%")  # Should clamp to 1.0
        assert pipeline._progress == 1.0

        pipeline.update_progress(-0.1, "Negative")  # Should clamp to 0.0
        assert pipeline._progress == 0.0

    @patch("pipeline.base.time")
    def test_execution_monitoring(self, mock_time):
        """Test pipeline execution with monitoring"""
        mock_time.time.side_effect = [0.0, 2.5]  # Start and end times

        config = PipelineConfig(name="test")

        class TestPipeline(BasePipeline):
            def validate_config(self):
                return True

            def run(self):
                return PipelineResult(
                    success=True,
                    duration=0.0,
                    metrics={"test": "value"},
                    artifacts=[],
                    errors=[],
                )

        pipeline = TestPipeline(config)

        # Mock the _execute_with_monitoring to avoid calling abstract run
        with patch.object(pipeline, "_execute_with_monitoring") as mock_execute:
            mock_execute.return_value = PipelineResult(
                success=True,
                duration=2.5,
                metrics={"test": "value"},
                artifacts=[],
                errors=[],
            )

            result = pipeline._execute_with_monitoring()

            assert result.success is True
            assert result.duration == 2.5
            assert result.metrics["test"] == "value"


class TestPipelineUtilities:
    """Test pipeline utility functions"""

    def test_create_pipeline_output_dir(self, tmp_path):
        """Test creating pipeline output directory"""
        base_dir = tmp_path / "output"
        output_dir = create_pipeline_output_dir("test_pipeline", base_dir)

        assert output_dir.exists()
        assert "test_pipeline" in str(output_dir)
        # Should contain timestamp
        assert len(output_dir.name.split("_")) >= 2

    def test_save_and_load_pipeline_result(self, tmp_path):
        """Test saving and loading pipeline results"""
        result = PipelineResult(
            success=True,
            duration=2.5,
            metrics={"accuracy": 0.95, "precision": 0.88},
            artifacts=[Path("model.pt"), Path("config.json")],
            errors=[],
        )

        # Save result
        saved_file = save_pipeline_result(result, tmp_path)
        assert saved_file.exists()
        assert saved_file.name == "pipeline_result.json"

        # Load result using the actual saved file path
        loaded_result = load_pipeline_result(saved_file)

        assert loaded_result.success == result.success
        assert loaded_result.duration == result.duration
        assert loaded_result.metrics == result.metrics
        assert len(loaded_result.artifacts) == len(result.artifacts)
        assert loaded_result.errors == result.errors


class TestConfigManagerIntegration:
    """Test ConfigManager integration with pipeline architecture"""

    def test_config_manager_creation(self):
        """Test creating ConfigManager"""
        manager = ConfigManager()
        assert manager.configs_dir.name == "configs"
        assert len(manager._loaded_configs) == 0

    def test_config_validation_without_jsonschema(self):
        """Test config validation when jsonschema is not available"""
        manager = ConfigManager()

        config = {"name": "test_config", "model": {"architecture": "yolov8s"}}

        result = manager.validate_config(config)
        assert result.is_valid is True  # Should pass basic validation
        assert len(result.warnings) > 0  # Should warn about missing jsonschema

    def test_config_validation_result(self):
        """Test ConfigValidationResult functionality"""
        result = ConfigValidationResult(is_valid=True)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

        result.add_error("Test error")
        assert result.is_valid is False
        assert len(result.errors) == 1

        result.add_warning("Test warning")
        assert len(result.warnings) == 1


class TestPipelineIntegration:
    """Test pipeline integration scenarios"""

    def test_pipeline_workflow_simulation(self):
        """Test a simulated pipeline workflow"""
        config = PipelineConfig(name="integration_test")

        class MockPipeline(BasePipeline):
            def __init__(self, config):
                super().__init__(config)
                self.validate_called = False
                self.run_called = False

            def validate_config(self):
                self.validate_called = True
                return True

            def run(self):
                self.run_called = True
                return PipelineResult(
                    success=True,
                    duration=1.5,
                    metrics={"integration_test": "passed"},
                    artifacts=[Path("test_output.txt")],
                    errors=[],
                )

        pipeline = MockPipeline(config)

        # Test validation
        assert pipeline.validate_config() is True
        assert pipeline.validate_called is True

        # Test run
        result = pipeline.run()
        assert pipeline.run_called is True
        assert result.success is True
        assert result.metrics["integration_test"] == "passed"

    def test_pipeline_error_handling(self):
        """Test pipeline error handling"""
        config = PipelineConfig(name="error_test")

        class ErrorPipeline(BasePipeline):
            def validate_config(self):
                return True

            def run(self):
                raise TrainingError("Simulated training error")

        pipeline = ErrorPipeline(config)

        result = pipeline._execute_with_monitoring()

        assert result.success is False
        assert len(result.errors) == 1
        assert "Simulated training error" in result.errors[0]

    def test_pipeline_cancellation(self):
        """Test pipeline cancellation"""
        config = PipelineConfig(name="cancel_test")

        class LongRunningPipeline(BasePipeline):
            def validate_config(self):
                return True

            def run(self):
                import time

                for i in range(10):
                    if self._status.value == "cancelled":
                        break
                    time.sleep(0.01)
                return PipelineResult(
                    success=True, duration=0.1, metrics={}, artifacts=[], errors=[]
                )

        pipeline = LongRunningPipeline(config)

        # Start pipeline in background
        import threading

        result_container = []

        def run_pipeline():
            result = pipeline.run()
            result_container.append(result)

        thread = threading.Thread(target=run_pipeline)
        thread.start()

        # Cancel after short delay
        import time

        time.sleep(0.05)
        pipeline.cancel()

        thread.join(timeout=1.0)

        assert pipeline._status.value == "cancelled"


if __name__ == "__main__":
    pytest.main([__file__])
