"""
Training Pipeline Implementation
Consolidated training functionality with unified interface
"""

from pathlib import Path
from typing import Dict, Any, Optional

from ..base import BasePipeline, PipelineConfig, PipelineResult
from core.config.manager import load_config
from utils.logger_config import get_logger

logger = get_logger(__name__)


class TrainingPipeline(BasePipeline):
    """
    Unified training pipeline for YOLO models
    Consolidates all training-related functionality
    """

    def __init__(self, config: PipelineConfig, data_yaml: str):
        """
        Initialize training pipeline

        Args:
            config: Pipeline configuration
            data_yaml: Path to data YAML file
        """
        super().__init__(config)
        self.data_yaml = data_yaml
        self.training_config: Optional[Dict[str, Any]] = None

    def validate_config(self) -> bool:
        """
        Validate training configuration

        Returns:
            True if configuration is valid
        """
        try:
            # Check if data YAML exists
            if not Path(self.data_yaml).exists():
                self.logger.error(f"Data YAML file not found: {self.data_yaml}")
                return False

            # Load training configuration
            self.training_config = load_config("default")

            # Validate required fields
            required_fields = ["model", "training"]
            for field in required_fields:
                if field not in self.training_config:
                    self.logger.error(f"Missing required configuration field: {field}")
                    return False

            # Validate model architecture
            model_arch = self.training_config.get("model", {}).get("architecture")
            if not model_arch or not model_arch.startswith("yolov8"):
                self.logger.error(f"Unsupported model architecture: {model_arch}")
                return False

            self.logger.info(" Training configuration validated")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def run(self) -> PipelineResult:
        """
        Execute training pipeline

        Returns:
            PipelineResult with training results
        """
        result = PipelineResult(
            success=False, duration=0.0, metrics={}, artifacts=[], errors=[]
        )

        try:
            self.logger.info(" Starting training pipeline execution")

            # Update progress
            self.update_progress(0.1, "Initializing training environment")

            # Import training components
            from core.trainer import ModelTrainer

            # Create trainer
            trainer = ModelTrainer(self.training_config)

            # Update progress
            self.update_progress(0.2, "Starting model training")

            # Run training
            training_result = trainer.train(
                data_yaml=self.data_yaml, experiment_name=self.config.name
            )

            # Update progress
            self.update_progress(0.9, "Training completed, saving results")

            # Process results
            if training_result.get("success"):
                result.success = True

                # Add artifacts
                if training_result.get("model_path"):
                    result.add_artifact(Path(training_result["model_path"]))

                if training_result.get("save_dir"):
                    result.add_artifact(Path(training_result["save_dir"]))

                # Add metrics
                result.add_metric("training_success", True)
                result.add_metric(
                    "final_model_path", training_result.get("model_path", "")
                )

                self.logger.info(" Training pipeline completed successfully")

            else:
                result.add_error("Training failed")
                result.add_error(training_result.get("error", "Unknown training error"))
                self.logger.error(" Training pipeline failed")

        except Exception as e:
            result.add_error(f"Training pipeline error: {e}")
            self.logger.error(f" Training pipeline failed: {e}")

        finally:
            self.update_progress(1.0, "Pipeline execution completed")

        return result