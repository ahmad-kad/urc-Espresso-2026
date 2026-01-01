"""
Generic training framework for object detection models
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

from detector import ObjectDetector
from utils.logger_config import get_logger

logger = get_logger(__name__, debug=os.getenv("DEBUG") == "1")


@dataclass
class TrainingResult:
    """Structured training result"""

    success: bool
    save_dir: str
    model_path: str
    results: Any
    error: Optional[str] = None


class ModelTrainer:
    """
    Generic trainer for object detection models
    """

    def __init__(self, config: Dict):
        self.config = config
        self.detector = ObjectDetector(config)

    def train(
        self,
        data_yaml: str,
        experiment_name: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        resume: bool = False,
        resume_path: Optional[str] = None,
    ) -> Dict:
        """
        Train the model with comprehensive logging and error handling

        Args:
            data_yaml: Path to data configuration YAML
            experiment_name: Optional experiment name for output organization

        Returns:
            Training results dictionary
        """

        logger.info("Starting model training...")
        logger.info(f"Data configuration: {data_yaml}")

        try:
            # Set project name for organized outputs
            if experiment_name:
                project_name = f"{self.config.get('project', {}).get('name', 'object_detection')}_{experiment_name}"
            else:
                project_name = self.config.get("project", {}).get(
                    "name", "object_detection"
                )

            # Train the model - use output directory as base
            output_base = Path("output")
            train_kwargs = {
                "project": str(output_base / "models"),
                "name": project_name,
            }
            if project:
                train_kwargs["project"] = str(output_base / project)
            if name:
                train_kwargs["name"] = name

            # Add resume parameters if requested
            if resume:
                if resume_path:
                    train_kwargs["resume"] = resume_path
                else:
                    train_kwargs["resume"] = True

            results = self.detector.train(
                data_yaml, model_name=name or project_name, **train_kwargs
            )

            logger.info("Training completed successfully!")

            # Extract save directory from results
            save_dir = self._extract_save_dir(results, project, name, project_name)

            logger.info(f"Model saved to: {save_dir}")

            return TrainingResult(
                success=True,
                save_dir=str(save_dir),
                model_path=str(Path(save_dir) / "weights" / "best.pt"),
                results=results,
            ).__dict__

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return TrainingResult(
                success=False, save_dir="", model_path="", results=None, error=str(e)
            ).__dict__

    def _extract_save_dir(
        self, results, project: Optional[str], name: Optional[str], project_name: str
    ) -> Path:
        """Extract save directory from training results with fallback logic"""
        # Try to get save_dir from results object
        if hasattr(results, "save_dir"):
            return Path(results.save_dir)

        if isinstance(results, dict):
            if "save_dir" in results:
                return Path(results["save_dir"])
            if results.get("success") is False:
                raise Exception(results.get("error", "Unknown training failure"))

        # Fallback: construct expected directory
        output_base = Path("output")
        if project:
            expected_dir = output_base / project / (name or project_name)
        else:
            expected_dir = output_base / "models" / (name or project_name)

        if expected_dir.exists():
            return expected_dir

        logger.warning(
            "Could not determine save_dir from results, using expected directory"
        )
        return expected_dir





    # quantize_and_evaluate method removed - unused

