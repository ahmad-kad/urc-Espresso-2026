"""
Generic training framework for object detection models
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger_config import get_logger

from .classification_trainer import ClassificationTrainer
from .models import ObjectDetector

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
        self.classification_trainer = None
        self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """Detect if we're training YOLO or classification model"""
        architecture = self.config.get("model", {}).get("architecture", "yolov8s")
        if architecture.startswith("yolov8"):
            return "yolo"
        elif architecture in [
            "mobilenetv2",
            "mobilenetv3",
            "resnet18",
            "resnet34",
            "efficientnet_lite0",
            "efficientnet_lite1",
        ]:
            return "classification"
        else:
            logger.warning(f"Unknown architecture '{architecture}', assuming YOLO")
            return "yolo"

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
            # Handle different training types based on model architecture
            if self.model_type == "classification":
                # Classification training with torchvision models
                logger.info(
                    "Starting classification training with torchvision model..."
                )

                # Setup classification trainer
                self.classification_trainer = ClassificationTrainer(self.config)
                model = self.detector.model  # Get the loaded torchvision model
                self.classification_trainer.setup_model(model)
                self.classification_trainer.setup_data_loaders()

                # Set project name for organized outputs
                if experiment_name:
                    project_name = f"{self.config.get('project', {}).get('name', 'classification')}_{experiment_name}"
                else:
                    project_name = self.config.get("project", {}).get(
                        "name", "classification"
                    )

                # Train the model
                output_base = Path("output")
                save_path = (
                    output_base / "models" / project_name / "weights" / "best.pth"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)

                num_epochs = self.config.get("training", {}).get("epochs", 25)
                results = self.classification_trainer.train(
                    num_epochs=num_epochs, save_path=str(save_path)
                )

                # Create mock results object for compatibility
                class MockResults:
                    def __init__(self, save_dir, results_dict):
                        self.save_dir = save_dir
                        self.results = results_dict

                mock_results = MockResults(str(save_path.parent.parent), results)

            else:
                # YOLO training (existing logic)
                logger.info("Starting YOLO training...")

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
                mock_results = results

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

    def evaluate_model_performance(
        self, model_path: str, data_yaml: str, input_size: int = 416
    ) -> Dict:
        """
        Evaluate model performance on validation set

        Args:
            model_path: Path to model file
            data_yaml: Path to data YAML
            input_size: Input image size

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model performance: {model_path}")

        try:
            # Use YOLO CLI for evaluation instead of Python API to avoid serialization issues
            import re
            import subprocess

            cmd = [
                "yolo",
                "val",
                f"model={model_path}",
                f"data={data_yaml}",
                "--verbose=False",
            ]

            # For ONNX models, specify the correct image size
            if model_path.endswith(".onnx"):
                cmd.append("imgsz=416")

            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                cwd=".",
            )

            if result.returncode != 0:
                logger.error(f"YOLO CLI failed: {result.stderr}")
                return {"error": f"YOLO CLI failed: {result.stderr}"}

            # Parse the output to extract metrics
            metrics = {}
            output = result.stdout
            if output is None:
                logger.error("No output from YOLO CLI")
                return {"error": "No output from YOLO CLI"}

            # Look for the 'all' line which contains overall metrics
            lines = output.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("all") and len(line.split()) >= 7:
                    # Parse the 'all' line which contains overall metrics
                    # Format: "all        320        327      0.93      0.838      0.881      0.486"
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            # parts[-2] is mAP50, parts[-1] is mAP50-95
                            metrics["mAP50"] = float(parts[-2])
                            metrics["mAP50_95"] = float(parts[-1])
                            break
                        except (ValueError, IndexError) as e:
                            logger.warning(
                                f"Failed to parse metrics from line: {line}, error: {e}"
                            )
                            continue

            logger.info(
                f"Evaluation completed: mAP50={metrics.get('mAP50', 'N/A')}, mAP50_95={metrics.get('mAP50_95', 'N/A')}"
            )
            return metrics

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

    def measure_inference_speed(
        self, model_path: str, input_size: int = 416, num_runs: int = 100
    ) -> Dict:
        """
        Measure inference speed

        Args:
            model_path: Path to model file
            input_size: Input image size
            num_runs: Number of inference runs

        Returns:
            Dictionary with speed metrics
        """
        import time

        import numpy as np

        logger.info(f"Measuring inference speed for {num_runs} runs")

        try:
            # Create dummy input
            dummy_input = np.random.randint(
                0, 255, (input_size, input_size, 3), dtype=np.uint8
            )

            # Warmup
            for _ in range(10):
                _ = self.detector.predict(dummy_input, verbose=False)

            # Measure
            latencies = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = self.detector.predict(dummy_input, verbose=False)
                latencies.append((time.perf_counter() - start) * 1000)  # ms

            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0

            metrics = {
                "avg_latency_ms": avg_latency,
                "std_latency_ms": std_latency,
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency,
                "fps": fps,
            }

            logger.info(
                f"Speed measurement completed: {fps:.1f} FPS, {avg_latency:.2f}ms avg"
            )
            return metrics

        except Exception as e:
            logger.error(f"Speed measurement failed: {e}")
            return {"error": str(e)}

    def get_model_size_mb(self, model_path: str) -> float:
        """
        Get model file size in MB

        Args:
            model_path: Path to model file

        Returns:
            Size in MB
        """
        try:
            from pathlib import Path

            size_bytes = Path(model_path).stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            logger.debug(f"Model size: {size_mb:.2f} MB")
            return size_mb
        except Exception as e:
            logger.error(f"Failed to get model size: {e}")
            return 0.0

    def measure_onnx_speed(
        self, onnx_path: str, input_size: int = 416, num_runs: int = 100
    ) -> Dict:
        """
        Measure ONNX model inference speed

        Args:
            onnx_path: Path to ONNX model
            input_size: Input image size
            num_runs: Number of inference runs

        Returns:
            Dictionary with speed metrics
        """
        import time

        import numpy as np

        logger.info(f"Measuring ONNX speed for {num_runs} runs")

        try:
            import onnxruntime as ort

            # Create session
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name

            # Create dummy input
            dummy_input = np.random.randn(1, 3, input_size, input_size).astype(
                np.float32
            )

            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input})

            # Measure
            latencies = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = session.run(None, {input_name: dummy_input})
                latencies.append((time.perf_counter() - start) * 1000)  # ms

            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0

            metrics = {
                "avg_latency_ms": avg_latency,
                "std_latency_ms": std_latency,
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency,
                "fps": fps,
            }

            logger.info(
                f"ONNX speed measurement completed: {fps:.1f} FPS, {avg_latency:.2f}ms avg"
            )
            return metrics

        except ImportError:
            logger.warning("ONNX runtime not available for speed measurement")
            return {"error": "onnxruntime not available"}
        except Exception as e:
            logger.error(f"ONNX speed measurement failed: {e}")
            return {"error": str(e)}
