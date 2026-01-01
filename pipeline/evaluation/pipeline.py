"""
Evaluation Pipeline Implementation
Consolidated evaluation and benchmarking functionality
"""

from pathlib import Path
from typing import Dict, List, Any

from ..base import BasePipeline, PipelineConfig, PipelineResult
from core.config.manager import load_config
from utils.logger_config import get_logger

logger = get_logger(__name__)


class EvaluationPipeline(BasePipeline):
    """
    Unified evaluation pipeline for YOLO models
    Consolidates evaluation, benchmarking, and metrics calculation
    """

    def __init__(self, config: PipelineConfig, data_yaml: str):
        """
        Initialize evaluation pipeline

        Args:
            config: Pipeline configuration
            data_yaml: Path to data YAML file
        """
        super().__init__(config)
        self.data_yaml = data_yaml
        self.models_to_evaluate: List[str] = []
        self.evaluation_types: List[str] = ["accuracy"]
        self.batch_size: int = 8
        self.num_runs: int = 100

    def set_models(self, model_paths: List[str]) -> None:
        """Set models to evaluate"""
        self.models_to_evaluate = model_paths

    def set_evaluation_types(self, eval_types: List[str]) -> None:
        """Set evaluation types to run"""
        self.evaluation_types = eval_types

    def set_batch_size(self, batch_size: int) -> None:
        """Set batch size for evaluation"""
        self.batch_size = batch_size

    def set_num_runs(self, num_runs: int) -> None:
        """Set number of runs for benchmarking"""
        self.num_runs = num_runs

    def validate_config(self) -> bool:
        """
        Validate evaluation configuration

        Returns:
            True if configuration is valid
        """
        try:
            # Check if data YAML exists
            if not Path(self.data_yaml).exists():
                self.logger.error(f"Data YAML file not found: {self.data_yaml}")
                return False

            # Check if models exist
            for model_path in self.models_to_evaluate:
                if not Path(model_path).exists():
                    self.logger.error(f"Model file not found: {model_path}")
                    return False

            # Validate evaluation types
            valid_types = ["accuracy", "speed", "memory"]
            for eval_type in self.evaluation_types:
                if eval_type not in valid_types:
                    self.logger.error(f"Invalid evaluation type: {eval_type}")
                    return False

            self.logger.info(" Evaluation configuration validated")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def run(self) -> PipelineResult:
        """
        Execute evaluation pipeline

        Returns:
            PipelineResult with evaluation results
        """
        result = PipelineResult(
            success=False, duration=0.0, metrics={}, artifacts=[], errors=[]
        )

        try:
            self.logger.info(" Starting evaluation pipeline execution")

            # Update progress
            self.update_progress(
                0.1, f"Evaluating {len(self.models_to_evaluate)} models"
            )

            total_steps = len(self.models_to_evaluate) * len(self.evaluation_types)
            current_step = 0

            all_results = {}

            for model_path in self.models_to_evaluate:
                model_results = {}

                for eval_type in self.evaluation_types:
                    current_step += 1
                    progress = 0.1 + (current_step / total_steps) * 0.8
                    self.update_progress(
                        progress,
                        f"Running {eval_type} evaluation on {Path(model_path).name}",
                    )

                    try:
                        if eval_type == "accuracy":
                            metrics = self._evaluate_accuracy(model_path)
                        elif eval_type == "speed":
                            metrics = self._evaluate_speed(model_path)
                        elif eval_type == "memory":
                            metrics = self._evaluate_memory(model_path)
                        else:
                            metrics = {"error": f"Unknown evaluation type: {eval_type}"}

                        model_results[eval_type] = metrics

                    except Exception as e:
                        model_results[eval_type] = {"error": str(e)}
                        result.add_error(
                            f"Failed to evaluate {eval_type} for {model_path}: {e}"
                        )

                all_results[model_path] = model_results

            # Process results
            result.success = True
            result.metrics = all_results

            # Save evaluation results
            self._save_evaluation_results(all_results, result)

            self.logger.info(" Evaluation pipeline completed successfully")

        except Exception as e:
            result.add_error(f"Evaluation pipeline error: {e}")
            self.logger.error(f" Evaluation pipeline failed: {e}")

        finally:
            self.update_progress(1.0, "Pipeline execution completed")

        return result

    def _evaluate_accuracy(self, model_path: str) -> Dict[str, Any]:
        """Evaluate model accuracy"""
        try:
            from core.trainer import ModelTrainer

            # Load evaluation config
            eval_config = load_config("default")

            # Create trainer for evaluation
            trainer = ModelTrainer(eval_config)

            # Run evaluation
            metrics = trainer.evaluate_model_performance(
                model_path=model_path,
                data_yaml=self.data_yaml,
                input_size=eval_config.get("model", {}).get("input_size", 416),
            )

            return metrics or {"mAP50": 0.0, "mAP50_95": 0.0}

        except Exception as e:
            return {"error": f"Accuracy evaluation failed: {e}"}

    def _evaluate_speed(self, model_path: str) -> Dict[str, Any]:
        """Evaluate model speed"""
        try:
            from core.trainer import ModelTrainer

            eval_config = load_config("default")
            trainer = ModelTrainer(eval_config)

            # Run speed evaluation
            metrics = trainer.measure_inference_speed(
                model_path=model_path,
                input_size=eval_config.get("model", {}).get("input_size", 416),
                num_runs=self.num_runs,
            )

            return metrics or {"fps": 0.0, "avg_latency_ms": 0.0}

        except Exception as e:
            return {"error": f"Speed evaluation failed: {e}"}

    def _evaluate_memory(self, model_path: str) -> Dict[str, Any]:
        """Evaluate model memory usage"""
        try:
            # Import memory evaluation function (to be implemented)
            # For now, return placeholder
            return {
                "model_size_mb": Path(model_path).stat().st_size / (1024 * 1024),
                "gpu_memory_mb": 0.0,  # Placeholder
                "cpu_memory_mb": 0.0,  # Placeholder
            }

        except Exception as e:
            return {"error": f"Memory evaluation failed: {e}"}

    def _save_evaluation_results(
        self, all_results: Dict[str, Any], result: PipelineResult
    ) -> None:
        """Save evaluation results to files"""
        try:
            import json

            # Save detailed results
            results_file = self.config.output_dir / "evaluation_results.json"
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            result.add_artifact(results_file)

            # Create summary report
            summary = self._create_evaluation_summary(all_results)
            summary_file = self.config.output_dir / "evaluation_summary.txt"
            with open(summary_file, "w") as f:
                f.write(summary)

            result.add_artifact(summary_file)

        except Exception as e:
            self.logger.warning(f"Failed to save evaluation results: {e}")

    def _create_evaluation_summary(self, results: Dict[str, Any]) -> str:
        """Create human-readable evaluation summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL EVALUATION SUMMARY")
        lines.append("=" * 80)

        for model_path, model_results in results.items():
            lines.append(f"\nModel: {Path(model_path).name}")

            for eval_type, metrics in model_results.items():
                lines.append(f" {eval_type.upper()}:")
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if not key.startswith("_"):  # Skip private keys
                            lines.append(f"   {key}: {value}")
                else:
                    lines.append(f"   Result: {metrics}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)