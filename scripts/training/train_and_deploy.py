#!/usr/bin/env python3
"""
End-to-end training pipeline with ONNX conversion and Raspberry Pi deployment preparation
Trains model, converts to ONNX, compares performance, and creates deployment package
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import os

# Removed broken import - functionality moved to ModelTrainer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.trainer import ModelTrainer
from utils.logger_config import get_logger
from utils.output_utils import OutputManager

logger = get_logger(__name__, debug=os.getenv("DEBUG") == "1")


class TrainingPipeline:
    """Complete training to deployment pipeline"""

    def __init__(self, config_path: str, data_yaml: str):
        """
        Initialize pipeline

        Args:
            config_path: Path to training configuration
            data_yaml: Path to dataset YAML file
        """
        from core.config.manager import load_config

        self.config = load_config(config_path)
        self.data_yaml = data_yaml
        self.trainer = ModelTrainer(self.config)
        self.output_manager = OutputManager()
        self.results = {}

    def train_model(self, experiment_name: Optional[str] = None) -> Dict:
        """
        Train the model

        Args:
            experiment_name: Optional name for the experiment

        Returns:
            Training results
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Training Model")
        logger.info("=" * 60)

        start_time = time.time()

        train_result = self.trainer.train(
            self.data_yaml, experiment_name=experiment_name or "e2e_training"
        )

        training_time = time.time() - start_time

        if not train_result.get("success"):
            raise RuntimeError(f"Training failed: {train_result.get('error')}")

        self.results["training"] = {
            "success": True,
            "model_path": train_result["model_path"],
            "save_dir": train_result["save_dir"],
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
        }

        logger.info(
            f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)"
        )
        logger.info(f"Model saved to: {train_result['model_path']}")

        return self.results["training"]

    def evaluate_pytorch_model(self, model_path: str, input_size: int = 224) -> Dict:
        """
        Evaluate PyTorch model performance

        Args:
            model_path: Path to trained model
            input_size: Model input size

        Returns:
            Evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Evaluating PyTorch Model")
        logger.info("=" * 60)

        metrics = self.trainer.evaluate_model_performance(
            model_path, self.data_yaml, input_size=input_size
        )

        speed_metrics = self.trainer.measure_inference_speed(
            model_path, input_size=input_size, num_runs=100
        )

        model_size = self.trainer.get_model_size_mb(model_path)

        self.results["pytorch_evaluation"] = {
            **metrics,
            **speed_metrics,
            "model_size_mb": model_size,
        }

        logger.info(f"PyTorch Model Evaluation:")
        logger.info(f"   mAP50: {metrics.get('mAP50', 0):.3f}")
        logger.info(f"   mAP: {metrics.get('mAP', 0):.3f}")
        logger.info(f"   FPS: {speed_metrics.get('fps', 0):.2f}")
        logger.info(f"   Model Size: {model_size:.2f} MB")

        return self.results["pytorch_evaluation"]

    def convert_to_onnx(self, model_path: str, input_size: int = 224) -> str:
        """
        Convert model to ONNX format

        Args:
            model_path: Path to PyTorch model
            input_size: Model input size

        Returns:
            Path to ONNX model
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Converting to ONNX")
        logger.info("=" * 60)

        from scripts.convert_to_onnx import convert_model

        output_dir = Path("output/onnx")
        output_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = output_dir / f"{Path(model_path).stem}.onnx"
        convert_model(str(model_path), str(onnx_path), input_size=input_size)

        self.results["onnx_conversion"] = {
            "success": True,
            "onnx_path": str(onnx_path),
            "input_size": input_size,
        }

        logger.info(f"ONNX conversion completed")
        logger.info(f"ONNX model saved to: {onnx_path}")

        return str(onnx_path)

    def evaluate_onnx_model(self, onnx_path: str, input_size: int = 224) -> Dict:
        """
        Evaluate ONNX model performance

        Args:
            onnx_path: Path to ONNX model
            input_size: Model input size

        Returns:
            Evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Evaluating ONNX Model")
        logger.info("=" * 60)

        speed_metrics = self.trainer.measure_onnx_speed(
            onnx_path, input_size, num_runs=100
        )

        onnx_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB

        self.results["onnx_evaluation"] = {**speed_metrics, "model_size_mb": onnx_size}

        logger.info(f"ONNX Model Evaluation:")
        logger.info(f"   FPS: {speed_metrics.get('fps', 0):.2f}")
        logger.info(f"   Avg Latency: {speed_metrics.get('avg_latency_ms', 0):.2f} ms")
        logger.info(f"   Model Size: {onnx_size:.2f} MB")

        return self.results["onnx_evaluation"]

    def compare_models(self) -> Dict:
        """
        Compare PyTorch and ONNX model performance

        Returns:
            Comparison metrics
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Comparing Models")
        logger.info("=" * 60)

        pytorch = self.results.get("pytorch_evaluation", {})
        onnx = self.results.get("onnx_evaluation", {})

        comparison = {
            "pytorch_fps": pytorch.get("fps", 0),
            "onnx_fps": onnx.get("fps", 0),
            "speedup": (
                onnx.get("fps", 0) / pytorch.get("fps", 1)
                if pytorch.get("fps", 0) > 0
                else 0
            ),
            "pytorch_size_mb": pytorch.get("model_size_mb", 0),
            "onnx_size_mb": onnx.get("model_size_mb", 0),
            "size_reduction": (
                (1 - onnx.get("model_size_mb", 0) / pytorch.get("model_size_mb", 1))
                * 100
                if pytorch.get("model_size_mb", 0) > 0
                else 0
            ),
            "pytorch_latency_ms": pytorch.get("avg_latency_ms", 0),
            "onnx_latency_ms": onnx.get("avg_latency_ms", 0),
            "latency_improvement": (
                (
                    (pytorch.get("avg_latency_ms", 0) - onnx.get("avg_latency_ms", 0))
                    / pytorch.get("avg_latency_ms", 1)
                )
                * 100
                if pytorch.get("avg_latency_ms", 0) > 0
                else 0
            ),
        }

        self.results["comparison"] = comparison

        logger.info("Model Comparison:")
        logger.info(f"   PyTorch FPS: {comparison['pytorch_fps']:.2f}")
        logger.info(f"   ONNX FPS: {comparison['onnx_fps']:.2f}")
        logger.info(f"   Speedup: {comparison['speedup']:.2f}x")
        logger.info(f"   PyTorch Size: {comparison['pytorch_size_mb']:.2f} MB")
        logger.info(f"   ONNX Size: {comparison['onnx_size_mb']:.2f} MB")
        logger.info(f"   Size Reduction: {comparison['size_reduction']:.1f}%")
        logger.info(f"   Latency Improvement: {comparison['latency_improvement']:.1f}%")

        return comparison

    def prepare_deployment_package(
        self, onnx_path: str, output_dir: str = "output/deployment"
    ) -> str:
        """
        Prepare deployment package for Raspberry Pi

        Args:
            onnx_path: Path to ONNX model
            output_dir: Output directory for deployment package

        Returns:
            Path to deployment package directory
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Preparing Deployment Package")
        logger.info("=" * 60)

        # Deployment package building integrated into main pipeline
        from pathlib import Path
        import shutil

        class DeploymentPackageBuilder:
            def __init__(self, output_dir):
                self.output_dir = Path(output_dir)

            def create_package(self, onnx_path, config, results):
                # Simple implementation - just copy files
                package_dir = self.output_dir / "deployment_package"
                package_dir.mkdir(parents=True, exist_ok=True)

                # Copy ONNX model
                if onnx_path:
                    shutil.copy2(onnx_path, package_dir / "model.onnx")

                # Save config
                import json

                with open(package_dir / "config.json", "w") as f:
                    json.dump(config, f, indent=2, default=str)

                return str(package_dir)

        builder = DeploymentPackageBuilder(output_dir)
        package_path = builder.create_package(
            onnx_path=onnx_path, config=self.config, results=self.results
        )

        self.results["deployment"] = {
            "package_path": package_path,
            "onnx_model": onnx_path,
        }

        logger.info(f"Deployment package created: {package_path}")

        return package_path

    def run_full_pipeline(self, experiment_name: Optional[str] = None) -> Dict:
        """
        Run complete pipeline: train -> evaluate -> convert -> compare -> deploy

        Args:
            experiment_name: Optional name for the experiment

        Returns:
            Complete results dictionary
        """
        logger.info("=" * 60)
        logger.info("END-TO-END TRAINING & DEPLOYMENT PIPELINE")
        logger.info("=" * 60)

        pipeline_start = time.time()

        try:
            # Step 1: Train
            training_result = self.train_model(experiment_name)
            model_path = training_result["model_path"]
            input_size = self.config.get("model", {}).get("input_size", 224)

            # Step 2: Evaluate PyTorch
            self.evaluate_pytorch_model(model_path, input_size)

            # Step 3: Convert to ONNX
            onnx_path = self.convert_to_onnx(model_path, input_size)

            # Step 4: Evaluate ONNX
            self.evaluate_onnx_model(onnx_path, input_size)

            # Step 5: Compare
            self.compare_models()

            # Step 6: Prepare deployment
            self.prepare_deployment_package(onnx_path)

            # Save complete results
            self.results["pipeline"] = {
                "success": True,
                "total_time_seconds": time.time() - pipeline_start,
                "total_time_minutes": (time.time() - pipeline_start) / 60,
                "timestamp": time.time(),
            }

            # Save results
            self._save_results()

            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(
                f"Total time: {self.results['pipeline']['total_time_minutes']:.2f} minutes"
            )
            logger.info(
                f"Results saved to: output/training/e2e_results_{int(time.time())}.json"
            )

            return self.results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.results["pipeline"] = {
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
            }
            self._save_results()
            raise

    def _save_results(self):
        """Save all results to JSON and text report"""
        timestamp = int(time.time())

        # Save JSON
        results_path = Path("output/training") / f"e2e_results_{timestamp}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save text report
        report = self._generate_report()
        if report:
            self.output_manager.save_text_report(
                report, f"e2e_training_report_{timestamp}", "training"
            )

    def _generate_report(self) -> str:
        """Generate a comprehensive training report"""
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("END-TO-END TRAINING PIPELINE REPORT")
            report_lines.append("=" * 80)

            # Training results
            if "training" in self.results:
                train_result = self.results["training"]
                report_lines.append("\nTRAINING RESULTS:")
                report_lines.append("-" * 40)
                report_lines.append(f"Success: {train_result.get('success', False)}")
                if "error" in train_result:
                    report_lines.append(f"Error: {train_result['error']}")

            # PyTorch evaluation
            if "pytorch_evaluation" in self.results:
                eval_result = self.results["pytorch_evaluation"]
                report_lines.append("\nPYTORCH MODEL EVALUATION:")
                report_lines.append("-" * 40)
                report_lines.append(f"mAP50: {eval_result.get('mAP50', 'N/A')}")
                report_lines.append(f"mAP50-95: {eval_result.get('mAP50_95', 'N/A')}")
                report_lines.append(f"FPS: {eval_result.get('fps', 'N/A'):.1f}")
                report_lines.append(
                    f"Model Size: {eval_result.get('model_size_mb', 'N/A'):.2f} MB"
                )

            # ONNX evaluation
            if "onnx_evaluation" in self.results:
                onnx_result = self.results["onnx_evaluation"]
                report_lines.append("\nONNX MODEL EVALUATION:")
                report_lines.append("-" * 40)
                report_lines.append(f"FPS: {onnx_result.get('fps', 'N/A'):.1f}")
                report_lines.append(
                    f"Avg Latency: {onnx_result.get('avg_latency_ms', 'N/A'):.2f} ms"
                )

            # Deployment
            if "deployment" in self.results:
                deploy_result = self.results["deployment"]
                report_lines.append("\nDEPLOYMENT:")
                report_lines.append("-" * 40)
                report_lines.append(
                    f"Package Path: {deploy_result.get('package_path', 'N/A')}"
                )

            report_lines.append("\n" + "=" * 80)
            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return f"Error generating report: {e}"


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end training and deployment pipeline"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing model)",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to existing model (if skipping training)"
    )

    args = parser.parse_args()

    pipeline = TrainingPipeline(args.config, args.data)

    if args.skip_training and args.model_path:
        # Load existing model and continue from evaluation
        logger.info("Skipping training, using existing model...")
        model_path = args.model_path
        input_size = pipeline.config.get("model", {}).get("input_size", 224)

        pipeline.evaluate_pytorch_model(model_path, input_size)
        onnx_path = pipeline.convert_to_onnx(model_path, input_size)
        pipeline.evaluate_onnx_model(onnx_path, input_size)
        pipeline.compare_models()
        pipeline.prepare_deployment_package(onnx_path)
        pipeline._save_results()
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(args.name)


if __name__ == "__main__":
    main()
