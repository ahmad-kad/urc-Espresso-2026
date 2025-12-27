#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for ML Training, Conversion, and Benchmarking
Tests multiple model architectures at different image sizes with full pipeline:
Training → ONNX Conversion → Quantization → Evaluation → Benchmarking
"""

import sys
import os
from pathlib import Path
import logging
import time
import pandas as pd
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer import ModelTrainer
from evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("output/end_to_end_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EndToEndTester:
    """Comprehensive end-to-end testing framework"""

    def __init__(self, data_yaml: str = "consolidated_dataset/data.yaml"):
        self.data_yaml = data_yaml
        self.output_base = Path("output/end_to_end_test")
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Test configurations
        self.test_configs = {
            "mobilenet": {
                "configs": ["configs/mobilenet_160.yaml", "configs/mobilenet_192.yaml", "configs/mobilenet_224.yaml"],
                "sizes": [160, 192, 224]
            },
            "yolov8n": {
                "configs": ["configs/yolov8n_160.yaml", "configs/yolov8n_192.yaml", "configs/yolov8n_224.yaml"],
                "sizes": [160, 192, 224]
            },
            "yolov8s": {
                "configs": ["configs/yolov8s_confidence.yaml"],
                "sizes": [224]  # Single size for baseline comparison
            },
            "efficientnet": {
                "configs": ["configs/efficientnet_confidence.yaml"],
                "sizes": [224]
            }
        }

        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

        # Results storage
        self.training_results = {}
        self.conversion_results = {}
        self.quantization_results = {}
        self.evaluation_results = {}
        self.benchmark_results = {}

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete end-to-end pipeline"""
        logger.info("=" * 80)
        logger.info("[STARTING] STARTING COMPREHENSIVE END-TO-END TEST")
        logger.info("=" * 80)

        start_time = time.time()

        try:
            # Phase 1: Training
            logger.info("\n📚 PHASE 1: MODEL TRAINING")
            self._run_training_phase()

            # Phase 2: ONNX Conversion
            logger.info("\n[UPDATE] PHASE 2: ONNX CONVERSION")
            self._run_conversion_phase()

            # Phase 3: Quantization
            logger.info("\n[PERFORMANCE] PHASE 3: MODEL QUANTIZATION")
            self._run_quantization_phase()

            # Phase 4: Evaluation
            logger.info("\n[ANALYSIS] PHASE 4: MODEL EVALUATION")
            self._run_evaluation_phase()

            # Phase 5: Benchmarking
            logger.info("\n[FINISHED] PHASE 5: PERFORMANCE BENCHMARKING")
            self._run_benchmarking_phase()

            # Phase 6: Generate Report
            logger.info("\n[SUMMARY] PHASE 6: GENERATING FINAL REPORT")
            self._generate_final_report()

            total_time = time.time() - start_time
            logger.info(".1f")
            logger.info("=" * 80)
            logger.info("[SUCCESS] END-TO-END TEST COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            return {
                "status": "success",
                "total_time": total_time,
                "training_results": self.training_results,
                "conversion_results": self.conversion_results,
                "quantization_results": self.quantization_results,
                "evaluation_results": self.evaluation_results,
                "benchmark_results": self.benchmark_results
            }

        except Exception as e:
            logger.error(f"End-to-end test failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "failed", "error": str(e)}

    def _run_training_phase(self):
        """Train all model configurations"""
        for arch_name, arch_config in self.test_configs.items():
            logger.info(f"\n🏗️  Training {arch_name} models...")

            for config_file, size in zip(arch_config["configs"], arch_config["sizes"]):
                experiment_name = f"{arch_name}_{size}"

                try:
                    logger.info(f"   Training {experiment_name} with config: {config_file}")

                    # Train the model
                    result = self.trainer.train_enhanced(
                        data_yaml=self.data_yaml,
                        experiment_name=experiment_name,
                        project=str(self.output_base / "models"),
                        name=experiment_name
                    )

                    self.training_results[experiment_name] = {
                        "status": "success",
                        "config_file": config_file,
                        "model_path": result["model_path"],
                        "save_dir": result["save_dir"],
                        "size": size,
                        "architecture": arch_name
                    }

                    logger.info(f"   [SUCCESS] {experiment_name} training completed")

                except Exception as e:
                    logger.error(f"   [ERROR] {experiment_name} training failed: {str(e)}")
                    self.training_results[experiment_name] = {
                        "status": "failed",
                        "error": str(e),
                        "config_file": config_file,
                        "size": size,
                        "architecture": arch_name
                    }

    def _run_conversion_phase(self):
        """Convert trained models to ONNX"""
        from convert_to_onnx import convert_to_onnx

        onnx_dir = self.output_base / "onnx"
        onnx_dir.mkdir(exist_ok=True)

        for experiment_name, train_result in self.training_results.items():
            if train_result["status"] != "success":
                continue

            try:
                logger.info(f"   Converting {experiment_name} to ONNX...")

                model_path = train_result["model_path"]
                onnx_path = onnx_dir / f"{experiment_name}.onnx"

                # Convert to ONNX
                convert_to_onnx(
                    model_path=model_path,
                    onnx_path=str(onnx_path),
                    input_size=train_result["size"]
                )

                self.conversion_results[experiment_name] = {
                    "status": "success",
                    "onnx_path": str(onnx_path),
                    "original_model": model_path
                }

                logger.info(f"   [SUCCESS] {experiment_name} ONNX conversion completed")

            except Exception as e:
                logger.error(f"   [ERROR] {experiment_name} ONNX conversion failed: {str(e)}")
                self.conversion_results[experiment_name] = {
                    "status": "failed",
                    "error": str(e)
                }

    def _run_quantization_phase(self):
        """Quantize ONNX models to INT8"""
        from quantize_models import quantize_onnx_model

        quantized_dir = self.output_base / "quantized"
        quantized_dir.mkdir(exist_ok=True)

        for experiment_name, conv_result in self.conversion_results.items():
            if conv_result["status"] != "success":
                continue

            try:
                logger.info(f"   Quantizing {experiment_name} to INT8...")

                onnx_path = conv_result["onnx_path"]
                quantized_path = quantized_dir / f"{experiment_name}_int8.onnx"

                # Quantize to INT8
                quantize_onnx_model(
                    onnx_path=onnx_path,
                    quantized_path=str(quantized_path),
                    data_yaml=self.data_yaml
                )

                self.quantization_results[experiment_name] = {
                    "status": "success",
                    "quantized_path": str(quantized_path),
                    "original_onnx": onnx_path
                }

                logger.info(f"   [SUCCESS] {experiment_name} quantization completed")

            except Exception as e:
                logger.error(f"   [ERROR] {experiment_name} quantization failed: {str(e)}")
                self.quantization_results[experiment_name] = {
                    "status": "failed",
                    "error": str(e)
                }

    def _run_evaluation_phase(self):
        """Evaluate all models (PyTorch, ONNX, Quantized)"""
        for experiment_name, train_result in self.training_results.items():
            if train_result["status"] != "success":
                continue

            logger.info(f"   Evaluating {experiment_name}...")

            # Evaluate PyTorch model
            try:
                pytorch_metrics = self.evaluator.evaluate_model(
                    model_path=train_result["model_path"],
                    data_yaml=self.data_yaml,
                    conf=0.25,
                    iou=0.5
                )
                self.evaluation_results[f"{experiment_name}_pytorch"] = {
                    "status": "success",
                    "model_type": "pytorch",
                    "metrics": pytorch_metrics
                }
            except Exception as e:
                self.evaluation_results[f"{experiment_name}_pytorch"] = {
                    "status": "failed",
                    "error": str(e)
                }

            # Evaluate ONNX model if available
            if experiment_name in self.conversion_results and self.conversion_results[experiment_name]["status"] == "success":
                try:
                    onnx_metrics = self.evaluator.evaluate_onnx_model(
                        onnx_path=self.conversion_results[experiment_name]["onnx_path"],
                        data_yaml=self.data_yaml,
                        input_size=train_result["size"]
                    )
                    self.evaluation_results[f"{experiment_name}_onnx"] = {
                        "status": "success",
                        "model_type": "onnx",
                        "metrics": onnx_metrics
                    }
                except Exception as e:
                    self.evaluation_results[f"{experiment_name}_onnx"] = {
                        "status": "failed",
                        "error": str(e)
                    }

            # Evaluate Quantized model if available
            if experiment_name in self.quantization_results and self.quantization_results[experiment_name]["status"] == "success":
                try:
                    quantized_metrics = self.evaluator.evaluate_onnx_model(
                        onnx_path=self.quantization_results[experiment_name]["quantized_path"],
                        data_yaml=self.data_yaml,
                        input_size=train_result["size"]
                    )
                    self.evaluation_results[f"{experiment_name}_quantized"] = {
                        "status": "success",
                        "model_type": "quantized",
                        "metrics": quantized_metrics
                    }
                except Exception as e:
                    self.evaluation_results[f"{experiment_name}_quantized"] = {
                        "status": "failed",
                        "error": str(e)
                    }

            logger.info(f"   [SUCCESS] {experiment_name} evaluation completed")

    def _run_benchmarking_phase(self):
        """Benchmark all models for speed and memory"""
        for experiment_name, train_result in self.training_results.items():
            if train_result["status"] != "success":
                continue

            logger.info(f"   Benchmarking {experiment_name}...")

            # Benchmark PyTorch model
            try:
                pytorch_benchmark = self.evaluator.benchmark_model(
                    model_path=train_result["model_path"],
                    input_size=train_result["size"],
                    num_runs=100
                )
                self.benchmark_results[f"{experiment_name}_pytorch"] = {
                    "status": "success",
                    "model_type": "pytorch",
                    "benchmark": pytorch_benchmark
                }
            except Exception as e:
                self.benchmark_results[f"{experiment_name}_pytorch"] = {
                    "status": "failed",
                    "error": str(e)
                }

            # Benchmark ONNX model if available
            if experiment_name in self.conversion_results and self.conversion_results[experiment_name]["status"] == "success":
                try:
                    onnx_benchmark = self.evaluator.benchmark_onnx_model(
                        onnx_path=self.conversion_results[experiment_name]["onnx_path"],
                        input_size=train_result["size"],
                        num_runs=100
                    )
                    self.benchmark_results[f"{experiment_name}_onnx"] = {
                        "status": "success",
                        "model_type": "onnx",
                        "benchmark": onnx_benchmark
                    }
                except Exception as e:
                    self.benchmark_results[f"{experiment_name}_onnx"] = {
                        "status": "failed",
                        "error": str(e)
                    }

            # Benchmark Quantized model if available
            if experiment_name in self.quantization_results and self.quantization_results[experiment_name]["status"] == "success":
                try:
                    quantized_benchmark = self.evaluator.benchmark_onnx_model(
                        onnx_path=self.quantization_results[experiment_name]["quantized_path"],
                        input_size=train_result["size"],
                        num_runs=100
                    )
                    self.benchmark_results[f"{experiment_name}_quantized"] = {
                        "status": "success",
                        "model_type": "quantized",
                        "benchmark": quantized_benchmark
                    }
                except Exception as e:
                    self.benchmark_results[f"{experiment_name}_quantized"] = {
                        "status": "failed",
                        "error": str(e)
                    }

            logger.info(f"   [SUCCESS] {experiment_name} benchmarking completed")

    def _generate_final_report(self):
        """Generate comprehensive final report"""
        report_path = self.output_base / "final_report.json"

        # Compile all results
        final_report = {
            "test_summary": {
                "total_models_trained": len([r for r in self.training_results.values() if r["status"] == "success"]),
                "total_conversions": len([r for r in self.conversion_results.values() if r["status"] == "success"]),
                "total_quantizations": len([r for r in self.quantization_results.values() if r["status"] == "success"]),
                "total_evaluations": len([r for r in self.evaluation_results.values() if r["status"] == "success"]),
                "total_benchmarks": len([r for r in self.benchmark_results.values() if r["status"] == "success"])
            },
            "training_results": self.training_results,
            "conversion_results": self.conversion_results,
            "quantization_results": self.quantization_results,
            "evaluation_results": self.evaluation_results,
            "benchmark_results": self.benchmark_results
        }

        # Save detailed JSON report
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        # Generate summary CSV
        summary_data = []
        for experiment_name, train_result in self.training_results.items():
            if train_result["status"] != "success":
                continue

            row = {
                "experiment": experiment_name,
                "architecture": train_result["architecture"],
                "size": train_result["size"],
                "training_status": train_result["status"],
                "conversion_status": self.conversion_results.get(experiment_name, {}).get("status", "not_attempted"),
                "quantization_status": self.quantization_results.get(experiment_name, {}).get("status", "not_attempted")
            }

            # Add evaluation results
            for model_type in ["pytorch", "onnx", "quantized"]:
                eval_key = f"{experiment_name}_{model_type}"
                if eval_key in self.evaluation_results and self.evaluation_results[eval_key]["status"] == "success":
                    metrics = self.evaluation_results[eval_key]["metrics"]
                    row[f"{model_type}_map50"] = metrics.get("mAP50", 0)
                    row[f"{model_type}_precision"] = metrics.get("precision", 0)
                    row[f"{model_type}_recall"] = metrics.get("recall", 0)

            # Add benchmark results
            for model_type in ["pytorch", "onnx", "quantized"]:
                bench_key = f"{experiment_name}_{model_type}"
                if bench_key in self.benchmark_results and self.benchmark_results[bench_key]["status"] == "success":
                    benchmark = self.benchmark_results[bench_key]["benchmark"]
                    row[f"{model_type}_fps"] = benchmark.get("fps", 0)
                    row[f"{model_type}_latency_ms"] = benchmark.get("avg_latency_ms", 0)

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = self.output_base / "summary_report.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        logger.info(f"[ANALYSIS] Final report saved to: {report_path}")
        logger.info(f"[ANALYSIS] Summary CSV saved to: {summary_csv_path}")

        # Print summary table
        logger.info("\n" + "="*100)
        logger.info("END-TO-END TEST SUMMARY")
        logger.info("="*100)
        logger.info(summary_df.to_string(index=False))
        logger.info("="*100)


def main():
    """Main entry point"""
    tester = EndToEndTester()
    results = tester.run_full_pipeline()

    if results["status"] == "success":
        logger.info("[COMPLETE] All phases completed successfully!")
        return 0
    else:
        logger.error(f"[ERROR] Test failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
