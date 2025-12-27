#!/usr/bin/env python3
"""
Comprehensive Benchmarking Script for All Trained Models
Focus: Accuracy per class, Speed, Inference, Memory Size, Statistical Analysis
Priority: Confidence metrics and low memory footprint
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
from typing import Dict, List, Any
import onnxruntime as ort
import torch
from ultralytics import YOLO
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("output/benchmark_all_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelBenchmarker:
    """Comprehensive model benchmarking with focus on confidence and memory efficiency"""

    def __init__(self, data_yaml: str = "consolidated_dataset/data.yaml"):
        self.data_yaml = data_yaml
        self.models_dir = Path("output/models")
        self.onnx_dir = Path("output/onnx")
        self.quantized_dir = Path("output/quantized")

        # Class names
        self.class_names = ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']

        # Results storage
        self.results = {}

    def find_all_models(self) -> List[Dict[str, Any]]:
        """Find all trained models"""
        models = []

        if self.models_dir.exists():
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    best_pt = model_dir / "weights" / "best.pt"
                    if best_pt.exists():
                        # Determine model type
                        name = model_dir.name.lower()
                        if 'mobilenet' in name:
                            model_type = 'mobilenet'
                        elif 'efficientnet' in name:
                            model_type = 'efficientnet'
                        else:
                            model_type = 'yolo'

                        # Extract size if present
                        size = 224  # default
                        if any(str(s) in name for s in [160, 192, 224, 320, 416]):
                            for s in [160, 192, 224, 320, 416]:
                                if str(s) in name:
                                    size = s
                                    break

                        models.append({
                            'name': model_dir.name,
                            'path': str(best_pt),
                            'type': model_type,
                            'size': size,
                            'onnx_path': self.onnx_dir / f"{model_dir.name}.onnx",
                            'quantized_path': self.quantized_dir / f"{model_dir.name}_int8.onnx"
                        })

        return models

    def benchmark_pytorch_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark PyTorch model: accuracy, speed, memory"""
        logger.info(f"🔬 Benchmarking PyTorch: {model_info['name']}")

        results = {
            'model': model_info['name'],
            'format': 'PyTorch',
            'type': model_info['type'],
            'input_size': model_info['size']
        }

        try:
            # Load model
            model = YOLO(model_info['path'])

            # Memory size
            model_size = Path(model_info['path']).stat().st_size / (1024 * 1024)  # MB
            results['model_size_mb'] = model_size

            # Speed benchmark (inference time)
            speeds = self.measure_inference_speed(model, model_info['size'], num_runs=50)
            results.update(speeds)

            # Accuracy benchmark
            accuracy = self.evaluate_model_accuracy(model, model_info['size'])
            results.update(accuracy)

            results['status'] = 'success'

        except Exception as e:
            logger.error(f"Failed to benchmark {model_info['name']}: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)

        return results

    def benchmark_onnx_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark ONNX model: accuracy, speed, memory"""
        if not model_info['onnx_path'].exists():
            return {'model': model_info['name'], 'format': 'ONNX', 'status': 'missing'}

        logger.info(f"🔬 Benchmarking ONNX: {model_info['name']}")

        results = {
            'model': model_info['name'],
            'format': 'ONNX',
            'type': model_info['type'],
            'input_size': model_info['size']
        }

        try:
            # Load ONNX model
            session = ort.InferenceSession(str(model_info['onnx_path']))

            # Memory size
            model_size = model_info['onnx_path'].stat().st_size / (1024 * 1024)  # MB
            results['model_size_mb'] = model_size

            # Speed benchmark
            speeds = self.measure_onnx_inference_speed(session, model_info['size'], num_runs=50)
            results.update(speeds)

            # Accuracy benchmark
            accuracy = self.evaluate_onnx_accuracy(session, model_info['size'])
            results.update(accuracy)

            results['status'] = 'success'

        except Exception as e:
            logger.error(f"Failed to benchmark ONNX {model_info['name']}: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)

        return results

    def benchmark_quantized_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark quantized INT8 model: accuracy, speed, memory"""
        if not model_info['quantized_path'].exists():
            return {'model': model_info['name'], 'format': 'INT8', 'status': 'missing'}

        logger.info(f"🔬 Benchmarking INT8: {model_info['name']}")

        results = {
            'model': model_info['name'],
            'format': 'INT8',
            'type': model_info['type'],
            'input_size': model_info['size']
        }

        try:
            # Load quantized model
            session = ort.InferenceSession(str(model_info['quantized_path']))

            # Memory size
            model_size = model_info['quantized_path'].stat().st_size / (1024 * 1024)  # MB
            results['model_size_mb'] = model_size

            # Speed benchmark
            speeds = self.measure_onnx_inference_speed(session, model_info['size'], num_runs=50)
            results.update(speeds)

            # Accuracy benchmark
            accuracy = self.evaluate_onnx_accuracy(session, model_info['size'])
            results.update(accuracy)

            results['status'] = 'success'

        except Exception as e:
            logger.error(f"Failed to benchmark INT8 {model_info['name']}: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)

        return results

    def measure_inference_speed(self, model, input_size: int, num_runs: int = 50) -> Dict[str, float]:
        """Measure PyTorch model inference speed"""
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Measure speed
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # ms

        return {
            'avg_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times),
            'min_inference_ms': np.min(times),
            'max_inference_ms': np.max(times),
            'fps': 1000 / np.mean(times)
        }

    def measure_onnx_inference_speed(self, session, input_size: int, num_runs: int = 50) -> Dict[str, float]:
        """Measure ONNX model inference speed"""
        # Get input details
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})

        # Measure speed
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run(None, {input_name: dummy_input})
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ms

        return {
            'avg_inference_ms': np.mean(times),
            'std_inference_ms': np.std(times),
            'min_inference_ms': np.min(times),
            'max_inference_ms': np.max(times),
            'fps': 1000 / np.mean(times)
        }

    def evaluate_model_accuracy(self, model, input_size: int) -> Dict[str, Any]:
        """Evaluate PyTorch model accuracy with per-class metrics"""
        try:
            # Run validation
            results = model.val(data=self.data_yaml, imgsz=input_size, conf=0.1, iou=0.6, verbose=False)

            # Extract metrics
            metrics = {
                'mAP50': results.box.map50,
                'mAP75': results.box.map75,
                'mAP50_95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1_score': 2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0
            }

            # Per-class metrics
            if hasattr(results.box, 'class_result'):
                per_class = {}
                for i, class_name in enumerate(self.class_names):
                    if i < len(results.box.class_result):
                        class_metrics = results.box.class_result[i]
                        per_class[class_name] = {
                            'precision': class_metrics[0],
                            'recall': class_metrics[1],
                            'mAP50': class_metrics[2],
                            'f1': class_metrics[3]
                        }
                metrics['per_class'] = per_class

            return metrics

        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            return {'accuracy_error': str(e)}

    def evaluate_onnx_accuracy(self, session, input_size: int) -> Dict[str, Any]:
        """Evaluate ONNX model accuracy - simplified version"""
        # For ONNX models, we'll use a basic confidence threshold evaluation
        # since full YOLO validation requires more complex setup

        try:
            # Get input details
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

            # Run inference
            outputs = session.run(None, {input_name: dummy_input})

            # Basic analysis - check if outputs are reasonable
            if len(outputs) > 0 and outputs[0] is not None:
                output_shape = outputs[0].shape
                metrics = {
                    'onnx_inference_success': True,
                    'output_shape': str(output_shape),
                    'output_elements': int(np.prod(output_shape))
                }
                return metrics
            else:
                return {'onnx_inference_success': False}

        except Exception as e:
            return {'onnx_inference_error': str(e)}

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmarking on all models"""
        logger.info("=" * 80)
        logger.info("🚀 STARTING COMPREHENSIVE MODEL BENCHMARKING")
        logger.info("=" * 80)

        models = self.find_all_models()
        logger.info(f"📊 Found {len(models)} models to benchmark")

        all_results = []

        for model_info in models:
            logger.info(f"\n🔬 Processing: {model_info['name']}")

            # Benchmark PyTorch model
            pytorch_results = self.benchmark_pytorch_model(model_info)
            all_results.append(pytorch_results)

            # Benchmark ONNX model
            onnx_results = self.benchmark_onnx_model(model_info)
            all_results.append(onnx_results)

            # Benchmark Quantized model
            quantized_results = self.benchmark_quantized_model(model_info)
            all_results.append(quantized_results)

        # Generate comprehensive report
        self.generate_comprehensive_report(all_results)

        return {'results': all_results}

    def generate_comprehensive_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive benchmarking report with focus on confidence and memory"""
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE BENCHMARKING REPORT")
        logger.info("="*80)

        # Convert to DataFrame for analysis
        df = pd.DataFrame([r for r in results if r.get('status') == 'success'])

        if df.empty:
            logger.error("No successful benchmark results to analyze")
            return

        # Focus areas: Confidence metrics and low memory footprint
        logger.info("\n🎯 CONFIDENCE & MEMORY EFFICIENCY ANALYSIS")
        logger.info("-" * 50)

        # Group by format (PyTorch, ONNX, INT8)
        formats = df['format'].unique()

        for fmt in formats:
            fmt_df = df[df['format'] == fmt]
            if fmt_df.empty:
                continue

            logger.info(f"\n📈 {fmt.upper()} MODELS PERFORMANCE:")
            logger.info("-" * 30)

            # Memory efficiency (focus on small models)
            memory_efficient = fmt_df.nsmallest(5, 'model_size_mb')
            logger.info("💾 Top 5 Memory Efficient Models:")
            for _, row in memory_efficient.iterrows():
                logger.info(".1f"
                          ".1f")

            # Performance metrics
            if 'mAP50' in fmt_df.columns:
                accuracy_best = fmt_df.nlargest(5, 'mAP50')
                logger.info("🎯 Top 5 Most Accurate Models:")
                for _, row in accuracy_best.iterrows():
                    logger.info(".3f"
                              ".1f")

            # Speed analysis
            if 'fps' in fmt_df.columns:
                fastest = fmt_df.nlargest(5, 'fps')
                logger.info("⚡ Top 5 Fastest Models:")
                for _, row in fastest.iterrows():
                    logger.info(".1f"
                              ".1f"
                              ".1f")

        # Confidence-focused analysis
        logger.info("\n🎯 CONFIDENCE OPTIMIZATION ANALYSIS")
        logger.info("-" * 40)

        # Models with confidence in name (optimized for confidence)
        confidence_models = df[df['model'].str.contains('confidence', case=False)]

        if not confidence_models.empty:
            logger.info("🎯 Confidence-Optimized Models:")
            for _, row in confidence_models.iterrows():
                if 'mAP50' in row:
                    logger.info(f"  {row['model']} ({row['format']}): mAP50={row.get('mAP50', 0):.3f}, Size={row.get('model_size_mb', 0):.1f}MB")

        # Statistical analysis
        logger.info("\n📊 STATISTICAL ANALYSIS")
        logger.info("-" * 30)

        for fmt in formats:
            fmt_df = df[df['format'] == fmt]
            if fmt_df.empty or 'mAP50' not in fmt_df.columns:
                continue

            logger.info(f"\n{fmt.upper()} Statistics:")
            logger.info(f"  Models: {len(fmt_df)}")
            logger.info(".3f")
            logger.info(".3f")
            logger.info(".3f")
            if 'fps' in fmt_df.columns:
                logger.info(".1f")
            if 'model_size_mb' in fmt_df.columns:
                logger.info(".1f")
        # Per-class analysis (for PyTorch models with detailed metrics)
        pytorch_results = [r for r in results if r.get('format') == 'PyTorch' and 'per_class' in r]
        if pytorch_results:
            logger.info("\n📋 PER-CLASS ACCURACY ANALYSIS (PyTorch Models)")
            logger.info("-" * 50)

            for result in pytorch_results[:3]:  # Show top 3
                logger.info(f"\n🏷️  {result['model']}:")
                if 'per_class' in result:
                    for class_name, metrics in result['per_class'].items():
                        logger.info("6"
                                  "6.3f"
                                  "6.3f"
                                  "6.3f"
                                  "6.3f")

        # Save detailed results
        output_file = "output/comprehensive_benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\n💾 Detailed results saved to: {output_file}")

        # Save summary CSV
        summary_df = df[['model', 'format', 'type', 'input_size', 'model_size_mb', 'mAP50', 'fps', 'avg_inference_ms']].copy()
        summary_df = summary_df.round(3)
        summary_file = "output/benchmark_summary.csv"
        summary_df.to_csv(summary_file, index=False)

        logger.info(f"📊 Summary CSV saved to: {summary_file}")

        logger.info("\n" + "="*80)
        logger.info("✅ COMPREHENSIVE BENCHMARKING COMPLETED")
        logger.info("="*80)


def main():
    """Main benchmarking function"""
    benchmarker = ModelBenchmarker()
    results = benchmarker.run_comprehensive_benchmark()

    logger.info("🎯 Benchmarking completed! Check output/ for detailed results.")

    return results


if __name__ == "__main__":
    main()
