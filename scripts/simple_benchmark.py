#!/usr/bin/env python3
"""
Simple Benchmarking Script for All Models
Focus: Accuracy, Speed, Memory Size with emphasis on confidence and low memory footprint
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleBenchmarker:
    """Simple benchmarking focused on key metrics"""

    def __init__(self):
        self.class_names = ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        self.results = []

    def find_models(self) -> List[Dict[str, Any]]:
        """Find all available models"""
        models = []

        # PyTorch models
        models_dir = Path("output/models")
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    pt_path = model_dir / "weights" / "best.pt"
                    if pt_path.exists():
                        models.append({
                            'name': model_dir.name,
                            'type': 'pytorch',
                            'path': str(pt_path),
                            'size': self.extract_size(model_dir.name)
                        })

        # ONNX models
        onnx_dir = Path("output/onnx")
        if onnx_dir.exists():
            for onnx_file in onnx_dir.glob("*.onnx"):
                models.append({
                    'name': onnx_file.stem,
                    'type': 'onnx',
                    'path': str(onnx_file),
                    'size': self.extract_size(onnx_file.stem)
                })

        # Quantized models
        quantized_dir = Path("output/quantized")
        if quantized_dir.exists():
            for quantized_file in quantized_dir.glob("*_int8.onnx"):
                name = quantized_file.stem.replace('_int8', '')
                models.append({
                    'name': name,
                    'type': 'int8',
                    'path': str(quantized_file),
                    'size': self.extract_size(name)
                })

        return models

    def extract_size(self, name: str) -> int:
        """Extract input size from model name"""
        for size in [160, 192, 224, 320, 416]:
            if str(size) in name:
                return size
        return 224  # default

    def benchmark_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single model"""
        logger.info(f"Benchmarking {model_info['type'].upper()}: {model_info['name']}")

        result = {
            'name': model_info['name'],
            'type': model_info['type'],
            'size': model_info['size']
        }

        try:
            if model_info['type'] == 'pytorch':
                metrics = self.benchmark_pytorch(model_info)
            elif model_info['type'] == 'onnx':
                metrics = self.benchmark_onnx(model_info)
            elif model_info['type'] == 'int8':
                metrics = self.benchmark_int8(model_info)
            else:
                return result

            result.update(metrics)
            result['status'] = 'success'

        except Exception as e:
            logger.error(f"Failed {model_info['name']}: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)

        return result

    def _extract_accuracy_metrics(self, results) -> Dict[str, Any]:
        """Safely extract accuracy metrics from YOLO validation results"""
        metrics = {}

        try:
            # Check if results.box exists and has the expected attributes
            if hasattr(results, 'box') and results.box is not None:
                box_results = results.box

                # Extract main metrics safely
                metrics['mAP50'] = float(getattr(box_results, 'map50', 0))
                metrics['precision'] = float(getattr(box_results, 'mp', 0))
                metrics['recall'] = float(getattr(box_results, 'mr', 0))

                # Calculate F1 safely
                mp = getattr(box_results, 'mp', 0)
                mr = getattr(box_results, 'mr', 0)
                metrics['f1_score'] = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0

            else:
                logger.warning("No box results available from validation")

        except Exception as e:
            logger.error(f"Error extracting accuracy metrics: {e}")
            metrics['extraction_error'] = str(e)

        return metrics

    def benchmark_pytorch(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark PyTorch model"""
        model = YOLO(model_info['path'])

        # Size
        size_mb = Path(model_info['path']).stat().st_size / (1024 * 1024)

        # Speed
        speeds = self.measure_speed(model, model_info['size'])

        # Accuracy (simplified)
        try:
            results = model.val(data="consolidated_dataset/data.yaml", imgsz=model_info['size'], verbose=False)
            accuracy = self._extract_accuracy_metrics(results)
        except Exception as e:
            logger.warning(f"Accuracy evaluation failed for {model_info['name']}: {e}")
            accuracy = {'accuracy_error': str(e)}

        return {
            'size_mb': round(size_mb, 2),
            **speeds,
            **accuracy
        }

    def benchmark_onnx(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark ONNX model"""
        session = ort.InferenceSession(model_info['path'])

        # Size
        size_mb = Path(model_info['path']).stat().st_size / (1024 * 1024)

        # Speed
        speeds = self.measure_onnx_speed(session, model_info['size'])

        return {
            'size_mb': round(size_mb, 2),
            **speeds
        }

    def benchmark_int8(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark INT8 model"""
        session = ort.InferenceSession(model_info['path'])

        # Size
        size_mb = Path(model_info['path']).stat().st_size / (1024 * 1024)

        # Speed
        speeds = self.measure_onnx_speed(session, model_info['size'])

        return {
            'size_mb': round(size_mb, 2),
            **speeds
        }

    def measure_speed(self, model, size: int, runs: int = 30) -> Dict[str, float]:
        """Measure PyTorch inference speed"""
        dummy = torch.randn(1, 3, size, size)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy)

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.time()
                _ = model(dummy)
                times.append((time.time() - start) * 1000)

        return {
            'avg_ms': round(np.mean(times), 2),
            'fps': round(1000 / np.mean(times), 1)
        }

    def measure_onnx_speed(self, session, size: int, runs: int = 30) -> Dict[str, float]:
        """Measure ONNX inference speed"""
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 3, size, size).astype(np.float32)

        # Warmup
        for _ in range(5):
            _ = session.run(None, {input_name: dummy})

        # Measure
        times = []
        for _ in range(runs):
            start = time.time()
            _ = session.run(None, {input_name: dummy})
            times.append((time.time() - start) * 1000)

        return {
            'avg_ms': round(np.mean(times), 2),
            'fps': round(1000 / np.mean(times), 1)
        }

    def run_benchmark(self):
        """Run complete benchmark"""
        logger.info("="*80)
        logger.info("🚀 STARTING COMPREHENSIVE MODEL BENCHMARK")
        logger.info("="*80)

        models = self.find_models()
        logger.info(f"📊 Found {len(models)} models to benchmark")

        for model in models:
            result = self.benchmark_model(model)
            self.results.append(result)

        self.generate_report()

    def generate_report(self):
        """Generate comprehensive report"""
        logger.info("\n" + "="*80)
        logger.info("📊 BENCHMARK RESULTS")
        logger.info("="*80)

        # Convert to DataFrame
        df = pd.DataFrame([r for r in self.results if r.get('status') == 'success'])

        if df.empty:
            logger.error("No successful results")
            return

        # Group by type
        for model_type in ['pytorch', 'onnx', 'int8']:
            type_df = df[df['type'] == model_type]
            if type_df.empty:
                continue

            logger.info(f"\n🎯 {model_type.upper()} MODELS:")
            logger.info("-" * 40)

            # Sort by size (memory efficiency)
            sorted_df = type_df.sort_values('size_mb')

            for _, row in sorted_df.iterrows():
                size_mb = row.get('size_mb', 0)
                fps = row.get('fps', 0)
                avg_ms = row.get('avg_ms', 0)
                map50 = row.get('mAP50', 'N/A')

                logger.info(f"  {row['name']}: {size_mb:.1f}MB, {fps:.1f} FPS, {avg_ms:.1f}ms, mAP50={map50}")

        # Focus on confidence models
        logger.info("\n🎯 CONFIDENCE OPTIMIZED MODELS:")
        logger.info("-" * 40)

        confidence_df = df[df['name'].str.contains('confidence', case=False)]
        if not confidence_df.empty:
            for _, row in confidence_df.iterrows():
                size_mb = row.get('size_mb', 0)
                fps = row.get('fps', 0)
                map50 = row.get('mAP50', 'N/A')
                logger.info(f"  {row['name']} ({row['type']}): {size_mb:.1f}MB, {fps:.1f} FPS, mAP50={map50}")

        # Best low-memory models
        logger.info("\n💾 LOW MEMORY MODELS (< 15MB):")
        logger.info("-" * 40)

        small_df = df[df['size_mb'] < 15].sort_values('size_mb')
        for _, row in small_df.iterrows():
            size_mb = row.get('size_mb', 0)
            fps = row.get('fps', 0)
            map50 = row.get('mAP50', 'N/A')
            logger.info(f"  {row['name']} ({row['type']}): {size_mb:.1f}MB, {fps:.1f} FPS, mAP50={map50}")

        # Save results
        with open('output/benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        df.to_csv('output/benchmark_summary.csv', index=False)

        logger.info("\n💾 Results saved to output/benchmark_results.json and output/benchmark_summary.csv")
        logger.info("="*80)


def main():
    benchmarker = SimpleBenchmarker()
    benchmarker.run_benchmark()


if __name__ == "__main__":
    main()
