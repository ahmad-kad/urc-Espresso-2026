#!/usr/bin/env python3
"""
Simple Model Conversion and Benchmarking Script
Converts all trained models to ONNX format and benchmarks them to find the best performer.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import time
import torch
import numpy as np

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_model_info(model_path: str) -> Tuple[str, int]:
    """
    Parse model name to extract architecture and input size.

    Returns:
        tuple: (model_name, input_size)
    """
    path = Path(model_path)
    model_dir = path.parent.parent.name  # e.g., "mobilenet_224"

    # Extract input size from directory name
    parts = model_dir.split('_')
    input_size = 224  # default

    # Look for numeric input size in the name
    for part in parts:
        if part.isdigit() and len(part) == 3:  # 160, 192, 224
            input_size = int(part)
            break

    return model_dir, input_size


def get_all_best_models(models_dir: str = "output/models") -> List[Dict]:
    """
    Get all best.pt models with their metadata.

    Returns:
        List of dicts with model info
    """
    models = []
    models_path = Path(models_dir)

    if not models_path.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return []

    # Find all best.pt files
    for best_pt_path in models_path.rglob("best.pt"):
        try:
            model_name, input_size = parse_model_info(str(best_pt_path))

            # Get model size
            size_mb = best_pt_path.stat().st_size / (1024 * 1024)

            model_info = {
                'name': model_name,
                'path': str(best_pt_path),
                'input_size': input_size,
                'size_mb': round(size_mb, 2),
                'onnx_path': None,
                'benchmark_results': None
            }

            models.append(model_info)
            logger.info(f"Found model: {model_name} ({input_size}x{input_size}, {size_mb:.1f} MB)")

        except Exception as e:
            logger.warning(f"Could not process {best_pt_path}: {e}")

    return sorted(models, key=lambda x: x['name'])


def convert_model_to_onnx(model_info: Dict, onnx_output_dir: str = "output/onnx") -> bool:
    """
    Convert a single model to ONNX format with correct input size.
    """
    try:
        model_path = model_info['path']
        input_size = model_info['input_size']
        model_name = model_info['name']

        logger.info(f"Converting {model_name} to ONNX ({input_size}x{input_size})...")

        # Ensure output directory exists
        Path(onnx_output_dir).mkdir(parents=True, exist_ok=True)

        # Load model
        model = YOLO(model_path)

        # Create ONNX filename
        onnx_filename = f"{model_name}_{input_size}.onnx"
        onnx_path = os.path.join(onnx_output_dir, onnx_filename)

        # Export with correct settings for benchmarking
        model.export(
            format="onnx",
            imgsz=input_size,
            dynamic=False,  # Fixed size for consistent benchmarking
            simplify=True,
            opset=11,  # Good compatibility
        )

        # YOLOv8 exports to current directory, need to move
        exported_path = model_path.replace(".pt", ".onnx")
        if os.path.exists(exported_path) and exported_path != onnx_path:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            os.rename(exported_path, onnx_path)

        # Verify the export worked
        if os.path.exists(onnx_path):
            onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            model_info['onnx_path'] = onnx_path
            model_info['onnx_size_mb'] = round(onnx_size_mb, 2)
            logger.info(f"✅ ONNX export successful: {onnx_filename} ({onnx_size_mb:.1f} MB)")
            return True
        else:
            logger.error(f"❌ ONNX export failed - file not found: {onnx_path}")
            return False

    except Exception as e:
        logger.error(f"❌ ONNX conversion failed for {model_info['name']}: {str(e)}")
        return False


def benchmark_pytorch_model(model_info: Dict, num_runs: int = 100) -> Dict:
    """
    Benchmark a PyTorch model using inference time measurements.
    """
    try:
        model_path = model_info['path']
        input_size = model_info['input_size']
        model_name = model_info['name']

        logger.info(f"Benchmarking PyTorch model {model_name}...")

        # Load model
        model = YOLO(model_path)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)

        # Warm up
        logger.info("Warming up...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark
        logger.info(f"Running {num_runs} inference runs...")
        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time  # Frames per second

        results = {
            'model': model_name,
            'format': 'PyTorch',
            'input_size': input_size,
            'model_size_mb': model_info['size_mb'],
            'avg_inference_ms': round(avg_time, 3),
            'std_inference_ms': round(std_time, 3),
            'min_inference_ms': round(min_time, 3),
            'max_inference_ms': round(max_time, 3),
            'fps': round(fps, 2),
            'num_runs': num_runs
        }

        logger.info(f"✅ Benchmark complete: {fps:.1f} FPS ({avg_time:.2f}ms avg)")
        return results

    except Exception as e:
        logger.error(f"❌ Benchmark failed for {model_name}: {str(e)}")
        return {}


def benchmark_onnx_model(model_info: Dict, num_runs: int = 100) -> Dict:
    """
    Benchmark an ONNX model using ONNX Runtime.
    """
    try:
        import onnxruntime as ort

        onnx_path = model_info['onnx_path']
        input_size = model_info['input_size']
        model_name = model_info['name']

        logger.info(f"Benchmarking ONNX model {model_name}...")

        # Create ONNX session
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # Get input name
        input_name = session.get_inputs()[0].name

        # Create dummy input
        dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

        # Warm up
        logger.info("Warming up...")
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})

        # Benchmark
        logger.info(f"Running {num_runs} inference runs...")
        times = []

        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run(None, {input_name: dummy_input})
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time  # Frames per second

        results = {
            'model': model_name,
            'format': 'ONNX',
            'input_size': input_size,
            'model_size_mb': model_info.get('onnx_size_mb', model_info['size_mb']),
            'avg_inference_ms': round(avg_time, 3),
            'std_inference_ms': round(std_time, 3),
            'min_inference_ms': round(min_time, 3),
            'max_inference_ms': round(max_time, 3),
            'fps': round(fps, 2),
            'num_runs': num_runs
        }

        logger.info(f"✅ ONNX Benchmark complete: {fps:.1f} FPS ({avg_time:.2f}ms avg)")
        return results

    except Exception as e:
        logger.error(f"❌ ONNX Benchmark failed for {model_name}: {str(e)}")
        return {}


def run_comprehensive_conversion_and_benchmark(models_dir: str = "output/models"):
    """
    Main function to convert all models to ONNX and benchmark them.
    """
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL CONVERSION & BENCHMARKING")
    logger.info("=" * 80)

    # Step 1: Discover all models
    logger.info("\n[STEP 1] Discovering trained models...")
    models = get_all_best_models(models_dir)

    if not models:
        logger.error("No models found!")
        return []

    logger.info(f"Found {len(models)} models to process")

    # Step 2: Convert all models to ONNX
    logger.info("\n[STEP 2] Converting models to ONNX format...")
    successful_conversions = 0

    for i, model in enumerate(models, 1):
        logger.info(f"\n[{i}/{len(models)}] Processing {model['name']}...")
        if convert_model_to_onnx(model):
            successful_conversions += 1

    logger.info(f"\n✅ ONNX Conversion Summary: {successful_conversions}/{len(models)} successful")

    # Step 3: Benchmark all models
    logger.info("\n[STEP 3] Benchmarking all models...")
    benchmark_results = []

    for i, model in enumerate(models, 1):
        logger.info(f"\n[{i}/{len(models)}] Benchmarking {model['name']}...")

        # Benchmark PyTorch version
        pytorch_results = benchmark_pytorch_model(model)
        if pytorch_results:
            benchmark_results.append(pytorch_results)

        # Benchmark ONNX version if available
        if model['onnx_path'] and os.path.exists(model['onnx_path']):
            onnx_results = benchmark_onnx_model(model)
            if onnx_results:
                benchmark_results.append(onnx_results)

    # Step 4: Analyze and rank results
    logger.info("\n[STEP 4] Analyzing results...")

    if benchmark_results:
        # Convert to DataFrame for analysis
        df = pd.DataFrame(benchmark_results)

        # Calculate efficiency score (higher FPS and lower latency is better)
        df['efficiency_score'] = df['fps'] / df['avg_inference_ms']
        df['size_efficiency'] = df['fps'] / df['model_size_mb']  # FPS per MB

        # Sort by FPS (primary), then by efficiency
        df_sorted = df.sort_values(['fps', 'efficiency_score', 'model_size_mb'],
                                 ascending=[False, False, True])

        # Save comprehensive results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        results_file = output_dir / "final_model_comparison.json"
        df_sorted.to_json(results_file, orient='records', indent=2)

        csv_file = output_dir / "final_model_comparison.csv"
        df_sorted.to_csv(csv_file, index=False)

        # Find the best model
        best_model = df_sorted.iloc[0]

        logger.info("=" * 80)
        logger.info("🏆 BEST MODEL RESULTS")
        logger.info("=" * 80)
        logger.info(f"Winner: {best_model['model']}")
        logger.info(f"Format: {best_model['format']}")
        logger.info(f"Input Size: {best_model['input_size']}x{best_model['input_size']}")
        logger.info(f"FPS: {best_model['fps']:.1f}")
        logger.info(f"Avg Latency: {best_model['avg_inference_ms']:.2f} ms")
        logger.info(f"Model Size: {best_model['model_size_mb']:.1f} MB")
        logger.info(f"Efficiency Score: {best_model['efficiency_score']:.4f}")

        logger.info("\n📊 TOP 5 MODELS BY FPS:")
        logger.info("-" * 50)
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            logger.info(f"{i}. {row['model']} ({row['format']}) - {row['fps']:.1f} FPS, {row['avg_inference_ms']:.1f}ms")

        logger.info("\n📊 TOP 5 MOST EFFICIENT (FPS per MB):")
        logger.info("-" * 50)
        efficiency_sorted = df.sort_values('size_efficiency', ascending=False)
        for i, (_, row) in enumerate(efficiency_sorted.head(5).iterrows(), 1):
            logger.info(f"{i}. {row['model']} ({row['format']}) - {row['size_efficiency']:.2f} FPS/MB")

        # Group by model family for comparison
        logger.info("\n📊 PERFORMANCE BY MODEL FAMILY:")
        logger.info("-" * 50)

        # Group PyTorch models only for fair comparison
        pytorch_only = df[df['format'] == 'PyTorch']
        if not pytorch_only.empty:
            family_stats = pytorch_only.groupby(pytorch_only['model'].str.split('_').str[0]).agg({
                'fps': ['mean', 'max'],
                'avg_inference_ms': 'mean',
                'model_size_mb': 'mean'
            }).round(2)

            logger.info("PyTorch Models Performance:")
            for family, stats in family_stats.iterrows():
                logger.info(f"  {family}: {stats['fps']['mean']:.1f} avg FPS, {stats['fps']['max']:.1f} max FPS, {stats['model_size_mb']['mean']:.1f}MB avg size")

        logger.info(f"\n💾 Results saved to:")
        logger.info(f"   JSON: {results_file}")
        logger.info(f"   CSV: {csv_file}")

        return df_sorted

    else:
        logger.error("❌ No benchmark results obtained!")
        return []


def main():
    """Main entry point"""
    try:
        results = run_comprehensive_conversion_and_benchmark()

        if results:
            logger.info("\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
            return 0
        else:
            logger.error("\n❌ Analysis failed!")
            return 1

    except Exception as e:
        logger.error(f"❌ Script failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
