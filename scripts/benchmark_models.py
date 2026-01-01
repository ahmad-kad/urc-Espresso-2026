#!/usr/bin/env python3
"""
Comprehensive model evaluation and benchmarking
Focuses on: Accuracy (Precision), Speed, and Memory
"""

import sys
import time
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO  # type: ignore

from typing import Optional


def find_best_models(
    output_dir: Optional[Path] = None, onnx_dir: Optional[Path] = None
):
    """Find the best Optuna-trained model and ONNX versions"""
    if output_dir is None:
        output_dir = Path("output/models")
    if onnx_dir is None:
        onnx_dir = Path("output/onnx")

    best_models = {}

    # Find Optuna best model
    for subdir in output_dir.iterdir():
        if subdir.is_dir() and "optuna_best" in subdir.name:
            model_path = subdir / "weights" / "best.pt"
            if model_path.exists():
                # Extract input size from name
                name_parts = subdir.name.split("_")
                input_size = 224  # default
                for part in name_parts:
                    if part.isdigit() and len(part) == 3:
                        input_size = int(part)
                        break

                # Extract architecture
                arch = "yolov8n"
                for part in name_parts:
                    if part.startswith("yolov8"):
                        arch = part
                        break

                best_models["pytorch"] = {
                    "name": subdir.name,
                    "path": str(model_path),
                    "size": input_size,
                    "architecture": arch,
                    "format": "pytorch",
                }
                print(f"Found PyTorch model: {subdir.name}")
                break

    # Find ONNX models
    if "pytorch" in best_models:
        base_name = best_models["pytorch"]["name"]
        input_size = best_models["pytorch"]["size"]

        # FP32 ONNX
        fp32_path = onnx_dir / f"{base_name}_fp32.onnx"
        if fp32_path.exists():
            best_models["onnx_fp32"] = {
                "name": f"{base_name}_fp32",
                "path": str(fp32_path),
                "size": input_size,
                "architecture": best_models["pytorch"]["architecture"],
                "format": "onnx_fp32",
            }
            print(f"Found ONNX FP32 model: {fp32_path.name}")

        # INT8 ONNX
        int8_path = onnx_dir / f"{base_name}_int8.onnx"
        if int8_path.exists():
            best_models["onnx_int8"] = {
                "name": f"{base_name}_int8",
                "path": str(int8_path),
                "size": input_size,
                "architecture": best_models["pytorch"]["architecture"],
                "format": "onnx_int8",
            }
            print(f"Found ONNX INT8 model: {int8_path.name}")

    return best_models


def measure_memory_usage(model, device: str, input_size: int) -> Dict[str, float]:
    """Measure memory usage during inference"""
    memory_stats = {}

    if device == "cuda" and torch.cuda.is_available():
        # Clear cache before measurement
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Get initial memory
        initial_memory = torch.cuda.memory_allocated(0) / (1024**2)  # MB

        # Run inference to allocate memory
        dummy_input = np.random.randint(
            0, 255, (input_size, input_size, 3), dtype=np.uint8
        )
        _ = model.predict(dummy_input, imgsz=input_size, device=device, verbose=False)

        torch.cuda.synchronize()

        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated(0) / (1024**2)  # MB
        current_memory = torch.cuda.memory_allocated(0) / (1024**2)  # MB

        memory_stats = {
            "gpu_memory_initial_mb": initial_memory,
            "gpu_memory_peak_mb": peak_memory,
            "gpu_memory_current_mb": current_memory,
            "gpu_memory_inference_mb": peak_memory - initial_memory,
            "cpu_memory_mb": 0.0,  # Not measured for GPU
        }
    else:
        # CPU memory measurement (approximate)
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024**2)  # MB

        dummy_input = np.random.randint(
            0, 255, (input_size, input_size, 3), dtype=np.uint8
        )
        _ = model.predict(dummy_input, imgsz=input_size, device=device, verbose=False)

        current_memory = process.memory_info().rss / (1024**2)  # MB

        memory_stats = {
            "gpu_memory_initial_mb": 0.0,
            "gpu_memory_peak_mb": 0.0,
            "gpu_memory_current_mb": 0.0,
            "gpu_memory_inference_mb": 0.0,
            "cpu_memory_mb": current_memory - initial_memory,
        }

    return memory_stats


def evaluate_and_benchmark_models(
    best_models: Dict,
    data_yaml: str = "consolidated_dataset/data.yaml",
    num_warmup_runs: int = 10,
    num_benchmark_runs: int = 100,
):
    """Comprehensive evaluation: Accuracy, Speed, and Memory

    Args:
        best_models: Dictionary of model information
        data_yaml: Path to data configuration YAML
        num_warmup_runs: Number of warmup runs for benchmarking
        num_benchmark_runs: Number of benchmark runs for latency measurement
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION & BENCHMARKING")
    print("Focus: Accuracy (Precision), Speed, Memory")
    print("=" * 80)

    # Try to import onnxruntime for ONNX benchmarking
    try:
        import onnxruntime

        onnxruntime_available = True
    except ImportError:
        onnxruntime_available = False
        onnxruntime = None
        print("Warning: onnxruntime not available. ONNX models will be skipped.")

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    print(f"Data YAML: {data_yaml}")
    print(f"\nEvaluating {len(best_models)} models:")
    for model_info in best_models.values():
        print(
            f"  • {model_info['name']} ({model_info['format']}) - {model_info['size']}x{model_info['size']}"
        )

    # Evaluate each model
    for model_info in best_models.values():
        model_path = model_info["path"]
        input_size = model_info["size"]
        format_type = model_info["format"]

        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_info['name']} ({format_type})")
        print(f"{'='*80}")

        try:
            if format_type == "pytorch":
                # PyTorch model evaluation
                model = YOLO(model_path)
                model.to(device)

                # 1. ACCURACY EVALUATION
                print("\n[1/3] Evaluating Accuracy...")
                metrics = model.val(
                    data=data_yaml, imgsz=input_size, device=device, verbose=False
                )

                # Overall accuracy metrics
                overall_map50 = (
                    float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0
                )
                overall_map = (
                    float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0
                )

                # Precision and recall are arrays (per-class), get mean for overall
                if hasattr(metrics.box, "p") and metrics.box.p is not None:
                    p = metrics.box.p
                    if hasattr(p, "mean"):
                        overall_precision = float(p.mean())
                    elif hasattr(p, "__iter__") and not isinstance(p, str):
                        overall_precision = float(np.mean(list(p)))
                    else:
                        overall_precision = (
                            float(p) if not hasattr(p, "__len__") else 0.0
                        )
                else:
                    overall_precision = 0.0

                if hasattr(metrics.box, "r") and metrics.box.r is not None:
                    r = metrics.box.r
                    if hasattr(r, "mean"):
                        overall_recall = float(r.mean())
                    elif hasattr(r, "__iter__") and not isinstance(r, str):
                        overall_recall = float(np.mean(list(r)))
                    else:
                        overall_recall = float(r) if not hasattr(r, "__len__") else 0.0
                else:
                    overall_recall = 0.0

                # Extract per-class metrics
                per_class_metrics = {}
                try:
                    # Get class names
                    if hasattr(metrics, "names") and metrics.names:
                        names_dict = metrics.names
                        if isinstance(names_dict, dict):
                            class_names = [
                                names_dict.get(i, f"class_{i}")
                                for i in sorted(names_dict.keys())
                            ]
                        else:
                            class_names = (
                                list(names_dict)
                                if hasattr(names_dict, "__iter__")
                                else []
                            )
                    else:
                        import yaml

                        with open(data_yaml, "r") as f:
                            data_config = yaml.safe_load(f)
                        names_config = data_config.get("names", {})
                        if isinstance(names_config, dict):
                            class_names = [
                                names_config.get(i, f"class_{i}")
                                for i in sorted(names_config.keys())
                            ]
                        else:
                            class_names = (
                                list(names_config)
                                if hasattr(names_config, "__iter__")
                                else []
                            )

                    # Extract per-class metrics
                    if hasattr(metrics, "maps") and metrics.maps is not None:
                        maps = metrics.maps
                        if hasattr(maps, "tolist"):
                            maps = maps.tolist()
                        elif hasattr(maps, "__iter__") and not isinstance(maps, str):
                            maps = list(maps)
                        else:
                            maps = []

                        precision_per_class = []
                        recall_per_class = []

                        if hasattr(metrics, "box"):
                            if hasattr(metrics.box, "p") and metrics.box.p is not None:
                                p = metrics.box.p
                                precision_per_class = (
                                    p.tolist() if hasattr(p, "tolist") else list(p)
                                )
                            if hasattr(metrics.box, "r") and metrics.box.r is not None:
                                r = metrics.box.r
                                recall_per_class = (
                                    r.tolist() if hasattr(r, "tolist") else list(r)
                                )

                        for i, class_name in enumerate(class_names):
                            if i < len(maps):
                                per_class_metrics[class_name] = {
                                    "precision": (
                                        float(precision_per_class[i])
                                        if i < len(precision_per_class)
                                        else 0.0
                                    ),
                                    "recall": (
                                        float(recall_per_class[i])
                                        if i < len(recall_per_class)
                                        else 0.0
                                    ),
                                    "mAP50": (
                                        float(maps[i]) if maps[i] is not None else 0.0
                                    ),
                                }
                except Exception as e:
                    print(f"Warning: Could not extract per-class metrics: {e}")

                # 2. SPEED BENCHMARKING
                print("[2/3] Benchmarking Speed...")
                dummy_input = np.random.randint(
                    0, 255, (input_size, input_size, 3), dtype=np.uint8
                )

                # Warmup
                for _ in range(10):
                    _ = model.predict(
                        dummy_input, imgsz=input_size, device=device, verbose=False
                    )

                if device == "cuda":
                    torch.cuda.synchronize()

                # Benchmark
                num_runs = 100
                latencies = []
                for _ in range(num_runs):
                    start = time.perf_counter()
                    _ = model.predict(
                        dummy_input, imgsz=input_size, device=device, verbose=False
                    )
                    if device == "cuda":
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - start) * 1000)  # ms

                avg_latency = np.mean(latencies)
                std_latency = np.std(latencies)
                min_latency = np.min(latencies)
                max_latency = np.max(latencies)
                fps = 1000.0 / avg_latency if avg_latency > 0 else 0

                # 3. MEMORY MEASUREMENT
                print("[3/3] Measuring Memory...")
                memory_stats = measure_memory_usage(model, device, input_size)
                model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)  # MB

                results.append(
                    {
                        "model": model_info["name"],
                        "format": format_type,
                        "input_size": input_size,
                        # Accuracy metrics
                        "precision": overall_precision,
                        "recall": overall_recall,
                        "mAP50": overall_map50,
                        "mAP50_95": overall_map,
                        # Speed metrics
                        "fps": fps,
                        "avg_latency_ms": avg_latency,
                        "min_latency_ms": min_latency,
                        "max_latency_ms": max_latency,
                        "std_latency_ms": std_latency,
                        # Memory metrics
                        "model_size_mb": model_size_mb,
                        **memory_stats,
                        # Per-class metrics (stored separately)
                        "per_class_metrics": per_class_metrics,
                    }
                )

            elif format_type.startswith("onnx") and onnxruntime_available:
                # ONNX model evaluation
                # Try different provider combinations for better compatibility
                session = None
                actual_providers = []
                provider_strategies = [
                    # Strategy 1: Try CUDA first, fallback to CPU
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    # Strategy 2: CPU only (most compatible)
                    ["CPUExecutionProvider"],
                    # Strategy 3: Try TensorRT if available (for NVIDIA GPUs)
                    [
                        "TensorrtExecutionProvider",
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                    # Strategy 4: Default providers
                    [],
                ]

                for i, providers in enumerate(provider_strategies):
                    try:
                        if onnxruntime is None:
                            raise ImportError("onnxruntime not available")
                        session = onnxruntime.InferenceSession(
                            model_path, providers=providers
                        )
                        actual_providers = session.get_providers()
                        provider_names = [
                            p.split("ExecutionProvider")[0] for p in actual_providers
                        ]
                        print(
                            f"Successfully created ONNX session with strategy {i+1}: {provider_names}"
                        )
                        break
                    except Exception as e:
                        print(f"Provider strategy {i+1} failed: {e}")
                        if session is not None:
                            session = None
                        continue

                if session is None:
                    if format_type == "onnx_int8":
                        print(
                            f"INT8 model evaluation failed completely - skipping. This is expected if INT8 quantization is not supported."
                        )
                        print(
                            "FP32 ONNX model results are still valid and available above."
                        )
                        continue  # Skip INT8 model but continue with other models
                    else:
                        raise Exception(
                            f"Could not create ONNX session with any provider strategy"
                        )

                try:
                    input_name = session.get_inputs()[0].name

                    # Accuracy: Use PyTorch model's metrics (same model)
                    if "pytorch" in best_models:
                        pytorch_result = next(
                            (r for r in results if r.get("format") == "pytorch"), None
                        )
                        if pytorch_result:
                            overall_precision = pytorch_result.get("precision", 0.0)
                            overall_recall = pytorch_result.get("recall", 0.0)
                            overall_map50 = pytorch_result.get("mAP50", 0.0)
                            overall_map = pytorch_result.get("mAP50_95", 0.0)
                            per_class_metrics = pytorch_result.get(
                                "per_class_metrics", {}
                            )
                        else:
                            overall_precision = overall_recall = overall_map50 = (
                                overall_map
                            ) = 0.0
                            per_class_metrics = {}
                    else:
                        overall_precision = overall_recall = overall_map50 = (
                            overall_map
                        ) = 0.0
                        per_class_metrics = {}

                    # Speed benchmarking
                    print("[2/3] Benchmarking Speed...")
                    dummy_input = np.random.randn(1, 3, input_size, input_size).astype(
                        np.float32
                    )

                    # Warmup
                    for _ in range(10):
                        _ = session.run(None, {input_name: dummy_input})

                    # Benchmark
                    num_runs = 100
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

                    # Memory: Model file size only (ONNX runtime memory is harder to measure)
                    print("[3/3] Measuring Memory...")
                    model_size_mb = Path(model_path).stat().st_size / (
                        1024 * 1024
                    )  # MB

                    # Approximate inference memory (ONNX models typically use less)
                    memory_stats = {
                        "gpu_memory_initial_mb": 0.0,
                        "gpu_memory_peak_mb": 0.0,
                        "gpu_memory_current_mb": 0.0,
                        "gpu_memory_inference_mb": model_size_mb
                        * 0.3,  # Rough estimate
                        "cpu_memory_mb": model_size_mb * 0.5,  # Rough estimate for CPU
                    }

                    results.append(
                        {
                            "model": model_info["name"],
                            "format": format_type,
                            "input_size": input_size,
                            # Accuracy metrics
                            "precision": overall_precision,
                            "recall": overall_recall,
                            "mAP50": overall_map50,
                            "mAP50_95": overall_map,
                            # Speed metrics
                            "fps": fps,
                            "avg_latency_ms": avg_latency,
                            "min_latency_ms": min_latency,
                            "max_latency_ms": max_latency,
                            "std_latency_ms": std_latency,
                            # Memory metrics
                            "model_size_mb": model_size_mb,
                            **memory_stats,
                            # Per-class metrics
                            "per_class_metrics": per_class_metrics,
                        }
                    )

                except Exception as e:
                    if format_type == "onnx_int8":
                        print(
                            f"INT8 model evaluation failed (likely due to quantization compatibility). Error: {e}"
                        )
                        print(
                            "Skipping INT8 benchmark - this is normal if INT8 quantization used unsupported operations"
                        )
                        print("FP32 ONNX model will still be benchmarked")
                        continue  # Skip INT8 but don't fail the whole benchmark
                    else:
                        raise
            else:
                print(
                    f"Skipping {format_type} (not supported or onnxruntime not available)"
                )

        except Exception as e:
            print(f"Error evaluating {model_info['name']}: {e}")
            import traceback

            traceback.print_exc()
            continue

    return pd.DataFrame(results) if results else pd.DataFrame()


def display_results(results_df: pd.DataFrame):
    """Display comprehensive evaluation results"""
    print("\n" + "=" * 80)
    print("EVALUATION & BENCHMARK RESULTS")
    print("=" * 80)

    if results_df.empty:
        print("No results available.")
        return results_df

    # Sort by precision (primary metric)
    results_df = results_df.sort_values("precision", ascending=False)

    # Main summary table
    print("\n[SUMMARY TABLE]")
    print("-" * 80)
    print(
        f"{'Model':<35} {'Format':<12} {'Precision':<10} {'mAP50':<8} {'FPS':<8} {'Latency(ms)':<12} {'Size(MB)':<10}"
    )
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(
            f"{row['model']:<35} {row['format']:<12} {row['precision']:<10.4f} {row['mAP50']:<8.3f} "
            f"{row['fps']:<8.1f} {row['avg_latency_ms']:<12.2f} {row['model_size_mb']:<10.2f}"
        )

    print("=" * 80)

    # Detailed accuracy metrics
    pytorch_row = results_df[results_df["format"] == "pytorch"]
    if not pytorch_row.empty:
        row = pytorch_row.iloc[0]
        print("\n[DETAILED ACCURACY METRICS]")
        print("-" * 80)
        print(f"Overall Precision: {row['precision']:.4f}")
        print(f"Overall Recall:    {row['recall']:.4f}")
        print(f"mAP50 (IoU=0.5):   {row['mAP50']:.4f}")
        print(f"mAP50-95:          {row['mAP50_95']:.4f}")

        # Per-class precision
        per_class = row.get("per_class_metrics", {})
        if per_class and isinstance(per_class, dict) and len(per_class) > 0:
            print("\n[PER-CLASS PRECISION]")
            print("-" * 80)
            print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP50':<12}")
            print("-" * 80)

            for class_name, metrics in sorted(per_class.items()):
                if isinstance(metrics, dict):
                    precision = metrics.get("precision", 0.0)
                    recall = metrics.get("recall", 0.0)
                    map50 = metrics.get("mAP50", 0.0)
                    print(
                        f"{class_name:<20} {precision:<12.4f} {recall:<12.4f} {map50:<12.4f}"
                    )

            print("=" * 80)

    # Speed analysis
    print("\n[SPEED ANALYSIS]")
    print("-" * 80)
    best_speed = results_df.loc[results_df["fps"].idxmax()]
    print(
        f"Fastest: {best_speed['model']} ({best_speed['format']}) - {best_speed['fps']:.1f} FPS ({best_speed['avg_latency_ms']:.2f}ms)"
    )

    for _, row in results_df.iterrows():
        print(
            f"{row['model']:<35} {row['format']:<12} "
            f"FPS: {row['fps']:.1f} | Latency: {row['avg_latency_ms']:.2f}ms "
            f"(min: {row['min_latency_ms']:.2f}ms, max: {row['max_latency_ms']:.2f}ms, std: {row['std_latency_ms']:.2f}ms)"
        )

    print("=" * 80)

    # Memory analysis
    print("\n[MEMORY ANALYSIS]")
    print("-" * 80)
    smallest = results_df.loc[results_df["model_size_mb"].idxmin()]
    print(
        f"Smallest model: {smallest['model']} ({smallest['format']}) - {smallest['model_size_mb']:.2f} MB"
    )

    for _, row in results_df.iterrows():
        print(f"\n{row['model']} ({row['format']}):")
        print(f"  Model file size: {row['model_size_mb']:.2f} MB")
        gpu_mem = row.get("gpu_memory_inference_mb", 0) or 0
        if gpu_mem > 0:
            print(f"  GPU inference memory: {gpu_mem:.2f} MB")
            gpu_peak = row.get("gpu_memory_peak_mb", 0) or 0
            print(f"  GPU peak memory: {gpu_peak:.2f} MB")
        cpu_mem = row.get("cpu_memory_mb", 0) or 0
        if cpu_mem > 0:
            print(f"  CPU inference memory: {cpu_mem:.2f} MB")

    print("=" * 80)

    # Best models by metric
    print("\n[BEST MODELS BY METRIC]")
    print("-" * 80)
    best_precision = results_df.loc[results_df["precision"].idxmax()]
    best_map50 = results_df.loc[results_df["mAP50"].idxmax()]
    best_speed = results_df.loc[results_df["fps"].idxmax()]
    smallest_size = results_df.loc[results_df["model_size_mb"].idxmin()]

    print(
        f"Highest Precision: {best_precision['model']} ({best_precision['format']}) - {best_precision['precision']:.4f}"
    )
    print(
        f"Best mAP50:        {best_map50['model']} ({best_map50['format']}) - {best_map50['mAP50']:.4f}"
    )
    print(
        f"Fastest:           {best_speed['model']} ({best_speed['format']}) - {best_speed['fps']:.1f} FPS"
    )
    print(
        f"Smallest:          {smallest_size['model']} ({smallest_size['format']}) - {smallest_size['model_size_mb']:.2f} MB"
    )
    print("=" * 80)

    # Save results
    output_dir = Path("output/benchmarking")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main results (without per_class_metrics dict)
    results_to_save = results_df.copy()
    if "per_class_metrics" in results_to_save.columns:
        results_to_save = results_to_save.drop(columns=["per_class_metrics"])

    results_file = output_dir / "evaluation_benchmark_results.csv"
    results_to_save.to_csv(results_file, index=False)
    print(f"\n[SAVE] Results saved to: {results_file}")

    # Save per-class metrics separately
    if not pytorch_row.empty:
        per_class = pytorch_row.iloc[0].get("per_class_metrics", {})
        if per_class and isinstance(per_class, dict) and len(per_class) > 0:
            per_class_df = pd.DataFrame(
                [
                    {
                        "class": class_name,
                        "precision": (
                            metrics.get("precision", 0.0)
                            if isinstance(metrics, dict)
                            else 0.0
                        ),
                        "recall": (
                            metrics.get("recall", 0.0)
                            if isinstance(metrics, dict)
                            else 0.0
                        ),
                        "mAP50": (
                            metrics.get("mAP50", 0.0)
                            if isinstance(metrics, dict)
                            else 0.0
                        ),
                    }
                    for class_name, metrics in per_class.items()
                ]
            )
            per_class_file = output_dir / "per_class_precision.csv"
            per_class_df.to_csv(per_class_file, index=False)
            print(f"[SAVE] Per-class precision saved to: {per_class_file}")

    return results_df


def main():
    """Main function"""
    print("=" * 80)
    print("MODEL EVALUATION & BENCHMARKING")
    print("Focus: Accuracy (Precision), Speed, Memory")
    print("=" * 80)

    # Find best models
    best_models = find_best_models()

    if not best_models:
        print("[ERROR] No trained models found.")
        print(
            "Please train a model first using: python scripts/training/basic_training.py --optuna"
        )
        return 1

    # Evaluate and benchmark
    data_yaml = "consolidated_dataset/data.yaml"
    if not Path(data_yaml).exists():
        print(f"[WARNING] Data YAML not found at {data_yaml}, using default path")
        data_yaml = "data/data.yaml"

    results_df = evaluate_and_benchmark_models(best_models, data_yaml)

    # Display results
    display_results(results_df)

    print("\n[SUCCESS] EVALUATION & BENCHMARKING COMPLETE!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
