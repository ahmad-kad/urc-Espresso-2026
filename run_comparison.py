#!/usr/bin/env python3
"""
Run benchmark comparison between PyTorch and ONNX models
"""

import sys
from pathlib import Path
import yaml
from scripts.evaluation.model_evaluator import ModelEvaluator

def load_config_direct(config_path: str):
    """Load config directly from file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("=" * 60)
    print("MODEL BENCHMARK COMPARISON")
    print("=" * 60)
    
    # Load configuration
    print("\nLoading configuration...")
    config_path = Path("configs/default.yaml")
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1
    
    config = load_config_direct(str(config_path))
    input_size = config.get('model', {}).get('input_size', 416)
    print(f"Using input size: {input_size}x{input_size}")
    
    # Find models
    print("\nLocating models...")
    pytorch_model = Path("output/output/models/object_detection_pipeline_test5/weights/best.pt")
    onnx_model = Path("output/onnx/best_fp32.onnx")
    
    # Try alternative paths
    if not pytorch_model.exists():
        # Try other possible locations
        possible_paths = [
            Path("output/models/object_detection_pipeline_test/weights/best.pt"),
            Path("output/models/object_detection_pipeline_test5/weights/best.pt"),
        ]
        for path in possible_paths:
            if path.exists():
                pytorch_model = path
                break
    
    if not pytorch_model.exists():
        print(f"PyTorch model not found. Checked: {pytorch_model}")
        return 1
    
    if not onnx_model.exists():
        # Try alternative ONNX path
        alt_onnx = Path("output/output/models/object_detection_pipeline_test5/weights/best.onnx")
        if alt_onnx.exists():
            onnx_model = alt_onnx
        else:
            print(f"ONNX model not found. Checked: {onnx_model}")
            return 1
    
    print(f"PyTorch model: {pytorch_model}")
    print(f"ONNX model: {onnx_model}")
    
    # Run comparison
    print("\n" + "=" * 60)
    print("BENCHMARKING MODELS")
    print("=" * 60)
    
    try:
        # Evaluate PyTorch model
        print("\nEvaluating PyTorch model...")
        evaluator_pt = ModelEvaluator(str(pytorch_model), model_type="pytorch")
        results_pt = evaluator_pt.benchmark_inference_speed(num_runs=100, input_size=input_size)
        print(f"  PyTorch - FPS: {results_pt['fps']:.2f}, Avg time: {results_pt['avg_inference_time']*1000:.2f}ms")
        
        # Evaluate ONNX model
        print("\nEvaluating ONNX model...")
        evaluator_onnx = ModelEvaluator(str(onnx_model), model_type="onnx")
        results_onnx = evaluator_onnx.benchmark_inference_speed(num_runs=100, input_size=input_size)
        print(f"  ONNX - FPS: {results_onnx['fps']:.2f}, Avg time: {results_onnx['avg_inference_time']*1000:.2f}ms")
        
        # Comparison
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        speedup = results_onnx['fps'] / results_pt['fps'] if results_pt['fps'] > 0 else 0
        print(f"PyTorch FPS:     {results_pt['fps']:.2f}")
        print(f"ONNX FPS:        {results_onnx['fps']:.2f}")
        print(f"Speedup:         {speedup:.2f}x")
        print(f"PyTorch Latency: {results_pt['avg_inference_time']*1000:.2f}ms")
        print(f"ONNX Latency:    {results_onnx['avg_inference_time']*1000:.2f}ms")
        print("=" * 60)
        
        if speedup > 1.1:
            print("\n✓ ONNX is faster!")
        elif speedup < 0.9:
            print("\n✓ PyTorch is faster!")
        else:
            print("\n✓ Performance is similar!")
        
        # Additional metrics
        print("\nDetailed Metrics:")
        print(f"  PyTorch - Min: {results_pt.get('min_time', 0)*1000:.2f}ms, Max: {results_pt.get('max_time', 0)*1000:.2f}ms, Std: {results_pt.get('std_time', 0)*1000:.2f}ms")
        print(f"  ONNX    - Min: {results_onnx.get('min_time', 0)*1000:.2f}ms, Max: {results_onnx.get('max_time', 0)*1000:.2f}ms, Std: {results_onnx.get('std_time', 0)*1000:.2f}ms")
        
    except Exception as e:
        print(f"\nComparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✓ Benchmark comparison completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
