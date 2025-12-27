#!/usr/bin/env python3
"""
Detailed memory usage analysis for the trained models
"""

import sys
import os
import psutil
import tracemalloc
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from trainer import ModelTrainer


def get_system_memory_info():
    """Get system memory information"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024**3),
        'available_gb': mem.available / (1024**3),
        'used_gb': mem.used / (1024**3),
        'percentage': mem.percent
    }


def analyze_model_memory_footprint():
    """Analyze memory footprint of different model formats"""
    print("[MEMORY] DETAILED MEMORY ANALYSIS")
    print("="*50)

    # System memory info
    system_mem = get_system_memory_info()
    print("💻 System Memory:")
    print(f"   Total RAM: {system_mem['total_gb']:.1f} GB")
    print(f"   Available: {system_mem['available_gb']:.1f} GB")
    print(f"   Used: {system_mem['used_gb']:.1f} GB")
    print(f"   Usage: {system_mem['percentage']:.1f}%")
    print()

    trainer = ModelTrainer({'training': {'device': 'cpu'}})

    models_to_analyze = [
        {
            'path': 'output/models/mobilenet_vit_rover_opt_320/weights/best.pt',
            'name': 'PyTorch FP32',
            'format': 'pytorch'
        },
        {
            'path': 'output/quantized_test/best.onnx',
            'name': 'ONNX FP32',
            'format': 'onnx'
        },
        {
            'path': 'output/quantized_test/best_int8.onnx',
            'name': 'ONNX INT8',
            'format': 'onnx'
        }
    ]

    results = []

    for model_config in models_to_analyze:
        model_path = model_config['path']
        model_name = model_config['name']
        model_format = model_config['format']

        if not os.path.exists(model_path):
            print(f"[WARNING]  Model not found: {model_path}")
            continue

        print(f"[ANALYSIS] Analyzing {model_name}...")

        # Get file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        # Measure loading memory
        tracemalloc.start()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

        try:
            if model_format == 'pytorch':
                # Load PyTorch model
                model = torch.load(model_path, map_location='cpu')
                loaded_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                model_memory_mb = loaded_memory - initial_memory

                # Measure inference memory
                dummy_input = torch.randn(1, 3, 320, 320)
                peak_memory = loaded_memory

                for _ in range(10):
                    with torch.no_grad():
                        _ = model(dummy_input)
                    current_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    peak_memory = max(peak_memory, current_mem)

                inference_memory_mb = peak_memory - loaded_memory

            else:  # ONNX
                import onnxruntime as ort

                # Load ONNX model
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                loaded_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                model_memory_mb = loaded_memory - initial_memory

                # Measure inference memory
                dummy_input = {session.get_inputs()[0].name: torch.randn(1, 3, 320, 320).numpy()}
                peak_memory = loaded_memory

                for _ in range(10):
                    _ = session.run(None, dummy_input)
                    current_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    peak_memory = max(peak_memory, current_mem)

                inference_memory_mb = peak_memory - loaded_memory

        except Exception as e:
            print(f"[ERROR] Error analyzing {model_name}: {str(e)}")
            tracemalloc.stop()
            continue

        tracemalloc.stop()

        # Calculate totals
        total_memory_mb = model_memory_mb + inference_memory_mb

        result = {
            'model': model_name,
            'file_size_mb': file_size_mb,
            'model_memory_mb': model_memory_mb,
            'inference_memory_mb': inference_memory_mb,
            'total_memory_mb': total_memory_mb,
            'peak_memory_mb': peak_memory
        }

        results.append(result)

        print(f"   File Size: {file_size_mb:.1f} MB")
        print(f"   Model Memory: {model_memory_mb:.1f} MB")
        print(f"   Inference Memory: {inference_memory_mb:.1f} MB")
        print(f"   Total Memory: {total_memory_mb:.1f} MB")
        print(f"   Peak Memory: {peak_memory:.1f} MB")
        print()

    # Comparison summary
    if results:
        print("[ANALYSIS] MEMORY COMPARISON SUMMARY")
        print("-" * 40)

        # Create comparison table
        print("<20")
        print("-" * 60)

        for result in results:
            print("<20")

        print("\n[TIP] INSIGHTS:")
        print("-" * 20)

        # Find most memory efficient
        most_efficient = min(results, key=lambda x: x['total_memory_mb'])
        print(f"• Most memory efficient: {most_efficient['model']} ({most_efficient['total_memory_mb']:.1f} MB total)")

        # Check if models fit in typical constraints
        for result in results:
            if result['total_memory_mb'] < 100:  # Less than 100MB
                print(f"• {result['model']}: Suitable for mobile/embedded devices")
            elif result['total_memory_mb'] < 500:  # Less than 500MB
                print(f"• {result['model']}: Suitable for desktop/workstation")
            else:
                print(f"• {result['model']}: High memory requirements")

        # Quantization savings
        pytorch_result = next((r for r in results if 'PyTorch' in r['model']), None)
        int8_result = next((r for r in results if 'INT8' in r['model']), None)

        if pytorch_result and int8_result:
            memory_savings = pytorch_result['total_memory_mb'] - int8_result['total_memory_mb']
            savings_percent = (memory_savings / pytorch_result['total_memory_mb']) * 100
            print(f"• Quantization memory savings: {memory_savings:.1f} MB ({savings_percent:.1f}%)")
    return results


def main():
    """Main function"""
    try:
        results = analyze_model_memory_footprint()

        print("\n[SUCCESS] MEMORY ANALYSIS COMPLETE")
        print("All measurements taken on CPU with 320x320 input resolution")

        # Save results
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            results_file = output_dir / 'memory_analysis_results.csv'
            df.to_csv(results_file, index=False)
            print(f"[SAVE] Results saved to: {results_file}")

    except Exception as e:
        print(f"[ERROR] Memory analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    sys.exit(main())
