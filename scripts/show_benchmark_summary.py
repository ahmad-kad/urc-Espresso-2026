#!/usr/bin/env python3
"""Quick script to display benchmark summary"""

import json
from pathlib import Path

# Load results
results_file = Path("output/final_model_comparison.json")
with open(results_file) as f:
    data = json.load(f)

# Sort by FPS
sorted_data = sorted(data, key=lambda x: x['fps'], reverse=True)

print("=" * 80)
print("COMPREHENSIVE BENCHMARK RESULTS - TOP 5 MODELS")
print("=" * 80)
print()

print("TOP 5 BY FPS:")
print("-" * 80)
for i, m in enumerate(sorted_data[:5], 1):
    print(f"{i}. {m['model']:30s} ({m['format']:8s}) - {m['fps']:6.1f} FPS | "
          f"{m['avg_inference_ms']:5.2f}ms | {m['model_size_mb']:5.1f}MB | "
          f"Efficiency: {m['efficiency_score']:.2f}")

print()
print("BEST MODEL FOR IMX500:")
print("-" * 80)
best = sorted_data[0]
print(f"Model: {best['model']}")
print(f"Format: {best['format']}")
print(f"Input Size: {best['input_size']}x{best['input_size']}")
print(f"FPS: {best['fps']:.1f}")
print(f"Latency: {best['avg_inference_ms']:.2f} ms")
print(f"Model Size: {best['model_size_mb']:.1f} MB")
print(f"Size Efficiency: {best['size_efficiency']:.2f} FPS/MB")

print()
print("ONNX vs PyTorch Comparison (Top Models):")
print("-" * 80)
onnx_models = [m for m in sorted_data if m['format'] == 'ONNX'][:5]
pytorch_models = [m for m in sorted_data if m['format'] == 'PyTorch'][:5]

print("\nTop 5 ONNX:")
for i, m in enumerate(onnx_models, 1):
    print(f"  {i}. {m['model']:30s} - {m['fps']:6.1f} FPS")

print("\nTop 5 PyTorch:")
for i, m in enumerate(pytorch_models, 1):
    print(f"  {i}. {m['model']:30s} - {m['fps']:6.1f} FPS")

print()
print(f"Full results saved to: {results_file}")
print(f"Summary report: output/benchmark_summary_report.md")
