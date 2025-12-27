#!/usr/bin/env python3
"""
Quantize all three optimal models (MobileNet-ViT, EfficientNet, YOLOv8s) and benchmark them
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from trainer import ModelTrainer


def get_optimal_models():
    """Get all three optimal models trained at 320x320"""
    models = [
        {
            'name': 'MobileNet-ViT',
            'path': 'output/models/mobilenet_vit_rover_opt_320/weights/best.pt',
            'input_size': 320,
            'architecture': 'mobilenet_vit'
        },
        {
            'name': 'EfficientNet',
            'path': 'output/models/efficientnet_rover_opt_320/weights/best.pt',
            'input_size': 320,
            'architecture': 'efficientnet'
        },
        {
            'name': 'YOLOv8s',
            'path': 'output/models/yolov8s_rover_opt_320/weights/best.pt',
            'input_size': 320,
            'architecture': 'yolov8s'
        }
    ]

    # Verify models exist
    valid_models = []
    for model in models:
        if os.path.exists(model['path']):
            valid_models.append(model)
            size_mb = os.path.getsize(model['path']) / (1024 * 1024)
            print(".1f"        else:
            print(f"[WARNING]  Model not found: {model['path']}")

    return valid_models


def quantize_all_models(models, output_base_dir="output/quantized_all"):
    """Quantize all three optimal models"""
    print(f"\n{'='*60}")
    print("QUANTIZING ALL THREE OPTIMAL MODELS")
    print(f"{'='*60}")

    trainer = ModelTrainer({'training': {'device': 'cpu'}})
    quantized_models = []

    for model in models:
        print(f"\n[UPDATE] Processing: {model['name']}")
        print("-" * 40)

        # Create model-specific output directory
        model_output_dir = os.path.join(output_base_dir, model['name'].lower().replace('-', '_'))

        try:
            # Quantize and evaluate
            result = trainer.quantize_and_evaluate(
                model['path'],
                'consolidated_dataset/data.yaml',
                model['input_size'],
                model_output_dir
            )

            if result and 'quantized' in result:
                # Store quantized model path
                onnx_path = os.path.join(model_output_dir, "best.onnx")
                quantized_path = os.path.join(model_output_dir, "best_int8.onnx")

                quantized_model = {
                    'original_name': model['name'],
                    'quantized_name': f"{model['name']}_INT8",
                    'original_path': model['path'],
                    'onnx_path': onnx_path,
                    'quantized_path': quantized_path,
                    'input_size': model['input_size'],
                    'original_metrics': result.get('original', {}),
                    'quantized_metrics': result.get('quantized', {}),
                    'trade_offs': result.get('trade_offs', {})
                }

                quantized_models.append(quantized_model)
                print(f"[SUCCESS] Successfully quantized: {model['name']}")
                print(f"   ONNX: {onnx_path}")
                print(f"   INT8: {quantized_path}")
            else:
                print(f"[ERROR] Failed to quantize: {model['name']}")

        except Exception as e:
            print(f"[ERROR] Quantization failed for {model['name']}: {str(e)}")
            import traceback
            traceback.print_exc()

    return quantized_models


def benchmark_all_models(original_models, quantized_models):
    """Benchmark all models (original + quantized)"""
    print(f"\n{'='*60}")
    print("COMPREHENSIVE BENCHMARKING: ORIGINAL VS QUANTIZED")
    print(f"{'='*60}")

    trainer = ModelTrainer({'training': {'device': 'cpu'}})

    # Create combined model list for benchmarking
    all_models = []

    # Add original models
    for model in original_models:
        all_models.append({
            'name': model['name'],
            'path': model['path'],
            'input_size': model['input_size'],
            'type': 'Original'
        })

    # Add ONNX FP32 models
    for qm in quantized_models:
        if os.path.exists(qm['onnx_path']):
            all_models.append({
                'name': f"{qm['original_name']}_ONNX",
                'path': qm['onnx_path'],
                'input_size': qm['input_size'],
                'type': 'ONNX_FP32'
            })

    # Add INT8 quantized models
    for qm in quantized_models:
        if os.path.exists(qm['quantized_path']):
            all_models.append({
                'name': f"{qm['original_name']}_INT8",
                'path': qm['quantized_path'],
                'input_size': qm['input_size'],
                'type': 'INT8'
            })

    print(f"Benchmarking {len(all_models)} models:")
    for model in all_models:
        print(f"  • {model['name']} ({model['type']})")

    # Run comprehensive benchmarking
    data_yaml = 'consolidated_dataset/data.yaml'
    benchmark_df = trainer.benchmark_rover_models(all_models, data_yaml)

    return benchmark_df, all_models


def generate_comprehensive_report(original_models, quantized_models, benchmark_df):
    """Generate comprehensive comparison report"""
    print(f"\n{'='*80}")
    print("[ANALYSIS] COMPREHENSIVE MODEL ARCHITECTURE COMPARISON REPORT")
    print(f"{'='*80}")

    print("\n[SEARCH] MODEL ARCHITECTURE CLARIFICATION:")
    print("-" * 50)
    print("All three 'architectures' actually use YOLOv8 as the base model:")
    print("• MobileNet-ViT → YOLOv8 Nano with MobileNet-style training")
    print("• EfficientNet → YOLOv8 Small with EfficientNet-style training")
    print("• YOLOv8s → YOLOv8 Small with standard training")
    print("Different performance due to training hyperparameters, not architecture!")

    # Accuracy Comparison
    print(f"\n[TARGET] ACCURACY COMPARISON (mAP50 on real dataset)")
    print("-" * 50)
    accuracy_data = []

    for model in original_models:
        model_name = model['name']
        # Find corresponding quantized results
        quantized = next((qm for qm in quantized_models if qm['original_name'] == model_name), None)

        if quantized:
            orig_metrics = quantized['original_metrics']
            quant_metrics = quantized['quantized_metrics']

            accuracy_data.append({
                'Model': model_name,
                'Original_mAP50': orig_metrics.get('mAP50', 0),
                'INT8_mAP50': quant_metrics.get('mAP50', 0),
                'Accuracy_Loss': orig_metrics.get('mAP50', 0) - quant_metrics.get('mAP50', 0)
            })

    if accuracy_data:
        acc_df = pd.DataFrame(accuracy_data)
        print(acc_df.to_string(index=False, float_format='%.3f'))

    # Performance Comparison from benchmark
    print(f"\n[PERFORMANCE] PERFORMANCE COMPARISON")
    print("-" * 50)

    # Group by original model name
    for orig_model in original_models:
        model_base = orig_model['name']
        print(f"\n{model_base}:")
        print("-" * 20)

        # Get all variants for this model
        variants = benchmark_df[benchmark_df['model'].str.contains(model_base)]

        if not variants.empty:
            perf_data = []
            for _, row in variants.iterrows():
                variant_name = row['model'].replace(f"{model_base}", "").replace("_", "").strip()
                if not variant_name:
                    variant_name = "Original"
                elif "ONNX" in variant_name:
                    variant_name = "ONNX FP32"
                elif "INT8" in variant_name:
                    variant_name = "INT8"

                perf_data.append({
                    'Variant': variant_name,
                    'FPS': row['fps'],
                    'Latency_ms': row['avg_latency_ms'],
                    'Size_MB': row['model_size_mb']
                })

            perf_df = pd.DataFrame(perf_data)
            print(perf_df.to_string(index=False, float_format='%.1f'))

    # Trade-off Analysis
    print(f"\n{'='*80}")
    print("[RESULTS] QUANTIZATION TRADE-OFF ANALYSIS")
    print(f"{'='*80}")

    for qm in quantized_models:
        print(f"\n[MEMORY] {qm['original_name']} QUANTIZATION RESULTS:")
        print("-" * 40)

        trade_offs = qm.get('trade_offs', {})

        print(".1f"        print(".1f"        print(".1f"
        # Recommendation
        accuracy_loss = trade_offs.get('accuracy_loss', 0)
        size_reduction = trade_offs.get('size_reduction', 0)

        if size_reduction > 2 and accuracy_loss < 0.05:  # Good trade-off
            print("[SUCCESS] RECOMMENDATION: Excellent quantization candidate")
        elif size_reduction > 1 and accuracy_loss < 0.10:  # Acceptable
            print("👍 RECOMMENDATION: Good quantization candidate")
        else:
            print("[ERROR] RECOMMENDATION: Consider keeping original")

    # Overall Summary
    print(f"\n{'='*80}")
    print("[BEST] FINAL RECOMMENDATIONS")
    print(f"{'='*80}")

    # Find best overall model
    if not benchmark_df.empty:
        # Sort by efficiency (mAP50 / (size * latency))
        benchmark_df['efficiency'] = benchmark_df['mAP50'] / (benchmark_df['model_size_mb'] * benchmark_df['avg_latency_ms'])

        best_overall = benchmark_df.loc[benchmark_df['efficiency'].idxmax()]
        print("[TARGET] MOST EFFICIENT MODEL:"        print(".3f"        print(".1f"        print(".1f"        print(".4f"
        # Best quantized model
        quantized_only = benchmark_df[benchmark_df['model'].str.contains('INT8')]
        if not quantized_only.empty:
            best_quantized = quantized_only.loc[quantized_only['efficiency'].idxmax()]
            print("\n[UPDATE] BEST QUANTIZED MODEL:")
            print(f"   {best_quantized['model']}")
            print(".3f"            print(".1f"            print(".1f"            print(".4f"
    # Save detailed results
    try:
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

        benchmark_df.to_csv(output_dir / 'comprehensive_model_comparison.csv', index=False)

        # Create summary DataFrame
        summary_data = []
        for qm in quantized_models:
            summary_data.append({
                'model': qm['original_name'],
                'orig_map50': qm['original_metrics'].get('mAP50', 0),
                'int8_map50': qm['quantized_metrics'].get('mAP50', 0),
                'accuracy_loss_percent': (
                    (qm['original_metrics'].get('mAP50', 0) - qm['quantized_metrics'].get('mAP50', 0)) /
                    max(qm['original_metrics'].get('mAP50', 0), 0.001) * 100
                ),
                'size_reduction_mb': qm['trade_offs'].get('size_reduction', 0),
                'size_reduction_percent': (
                    qm['trade_offs'].get('size_reduction', 0) /
                    max(qm['original_metrics'].get('file_size_mb', 0), 0.001) * 100
                )
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'quantization_summary.csv', index=False)

        print("\n[SAVE] Results saved:")
        print(f"   • {output_dir / 'comprehensive_model_comparison.csv'}")
        print(f"   • {output_dir / 'quantization_summary.csv'}")
    except Exception as e:
        print(f"[WARNING]  Failed to save results: {e}")


def main():
    """Main function"""
    print("[STARTING] COMPREHENSIVE MODEL QUANTIZATION & BENCHMARKING")
    print("="*60)

    # Get all optimal models
    optimal_models = get_optimal_models()
    if not optimal_models:
        print("[ERROR] No optimal models found. Run rover optimization first.")
        return 1

    # Quantize all models
    quantized_models = quantize_all_models(optimal_models)

    if not quantized_models:
        print("[ERROR] No models were successfully quantized.")
        return 1

    # Benchmark all models
    benchmark_df, all_models = benchmark_all_models(optimal_models, quantized_models)

    # Generate comprehensive report
    generate_comprehensive_report(optimal_models, quantized_models, benchmark_df)

    print(f"\n[SUCCESS] COMPREHENSIVE ANALYSIS COMPLETE!")
    print("All models quantized and benchmarked on real espresso dataset")

    return 0


if __name__ == '__main__':
    sys.exit(main())
