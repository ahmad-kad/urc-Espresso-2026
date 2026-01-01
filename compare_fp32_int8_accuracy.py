#!/usr/bin/env python3
"""
Compare accuracy between FP32 and INT8 ONNX models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from deployment_package.tools.evaluate_accuracy_per_class import (
    load_ground_truth_labels,
    evaluate_onnx_model,
    calculate_per_class_metrics,
    print_comparison_results
)

def main():
    print("=" * 80)
    print("FP32 vs INT8 ONNX MODEL ACCURACY COMPARISON")
    print("=" * 80)
    
    # Model paths
    fp32_model = "output/onnx/best_fp32.onnx"
    int8_model = "output/onnx/best_int8.onnx"
    dataset = "consolidated_dataset"
    input_size = 416
    
    # Check models exist
    if not Path(fp32_model).exists():
        print(f"Error: FP32 model not found: {fp32_model}")
        return 1
    
    if not Path(int8_model).exists():
        print(f"Error: INT8 model not found: {int8_model}")
        return 1
    
    print(f"\nFP32 Model: {fp32_model}")
    print(f"INT8 Model: {int8_model}")
    print(f"Dataset: {dataset}")
    print(f"Input Size: {input_size}x{input_size}\n")
    
    # Load ground truth
    label_dir = Path(dataset) / 'val' / 'labels'
    ground_truth = load_ground_truth_labels(label_dir)
    image_paths = list(ground_truth.keys())
    
    print(f"Evaluating {len(image_paths)} images...\n")
    
    # Evaluate FP32 model
    print("Evaluating FP32 model...")
    fp32_predictions = evaluate_onnx_model(
        fp32_model, image_paths, ground_truth,
        input_size=input_size,
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Evaluate INT8 model
    print("Evaluating INT8 model...")
    int8_predictions = evaluate_onnx_model(
        int8_model, image_paths, ground_truth,
        input_size=input_size,
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Calculate metrics
    fp32_metrics = calculate_per_class_metrics(ground_truth, fp32_predictions, 0.45)
    int8_metrics = calculate_per_class_metrics(ground_truth, int8_predictions, 0.45)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("FP32 vs INT8 ACCURACY COMPARISON")
    print("=" * 80)
    print_comparison_results(fp32_metrics, int8_metrics)
    
    # Calculate differences
    print("\n" + "=" * 80)
    print("ACCURACY DIFFERENCE SUMMARY (FP32 - INT8)")
    print("=" * 80)
    
    # Calculate averages (only for classes with data)
    fp32_f1_values = [m.get('f1_score', 0) for m in fp32_metrics.values() if m.get('f1_score', 0) > 0 or m.get('total_predictions', 0) > 0]
    int8_f1_values = [m.get('f1_score', 0) for m in int8_metrics.values() if m.get('f1_score', 0) > 0 or m.get('total_predictions', 0) > 0]
    
    fp32_avg_f1 = sum(fp32_f1_values) / len(fp32_f1_values) if fp32_f1_values else 0
    int8_avg_f1 = sum(int8_f1_values) / len(int8_f1_values) if int8_f1_values else 0
    
    # Get AP from the metrics - check different possible keys
    fp32_ap_values = []
    int8_ap_values = []
    for class_name in fp32_metrics.keys():
        fp32_ap = fp32_metrics[class_name].get('average_precision', fp32_metrics[class_name].get('ap', 0))[1] if isinstance(fp32_metrics[class_name].get('average_precision'), tuple) else fp32_metrics[class_name].get('average_precision', fp32_metrics[class_name].get('ap', 0))
        int8_ap = int8_metrics[class_name].get('average_precision', int8_metrics[class_name].get('ap', 0)) if not isinstance(int8_metrics[class_name].get('average_precision'), tuple) else int8_metrics[class_name].get('average_precision')[1]
        if fp32_ap > 0 or int8_ap > 0:
            fp32_ap_values.append(fp32_ap)
            int8_ap_values.append(int8_ap)
    
    fp32_avg_ap = sum(fp32_ap_values) / len(fp32_ap_values) if fp32_ap_values else 0
    int8_avg_ap = sum(int8_ap_values) / len(int8_ap_values) if int8_ap_values else 0
    
    print(f"\nOverall Metrics:")
    print(f"  Average F1-Score:")
    print(f"    FP32: {fp32_avg_f1:.4f}")
    print(f"    INT8: {int8_avg_f1:.4f}")
    print(f"    Difference: {fp32_avg_f1 - int8_avg_f1:+.4f} ({((fp32_avg_f1 - int8_avg_f1) / fp32_avg_f1 * 100):+.2f}%)")
    
    print(f"\n  Average Precision (AP):")
    print(f"    FP32: {fp32_avg_ap:.4f}")
    print(f"    INT8: {int8_avg_ap:.4f}")
    if fp32_avg_ap > 0:
        print(f"    Difference: {fp32_avg_ap - int8_avg_ap:+.4f} ({((fp32_avg_ap - int8_avg_ap) / fp32_avg_ap * 100):+.2f}%)")
    else:
        print(f"    Difference: {fp32_avg_ap - int8_avg_ap:+.4f} (N/A - no AP data)")
    
    print("\n" + "=" * 80)
    print("PER-CLASS DIFFERENCES:")
    print("=" * 80)
    print(f"{'Class':<15} {'FP32 F1':<12} {'INT8 F1':<12} {'Difference':<12} {'FP32 AP':<12} {'INT8 AP':<12} {'AP Diff':<12}")
    print("-" * 80)
    
    for class_name in sorted(fp32_metrics.keys()):
        fp32_f1 = fp32_metrics[class_name].get('f1_score', 0)
        int8_f1 = int8_metrics[class_name].get('f1_score', 0)
        f1_diff = fp32_f1 - int8_f1
        
        fp32_ap = fp32_metrics[class_name].get('average_precision', fp32_metrics[class_name].get('ap', 0))
        if isinstance(fp32_ap, tuple):
            fp32_ap = fp32_ap[1] if len(fp32_ap) > 1 else 0
        int8_ap = int8_metrics[class_name].get('average_precision', int8_metrics[class_name].get('ap', 0))
        if isinstance(int8_ap, tuple):
            int8_ap = int8_ap[1] if len(int8_ap) > 1 else 0
        ap_diff = fp32_ap - int8_ap
        
        print(f"{class_name:<15} {fp32_f1:<12.4f} {int8_f1:<12.4f} {f1_diff:+.4f}        {fp32_ap:<12.4f} {int8_ap:<12.4f} {ap_diff:+.4f}")
    
    print("=" * 80)
    
    # Conclusion
    f1_loss = ((fp32_avg_f1 - int8_avg_f1) / fp32_avg_f1) * 100 if fp32_avg_f1 > 0 else 0
    ap_loss = ((fp32_avg_ap - int8_avg_ap) / fp32_avg_ap) * 100 if fp32_avg_ap > 0 else 0
    
    print(f"\nCONCLUSION:")
    print(f"  F1-Score Loss: {f1_loss:.2f}%")
    print(f"  AP Loss: {ap_loss:.2f}%")
    
    if f1_loss < 1.0 and ap_loss < 1.0:
        print(f"\n  INT8 quantization maintains >99% accuracy!")
        print(f"  Size reduction: 74.5% (42.59 MB -> 10.87 MB)")
        print(f"  INT8 is recommended for deployment with minimal accuracy loss.")
    elif f1_loss < 2.0 and ap_loss < 2.0:
        print(f"\n  INT8 quantization has <2% accuracy loss.")
        print(f"  Size reduction: 74.5% (42.59 MB -> 10.87 MB)")
        print(f"  INT8 is acceptable for deployment.")
    else:
        print(f"\n  INT8 quantization has >2% accuracy loss.")
        print(f"  Consider FP16 or FP32 for better accuracy.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

