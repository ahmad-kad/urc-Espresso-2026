#!/usr/bin/env python3
"""
Evaluate models and extract per-class accuracy metrics
"""

import sys
import os
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model_per_class(model_path, model_name, data_yaml="consolidated_dataset/data.yaml"):
    """Evaluate a single model and extract per-class metrics"""
    logger.info(f"Evaluating {model_name} for per-class accuracy...")

    try:
        # Load model
        model = YOLO(model_path)

        # Run validation
        results = model.val(data=data_yaml, conf=0.1, iou=0.6, verbose=False)

        # Extract overall metrics
        overall_metrics = {
            'model': model_name,
            'mAP50': float(results.box.map50),
            'mAP75': float(getattr(results.box, 'map75', 0)),
            'mAP50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        }

        # Extract per-class metrics
        per_class_data = []
        class_names = ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']

        if hasattr(results.box, 'class_result') and results.box.class_result is not None:
            class_results = results.box.class_result
            if hasattr(class_results, '__len__') and len(class_results) > 0:
                for i, class_name in enumerate(class_names):
                    if i < len(class_results):
                        class_metrics = class_results[i]
                        if hasattr(class_metrics, '__len__') and len(class_metrics) >= 4:
                            per_class_data.append({
                                'model': model_name,
                                'class': class_name,
                                'precision': float(class_metrics[0]),
                                'recall': float(class_metrics[1]),
                                'mAP50': float(class_metrics[2]),
                                'f1': float(class_metrics[3])
                            })

        return overall_metrics, per_class_data

    except Exception as e:
        logger.error(f"Failed to evaluate {model_name}: {e}")
        return None, []

def main():
    """Main evaluation function"""
    print("=" * 100)
    print("PER-CLASS ACCURACY EVALUATION")
    print("=" * 100)

    # Find all trained models
    models_dir = Path("output/models")
    evaluation_results = []
    per_class_results = []

    if not models_dir.exists():
        print("Models directory not found!")
        return

    # Evaluate each model
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            pt_path = model_dir / "weights" / "best.pt"
            if pt_path.exists():
                model_name = model_dir.name

                overall, per_class = evaluate_model_per_class(str(pt_path), model_name)

                if overall:
                    evaluation_results.append(overall)
                    per_class_results.extend(per_class)
                    print(f"✓ Evaluated {model_name}")
                else:
                    print(f"✗ Failed to evaluate {model_name}")

    # Create DataFrames
    overall_df = pd.DataFrame(evaluation_results)
    per_class_df = pd.DataFrame(per_class_results)

    # Display results
    print(f"\n{'='*100}")
    print("OVERALL MODEL PERFORMANCE")
    print(f"{'='*100}")

    if not overall_df.empty:
        # Sort by mAP50
        sorted_overall = overall_df.sort_values('mAP50', ascending=False)

        print("Model Performance Summary:")
        print("Model Name              | mAP50 | Precision | Recall | mAP75 | mAP50-95")
        print("------------------------|-------|-----------|--------|-------|----------")

        for _, row in sorted_overall.iterrows():
            print("22"
                  "7.3f"
                  "11.3f"
                  "8.3f"
                  "7.3f"
                  "9.3f")

    # Per-class analysis
    print(f"\n{'='*100}")
    print("PER-CLASS ACCURACY ANALYSIS")
    print(f"{'='*100}")

    if not per_class_df.empty:
        print("\nPer-Class Performance by Model:")
        print("Model                    | Class       | Precision | Recall | mAP50 | F1")
        print("-------------------------|-------------|-----------|--------|-------|-----")

        # Group by model and show top performers
        for model_name in per_class_df['model'].unique():
            model_data = per_class_df[per_class_df['model'] == model_name]
            print(f"\n{model_name}:")

            for _, row in model_data.iterrows():
                print("23"
                      "11"
                      "11.3f"
                      "8.3f"
                      "7.3f"
                      "5.3f")

        # Best class performance across all models
        print(f"\n{'='*50}")
        print("BEST PERFORMING CLASSES ACROSS ALL MODELS")
        print(f"{'='*50}")

        class_performance = per_class_df.groupby('class').agg({
            'mAP50': ['mean', 'max', 'min'],
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean'
        }).round(3)

        print("Class       | Avg mAP50 | Max mAP50 | Min mAP50 | Avg Prec | Avg Rec | Avg F1")
        print("------------|-----------|-----------|-----------|----------|---------|--------")

        for class_name in ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']:
            if class_name in class_performance.index:
                stats = class_performance.loc[class_name]
                print("12"
                      "11.3f"
                      "11.3f"
                      "11.3f"
                      "10.3f"
                      "9.3f"
                      "8.3f")

        # Model ranking by class
        print(f"\n{'='*50}")
        print("MODEL RANKING BY CLASS PERFORMANCE")
        print(f"{'='*50}")

        for class_name in ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']:
            class_data = per_class_df[per_class_df['class'] == class_name].sort_values('mAP50', ascending=False)

            if not class_data.empty:
                print(f"\n{class_name} Detection Ranking:")
                for i, (_, row) in enumerate(class_data.head(3).iterrows(), 1):
                    print("2"
                          ".3f"
                          ".3f"
                          ".3f"
                          ".3f")

    # Save results
    if not overall_df.empty:
        overall_df.to_csv('output/per_class_overall_results.csv', index=False)
        print(f"\n💾 Overall results saved to: output/per_class_overall_results.csv")

    if not per_class_df.empty:
        per_class_df.to_csv('output/per_class_detailed_results.csv', index=False)
        print(f"💾 Per-class results saved to: output/per_class_detailed_results.csv")

    print(f"\n{'='*100}")
    print("PER-CLASS ACCURACY EVALUATION COMPLETE")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()



