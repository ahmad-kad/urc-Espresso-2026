#!/usr/bin/env python3
"""
Benchmark the trained rover optimization models and show results
"""

import os
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trainer import ModelTrainer


def find_best_models():
    """Find the best model from each architecture based on validation results"""
    output_dir = Path("output/models")
    architectures = ["yolov8s"]
    best_models = {}

    for arch in architectures:
        # Find all models for this architecture
        arch_models = []
        for subdir in output_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith(f"{arch}_rover_opt_"):
                # Check if results.csv exists
                results_file = subdir / "results.csv"
                if results_file.exists():
                    try:
                        # Read the last row (best epoch results)
                        df = pd.read_csv(results_file)
                        if len(df) > 0:
                            last_row = df.iloc[-1]
                            mAP50 = last_row.get("metrics/mAP50(B)", 0)

                            # Extract size from name
                            size_part = subdir.name.split("_")[-1]
                            size = int(size_part)

                            arch_models.append(
                                {
                                    "name": subdir.name,
                                    "path": str(subdir / "weights" / "best.pt"),
                                    "size": size,
                                    "mAP50": mAP50,
                                }
                            )
                    except Exception as e:
                        print(f"Error reading {results_file}: {e}")

        if arch_models:
            # Find best performing model for this architecture
            best_model = max(arch_models, key=lambda x: x["mAP50"])
            best_models[arch] = best_model
            print(
                f"{arch}: Best model {best_model['name']} - mAP50: {best_model['mAP50']:.3f}"
            )

    return best_models


def benchmark_best_models(best_models):
    """Benchmark the best models from each architecture"""
    print("\n" + "=" * 60)
    print("BENCHMARKING BEST MODELS FROM EACH ARCHITECTURE")
    print("=" * 60)

    trainer = ModelTrainer(
        {
            "training": {
                "device": "cuda" if __import__("torch").cuda.is_available() else "cpu"
            }
        }
    )

    # Prepare model configs for benchmarking
    model_configs = []
    for arch, model_info in best_models.items():
        model_configs.append(
            {
                "name": model_info["name"],
                "path": model_info["path"],
                "input_size": model_info["size"],
                "architecture": arch,
            }
        )

    print(f"Benchmarking {len(model_configs)} models:")
    for config in model_configs:
        print(f"  • {config['name']} ({config['input_size']}x{config['input_size']})")

    # Simplified benchmarking - use model_evaluator instead
    import logging

    from scripts.evaluation.model_evaluator import ModelEvaluator

    logger = logging.getLogger(__name__)

    logger.warning("benchmark_rover_models removed - using ModelEvaluator instead")

    results = []
    for config in model_configs:
        try:
            evaluator = ModelEvaluator(config["path"], model_type="pytorch")
            eval_results = evaluator.run_comprehensive_evaluation(data_yaml)
            results.append(
                {
                    "model": config["name"],
                    "input_size": config.get("input_size", 224),
                    **eval_results,
                }
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {config['name']}: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()


def show_final_results(benchmark_df):
    """Display final results and recommendations"""
    print("\n" + "=" * 80)
    print("ROVER MODEL OPTIMIZATION RESULTS")
    print("=" * 80)

    # Sort by efficiency score
    results_df = benchmark_df.sort_values("efficiency_score", ascending=False)

    print("[BEST] TOP 3 RECOMMENDED MODELS FOR ROVER DEPLOYMENT:")
    print("(Ranked by efficiency: accuracy / (size × latency))")
    print("")

    for i, (_, model) in enumerate(results_df.head(3).iterrows(), 1):
        print(f"#{i} RANK: {model['model']}")
        print(f"   [ANALYSIS] Accuracy: {model['mAP50']:.3f} mAP50")
        print(
            f"   [PERFORMANCE] Speed: {model['fps']:.1f} FPS ({model['avg_latency_ms']:.1f}ms latency)"
        )
        print(f"   📦 Size: {model['model_size_mb']:.1f} MB")
        print(f"   [TARGET] Efficiency Score: {model['efficiency_score']:.4f}")
        print(
            f"   [TARGET] Precision: {model.get('precision', 0):.3f}, Recall: {model.get('recall', 0):.3f}"
        )
        print("")

    # Analysis
    print("[RESULTS] ANALYSIS:")
    best_accuracy = results_df.loc[results_df["mAP50"].idxmax()]
    best_speed = results_df.loc[results_df["fps"].idxmax()]
    smallest_size = results_df.loc[results_df["model_size_mb"].idxmin()]

    print(
        f"• Most Accurate: {best_accuracy['model']} ({best_accuracy['mAP50']:.3f} mAP50)"
    )
    print(f"• Fastest: {best_speed['model']} ({best_speed['fps']:.1f} FPS)")
    print(
        f"• Smallest: {smallest_size['model']} ({smallest_size['model_size_mb']:.1f} MB)"
    )

    # Save detailed results using OutputManager
    from utils.output_utils import output_manager

    results_file = output_manager.save_csv(
        results_df.to_dict("records"), "rover_final_benchmark_results", "benchmarking"
    )
    print(f"\n[SAVE] Detailed results saved to: {results_file}")

    return results_df


def main():
    """Main function"""
    print("[SEARCH] ANALYZING ROVER OPTIMIZATION RESULTS")
    print("=" * 50)

    # Find best models from each architecture
    best_models = find_best_models()

    if not best_models:
        print("[ERROR] No trained models found. The optimization may have failed.")
        return 1

    # Benchmark the best models
    benchmark_df = benchmark_best_models(best_models)

    # Show final results
    show_final_results(benchmark_df)

    print("\n[SUCCESS] ANALYSIS COMPLETE!")
    print("The rover optimization successfully trained and evaluated multiple models.")
    # Save results using OutputManager
    from utils.output_utils import output_manager

    if not benchmark_df.empty:
        csv_path = output_manager.save_csv(
            benchmark_df.to_dict("records"), "benchmark_results", "benchmarking"
        )
        print(f"\n[SUCCESS] Results saved to: {csv_path}")

        # Also save summary
        summary_content = "ROVER MODEL BENCHMARKING RESULTS\n"
        summary_content += "=" * 40 + "\n\n"
        summary_content += "Best models by architecture:\n"
        for arch, model_info in best_models.items():
            summary_content += (
                f"• {model_info['name']}: {model_info['mAP50']:.3f} mAP50\n"
            )
        summary_content += f"\nBenchmark results saved to: {csv_path}\n"

        summary_path = output_manager.save_text_report(
            summary_content, "benchmark_summary", "benchmarking"
        )
        print(f"Summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
