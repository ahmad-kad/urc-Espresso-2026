#!/usr/bin/env python3
"""
Run the complete rover model optimization pipeline
Trains multiple architectures at different sizes with early stopping, then benchmarks
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trainer import run_rover_optimization_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("rover_optimization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete rover optimization pipeline"""

    logger.info("=" * 100)
    logger.info("[STARTING] STARTING ROVER MODEL OPTIMIZATION PIPELINE")
    logger.info("=" * 100)

    # Base configuration optimized for rover deployment
    base_config = {
        "model": {
            "architecture": "yolov8s",  # Default, will be overridden
            "imgsz": 224,  # Default, will be overridden per size test
        },
        "training": {
            "epochs": 50,  # Sufficient for convergence with early stopping
            "batch_size": 16,  # Balanced for GPU memory
            "learning_rate": 0.001,
            "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
            "patience": 10,  # Early stopping patience as requested
        },
        "attention": {
            "enabled": False,  # Will be overridden for CBAM models
            "type": "cbam",
            "layers": ["model.3", "model.6", "model.9", "model.12"],
        },
    }

    # Dataset configuration
    data_yaml = "consolidated_dataset/data.yaml"

    # Verify dataset exists
    if not Path(data_yaml).exists():
        logger.error(f"Dataset configuration not found: {data_yaml}")
        logger.error("Please ensure the consolidated dataset is prepared.")
        return 1

    logger.info("Configuration:")
    logger.info(f"  • Dataset: {data_yaml}")
    logger.info(f"  • Device: {base_config['training']['device']}")
    logger.info(f"  • Max Epochs: {base_config['training']['epochs']}")
    logger.info(f"  • Early Stopping Patience: {base_config['training']['patience']}")
    logger.info(f"  • Batch Size: {base_config['training']['batch_size']}")
    logger.info("")

    try:
        # Run the complete optimization pipeline
        results = run_rover_optimization_pipeline(base_config, data_yaml)

        # Save results summary
        save_results_summary(results)

        logger.info("=" * 100)
        logger.info("[SUCCESS] ROVER OPTIMIZATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 100)

        # Final recommendations
        top_models = results["top_recommendations"]
        if len(top_models) > 0:
            best_model = top_models.iloc[0]
            logger.info("[BEST] WINNING MODEL FOR ROVER DEPLOYMENT:")
            logger.info(f"   {best_model['model']}")
            logger.info(f"   • mAP50: {best_model['mAP50']:.3f}")
            logger.info(f"   • FPS: {best_model['fps']:.1f}")
            logger.info(f"   • Size: {best_model['model_size_mb']:.1f} MB")
            logger.info(f"   • Efficiency: {best_model['efficiency_score']:.4f}")

        return 0

    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


def save_results_summary(results):
    """Save a summary of the optimization results"""
    try:
        import pandas as pd

        summary_file = "rover_optimization_summary.txt"
        with open(summary_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ROVER MODEL OPTIMIZATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Multi-size training results
            f.write("MULTI-SIZE TRAINING RESULTS:\n")
            f.write("-" * 40 + "\n")
            for arch, arch_results in results["multi_size_results"].items():
                f.write(f"{arch.upper()}:\n")
                f.write(f"  Best Size: {arch_results['best_size']}x{arch_results['best_size']}\n")
                f.write(f"  Best Performance: {arch_results['best_performance']:.3f} mAP50\n")
                f.write(f"  Tested Sizes: {arch_results['all_sizes']}\n\n")

            # Benchmark results
            f.write("BENCHMARK RESULTS:\n")
            f.write("-" * 40 + "\n")
            benchmark_df = results["benchmark_results"]
            f.write(benchmark_df.to_string(index=False))
            f.write("\n\n")

            # Top recommendations
            f.write("TOP 3 RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            top_models = results["top_recommendations"]
            for i, (_, model) in enumerate(top_models.iterrows(), 1):
                f.write(f"#{i}: {model['model']}\n")
                f.write(f"   Accuracy: {model['mAP50']:.3f} mAP50\n")
                f.write(f"   Speed: {model['fps']:.1f} FPS\n")
                f.write(f"   Size: {model['model_size_mb']:.1f} MB\n")
                f.write(f"   Efficiency: {model['efficiency_score']:.4f}\n\n")

        logger.info(f"Results summary saved to: {summary_file}")

        # Also save benchmark DataFrame
        benchmark_df.to_csv("rover_benchmark_results.csv", index=False)
        logger.info("Detailed benchmark results saved to: rover_benchmark_results.csv")

    except Exception as e:
        logger.warning(f"Failed to save results summary: {str(e)}")


if __name__ == "__main__":
    sys.exit(main())
