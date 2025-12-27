#!/usr/bin/env python3
"""
Convert ALL trained models to ONNX format and quantize them.
Creates ONNX FP32 and INT8 versions of every trained model.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent))  # Add root directory for trainer.py

from trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_all_trained_models(models_dir="output/models"):
    """Find all trained models with best.pt files"""
    models_path = Path(models_dir)
    trained_models = []

    if not models_path.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return []

    logger.info(f"[SEARCH] Scanning for trained models in {models_dir}")

    # Find all best.pt files
    for model_dir in models_path.rglob("*"):
        if model_dir.is_dir():
            best_pt_path = model_dir / "weights" / "best.pt"
            if best_pt_path.exists():
                # Extract model info from directory name
                model_name = model_dir.name

                # Parse model type and input size from name
                if "_rover_opt_" in model_name:
                    parts = model_name.split("_rover_opt_")
                    architecture = parts[0]
                    input_size = int(parts[1]) if parts[1].isdigit() else 320
                else:
                    architecture = model_name
                    input_size = 320  # default

                model_info = {
                    'name': model_name,
                    'architecture': architecture,
                    'input_size': input_size,
                    'pt_path': str(best_pt_path),
                    'model_dir': str(model_dir)
                }

                trained_models.append(model_info)
                logger.info(f"[NOTE] Found model: {model_name} (size: {input_size})")

    logger.info(f"[SUCCESS] Found {len(trained_models)} trained models")
    return trained_models


def convert_model_to_onnx(model_info, onnx_output_dir="output/onnx"):
    """Convert a single model to ONNX format"""
    model_name = model_info['name']
    pt_path = model_info['pt_path']
    input_size = model_info['input_size']

    # Create output path
    onnx_path = Path(onnx_output_dir) / f"{model_name}.onnx"
    Path(onnx_output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"[CONFIG] Converting {model_name} to ONNX...")

    try:
        # Use the existing conversion function
        from scripts.convert_to_onnx import convert_model

        # All models are YOLO models with different training configs
        # The names like "efficientnet_rover_opt" are just labels for training approach
        model_type = 'yolo'

        convert_model(
            model_path=pt_path,
            output_path=str(onnx_path),
            model_type=model_type,
            input_size=input_size
        )

        # Verify file was created
        if onnx_path.exists():
            file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"[SUCCESS] ONNX conversion complete: {model_name} ({file_size_mb:.1f} MB)")
            return str(onnx_path), file_size_mb
        else:
            logger.error(f"[ERROR] ONNX file not created: {onnx_path}")
            return None, 0

    except Exception as e:
        logger.error(f"[ERROR] Failed to convert {model_name}: {str(e)}")
        return None, 0


def quantize_onnx_model(onnx_path, model_name, input_size, quantized_output_dir="output/quantized"):
    """Quantize a single ONNX model to INT8"""
    quantized_path = Path(quantized_output_dir) / f"{model_name}_int8.onnx"
    Path(quantized_output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"[CONFIG] Quantizing {model_name} to INT8...")

    try:
        trainer = ModelTrainer({'training': {'device': 'cpu'}})

        # Create model config for quantization
        model_config = {
            'name': model_name,
            'path': onnx_path,
            'input_size': input_size,
            'architecture': 'onnx'  # Generic ONNX model
        }

        # Quantize the model
        result = trainer.quantize_and_evaluate(
            onnx_path,
            'consolidated_dataset/data.yaml',
            input_size,
            str(Path(quantized_output_dir) / model_name)
        )

        if result and 'quantized' in result:
            # Check if quantized file was created
            expected_path = Path(quantized_output_dir) / model_name / "best_int8.onnx"
            if expected_path.exists():
                file_size_mb = expected_path.stat().st_size / (1024 * 1024)
                logger.info(f"[SUCCESS] Quantization complete: {model_name} ({file_size_mb:.1f} MB)")

                return str(expected_path), file_size_mb, result.get('quantized', {}).get('mAP50', 0)
            else:
                logger.warning(f"[WARNING] Expected quantized file not found: {expected_path}")

        logger.error(f"[ERROR] Quantization failed for {model_name}")
        return None, 0, 0

    except Exception as e:
        logger.error(f"[ERROR] Failed to quantize {model_name}: {str(e)}")
        return None, 0, 0


def process_single_model(model_info, onnx_output_dir, quantized_output_dir, skip_quantization=False):
    """Process a single model: convert to ONNX and quantize"""
    model_name = model_info['name']
    results = {'model': model_name, 'status': 'failed'}

    try:
        # Step 1: Convert to ONNX
        logger.info(f"\n{'='*60}")
        logger.info(f"[STARTING] Processing: {model_name}")
        logger.info(f"{'='*60}")

        onnx_path, onnx_size = convert_model_to_onnx(model_info, onnx_output_dir)

        if onnx_path:
            results.update({
                'onnx_path': onnx_path,
                'onnx_size_mb': onnx_size,
                'onnx_status': 'success'
            })

            # Step 2: Quantize to INT8 (if not skipped)
            if not skip_quantization:
                quantized_path, quantized_size, quantized_map50 = quantize_onnx_model(
                    onnx_path, model_name, model_info['input_size'], quantized_output_dir
                )

                if quantized_path:
                    results.update({
                        'quantized_path': quantized_path,
                        'quantized_size_mb': quantized_size,
                        'quantized_map50': quantized_map50,
                        'quantized_status': 'success',
                        'status': 'success'
                    })
                else:
                    results['quantized_status'] = 'failed'
            else:
                results['status'] = 'success'

        logger.info(f"[COMPLETE] Finished processing: {model_name}")

    except Exception as e:
        logger.error(f"[ERROR] Failed to process {model_name}: {str(e)}")
        results['error'] = str(e)

    return results


def convert_all_models_parallel(models_list, onnx_output_dir="output/onnx", quantized_output_dir="output/quantized",
                               max_workers=4, skip_quantization=False):
    """Convert all models to ONNX and quantize them using parallel processing"""
    logger.info(f"\n{'='*80}")
    logger.info(f"[STARTING] CONVERTING ALL {len(models_list)} MODELS TO ONNX (+INT8)")
    logger.info(f"{'='*80}")

    all_results = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(process_single_model, model_info, onnx_output_dir, quantized_output_dir, skip_quantization): model_info
            for model_info in models_list
        }

        # Process results as they complete
        for future in as_completed(future_to_model):
            model_info = future_to_model[future]
            try:
                result = future.result()
                all_results.append(result)
                logger.info(f"[UPDATE] Completed: {model_info['name']} - {result['status']}")
            except Exception as e:
                logger.error(f"[ERROR] Exception processing {model_info['name']}: {str(e)}")
                all_results.append({
                    'model': model_info['name'],
                    'status': 'error',
                    'error': str(e)
                })

    return all_results


def generate_summary_report(results, output_file="output/model_conversion_summary.csv"):
    """Generate a summary report of all conversions"""
    logger.info(f"\n{'='*80}")
    logger.info(f"[ANALYSIS] CONVERSION SUMMARY REPORT")
    logger.info(f"{'='*80}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Summary statistics
    total_models = len(results)
    successful_onnx = sum(1 for r in results if r.get('onnx_status') == 'success')
    successful_quantized = sum(1 for r in results if r.get('quantized_status') == 'success')

    logger.info(f"[RESULTS] Total models processed: {total_models}")
    logger.info(f"[RESULTS] ONNX conversions successful: {successful_onnx}")
    logger.info(f"[RESULTS] INT8 quantizations successful: {successful_quantized}")

    # Size analysis
    if 'onnx_size_mb' in df.columns:
        avg_onnx_size = df['onnx_size_mb'].mean()
        logger.info(f"[RESULTS] Average ONNX model size: {avg_onnx_size:.1f} MB")

    if 'quantized_size_mb' in df.columns:
        avg_quantized_size = df['quantized_size_mb'].mean()
        logger.info(f"[RESULTS] Average INT8 model size: {avg_quantized_size:.1f} MB")

    # Save detailed results
    try:
        df.to_csv(output_file, index=False)
        logger.info(f"[SAVE] Detailed results saved to: {output_file}")
    except Exception as e:
        logger.error(f"[WARNING] Failed to save results: {e}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Convert all trained models to ONNX and quantize them')
    parser.add_argument('--models_dir', type=str, default='output/models',
                       help='Directory containing trained models')
    parser.add_argument('--onnx_output_dir', type=str, default='output/onnx',
                       help='Output directory for ONNX models')
    parser.add_argument('--quantized_output_dir', type=str, default='output/quantized',
                       help='Output directory for quantized models')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of parallel workers')
    parser.add_argument('--skip_quantization', action='store_true',
                       help='Skip quantization step, only convert to ONNX')
    parser.add_argument('--single_model', type=str,
                       help='Process only a specific model by name')

    args = parser.parse_args()

    try:
        # Find all trained models
        trained_models = find_all_trained_models(args.models_dir)

        if not trained_models:
            logger.error("[ERROR] No trained models found!")
            return 1

        # Filter to single model if requested
        if args.single_model:
            trained_models = [m for m in trained_models if m['name'] == args.single_model]
            if not trained_models:
                logger.error(f"[ERROR] Model '{args.single_model}' not found!")
                return 1

        # Convert all models
        results = convert_all_models_parallel(
            trained_models,
            args.onnx_output_dir,
            args.quantized_output_dir,
            args.max_workers,
            args.skip_quantization
        )

        # Generate summary report
        summary_df = generate_summary_report(results)

        logger.info(f"\n[SUCCESS] ALL MODEL CONVERSIONS COMPLETE!")
        logger.info(f"Processed {len(trained_models)} models")
        logger.info(f"ONNX models saved to: {args.onnx_output_dir}")
        if not args.skip_quantization:
            logger.info(f"INT8 models saved to: {args.quantized_output_dir}")

        return 0

    except Exception as e:
        logger.error(f"[ERROR] Conversion pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
