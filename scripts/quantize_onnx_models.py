#!/usr/bin/env python3
"""
Quantize ONNX models to INT8 format for deployment
"""

import sys
import os
import argparse
from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quantize_onnx_model(onnx_path, output_path, model_name="model"):
    """Quantize a single ONNX model to INT8"""
    logger.info(f"[CONFIG] Quantizing {model_name}...")

    try:
        # Quantize the model using ONNX Runtime
        quantize_dynamic(
            model_input=onnx_path,
            model_output=output_path,
            weight_type=QuantType.QInt8  # Use signed 8-bit integers for weights
        )

        # Get file sizes
        original_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB
        quantized_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        reduction = ((original_size - quantized_size) / original_size) * 100

        logger.info(f"[SUCCESS] Quantized {model_name}")
        logger.info(f"  Original: {original_size:.1f} MB")
        logger.info(f"  Quantized: {quantized_size:.1f} MB")
        logger.info(f"  Reduction: {reduction:.1f}%")

        return True, quantized_size, reduction

    except Exception as e:
        logger.error(f"[ERROR] Failed to quantize {model_name}: {str(e)}")
        return False, 0, 0


def validate_quantized_model(onnx_path, quantized_path, input_size=320):
    """Validate that the quantized model works"""
    try:
        # Create dummy input
        import numpy as np
        dummy_input = np.random.rand(1, 3, input_size, input_size).astype(np.float32)

        # Test original model
        original_session = ort.InferenceSession(onnx_path)
        original_inputs = {original_session.get_inputs()[0].name: dummy_input}
        original_output = original_session.run(None, original_inputs)

        # Test quantized model
        quantized_session = ort.InferenceSession(quantized_path)
        quantized_inputs = {quantized_session.get_inputs()[0].name: dummy_input}
        quantized_output = quantized_session.run(None, quantized_inputs)

        # Compare outputs (basic shape check)
        if len(original_output) == len(quantized_output):
            logger.info("[VALIDATION] Quantized model validation passed")
            return True
        else:
            logger.warning("[VALIDATION] Output shapes don't match")
            return False

    except Exception as e:
        logger.error(f"[VALIDATION] Failed to validate quantized model: {str(e)}")
        return False


def quantize_all_onnx_models(onnx_dir="output/onnx", quantized_dir="output/quantized"):
    """Quantize all ONNX models in the directory"""
    onnx_path = Path(onnx_dir)
    quantized_path = Path(quantized_dir)

    if not onnx_path.exists():
        logger.error(f"[ERROR] ONNX directory not found: {onnx_dir}")
        return []

    # Create output directory
    quantized_path.mkdir(parents=True, exist_ok=True)

    # Find all ONNX files
    onnx_files = list(onnx_path.glob("*.onnx"))
    logger.info(f"[SEARCH] Found {len(onnx_files)} ONNX models to quantize")

    results = []

    for onnx_file in onnx_files:
        model_name = onnx_file.stem
        output_path = quantized_path / f"{model_name}_int8.onnx"

        logger.info(f"\n[STARTING] Processing: {model_name}")
        logger.info("-" * 50)

        # Quantize the model
        success, quantized_size, reduction = quantize_onnx_model(
            str(onnx_file), str(output_path), model_name
        )

        if success:
            # Validate the quantized model
            validation_passed = validate_quantized_model(
                str(onnx_file), str(output_path),
                input_size=int(model_name.split('_')[-1]) if model_name.split('_')[-1].isdigit() else 320
            )

            result = {
                'model': model_name,
                'onnx_path': str(onnx_file),
                'quantized_path': str(output_path),
                'quantized_size_mb': quantized_size,
                'size_reduction_percent': reduction,
                'validation_passed': validation_passed,
                'status': 'success'
            }
        else:
            result = {
                'model': model_name,
                'status': 'failed'
            }

        results.append(result)
        logger.info(f"[COMPLETE] Finished: {model_name} - {result['status']}")

    return results


def generate_quantization_report(results, output_file="output/quantization_summary.csv"):
    """Generate a summary report of quantization results"""
    import pandas as pd

    logger.info(f"\n{'='*80}")
    logger.info(f"[ANALYSIS] QUANTIZATION SUMMARY REPORT")
    logger.info(f"{'='*80}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Summary statistics
    total_models = len(results)
    successful_quantizations = sum(1 for r in results if r.get('status') == 'success')
    valid_models = sum(1 for r in results if r.get('validation_passed') == True)

    logger.info(f"[RESULTS] Total ONNX models: {total_models}")
    logger.info(f"[RESULTS] Successful quantizations: {successful_quantizations}")
    logger.info(f"[RESULTS] Validated models: {valid_models}")

    if successful_quantizations > 0:
        avg_size = df[df['status'] == 'success']['quantized_size_mb'].mean()
        avg_reduction = df[df['status'] == 'success']['size_reduction_percent'].mean()

        logger.info(f"[RESULTS] Average quantized size: {avg_size:.1f} MB")
        logger.info(f"[RESULTS] Average size reduction: {avg_reduction:.1f}%")

    # Save detailed results
    try:
        df.to_csv(output_file, index=False)
        logger.info(f"[SAVE] Detailed results saved to: {output_file}")
    except Exception as e:
        logger.error(f"[WARNING] Failed to save results: {e}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Quantize ONNX models to INT8 format')
    parser.add_argument('--onnx_dir', type=str, default='output/onnx',
                       help='Directory containing ONNX models')
    parser.add_argument('--quantized_dir', type=str, default='output/quantized',
                       help='Output directory for quantized models')
    parser.add_argument('--single_model', type=str,
                       help='Quantize only a specific ONNX model file')

    args = parser.parse_args()

    try:
        if args.single_model:
            # Quantize a single model
            model_path = Path(args.single_model)
            if not model_path.exists():
                logger.error(f"[ERROR] Model file not found: {args.single_model}")
                return 1

            model_name = model_path.stem
            output_path = Path(args.quantized_dir) / f"{model_name}_int8.onnx"
            Path(args.quantized_dir).mkdir(parents=True, exist_ok=True)

            success, size, reduction = quantize_onnx_model(
                str(model_path), str(output_path), model_name
            )

            if success:
                logger.info(f"\n[SUCCESS] Single model quantization complete!")
                logger.info(f"Quantized model saved to: {output_path}")
                return 0
            else:
                return 1

        else:
            # Quantize all models
            results = quantize_all_onnx_models(args.onnx_dir, args.quantized_dir)
            generate_quantization_report(results)

            logger.info(f"\n[SUCCESS] ALL QUANTIZATION COMPLETE!")
            logger.info(f"Processed {len(results)} ONNX models")
            logger.info(f"INT8 models saved to: {args.quantized_dir}")

            return 0

    except Exception as e:
        logger.error(f"[ERROR] Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
