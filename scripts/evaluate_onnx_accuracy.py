#!/usr/bin/env python3
"""
Evaluate ONNX model accuracy and compare with PyTorch results
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional
import numpy as np

from ultralytics import YOLO
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_model_info(model_path: str) -> tuple:
    """Parse model name to extract architecture and input size"""
    path = Path(model_path)
    model_name = path.stem
    
    # Extract input size from filename (e.g., mobilenet_160_160.onnx -> 160)
    parts = model_name.split('_')
    input_size = 224  # default
    
    for part in parts:
        if part.isdigit() and len(part) == 3:  # 160, 192, 224
            input_size = int(part)
            break
    
    # Get base model name (remove size suffix)
    base_name = model_name
    for size in ['160', '192', '224']:
        if base_name.endswith(f'_{size}'):
            base_name = base_name[:-4]  # Remove '_160', '_192', or '_224'
            break
    
    return base_name, input_size


def evaluate_onnx_model(onnx_path: str, data_yaml: str, input_size: int) -> Optional[Dict]:
    """
    Evaluate ONNX model accuracy using YOLO validation
    """
    try:
        model_name = Path(onnx_path).stem
        logger.info(f"Evaluating ONNX model: {model_name} ({input_size}x{input_size})...")
        
        # Load ONNX model using YOLO (it can validate ONNX models)
        model = YOLO(onnx_path)
        
        # Run validation
        results = model.val(
            data=data_yaml,
            imgsz=input_size,
            conf=0.25,
            iou=0.6,
            verbose=False,
            plots=False
        )
        
        # Extract metrics
        metrics = {
            'model': model_name,
            'format': 'ONNX',
            'input_size': input_size,
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * float(results.box.mp) * float(results.box.mr) / (
                float(results.box.mp) + float(results.box.mr)
            ) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
        }
        
        logger.info(f"✅ {model_name}: mAP50={metrics['mAP50']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Failed to evaluate {onnx_path}: {str(e)}")
        return None


def evaluate_pytorch_model(model_path: str, data_yaml: str, input_size: int) -> Optional[Dict]:
    """
    Evaluate PyTorch model accuracy for comparison
    """
    try:
        model_name = Path(model_path).stem
        logger.info(f"Evaluating PyTorch model: {model_name} ({input_size}x{input_size})...")
        
        # Load PyTorch model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data=data_yaml,
            imgsz=input_size,
            conf=0.25,
            iou=0.6,
            verbose=False,
            plots=False
        )
        
        # Extract metrics
        metrics = {
            'model': model_name,
            'format': 'PyTorch',
            'input_size': input_size,
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * float(results.box.mp) * float(results.box.mr) / (
                float(results.box.mp) + float(results.box.mr)
            ) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
        }
        
        logger.info(f"✅ {model_name}: mAP50={metrics['mAP50']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Failed to evaluate {model_path}: {str(e)}")
        return None


def load_existing_pytorch_results() -> Dict:
    """Load existing PyTorch accuracy results from benchmark_results.json"""
    try:
        with open('output/benchmark_results.json', 'r') as f:
            data = json.load(f)
        
        # Filter PyTorch results and create lookup dict
        pytorch_results = {}
        for entry in data:
            if entry.get('type') == 'pytorch' and 'mAP50' in entry:
                model_name = entry['name']
                pytorch_results[model_name] = {
                    'mAP50': entry['mAP50'],
                    'precision': entry['precision'],
                    'recall': entry['recall'],
                    'f1_score': entry.get('f1_score', 0),
                    'input_size': entry['size']
                }
        
        return pytorch_results
    except Exception as e:
        logger.warning(f"Could not load existing PyTorch results: {e}")
        return {}


def find_all_onnx_models(onnx_dir: str = "output/onnx") -> List[Dict]:
    """Find all ONNX models to evaluate"""
    models = []
    onnx_path = Path(onnx_dir)
    
    if not onnx_path.exists():
        logger.error(f"ONNX directory not found: {onnx_dir}")
        return []
    
    for onnx_file in onnx_path.glob("*.onnx"):
        try:
            model_name, input_size = parse_model_info(str(onnx_file))
            
            model_info = {
                'onnx_path': str(onnx_file),
                'model_name': model_name,
                'input_size': input_size
            }
            
            models.append(model_info)
            logger.info(f"Found ONNX model: {model_name} ({input_size}x{input_size})")
            
        except Exception as e:
            logger.warning(f"Could not process {onnx_file}: {e}")
    
    return sorted(models, key=lambda x: (x['model_name'], x['input_size']))


def compare_accuracy_results(onnx_results: List[Dict], pytorch_results: Dict) -> pd.DataFrame:
    """Compare ONNX vs PyTorch accuracy and create comparison DataFrame"""
    comparison_data = []
    
    for onnx_result in onnx_results:
        model_name = onnx_result['model']
        input_size = onnx_result['input_size']
        
        # Try to find matching PyTorch result
        # Match by base model name (remove size suffix if present)
        base_name = model_name
        for size in ['_160', '_192', '_224']:
            if base_name.endswith(size):
                base_name = base_name[:-4]
                break
        
        # Look for matching PyTorch model
        pytorch_match = None
        for pytorch_name, pytorch_data in pytorch_results.items():
            # Check if names match (with or without size suffix)
            if pytorch_name == base_name or pytorch_name == model_name:
                if pytorch_data['input_size'] == input_size:
                    pytorch_match = pytorch_data
                    break
        
        comparison = {
            'model': base_name,
            'input_size': input_size,
            'onnx_mAP50': onnx_result['mAP50'],
            'onnx_precision': onnx_result['precision'],
            'onnx_recall': onnx_result['recall'],
            'onnx_f1': onnx_result['f1_score'],
        }
        
        if pytorch_match:
            comparison.update({
                'pytorch_mAP50': pytorch_match['mAP50'],
                'pytorch_precision': pytorch_match['precision'],
                'pytorch_recall': pytorch_match['recall'],
                'pytorch_f1': pytorch_match['f1_score'],
                'mAP50_diff': onnx_result['mAP50'] - pytorch_match['mAP50'],
                'precision_diff': onnx_result['precision'] - pytorch_match['precision'],
                'recall_diff': onnx_result['recall'] - pytorch_match['recall'],
                'f1_diff': onnx_result['f1_score'] - pytorch_match['f1_score'],
            })
        else:
            comparison.update({
                'pytorch_mAP50': None,
                'pytorch_precision': None,
                'pytorch_recall': None,
                'pytorch_f1': None,
                'mAP50_diff': None,
                'precision_diff': None,
                'recall_diff': None,
                'f1_diff': None,
            })
        
        comparison_data.append(comparison)
    
    return pd.DataFrame(comparison_data)


def main():
    """Main evaluation function"""
    logger.info("=" * 80)
    logger.info("ONNX MODEL ACCURACY EVALUATION")
    logger.info("=" * 80)
    
    data_yaml = "consolidated_dataset/data.yaml"
    
    # Step 1: Find all ONNX models
    logger.info("\n[STEP 1] Finding ONNX models...")
    onnx_models = find_all_onnx_models()
    
    if not onnx_models:
        logger.error("No ONNX models found!")
        return 1
    
    logger.info(f"Found {len(onnx_models)} ONNX models to evaluate")
    
    # Step 2: Load existing PyTorch results
    logger.info("\n[STEP 2] Loading existing PyTorch accuracy results...")
    pytorch_results = load_existing_pytorch_results()
    logger.info(f"Loaded {len(pytorch_results)} PyTorch model results")
    
    # Step 3: Evaluate all ONNX models
    logger.info("\n[STEP 3] Evaluating ONNX model accuracy...")
    onnx_results = []
    
    for i, model_info in enumerate(onnx_models, 1):
        logger.info(f"\n[{i}/{len(onnx_models)}] Processing {model_info['model_name']}...")
        
        result = evaluate_onnx_model(
            model_info['onnx_path'],
            data_yaml,
            model_info['input_size']
        )
        
        if result:
            onnx_results.append(result)
    
    if not onnx_results:
        logger.error("No ONNX models were successfully evaluated!")
        return 1
    
    logger.info(f"\n✅ Successfully evaluated {len(onnx_results)} ONNX models")
    
    # Step 4: Compare with PyTorch results
    logger.info("\n[STEP 4] Comparing ONNX vs PyTorch accuracy...")
    comparison_df = compare_accuracy_results(onnx_results, pytorch_results)
    
    # Step 5: Generate reports
    logger.info("\n[STEP 5] Generating reports...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save ONNX results
    onnx_df = pd.DataFrame(onnx_results)
    onnx_file = output_dir / "onnx_accuracy_results.json"
    onnx_df.to_json(onnx_file, orient='records', indent=2)
    onnx_csv = output_dir / "onnx_accuracy_results.csv"
    onnx_df.to_csv(onnx_csv, index=False)
    
    # Save comparison
    comparison_file = output_dir / "onnx_pytorch_accuracy_comparison.json"
    comparison_df.to_json(comparison_file, orient='records', indent=2)
    comparison_csv = output_dir / "onnx_pytorch_accuracy_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    
    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("ACCURACY COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    if not comparison_df.empty:
        # Show models with comparisons
        has_comparison = comparison_df['pytorch_mAP50'].notna()
        if has_comparison.any():
            logger.info("\n📊 ONNX vs PyTorch Accuracy Comparison:")
            logger.info("-" * 80)
            logger.info(f"{'Model':<25} {'Input':<8} {'ONNX mAP50':<12} {'PyTorch mAP50':<15} {'Difference':<12}")
            logger.info("-" * 80)
            
            for _, row in comparison_df[has_comparison].iterrows():
                diff = row['mAP50_diff']
                diff_str = f"{diff:+.4f}" if not pd.isna(diff) else "N/A"
                logger.info(
                    f"{row['model']:<25} {int(row['input_size']):<8} "
                    f"{row['onnx_mAP50']:<12.4f} {row['pytorch_mAP50']:<15.4f} {diff_str:<12}"
                )
            
            # Calculate average differences
            avg_diff = comparison_df[has_comparison]['mAP50_diff'].mean()
            logger.info("\n" + "-" * 80)
            logger.info(f"Average mAP50 difference (ONNX - PyTorch): {avg_diff:+.4f}")
            
            if abs(avg_diff) < 0.01:
                logger.info("✅ ONNX models maintain accuracy (difference < 1%)")
            elif abs(avg_diff) < 0.05:
                logger.info("⚠️  ONNX models show minor accuracy difference (< 5%)")
            else:
                logger.info("❌ ONNX models show significant accuracy difference (> 5%)")
        
        # Show top ONNX models by accuracy
        logger.info("\n📊 Top 5 ONNX Models by mAP50:")
        logger.info("-" * 80)
        top_onnx = onnx_df.nlargest(5, 'mAP50')
        for i, (_, row) in enumerate(top_onnx.iterrows(), 1):
            logger.info(
                f"{i}. {row['model']:<30} - mAP50: {row['mAP50']:.4f}, "
                f"Precision: {row['precision']:.4f}, Recall: {row['recall']:.4f}"
            )
    
    logger.info(f"\n💾 Results saved to:")
    logger.info(f"   ONNX Results: {onnx_file}")
    logger.info(f"   Comparison: {comparison_file}")
    logger.info(f"   CSV files also available in output/")
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
