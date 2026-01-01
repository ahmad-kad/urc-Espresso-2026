#!/usr/bin/env python3
"""
Complete pipeline: Training -> ONNX Conversion -> Comparison
"""

import sys
from pathlib import Path
import yaml
from trainer import ModelTrainer
from scripts.evaluation.model_evaluator import ModelEvaluator
from utils.logger_config import get_logger

logger = get_logger(__name__)

def load_config_direct(config_path: str):
    """Load config directly from file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("=" * 60)
    print("TRAINING -> ONNX CONVERSION -> COMPARISON PIPELINE")
    print("=" * 60)
    
    # Step 1: Load configuration
    print("\nStep 1: Loading configuration...")
    config_path = Path("configs/default.yaml")
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1
    
    config = load_config_direct(str(config_path))
    # Force CUDA if available, otherwise use auto (will fallback to CPU)
    import torch
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        print(f"CUDA available - using GPU for training")
    else:
        config['device'] = 'auto'
        print(f"CUDA not available - using CPU for training")
    print(f"Config loaded: {config.get('model', {}).get('architecture', 'unknown')}")
    
    # Step 2: Check for data.yaml
    print("\nStep 2: Checking for dataset...")
    data_yaml_paths = [
        "consolidated_dataset/data.yaml",
        "data/data.yaml",
        config.get('data', {}).get('yaml_path', '')
    ]
    
    data_yaml = None
    for path in data_yaml_paths:
        if path and Path(path).exists():
            data_yaml = path
            print(f"Found dataset: {data_yaml}")
            break
    
    if not data_yaml:
        print("No data.yaml found. Skipping training.")
        print("   Available paths checked:", data_yaml_paths)
        # Use existing model if available
        existing_model = Path("output/models/yolov8n_fixed_224/weights/best.pt")
        if existing_model.exists():
            print(f"Using existing model: {existing_model}")
            model_path = str(existing_model)
        else:
            print("No model found for conversion")
            return 1
    else:
        # Step 3: Training
        print("\nStep 3: Training model...")
        try:
            trainer = ModelTrainer(config)
            
            # Check for existing checkpoints to resume from
            experiment_name = "pipeline_test"
            project_name = f"{config.get('project', {}).get('name', 'object_detection')}_{experiment_name}"
            checkpoint_dir = Path("output/models") / project_name / "weights"
            last_checkpoint = checkpoint_dir / "last.pt" if checkpoint_dir.exists() else None
            
            resume = False
            resume_path = None
            if last_checkpoint and last_checkpoint.exists():
                resume = True
                resume_path = str(last_checkpoint)
                print(f"Found checkpoint: {resume_path}")
                print("Resuming training from checkpoint...")
            
            result = trainer.train(
                data_yaml=data_yaml,
                experiment_name=experiment_name,
                project="output/models",
                resume=resume,
                resume_path=resume_path
            )
            
            if result.get('success'):
                model_path = result.get('model_path', '')
                print(f"Training completed! Model: {model_path}")
            else:
                print(f"Training failed: {result.get('error', 'Unknown error')}")
                return 1
        except Exception as e:
            print(f"Training error: {e}")
            # Try to use existing model
            existing_model = Path("output/models/yolov8n_fixed_224/weights/best.pt")
            if existing_model.exists():
                print(f"Using existing model: {existing_model}")
                model_path = str(existing_model)
            else:
                return 1
    
    # Step 4: Convert to ONNX (all precisions)
    print("\nStep 4: Converting to ONNX (FP32, FP16, INT8)...")
    try:
        trainer = ModelTrainer(config)
        onnx_paths = trainer.convert_to_onnx(
            model_path=model_path,
            input_size=config.get('model', {}).get('input_size', 416),
            output_dir="output/onnx",
            precisions=['fp32', 'fp16', 'int8']
        )
        print(f"ONNX conversion completed!")
        for precision, path in onnx_paths.items():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"  {precision.upper()}: {path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Comparison (all precisions)
    print("\nStep 5: Running model comparison (all precisions)...")
    try:
        input_size = config.get('model', {}).get('input_size', 416)
        print(f"  Using input size: {input_size}x{input_size}")
        
        # Evaluate PyTorch model
        print("\n  Evaluating PyTorch model...")
        evaluator_pt = ModelEvaluator(model_path, model_type="pytorch")
        results_pt = evaluator_pt.benchmark_inference_speed(num_runs=50, input_size=input_size)
        print(f"    PyTorch - FPS: {results_pt['fps']:.2f}, Avg time: {results_pt['avg_inference_time']*1000:.2f}ms")
        
        # Evaluate all ONNX precisions
        all_results = {'pytorch': results_pt}
        
        for precision in ['fp32', 'fp16', 'int8']:
            if precision in onnx_paths:
                print(f"\n  Evaluating ONNX {precision.upper()} model...")
                try:
                    evaluator_onnx = ModelEvaluator(onnx_paths[precision], model_type="onnx")
                    results_onnx = evaluator_onnx.benchmark_inference_speed(num_runs=50, input_size=input_size)
                    all_results[precision] = results_onnx
                    print(f"    ONNX {precision.upper()} - FPS: {results_onnx['fps']:.2f}, Avg time: {results_onnx['avg_inference_time']*1000:.2f}ms")
                except Exception as e:
                    print(f"    Failed to evaluate {precision.upper()}: {e}")
        
        # Comparison table
        print("\n" + "=" * 80)
        print("COMPREHENSIVE COMPARISON RESULTS")
        print("=" * 80)
        print(f"{'Model':<15} {'FPS':<12} {'Latency (ms)':<15} {'Size (MB)':<12}")
        print("-" * 80)
        
        # PyTorch
        pt_size = Path(model_path).stat().st_size / (1024 * 1024)
        print(f"{'PyTorch':<15} {results_pt['fps']:<12.2f} {results_pt['avg_inference_time']*1000:<15.2f} {pt_size:<12.2f}")
        
        # ONNX models
        for precision in ['fp32', 'fp16', 'int8']:
            if precision in all_results:
                onnx_size = Path(onnx_paths[precision]).stat().st_size / (1024 * 1024)
                results = all_results[precision]
                print(f"{'ONNX ' + precision.upper():<15} {results['fps']:<12.2f} {results['avg_inference_time']*1000:<15.2f} {onnx_size:<12.2f}")
        
        print("=" * 80)
        
        # Calculate speedups
        if 'fp32' in all_results:
            speedup_fp32 = all_results['fp32']['fps'] / results_pt['fps'] if results_pt['fps'] > 0 else 0
            print(f"\nSpeedup vs PyTorch:")
            print(f"  ONNX FP32: {speedup_fp32:.2f}x")
            
            if 'fp16' in all_results:
                speedup_fp16 = all_results['fp16']['fps'] / results_pt['fps'] if results_pt['fps'] > 0 else 0
                print(f"  ONNX FP16: {speedup_fp16:.2f}x")
            
            if 'int8' in all_results:
                speedup_int8 = all_results['int8']['fps'] / results_pt['fps'] if results_pt['fps'] > 0 else 0
                print(f"  ONNX INT8: {speedup_int8:.2f}x")
        
        # Size reduction
        if 'fp32' in onnx_paths:
            fp32_size = Path(onnx_paths['fp32']).stat().st_size / (1024 * 1024)
            print(f"\nSize Reduction vs FP32:")
            if 'fp16' in onnx_paths:
                fp16_size = Path(onnx_paths['fp16']).stat().st_size / (1024 * 1024)
                reduction = (1 - fp16_size / fp32_size) * 100
                print(f"  FP16: {reduction:.1f}% smaller ({fp16_size:.2f} MB vs {fp32_size:.2f} MB)")
            if 'int8' in onnx_paths:
                int8_size = Path(onnx_paths['int8']).stat().st_size / (1024 * 1024)
                reduction = (1 - int8_size / fp32_size) * 100
                print(f"  INT8: {reduction:.1f}% smaller ({int8_size:.2f} MB vs {fp32_size:.2f} MB)")
            
    except Exception as e:
        print(f"  Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        print("   (This is okay if dataset is not available)")
    
    print("\n Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

