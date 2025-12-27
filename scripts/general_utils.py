#!/usr/bin/env python3
"""
Consolidated utility functions
Combines configuration utilities, retraining utilities, and miscellaneous helpers
"""

import yaml
import os
import subprocess
import time
from pathlib import Path


class ConfidenceTrainingConfig:
    """Generate confidence-optimized training configurations for all architectures"""

    def __init__(self):
        self.base_config = {
            'task': 'detect',
            'mode': 'train',
            'data': 'enhanced_dataset/data.yaml',
            'epochs': 100,  # 100 epochs as requested
            'patience': 20,
            'batch': 16,  # Larger batch for GPU training
            'imgsz': 416,
            'save': True,
            'save_period': 10,
            'cache': 'disk',  # Use disk cache for stability (deterministic)
            'device': '0',  # GPU training
            'workers': 4,  # Reduced for Windows multiprocessing stability
            'project': 'output',
            'name': 'confidence',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 15,
            'resume': False,
            'amp': True,  # Enable AMP for GPU acceleration
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': True,
            'compile': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.1,  # Slight dropout for confidence calibration
            'val': True,
            'split': 'val',
            'save_json': False,
            'conf': 0.1,  # Lower confidence threshold for training
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'plots': True
        }

    def generate_yolov8s_config(self):
        """Generate YOLOv8s confidence configuration"""
        config = self.base_config.copy()
        config.update({
            'model': 'models/yolov8s.pt',
            'name': 'yolov8s_confidence',
            'batch': 16,
            'imgsz': 416,
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        })
        return config

    def generate_yolov8m_config(self):
        """Generate YOLOv8m confidence configuration"""
        config = self.base_config.copy()
        config.update({
            'model': 'models/yolov8m.pt',
            'name': 'yolov8m_confidence',
            'batch': 12,
            'imgsz': 416,
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        })
        return config

    def generate_yolov8l_config(self):
        """Generate YOLOv8l confidence configuration"""
        config = self.base_config.copy()
        config.update({
            'model': 'models/yolov8l.pt',
            'name': 'yolov8l_confidence',
            'batch': 8,
            'imgsz': 416,
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        })
        return config

    def generate_efficientnet_config(self):
        """Generate EfficientNet confidence configuration"""
        config = self.base_config.copy()
        config.update({
            'model': 'models/efficientnet.pt',
            'name': 'efficientnet_confidence',
            'batch': 12,
            'imgsz': 416,
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'warmup_epochs': 5.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        })
        return config

    def generate_mobilenet_config(self):
        """Generate MobileNet-ViT confidence configuration"""
        config = self.base_config.copy()
        config.update({
            'model': 'models/mobilenet_vit.pt',
            'name': 'mobilenet_confidence',
            'batch': 16,
            'imgsz': 416,
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'warmup_epochs': 5.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        })
        return config

    def generate_all_configs(self, output_dir="configs/training"):
        """Generate all confidence configurations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        configs = {
            'yolov8s_confidence.yaml': self.generate_yolov8s_config(),
            'yolov8m_confidence.yaml': self.generate_yolov8m_config(),
            'yolov8l_confidence.yaml': self.generate_yolov8l_config(),
            'efficientnet_confidence.yaml': self.generate_efficientnet_config(),
            'mobilenet_confidence.yaml': self.generate_mobilenet_config()
        }

        for filename, config in configs.items():
            output_path = output_dir / filename
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Generated {output_path}")

        return list(configs.keys())


class RetrainingManager:
    """Manage retraining operations for confidence-optimized models"""

    def __init__(self, device='cpu'):
        self.device = device

    def run_training(self, config_file, model_name):
        """Run training for a specific model configuration"""
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        cmd = f"yolo train cfg=\"{config_file}\""

        try:
            print(f"Running: {cmd}")
            start_time = time.time()

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            end_time = time.time()
            duration = end_time - start_time

            print(f"Training completed in {duration:.1f} seconds")

            if result.returncode == 0:
                print(f"✓ {model_name} training successful")
            else:
                print(f"✗ {model_name} training failed")
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                print("STDERR:", result.stderr[-500:])  # Last 500 chars

            return result.returncode == 0

        except Exception as e:
            print(f"Error running {model_name}: {e}")
            return False

    def retrain_confidence_models_cpu(self):
        """Retrain all confidence models on CPU"""
        print("Starting confidence-focused retraining on CPU...")
        print("This will take significant time - monitor progress in output/confidence/")
        print(f"Working directory: {os.getcwd()}")

        models = [
            ("configs/training/yolov8s_confidence.yaml", "YOLOv8s (confidence-optimized)"),
            ("configs/training/yolov8m_confidence.yaml", "YOLOv8m (confidence-optimized)"),
            ("configs/training/yolov8l_confidence.yaml", "YOLOv8l (confidence-optimized)"),
            ("configs/training/efficientnet_confidence.yaml", "EfficientNet (confidence-optimized)"),
            ("configs/training/mobilenet_confidence.yaml", "MobileNet-ViT (confidence-optimized)")
        ]

        results = {}
        for config_file, model_name in models:
            success = self.run_training(config_file, model_name)
            results[model_name] = success

        # Summary
        print(f"\n{'='*60}")
        print("RETRAINING SUMMARY")
        print(f"{'='*60}")

        successful = sum(results.values())
        total = len(results)

        for model_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{model_name}: {status}")

        print(f"\nTotal: {successful}/{total} models trained successfully")

        if successful == total:
            print("[COMPLETE] All models retrained successfully!")
        else:
            print(f"[WARNING]  {total - successful} models failed to train")

        return results

    def retrain_confidence_models_gpu(self):
        """Retrain all confidence models on GPU"""
        print("Starting confidence-focused retraining on GPU...")
        print("This will take significant time - monitor progress in output/confidence/")
        print(f"Working directory: {os.getcwd()}")

        models = [
            ("configs/training/yolov8s_confidence.yaml", "YOLOv8s (confidence-optimized)"),
            ("configs/training/yolov8m_confidence.yaml", "YOLOv8m (confidence-optimized)"),
            ("configs/training/yolov8l_confidence.yaml", "YOLOv8l (confidence-optimized)"),
            ("configs/training/efficientnet_confidence.yaml", "EfficientNet (confidence-optimized)"),
            ("configs/training/mobilenet_confidence.yaml", "MobileNet-ViT (confidence-optimized)")
        ]

        results = {}
        for config_file, model_name in models:
            success = self.run_training(config_file, model_name)
            results[model_name] = success

        # Summary
        print(f"\n{'='*60}")
        print("RETRAINING SUMMARY")
        print(f"{'='*60}")

        successful = sum(results.values())
        total = len(results)

        for model_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{model_name}: {status}")

        print(f"\nTotal: {successful}/{total} models trained successfully")

        if successful == total:
            print("[COMPLETE] All models retrained successfully!")
        else:
            print(f"[WARNING]  {total - successful} models failed to train")

        return results


def setup_webcam_test(model_path, config_path="default", output_dir="webcam_test"):
    """Setup webcam testing environment"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create test configuration
    test_config = {
        'model_path': str(model_path),
        'config': config_path,
        'output_dir': str(output_dir),
        'test_videos': [
            str(output_dir / 'test_video.mp4'),
            str(output_dir / 'calibration_video.mp4')
        ],
        'test_images': str(output_dir / 'test_images'),
        'confidence_thresholds': [0.3, 0.5, 0.7],
        'temporal_filtering': True,
        'filter_window': 5
    }

    config_path = output_dir / 'webcam_test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)

    print(f"Webcam test setup complete. Configuration saved to {config_path}")
    return config_path


def main():
    """Main function for utilities"""
    import argparse

    parser = argparse.ArgumentParser(description='General utilities')
    parser.add_argument('--action', choices=[
        'generate_configs', 'retrain_cpu', 'retrain_gpu', 'setup_webcam'
    ], required=True, help='Action to perform')
    parser.add_argument('--model_path', help='Model path (for webcam setup)')
    parser.add_argument('--config', default='default', help='Configuration name')
    parser.add_argument('--output_dir', help='Output directory')

    args = parser.parse_args()

    if args.action == 'generate_configs':
        config_gen = ConfidenceTrainingConfig()
        configs = config_gen.generate_all_configs(args.output_dir)
        print(f"Generated {len(configs)} configuration files")

    elif args.action == 'retrain_cpu':
        manager = RetrainingManager(device='cpu')
        results = manager.retrain_confidence_models_cpu()

    elif args.action == 'retrain_gpu':
        manager = RetrainingManager(device='cuda')
        results = manager.retrain_confidence_models_gpu()

    elif args.action == 'setup_webcam':
        if not args.model_path:
            parser.error("--model_path required for webcam setup")
        config_path = setup_webcam_test(args.model_path, args.config, args.output_dir)
        print(f"Webcam test configured: {config_path}")


if __name__ == '__main__':
    main()
