#!/usr/bin/env python3
"""
Quick test of the training pipeline
"""

import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer import ModelTrainer

def load_config_from_file(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Simple config for quick test
    config = {
        "model": {
            "architecture": "yolov8n",
            "imgsz": 160,
        },
        "training": {
            "epochs": 2,  # Very quick test
            "batch_size": 8,
            "learning_rate": 0.001,
            "device": "cpu",
            "patience": 5,
        },
        "data": {
            "classes": ["object"]  # Placeholder
        }
    }

    trainer = ModelTrainer(config)
    print('Testing basic training with yolov8n_160 (2 epochs)...')

    try:
        result = trainer.train_enhanced(
            data_yaml='consolidated_dataset/data.yaml',
            experiment_name='test_yolov8n_160',
            project='output/test_models',
            name='test_yolov8n_160'
        )
        print('Training completed successfully!')
        print(f'Model saved to: {result["model_path"]}')
        return True
    except Exception as e:
        print(f'Training failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
