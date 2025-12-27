#!/usr/bin/env python3
"""
Complete training script for all model architectures
Supports YOLOv8 variants, MobileNet, and EfficientNet
Uses the proper trainer.py framework
"""

import sys
from pathlib import Path
import logging
import time
from typing import Dict, List
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_complete_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model configurations - each dict defines the model setup
MODEL_CONFIGS = {
    # YOLOv8 Nano variants
    'yolov8n_160': {
        'model': {
            'architecture': 'yolov8n',
            'imgsz': 160,
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },
    'yolov8n_192': {
        'model': {
            'architecture': 'yolov8n',
            'imgsz': 192,
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },
    'yolov8n_224': {
        'model': {
            'architecture': 'yolov8n',
            'imgsz': 224,
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },

    # YOLOv8 Small variants
    'yolov8s_confidence': {
        'model': {
            'architecture': 'yolov8s_baseline',
            'imgsz': 416,
        },
        'training': {
            'epochs': 100,
            'batch_size': 24,
            'learning_rate': 0.0005,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },
    'yolov8s_cbam_confidence': {
        'model': {
            'architecture': 'yolov8s_cbam',
            'imgsz': 416,
        },
        'training': {
            'epochs': 100,
            'batch_size': 20,
            'learning_rate': 0.0005,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },

    # YOLOv8 Medium and Large
    'yolov8m_confidence': {
        'model': {
            'architecture': 'yolov8m',
            'imgsz': 416,
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.0005,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },
    'yolov8l_confidence': {
        'model': {
            'architecture': 'yolov8l',
            'imgsz': 416,
        },
        'training': {
            'epochs': 100,
            'batch_size': 12,
            'learning_rate': 0.0005,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },

    # MobileNet variants
    'mobilenet_160': {
        'model': {
            'architecture': 'mobilenet_vit',
            'imgsz': 160,
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },
    'mobilenet_192': {
        'model': {
            'architecture': 'mobilenet_vit',
            'imgsz': 192,
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },
    'mobilenet_224': {
        'model': {
            'architecture': 'mobilenet_vit',
            'imgsz': 224,
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },
    'mobilenet_confidence': {
        'model': {
            'architecture': 'mobilenet_vit',
            'imgsz': 416,
        },
        'training': {
            'epochs': 100,
            'batch_size': 20,
            'learning_rate': 0.001,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },

    # EfficientNet
    'efficientnet_confidence': {
        'model': {
            'architecture': 'efficientnet',
            'imgsz': 416,
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'device': '0',
            'patience': 20,
        },
        'data': {
            'classes': ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
        }
    },
}

def train_single_model(model_name: str, config: Dict) -> bool:
    """
    Train a single model

    Args:
        model_name: Name of the model
        config: Model configuration dictionary

    Returns:
        bool: True if training completed successfully
    """
    logger.info(f"Starting training for {model_name}")

    try:
        # Create trainer
        trainer = ModelTrainer(config)

        # Train the model
        result = trainer.train_enhanced(
            data_yaml='consolidated_dataset/data.yaml',
            experiment_name=model_name,
            project='output/models',
            name=model_name
        )

        if result['success']:
            logger.info(f"Successfully completed training for {model_name}")
            logger.info(f"Model saved to: {result['model_path']}")
            return True
        else:
            logger.error(f"Training failed for {model_name}: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        logger.error(f"Unexpected error training {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def train_all_models(model_list: List[str] = None) -> Dict[str, bool]:
    """
    Train all specified models

    Args:
        model_list: List of model names to train (default: all)

    Returns:
        Dict mapping model names to success status
    """
    if model_list is None:
        model_list = list(MODEL_CONFIGS.keys())

    results = {}
    total_models = len(model_list)

    logger.info("="*80)
    logger.info("STARTING COMPREHENSIVE MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Training {total_models} models")
    logger.info("="*80)

    model_count = 0

    for model_name in model_list:
        if model_name not in MODEL_CONFIGS:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        model_count += 1
        logger.info(f"[{model_count}/{total_models}] Training {model_name}")

        # Get config
        config = MODEL_CONFIGS[model_name]

        # Train the model
        success = train_single_model(model_name, config)
        results[model_name] = success

        # Brief pause between models to allow system to settle
        if model_count < total_models:
            logger.info("Pausing for 60 seconds before next model...")
            time.sleep(60)

    return results

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train all models using the complete framework')
    parser.add_argument('--models', nargs='+',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Models to train (default: all)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')

    args = parser.parse_args()

    # Update epochs in all configs if specified
    if args.epochs != 100:
        for config in MODEL_CONFIGS.values():
            config['training']['epochs'] = args.epochs

    logger.info("="*80)
    logger.info("STARTING COMPREHENSIVE MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Models: {args.models or 'all'}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("="*80)

    # Run training
    start_time = time.time()
    results = train_all_models(args.models)
    end_time = time.time()

    # Log results
    logger.info("="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")

    successful = sum(results.values())
    total = len(results)

    logger.info(f"Results: {successful}/{total} models trained successfully")

    for model, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {model}: {status}")

    # Write summary file
    with open("output/full_training_complete_summary.txt", 'w') as f:
        f.write("Complete Model Training Summary\n")
        f.write("================================\n\n")
        f.write(f"Total models: {total}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {total - successful}\n")
        f.write(f"Total time: {end_time - start_time:.2f} seconds\n\n")

        f.write("Detailed Results:\n")
        for model, success in results.items():
            f.write(f"  {model}: {'SUCCESS' if success else 'FAILED'}\n")

    logger.info("Summary written to output/full_training_complete_summary.txt")

    # Exit with appropriate code
    sys.exit(0 if successful == total else 1)

if __name__ == "__main__":
    main()



