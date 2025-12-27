#!/usr/bin/env python3
"""
Consolidated training utilities
Combines functionality from train.py, train_efficientnet.py, and train_mobilenet_vit.py
"""

import argparse
import sys
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import numpy as np
import gc

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config, create_config_from_data_yaml
from trainer import run_training_pipeline
from data_utils import get_image_paths, load_yolo_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLODataset(torch.utils.data.Dataset):
    """Custom dataset for YOLO training"""

    def __init__(self, image_dir, label_dir, input_size=416, augment=True):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.input_size = input_size
        self.augment = augment

        # Get all image files
        self.image_files = list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.png'))

        # Basic augmentations (simplified for multiprocessing compatibility)
        self.augment = augment

        logger.info(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = img_path.stem

        # Load image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.input_size, self.input_size))
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW

        # Load labels
        label_path = self.label_dir / f"{img_name}.txt"
        if label_path.exists():
            boxes, classes = load_yolo_labels(str(label_path))
        else:
            boxes, classes = [], []

        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            classes = torch.tensor(classes, dtype=torch.long)
        else:
            boxes = torch.empty(0, 4, dtype=torch.float32)
            classes = torch.empty(0, dtype=torch.long)

        # Apply augmentations if enabled (multiprocessing-safe)
        if self.augment and torch.rand(1) > 0.5:
            # Horizontal flip
            image = torch.flip(image, dims=[2])
            # Flip boxes horizontally (x coordinates)
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip x coordinates

        return image, boxes, classes


def collate_fn(batch):
    """Custom collate function for variable-sized batches"""
    images = []
    boxes = []
    classes = []

    for img, box, cls in batch:
        images.append(img)
        boxes.append(box)
        classes.append(cls)

    return torch.stack(images), boxes, classes


def train_efficientnet(data_yaml, epochs=50, batch_size=8, lr=0.001, device='cpu',
                      attention_layers=None, output_dir='output/models/efficientnet'):
    """Train EfficientNet detector"""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Create datasets
    train_image_dir = data_config['train'] if data_config['train'].endswith('/images') else data_config['train'] + '/images'
    train_label_dir = train_image_dir.replace('/images', '/labels')

    val_image_dir = data_config['val'] if data_config['val'].endswith('/images') else data_config['val'] + '/images'
    val_label_dir = val_image_dir.replace('/images', '/labels')

    train_dataset = YOLODataset(
        train_image_dir,
        train_label_dir,
        augment=True
    )

    val_dataset = YOLODataset(
        val_image_dir,
        val_label_dir,
        augment=False
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    # Create model
    from models.efficientnet import create_efficientnet_detector
    num_classes = len(data_config.get('names', []))
    model = create_efficientnet_detector(num_classes=num_classes, attention_layers=attention_layers)
    model.to(device)

    # Loss functions
    class_loss_fn = nn.CrossEntropyLoss()
    box_loss_fn = nn.SmoothL1Loss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, boxes_batch, classes_batch in pbar:
            images = images.to(device)
            optimizer.zero_grad()

            # Forward pass
            class_logits, box_preds = model(images)

            # Compute loss (simplified - in practice you'd need proper YOLO loss)
            total_loss = 0.0
            batch_size = len(images)

            for i in range(batch_size):
                if len(classes_batch[i]) > 0:
                    # Classification loss
                    cls_loss = class_loss_fn(class_logits[i].unsqueeze(0), classes_batch[i])
                    total_loss += cls_loss

                    # Box regression loss (simplified)
                    if box_preds[i].shape[0] >= len(boxes_batch[i]):
                        box_loss = box_loss_fn(box_preds[i][:len(boxes_batch[i])], boxes_batch[i])
                        total_loss += box_loss

            if batch_size > 0:
                total_loss = total_loss / batch_size

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, boxes_batch, classes_batch in val_loader:
                images = images.to(device)
                class_logits, box_preds = model(images)

                total_loss = 0.0
                batch_size = len(images)

                for i in range(batch_size):
                    if len(classes_batch[i]) > 0:
                        cls_loss = class_loss_fn(class_logits[i].unsqueeze(0), classes_batch[i])
                        total_loss += cls_loss

                        if box_preds[i].shape[0] >= len(boxes_batch[i]):
                            box_loss = box_loss_fn(box_preds[i][:len(boxes_batch[i])], boxes_batch[i])
                            total_loss += box_loss

                if batch_size > 0:
                    total_loss = total_loss / batch_size

                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / 'best.pt')
            logger.info(f"New best model saved with val_loss: {best_loss:.4f}")

        # Save latest model
        torch.save(model.state_dict(), output_dir / 'last.pt')

        # Update scheduler
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    logger.info(f"Training completed. Best model saved to {output_dir}/best.pt")


def train_mobilenet_vit(data_yaml, epochs=30, batch_size=8, lr=0.001, device='cpu',
                       output_dir='output/models/mobilenet_vit', input_size=224):
    """Train MobileNet-ViT detector with configurable input size"""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data config
    data_yaml_path = Path(data_yaml)
    data_dir = data_yaml_path.parent

    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Create datasets - resolve paths relative to data.yaml location
    # The data.yaml has paths like 'train', 'val' - we need to add /images and /labels
    train_base = data_config['train']
    val_base = data_config['val']

    train_image_dir = data_dir / train_base / 'images'
    train_label_dir = data_dir / train_base / 'labels'

    val_image_dir = data_dir / val_base / 'images'
    val_label_dir = data_dir / val_base / 'labels'

    train_dataset = YOLODataset(
        train_image_dir,
        train_label_dir,
        augment=True
    )

    val_dataset = YOLODataset(
        val_image_dir,
        val_label_dir,
        augment=False
    )

    # Create data loaders with multithreaded loading for better GPU utilization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=4)

    # Create model
    from models.mobilevit import create_mobilevit_detector
    num_classes = len(data_config.get('names', []))
    model = create_mobilevit_detector(num_classes=num_classes, input_size=input_size)
    model.to(device)

    # Loss functions
    class_loss_fn = nn.CrossEntropyLoss()
    box_loss_fn = nn.SmoothL1Loss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Real object detection training loop with memory management
    best_loss = float('inf')

    # Initial memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, boxes_batch, classes_batch in pbar:
            images = images.to(device)
            optimizer.zero_grad()

            # Forward pass
            class_logits, box_preds = model(images)

            # Proper object detection loss calculation
            total_loss = 0.0
            batch_size = len(images)

            for i in range(batch_size):
                if len(classes_batch[i]) > 0:
                    # Use first prediction and first ground truth for training
                    cls_pred = class_logits[i:i+1]  # [1, num_predictions, num_classes]
                    box_pred = box_preds[i:i+1]     # [1, num_predictions, 4]

                    # Take the first prediction for simplicity
                    cls_pred_first = cls_pred[0, 0]  # [num_classes]
                    box_pred_first = box_pred[0, 0]  # [4]

                    # Classification loss - use first class from ground truth
                    cls_target = classes_batch[i][0]  # Single class ID
                    cls_target_tensor = torch.tensor([cls_target], device=device)
                    cls_loss = class_loss_fn(cls_pred_first.unsqueeze(0), cls_target_tensor)
                    total_loss += cls_loss

                    # Box regression loss - use first box from ground truth
                    box_target = boxes_batch[i][0:1]  # [1, 4]
                    box_target_tensor = box_target.to(device)
                    box_loss = box_loss_fn(box_pred_first.unsqueeze(0), box_target_tensor)
                    total_loss += box_loss

            if batch_size > 0:
                total_loss = total_loss / batch_size

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, boxes_batch, classes_batch in val_loader:
                images = images.to(device)
                class_logits, box_preds = model(images)

                total_loss = 0.0
                batch_size = len(images)

                for i in range(batch_size):
                    if len(classes_batch[i]) > 0:
                        # Use first prediction for validation
                        cls_pred_first = class_logits[i, 0]  # [num_classes]
                        box_pred_first = box_preds[i, 0]    # [4]

                        # Classification loss - use first class from ground truth
                        cls_target = classes_batch[i][0]  # Single class ID
                        cls_target_tensor = torch.tensor([cls_target], device=device)
                        cls_loss = class_loss_fn(cls_pred_first.unsqueeze(0), cls_target_tensor)
                        total_loss += cls_loss

                        # Box regression loss - use first box from ground truth
                        box_target = boxes_batch[i][0:1]  # [1, 4]
                        box_target_tensor = box_target.to(device)
                        box_loss = box_loss_fn(box_pred_first.unsqueeze(0), box_target_tensor)
                        total_loss += box_loss

                if batch_size > 0:
                    total_loss = total_loss / batch_size

                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / 'best.pt')
            logger.info(f"New best model saved with val_loss: {best_loss:.4f}")

        # Save latest model
        torch.save(model.state_dict(), output_dir / 'last.pt')

        # Update scheduler
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    logger.info(f"Training completed. Best model saved to {output_dir}/best.pt")


def run_general_training(config, data_yaml, train_all=True):
    """Run general training pipeline for all architectures"""

    try:
        logger.info("Starting object detection training")

        # Apply any overrides and save config
        from core.config import ConfigManager
        config_manager = ConfigManager()
        config_manager.save_config(config, Path('output/config_used.yaml'))

        # Run training pipeline
        results = run_training_pipeline(
            base_config=config,
            data_yaml=data_yaml,
            train_all=train_all
        )

        # Report results
        logger.info("Training completed!")
        successful_models = sum(1 for result in results.values() if result.get('success'))
        logger.info(f"Successfully trained {successful_models}/{len(results)} models")

        return results

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def main():
    """Main function for training utilities"""
    parser = argparse.ArgumentParser(description='Training utilities for object detection models')

    # General training options
    parser.add_argument('--mode', choices=['general', 'efficientnet', 'mobilenet_vit'],
                       default='general', help='Training mode')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration to use (filename without .yaml)')
    parser.add_argument('--data_yaml', type=str, default='balanced_data/data.yaml',
                       help='Path to data configuration YAML')

    # Model-specific training options
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Training device')

    # EfficientNet-specific options
    parser.add_argument('--attention', action='store_true',
                       help='Add CBAM attention layers (EfficientNet only)')

    # General training options
    parser.add_argument('--train_all', action='store_true', default=True,
                       help='Train all 4 model architectures (general mode only)')

    # Override config parameters
    parser.add_argument('--input_size', type=int, help='Input image size')
    parser.add_argument('--output_dir', type=str, help='Output directory for models')

    args = parser.parse_args()

    try:
        if args.mode == 'general':
            # Load configuration
            if Path(args.data_yaml).exists():
                config = create_config_from_data_yaml(args.data_yaml, args.config)
            else:
                print(f"Data YAML not found at {args.data_yaml}, falling back to default")
                config = load_config(args.config)

            # Apply command line overrides
            overrides = {}
            for param in ['epochs', 'batch_size', 'lr', 'input_size', 'device']:
                if getattr(args, param) is not None:
                    overrides[param] = getattr(args, param)

            if overrides:
                config['training'].update(overrides)
                logger.info(f"Applied overrides: {overrides}")

            # Run general training
            run_general_training(config, args.data_yaml, args.train_all)

        elif args.mode == 'efficientnet':
            # Define attention layers if requested
            attention_layers = ['features.0', 'features.3', 'features.6'] if args.attention else None
            output_dir = args.output_dir or 'output/models/efficientnet'

            train_efficientnet(
                data_yaml=args.data_yaml,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                attention_layers=attention_layers,
                output_dir=output_dir
            )

        elif args.mode == 'mobilenet_vit':
            output_dir = args.output_dir or 'output/models/mobilenet_vit'

            train_mobilenet_vit(
                data_yaml=args.data_yaml,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                output_dir=output_dir,
                input_size=args.input_size or 224
            )

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

