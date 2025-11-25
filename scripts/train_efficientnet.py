#!/usr/bin/env python3
"""
Train EfficientNet object detection model
Custom training script for EfficientNet-based detector
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.efficientnet import create_efficientnet_detector
from utils.data_utils import get_image_paths, load_yolo_labels

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

        # Basic augmentations
        self.transforms = []
        if augment:
            self.transforms = [
                lambda x: torch.flip(x, dims=[2]) if torch.rand(1) > 0.5 else x,  # horizontal flip
            ]

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


def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet detector')
    parser.add_argument('--data_yaml', type=str, default='balanced_data/data.yaml',
                       help='Path to data configuration YAML')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Training device')
    parser.add_argument('--attention', action='store_true',
                       help='Add CBAM attention layers')
    parser.add_argument('--output_dir', type=str, default='output/models/efficientnet',
                       help='Output directory for model weights')

    args = parser.parse_args()

    # Define attention layers if requested
    attention_layers = ['features.0', 'features.3', 'features.6'] if args.attention else None

    # Train the model
    train_efficientnet(
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        attention_layers=attention_layers,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
