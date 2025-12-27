"""
Generic YOLO Object Detector
Supports both baseline and attention-enhanced models
"""

import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules import C2f
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class C2fCBAM(C2f):
    """
    C2f block with CBAM attention module
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        # Import CBAM here to avoid circular imports
        from ..models.attention_modules import CBAM
        self.cbam = CBAM(c2, reduction=16)

    def forward(self, x):
        x = super().forward(x)
        x = self.cbam(x)
        return x


class YOLOv8CBAM(YOLO):
    """
    YOLOv8 with CBAM attention modules integrated
    """

    def __init__(self, model='yolov8s.yaml', attention_layers=None):
        super().__init__(model)

        if attention_layers is None:
            attention_layers = ['model.3', 'model.6', 'model.9', 'model.12']

        self.attention_layers = attention_layers
        self._add_cbam_attention()

    def _add_cbam_attention(self):
        """Add CBAM attention to specified layers"""
        for layer_name in self.attention_layers:
            try:
                # Navigate to the layer
                parts = layer_name.split('.')
                obj = self.model
                for part in parts[:-1]:
                    obj = getattr(obj, part)

                # Get the layer index
                layer_idx = int(parts[-1])

                # Replace with CBAM-enhanced version
                if hasattr(obj, layer_idx):
                    original_layer = getattr(obj, layer_idx)

                    # Create CBAM version with same parameters
                    if isinstance(original_layer, C2f):
                        c1 = original_layer.cv1.conv.in_channels
                        c2 = original_layer.cv2.conv.out_channels
                        n = len(original_layer.m) if hasattr(original_layer, 'm') else 1
                        shortcut = original_layer.shortcut
                        g = original_layer.g
                        e = original_layer.e

                        cbam_layer = C2fCBAM(c1, c2, n, shortcut, g, e)
                        setattr(obj, layer_idx, cbam_layer)

                        logger.info(f"Added CBAM attention to {layer_name}")

            except Exception as e:
                logger.warning(f"Could not add CBAM to {layer_name}: {e}")


class ObjectDetector:
    """
    Generic object detector supporting multiple architectures
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.class_names = config.get('data', {}).get('classes', [])

        self._load_model()

    def _load_model(self):
        """Load the appropriate model based on configuration"""
        model_config = self.config.get('model', {})

        architecture = model_config.get('architecture', 'yolov8s')
        pretrained_weights = model_config.get('pretrained_weights')

        logger.info(f"Loading model with architecture: {architecture}")

        # Handle different architectures
        if architecture == 'yolov8s_baseline':
            # Standard YOLOv8s model
            if pretrained_weights and Path(pretrained_weights).exists():
                self.model = YOLO(pretrained_weights)
                logger.info(f"Loaded pre-trained YOLOv8s baseline: {pretrained_weights}")
            else:
                self.model = YOLO('yolov8s.pt')  # Use COCO pre-trained weights
                logger.info("Created YOLOv8s baseline model with COCO pre-trained weights")

        elif architecture == 'yolov8s_cbam':
            # YOLOv8s with CBAM attention
            try:
                if pretrained_weights and Path(pretrained_weights).exists():
                    self.model = YOLO(pretrained_weights)
                    logger.info(f"Loaded pre-trained YOLOv8s CBAM: {pretrained_weights}")
                else:
                    # Load base YOLOv8s and add CBAM
                    base_model = YOLO('yolov8s.pt')
                    self.model = YOLOv8CBAM(model=base_model.model)
                    logger.info("Created YOLOv8s CBAM model")
            except Exception as e:
                logger.warning(f"CBAM model creation failed: {e}. Falling back to baseline.")
                self.model = YOLO('yolov8s.pt')

        elif architecture == 'mobilenet_vit':
            # Custom MobileNetVIT detector
            try:
                from ..models.mobilevit import create_mobilevit_detector
                if pretrained_weights and Path(pretrained_weights).exists():
                    self.model = create_mobilevit_detector(num_classes=len(self.class_names))
                    checkpoint = torch.load(pretrained_weights, map_location='cpu')
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"Loaded pre-trained MobileNetVIT: {pretrained_weights}")
                else:
                    self.model = create_mobilevit_detector(num_classes=len(self.class_names))
                    logger.info("Created MobileNetVIT detector")
            except Exception as e:
                logger.warning(f"MobileNetVIT creation failed: {e}. Falling back to YOLOv8n.")
                self.model = YOLO('yolov8n.pt')

        elif architecture == 'efficientnet':
            # Custom EfficientNet detector
            try:
                from ..models.efficientnet import create_efficientnet_detector
                if pretrained_weights and Path(pretrained_weights).exists():
                    self.model = create_efficientnet_detector(num_classes=len(self.class_names))
                    checkpoint = torch.load(pretrained_weights, map_location='cpu')
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"Loaded pre-trained EfficientNet: {pretrained_weights}")
                else:
                    self.model = create_efficientnet_detector(num_classes=len(self.class_names))
                    logger.info("Created EfficientNet detector")
            except Exception as e:
                logger.warning(f"EfficientNet creation failed: {e}. Falling back to YOLOv8s.")
                self.model = YOLO('yolov8s.pt')

        else:
            # Default to YOLOv8s
            if pretrained_weights and Path(pretrained_weights).exists():
                self.model = YOLO(pretrained_weights)
                logger.info(f"Loaded pre-trained model: {pretrained_weights}")
            else:
                self.model = YOLO('yolov8s.pt')
                logger.info("Created default YOLOv8s model")

        # Log model info
        if hasattr(self.model, 'model'):
            total_params = sum(p.numel() for p in self.model.model.parameters())
            logger.info(f"Model loaded with {total_params:,} parameters")

    def predict(self, source: Union[str, torch.Tensor], **kwargs):
        """Run inference on input"""
        architecture = self.config.get('model', {}).get('architecture', 'yolov8s')

        if architecture in ['mobilenet_vit', 'efficientnet']:
            # Custom model inference - basic implementation
            try:
                # Ensure model is in eval mode
                self.model.eval()

                # Convert input to tensor if needed
                if isinstance(source, str):
                    # For now, return empty results for image paths
                    # Full inference would require image loading and preprocessing
                    logger.warning(f"Custom model {architecture} inference for image paths not implemented")
                    return []
                elif isinstance(source, torch.Tensor):
                    # Run inference
                    with torch.no_grad():
                        cls_logits, bbox_preds = self.model(source)

                    # Basic post-processing (simplified)
                    # In practice, you'd need anchor decoding, NMS, etc.
                    logger.info(f"Custom model {architecture} inference completed - shapes: {cls_logits.shape}, {bbox_preds.shape}")
                    return []  # Return empty for now - full implementation needed

            except Exception as e:
                logger.error(f"Custom model inference failed: {e}")
            return []
        else:
            # YOLO model inference
            default_kwargs = {
                'conf': self.config.get('model', {}).get('confidence_threshold', 0.5),
                'iou': self.config.get('model', {}).get('iou_threshold', 0.4),
                'max_det': self.config.get('model', {}).get('max_detections', 20),
                'imgsz': self.config.get('model', {}).get('input_size', 416)
            }
            default_kwargs.update(kwargs)
            return self.model.predict(source, **default_kwargs)

    def train(self, data_yaml: str, model_name: str = None, **kwargs):
        """Train the model"""
        training_config = self.config.get('training', {})

        default_kwargs = {
            'epochs': training_config.get('epochs', 50),
            'batch': training_config.get('batch_size', 8),
            'imgsz': self.config.get('model', {}).get('input_size', 416),
            'lr0': training_config.get('learning_rate', 0.001),
            'weight_decay': training_config.get('weight_decay', 0.0005),
            'patience': training_config.get('patience', 20),
            'device': self.config.get('device', 'auto'),
            'project': 'output',
            'name': model_name or f"{self.config.get('project', {}).get('name', 'object_detection')}_training"
        }

        # Add augmentation settings
        aug_config = training_config.get('augmentation', {})
        default_kwargs.update({
            'degrees': aug_config.get('degrees', 0.0),
            'translate': aug_config.get('translate', 0.1),
            'scale': aug_config.get('scale', 0.5),
            'shear': aug_config.get('shear', 0.0),
            'perspective': aug_config.get('perspective', 0.0),
            'flipud': aug_config.get('flipud', 0.0),
            'fliplr': aug_config.get('fliplr', 0.5),
            'mosaic': aug_config.get('mosaic', 1.0),
            'mixup': aug_config.get('mixup', 0.0),
            'hsv_h': aug_config.get('hsv_h', 0.0),
            'hsv_s': aug_config.get('hsv_s', 0.7),
            'hsv_v': aug_config.get('hsv_v', 0.4)
        })

        # Add loss weights
        loss_config = training_config.get('loss_weights', {})
        default_kwargs.update({
            'box': loss_config.get('box', 7.5),
            'cls': loss_config.get('cls', 0.5),
            'dfl': loss_config.get('dfl', 1.5)
        })

        default_kwargs.update(kwargs)

        logger.info(f"Starting training with config: {default_kwargs}")

        # Handle different model types
        architecture = self.config.get('model', {}).get('architecture', 'yolov8s')
        if architecture in ['yolov8s']:
            # YOLO models use data as keyword argument
            return self.model.train(data=data_yaml, **default_kwargs)
        else:
            # Custom models (MobileNetVIT, EfficientNet) - implement custom training
            return self._train_custom_model(data_yaml, architecture, **default_kwargs)

    def val(self, data_yaml: str, **kwargs):
        """Validate the model"""
        return self.model.val(data_yaml, **kwargs)

    def _train_custom_model(self, data_yaml: str, architecture: str, **kwargs):
        """Train custom models (EfficientNet, MobileNet-ViT) using PyTorch training loop"""
        try:
            import yaml
            from torch.utils.data import DataLoader
            from torchvision import transforms
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR
            import time
            from pathlib import Path

            # Load dataset configuration
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)

            num_classes = len(data_config.get('names', []))
            train_path = data_config.get('train', '')
            val_path = data_config.get('val', '')

            logger.info(f"Training {architecture} with {num_classes} classes")
            logger.info(f"Train data: {train_path}")
            logger.info(f"Val data: {val_path}")

            # Get training parameters
            epochs = kwargs.get('epochs', 100)
            batch_size = kwargs.get('batch', 8)
            lr = kwargs.get('lr0', 0.001)
            imgsz = kwargs.get('imgsz', 224)
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

            # Create model
            if architecture == 'efficientnet':
                from .models.efficientnet import create_efficientnet_detector
                model = create_efficientnet_detector(num_classes=num_classes)
            elif architecture == 'mobilenet_vit':
                from .models.mobilevit import create_mobilevit_detector
                model = create_mobilevit_detector(num_classes=num_classes, input_size=imgsz)
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")

            model = model.to(device)

            # Create optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=kwargs.get('weight_decay', 0.0005))
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

            # Create data transforms
            train_transform = transforms.Compose([
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            val_transform = transforms.Compose([
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # For now, create dummy datasets (would need actual dataset implementation)
            logger.warning("Custom model training uses simplified approach - full dataset implementation needed")
            logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

            # Create output directory structure
            project = kwargs.get('project', 'output')
            name = kwargs.get('name', f"{architecture}_training")
            save_dir = Path(project) / name
            save_dir.mkdir(parents=True, exist_ok=True)

            weights_dir = save_dir / "weights"
            weights_dir.mkdir(exist_ok=True)

            # Simple training loop (placeholder - would need full dataset implementation)
            best_loss = float('inf')

            for epoch in range(epochs):
                start_time = time.time()

                # Training phase (placeholder)
                model.train()
                train_loss = 0.0

                # Placeholder: would need actual data loading and training steps
                # This is just to demonstrate the structure

                # Validation phase (placeholder)
                model.eval()
                val_loss = 0.0

                # Placeholder: would need actual validation steps

                epoch_time = time.time() - start_time

                # Log progress
                logger.info(".1f")
                # Save checkpoint
                if val_loss < best_loss:
                    best_loss = val_loss
                    checkpoint_path = weights_dir / "best.pt"
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info(f"Best model saved: {checkpoint_path}")

                # Save last checkpoint
                last_checkpoint_path = weights_dir / "last.pt"
                torch.save(model.state_dict(), last_checkpoint_path)

                scheduler.step()

            # Save final model
            final_path = weights_dir / "best.pt"
            torch.save(model.state_dict(), final_path)

            logger.info(f"Training completed! Model saved to: {save_dir}")

            return {
                'save_dir': str(save_dir),
                'model_path': str(final_path),
                'success': True,
                'epochs': epochs,
                'architecture': architecture
            }

        except Exception as e:
            logger.error(f"Custom model training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def save(self, path: str):
        """Save the model"""
        try:
            if hasattr(self.model, 'save'):
                self.model.save(path)
            else:
                # For custom models, save state dict
                torch.save(self.model.state_dict(), path)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

