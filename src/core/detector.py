"""
Generic YOLO Object Detector
Supports both baseline and attention-enhanced models
"""

import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules import C2f
import logging
import yaml
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
        attention_config = self.config.get('attention', {})

        architecture = model_config.get('architecture', 'yolov8s')
        pretrained_weights = model_config.get('pretrained_weights')

        # Handle different architectures
        if architecture == 'mobilenet_vit':
            # Custom MobileNetVIT detector
            from ..models.mobilenet_vit import create_mobilenet_vit_detector
            num_classes = len(self.class_names) if self.class_names else 5
            attention_layers = attention_config.get('layers', []) if attention_config.get('enabled', False) else None
            self.model = create_mobilenet_vit_detector(num_classes=num_classes, attention_layers=attention_layers)
            logger.info(f"Created MobileNetVIT detector with {num_classes} classes")

        elif architecture == 'efficientnet':
            # Custom EfficientNet detector
            from ..models.efficientnet import create_efficientnet_detector
            num_classes = len(self.class_names) if self.class_names else 5
            attention_layers = attention_config.get('layers', []) if attention_config.get('enabled', False) else None
            self.model = create_efficientnet_detector(num_classes=num_classes, attention_layers=attention_layers)
            logger.info(f"Created EfficientNet detector with {num_classes} classes")

        elif attention_config.get('enabled', False) and architecture == 'yolov8s':
            # CBAM enhancement temporarily disabled due to compatibility issues
            logger.warning("CBAM enhancement temporarily disabled. Using base YOLO model.")
            if pretrained_weights and Path(pretrained_weights).exists():
                self.model = YOLO(pretrained_weights)
                logger.info(f"Loaded pre-trained YOLO model: {pretrained_weights}")
            else:
                self.model = YOLO(f'{architecture}.yaml')
                logger.info(f"Created base YOLO {architecture} model")
        else:
            # Use baseline YOLO model
            if pretrained_weights and Path(pretrained_weights).exists():
                self.model = YOLO(pretrained_weights)
                logger.info(f"Loaded pre-trained model: {pretrained_weights}")
            else:
                self.model = YOLO(f'{architecture}.yaml')
                logger.info(f"Created baseline {architecture} model")

    def predict(self, source: Union[str, torch.Tensor], **kwargs):
        """Run inference on input"""
        architecture = self.config.get('model', {}).get('architecture', 'yolov8s')

        if architecture in ['mobilenet_vit', 'efficientnet']:
            # Custom model inference - simplified implementation
            # In practice, you'd need proper preprocessing and postprocessing
            logger.warning(f"Custom model {architecture} inference not fully implemented yet")
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
            # Custom models (MobileNetVIT, EfficientNet) - training not implemented yet
            logger.warning(f"Training for {architecture} not fully implemented yet. Skipping.")
            return {'success': False, 'error': f'Training not implemented for {architecture}'}  # Return a mock result

    def val(self, data_yaml: str, **kwargs):
        """Validate the model"""
        return self.model.val(data_yaml, **kwargs)

    def save(self, path: str):
        """Save the model"""
        self.model.save(path)

    @classmethod
    def from_config_file(cls, config_path: str) -> 'ObjectDetector':
        """Create detector from YAML configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Handle config inheritance
        if 'extends' in config:
            base_config_path = Path(config_path).parent / config['extends']
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
            base_config.update(config)
            config = base_config
            del config['extends']

        return cls(config)
