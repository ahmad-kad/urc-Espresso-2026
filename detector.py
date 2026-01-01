"""
Generic YOLO Object Detector
Supports both baseline and attention-enhanced models
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from ultralytics import YOLO

from utils.logger_config import get_logger

logger = get_logger(__name__, debug=os.getenv("DEBUG") == "1")


# Attention-enhanced classes removed due to missing attention_modules dependency
# If CBAM attention is needed in the future, implement attention_modules.py first


class ObjectDetector:
    """
    Generic object detector supporting multiple architectures
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.class_names = config.get("data", {}).get("classes", [])

        self._load_model()

    def _load_model(self):
        """Load the appropriate model based on configuration"""
        model_config = self.config.get("model", {})

        architecture = model_config.get("architecture", "yolov8s")
        pretrained_weights = model_config.get("pretrained_weights")

        logger.info(f"Loading model with architecture: {architecture}")

        # Handle different architectures
        if architecture == "yolov8s_baseline":
            # Standard YOLOv8s model
            if pretrained_weights and Path(pretrained_weights).exists():
                self.model = YOLO(pretrained_weights)
                logger.info(
                    f"Loaded pre-trained YOLOv8s baseline: {pretrained_weights}"
                )
            else:
                self.model = YOLO("yolov8s.pt")  # Use COCO pre-trained weights
                logger.info(
                    "Created YOLOv8s baseline model with COCO pre-trained weights"
                )

        elif architecture.startswith("yolov8"):
            # Generic YOLOv8 handling (n, s, m, l, x)
            if pretrained_weights and Path(pretrained_weights).exists():
                self.model = YOLO(pretrained_weights)
                logger.info(f"Loaded pre-trained {architecture}: {pretrained_weights}")
            else:
                model_name = f"{architecture}.pt"
                try:
                    self.model = YOLO(model_name)
                    logger.info(f"Created {architecture} model")
                except Exception as e:
                    logger.warning(
                        f"Failed to load {model_name}: {e}. Falling back to yolov8s.pt"
                    )
                    self.model = YOLO("yolov8s.pt")

        else:
            # Default to YOLOv8s
            if pretrained_weights and Path(pretrained_weights).exists():
                self.model = YOLO(pretrained_weights)
                logger.info(f"Loaded pre-trained model: {pretrained_weights}")
            else:
                self.model = YOLO("yolov8s.pt")
                logger.info("Created default YOLOv8s model")

        # Log model info
        if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
            try:
                total_params = sum(p.numel() for p in self.model.model.parameters())
                logger.info(f"Model loaded with {total_params:,} parameters")
            except (TypeError, AttributeError):
                logger.debug("Could not count model parameters")

    def predict(self, source: Union[str, torch.Tensor], **kwargs):
        """Run inference on input"""
        # YOLO model inference
        default_kwargs = self._get_default_predict_kwargs()
        default_kwargs.update(kwargs)
        return self.model.predict(source, **default_kwargs)

    def _get_default_predict_kwargs(self) -> Dict:
        """Get default prediction parameters from config"""
        model_config = self.config.get("model", {})
        return {
            "conf": model_config.get("confidence_threshold", 0.5),
            "iou": model_config.get("iou_threshold", 0.4),
            "max_det": model_config.get("max_detections", 20),
            "imgsz": model_config.get("input_size", 416),
        }

    def train(self, data_yaml: str, model_name: Optional[str] = None, **kwargs):
        """Train the model"""
        default_kwargs = self._build_training_kwargs(model_name)
        default_kwargs.update(kwargs)

        logger.info(f"Starting training with config: {default_kwargs}")

        # All models use YOLO training framework
        architecture = self.config.get("model", {}).get("architecture", "yolov8s")
        if not architecture.startswith("yolov8"):
            logger.warning(
                f"Architecture {architecture} not supported. Using YOLOv8 training."
            )
        return self.model.train(data=data_yaml, **default_kwargs)

    def _build_training_kwargs(self, model_name: Optional[str]) -> Dict:
        """Build training keyword arguments from config"""
        training_config = self.config.get("training", {})
        model_config = self.config.get("model", {})

        from utils.device_utils import resolve_device

        device = resolve_device(
            self.config.get("device"), training_config.get("device")
        )

        kwargs = {
            "epochs": training_config.get("epochs", 50),
            "batch": training_config.get("batch_size", 8),
            "imgsz": model_config.get("input_size", 416),
            "lr0": training_config.get("learning_rate", 0.001),
            "weight_decay": training_config.get("weight_decay", 0.0005),
            "patience": training_config.get("patience", 20),
            "device": device,
            "project": "output",
            "name": model_name
            or f"{self.config.get('project', {}).get('name', 'object_detection')}_training",
        }

        # Add augmentation settings
        aug_config = training_config.get("augmentation", {})
        kwargs.update(
            {
                "degrees": aug_config.get("degrees", 0.0),
                "translate": aug_config.get("translate", 0.1),
                "scale": aug_config.get("scale", 0.5),
                "shear": aug_config.get("shear", 0.0),
                "perspective": aug_config.get("perspective", 0.0),
                "flipud": aug_config.get("flipud", 0.0),
                "fliplr": aug_config.get("fliplr", 0.5),
                "mosaic": aug_config.get("mosaic", 1.0),
                "mixup": aug_config.get("mixup", 0.0),
                "hsv_h": aug_config.get("hsv_h", 0.0),
                "hsv_s": aug_config.get("hsv_s", 0.7),
                "hsv_v": aug_config.get("hsv_v", 0.4),
            }
        )

        # Add loss weights
        loss_config = training_config.get("loss_weights", {})
        kwargs.update(
            {
                "box": loss_config.get("box", 7.5),
                "cls": loss_config.get("cls", 0.5),
                "dfl": loss_config.get("dfl", 1.5),
            }
        )

        return kwargs

    def val(self, data_yaml: str, **kwargs):
        """Validate the model"""
        return self.model.val(data_yaml, **kwargs)

    def save(self, path: str):
        """Save the model"""
        try:
            if hasattr(self.model, "save"):
                self.model.save(path)
            else:
                logger.warning("Model does not have save method")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
