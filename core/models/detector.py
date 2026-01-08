"""
High-Performance YOLO Object Detector
Production-ready with advanced error handling, caching, and performance optimizations
"""

import hashlib
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from ultralytics import YOLO

from utils.logger_config import get_logger

logger = get_logger(__name__, debug=os.getenv("DEBUG") == "1")


# Custom exceptions for better error handling
class ModelLoadError(Exception):
    """Raised when model loading fails"""

    pass


class InferenceError(Exception):
    """Raised when inference fails"""

    pass


class TrainingError(Exception):
    """Raised when training fails"""

    pass


@contextmanager
def performance_monitor(operation_name: str):
    """Context manager to monitor operation performance"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.debug(f"{operation_name} completed in {elapsed:.3f}s")


class ModelCache:
    """Thread-safe model caching system"""

    def __init__(self, max_cache_size: int = 3):
        self._cache = {}
        self._max_size = max_cache_size
        self._lock = threading.RLock()
        self._access_times = {}

    def get(self, key: str) -> Optional[Any]:
        """Get model from cache"""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
        return None

    def put(self, key: str, model: Any) -> None:
        """Put model in cache with LRU eviction"""
        with self._lock:
            if len(self._cache) >= self._max_size:
                # Remove least recently used
                oldest_key = min(self._access_times, key=self._access_times.get)
                del self._cache[oldest_key]
                del self._access_times[oldest_key]

            self._cache[key] = model
            self._access_times[key] = time.time()


# Global model cache instance
model_cache = ModelCache()


def get_model_hash(config: Dict) -> str:
    """Generate hash for model configuration"""
    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ObjectDetector:
    """
    High-performance object detector with advanced caching, error handling, and monitoring

    Features:
    - Model caching for faster reloads
    - Comprehensive error handling with custom exceptions
    - Performance monitoring and metrics
    - Thread-safe operations
    - Automatic model validation
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.class_names = config.get("data", {}).get("classes", [])
        self.model_hash = get_model_hash(config.get("model", {}))
        self._lock = threading.RLock()
        self._performance_stats = {
            "inference_count": 0,
            "total_inference_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

        # Try to load from cache first
        cached_model = model_cache.get(self.model_hash)
        if cached_model:
            self.model = cached_model
            self._performance_stats["cache_hits"] += 1
            logger.info(f"Loaded model from cache (hash: {self.model_hash})")
        else:
            self._performance_stats["cache_misses"] += 1
            self._load_model()

        # Mark model as not validated yet - validation is lazy
        self._model_validated = False

    def _load_model(self):
        """Load the appropriate model based on configuration with enhanced error handling"""
        with performance_monitor("model_loading"):
            model_config = self.config.get("model", {})
            architecture = model_config.get("architecture", "yolov8s")
            pretrained_weights = model_config.get("pretrained_weights")

            logger.info(f"Loading model with architecture: {architecture}")

            try:
                # Handle different architectures with better error handling
                if architecture == "yolov8s_baseline":
                    self.model = self._load_yolov8_baseline(pretrained_weights)

                elif architecture.startswith("yolov8"):
                    self.model = self._load_yolov8_model(
                        architecture, pretrained_weights
                    )

                elif architecture in [
                    "mobilenetv2",
                    "mobilenetv3",
                    "resnet18",
                    "resnet34",
                    "efficientnet_lite0",
                    "efficientnet_lite1",
                ]:
                    self.model = self._load_torchvision_model(
                        architecture, pretrained_weights
                    )

                else:
                    # Default to YOLOv8s
                    logger.warning(
                        f"Unknown architecture '{architecture}', defaulting to yolov8s"
                    )
                    self.model = self._load_yolov8_model("yolov8s", pretrained_weights)

                # Cache the loaded model
                model_cache.put(self.model_hash, self.model)

                # Log model information
                self._log_model_info()

                # Store model type for later use
                self.model_type = (
                    "yolo" if architecture.startswith("yolov8") else "classification"
                )

            except Exception as e:
                self._performance_stats["errors"] += 1
                logger.error(f"Failed to load model: {e}")
                raise ModelLoadError(f"Model loading failed: {e}") from e

    def _load_torchvision_model(
        self, architecture: str, pretrained_weights: Optional[str] = None
    ):
        """Load torchvision classification model for IMX500 compatibility"""
        try:
            import torch.nn as nn
            import torchvision.models as models

            logger.info(f"Loading torchvision model: {architecture}")

            # Get number of classes from config or default to 3 (for consolidated dataset)
            num_classes = self.config.get("model", {}).get("num_classes", 3)

            if architecture == "mobilenetv2":
                model = models.mobilenet_v2(pretrained=True)
                # Modify classifier for our number of classes
                model.classifier[1] = nn.Linear(
                    model.classifier[1].in_features, num_classes
                )

            elif architecture == "mobilenetv3":
                # Load MobileNetV3-Large by default
                model = models.mobilenet_v3_large(pretrained=True)
                model.classifier[3] = nn.Linear(
                    model.classifier[3].in_features, num_classes
                )

            elif architecture == "resnet18":
                model = models.resnet18(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)

            elif architecture == "resnet34":
                model = models.resnet34(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)

            elif architecture.startswith("efficientnet"):
                # For EfficientNet-Lite variants, we'll use regular EfficientNet as approximation
                # since EfficientNet-Lite isn't in standard torchvision
                if architecture == "efficientnet_lite0":
                    model = models.efficientnet_b0(pretrained=True)
                elif architecture == "efficientnet_lite1":
                    model = models.efficientnet_b1(pretrained=True)
                else:
                    model = models.efficientnet_b0(pretrained=True)

                # Modify classifier
                model.classifier[1] = nn.Linear(
                    model.classifier[1].in_features, num_classes
                )

            else:
                raise ValueError(
                    f"Unsupported torchvision architecture: {architecture}"
                )

            # Load custom weights if provided
            if pretrained_weights and Path(pretrained_weights).exists():
                logger.info(f"Loading custom weights from: {pretrained_weights}")
                state_dict = torch.load(pretrained_weights, map_location="cpu")
                model.load_state_dict(state_dict)

            logger.info(
                f"Loaded torchvision {architecture} model with {num_classes} classes"
            )
            return model

        except ImportError:
            logger.error(
                "torchvision not available. Install with: pip install torchvision"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load torchvision model {architecture}: {e}")
            raise

    def _load_yolov8_baseline(self, pretrained_weights: Optional[str]) -> YOLO:
        """Load YOLOv8 baseline model"""
        if pretrained_weights and Path(pretrained_weights).exists():
            model = YOLO(pretrained_weights)
            logger.info(f"Loaded pre-trained YOLOv8s baseline: {pretrained_weights}")
            return model
        else:
            model = YOLO("yolov8s.pt")
            logger.info("Created YOLOv8s baseline model with COCO pre-trained weights")
            return model

    def _load_yolov8_model(
        self, architecture: str, pretrained_weights: Optional[str]
    ) -> YOLO:
        """Load generic YOLOv8 model with fallback"""
        if pretrained_weights and Path(pretrained_weights).exists():
            model = YOLO(pretrained_weights)
            logger.info(f"Loaded pre-trained {architecture}: {pretrained_weights}")
            return model
        else:
            model_name = f"{architecture}.pt"
            try:
                model = YOLO(model_name)
                logger.info(f"Created {architecture} model")
                return model
            except Exception as e:
                logger.warning(
                    f"Failed to load {model_name}: {e}. Falling back to yolov8s.pt"
                )
                model = YOLO("yolov8s.pt")
                logger.info("Created fallback YOLOv8s model")
                return model

    def _log_model_info(self):
        """Log detailed model information"""
        try:
            if (
                getattr(self, "model", None) is not None
                and hasattr(self.model, "model")
                and hasattr(self.model.model, "parameters")
                and callable(getattr(self.model.model, "parameters", None))
            ):
                try:
                    total_params = sum(p.numel() for p in self.model.model.parameters())
                    logger.info(f"Model loaded with {total_params:,} parameters")
                except Exception as e:
                    logger.debug(f"Could not compute model parameter count: {e}")

            # Log model size on disk
            ckpt_path = getattr(self.model, "ckpt_path", None)
            if ckpt_path and Path(ckpt_path).exists():
                model_size_mb = Path(ckpt_path).stat().st_size / (1024 * 1024)
                logger.info(f"Model file size: {model_size_mb:.2f} MB")

            # Log device information
            if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
                try:
                    device = next(self.model.model.parameters()).device
                    logger.info(f"Model loaded on device: {device}")
                except (StopIteration, RuntimeError):
                    logger.debug("Could not determine model device")

        except (TypeError, AttributeError, OSError) as e:
            logger.debug(f"Could not get detailed model info: {e}")

    def _validate_model(self):
        """Validate that the model is properly loaded and functional"""
        if self._model_validated:
            return  # Already validated

        try:
            # Quick validation - check if model has required attributes
            required_attrs = ["predict", "train", "val"]
            for attr in required_attrs:
                if not hasattr(self.model, attr):
                    raise ModelLoadError(f"Model missing required attribute: {attr}")

            # Try a quick dummy prediction to validate functionality
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = self.model.predict(dummy_input, verbose=False)

            self._model_validated = True
            logger.debug("Model validation successful")

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise ModelLoadError(f"Model validation failed: {e}") from e

    def predict(self, source: Union[str, torch.Tensor], **kwargs) -> Any:
        """
        Run inference on input with enhanced error handling and performance monitoring

        Args:
            source: Input image path, PIL image, or tensor
            **kwargs: Additional prediction parameters

        Returns:
            YOLO prediction results

        Raises:
            InferenceError: If inference fails
        """
        with self._lock:  # Thread-safe inference
            start_time = time.perf_counter()

            try:
                # Validate model on first use
                self._validate_model()

                # Get prediction parameters
                predict_kwargs = self._get_default_predict_kwargs()
                predict_kwargs.update(kwargs)

                # Log prediction start for debugging
                logger.debug(f"Starting inference on {type(source).__name__}")

                # Run inference
                with performance_monitor("model_inference"):
                    results = self.model.predict(source, **predict_kwargs)

                # Update performance statistics
                inference_time = time.perf_counter() - start_time
                self._performance_stats["inference_count"] += 1
                self._performance_stats["total_inference_time"] += inference_time

                # Log performance metrics
                avg_inference_time = (
                    self._performance_stats["total_inference_time"]
                    / self._performance_stats["inference_count"]
                )
                logger.debug(
                    f"Inference completed in {inference_time:.3f}s (avg: {avg_inference_time:.3f}s)"
                )

                return results

            except Exception as e:
                self._performance_stats["errors"] += 1
                logger.error(f"Inference failed: {e}")
                raise InferenceError(f"Model inference failed: {e}") from e

    def _get_default_predict_kwargs(self) -> Dict:
        """Get default prediction parameters from config"""
        model_config = self.config.get("model", {})
        return {
            "conf": model_config.get("confidence_threshold", 0.5),
            "iou": model_config.get("iou_threshold", 0.4),
            "max_det": model_config.get("max_detections", 20),
            "imgsz": model_config.get("input_size", 416),
        }

    def train(self, data_yaml: str, model_name: Optional[str] = None, **kwargs) -> Any:
        """
        Train the model with enhanced error handling and monitoring

        Args:
            data_yaml: Path to data configuration YAML
            model_name: Optional name for the training run
            **kwargs: Additional training parameters

        Returns:
            Training results

        Raises:
            TrainingError: If training fails
        """
        with self._lock:  # Thread-safe training
            try:
                # Validate data YAML exists
                if not Path(data_yaml).exists():
                    raise TrainingError(f"Data YAML file not found: {data_yaml}")

                # Build training configuration
                training_kwargs = self._build_training_kwargs(model_name)
                training_kwargs.update(kwargs)

                logger.info(f"Starting training with config: {training_kwargs}")
                logger.info(f"Training data: {data_yaml}")

                # Validate architecture
                architecture = self.config.get("model", {}).get(
                    "architecture", "yolov8s"
                )
                if not architecture.startswith("yolov8"):
                    logger.warning(
                        f"Architecture {architecture} not supported, using YOLOv8 training"
                    )

                # Run training with performance monitoring
                with performance_monitor("model_training"):
                    results = self.model.train(data=data_yaml, **training_kwargs)

                logger.info("Training completed successfully")
                return results

            except Exception as e:
                logger.error(f"Training failed: {e}")
                raise TrainingError(f"Model training failed: {e}") from e

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
            "workers": training_config.get("workers", 8),
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
                "copy_paste": aug_config.get("copy_paste", 0.0),
                "hsv_h": aug_config.get("hsv_h", 0.0),
                "hsv_s": aug_config.get("hsv_s", 0.7),
                "hsv_v": aug_config.get("hsv_v", 0.4),
            }
        )

        # Add multi-scale training if enabled
        if training_config.get("multi_scale", False):
            # YOLO supports multi-scale through imgsz range
            min_size = training_config.get("multi_scale_min", 320)
            max_size = training_config.get("multi_scale_max", 640)
            step = training_config.get("multi_scale_step", 32)
            # Note: YOLO's multi-scale is typically handled internally, but we can set imgsz to a range
            # For now, we'll use the max size as base and let YOLO handle multi-scale internally
            kwargs["imgsz"] = max_size

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

    def save(self, path: str) -> None:
        """
        Save the model to disk

        Args:
            path: Path to save the model

        Raises:
            Exception: If saving fails
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            if hasattr(self.model, "save"):
                with performance_monitor("model_save"):
                    self.model.save(path)
                logger.info(f"Model saved successfully to: {path}")
            else:
                logger.warning("Model does not have save method")
                raise ValueError("Model does not support saving")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for monitoring

        Returns:
            Dictionary with performance metrics
        """
        stats = self._performance_stats.copy()

        # Calculate derived metrics
        if stats["inference_count"] > 0:
            stats["avg_inference_time"] = (
                stats["total_inference_time"] / stats["inference_count"]
            )
            stats["cache_hit_rate"] = stats["cache_hits"] / (
                stats["cache_hits"] + stats["cache_misses"]
            )
        else:
            stats["avg_inference_time"] = 0.0
            stats["cache_hit_rate"] = 0.0

        return stats

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the model

        Returns:
            Health status information
        """
        health_info = {
            "model_loaded": self.model is not None,
            "cache_hash": self.model_hash,
            "performance_stats": self.get_performance_stats(),
            "device": None,
            "model_size_mb": None,
            "status": "unknown",
        }

        try:
            if self.model and hasattr(self.model, "model"):
                # Get device information (handle mock objects gracefully)
                try:
                    if hasattr(self.model.model, "parameters"):
                        device = next(self.model.model.parameters()).device
                        health_info["device"] = str(device)
                    else:
                        health_info["device"] = "unknown"
                except (TypeError, AttributeError, StopIteration):
                    health_info["device"] = "unknown"

                # Get model size if checkpoint path exists
                ckpt_path = getattr(self.model, "ckpt_path", None)
                if ckpt_path and isinstance(ckpt_path, (str, Path)):
                    try:
                        ckpt_path_obj = Path(ckpt_path)
                        if ckpt_path_obj.exists():
                            size_mb = ckpt_path_obj.stat().st_size / (1024 * 1024)
                            health_info["model_size_mb"] = round(size_mb, 2)
                    except (OSError, ValueError, TypeError):
                        logger.debug("Could not determine model file size")

                # Test model functionality with dummy input only if not already validated
                if not getattr(self, "_model_validated", False):
                    self._validate_model()

            health_info["status"] = "healthy"

        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            logger.warning(f"Health check failed: {e}")

        return health_info

    def warmup(
        self, input_size: Tuple[int, int] = (224, 224), num_runs: int = 5
    ) -> None:
        """
        Warm up the model with dummy inferences

        Args:
            input_size: Size of dummy input images (height, width)
            num_runs: Number of warmup runs
        """
        logger.info(f"Warming up model with {num_runs} dummy inferences")

        try:
            with torch.no_grad():
                for i in range(num_runs):
                    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
                    _ = self.predict(dummy_input, verbose=False)

            logger.info("Model warmup completed")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def clear_cache(self) -> None:
        """Clear the model cache"""
        model_cache._cache.clear()
        model_cache._access_times.clear()
        logger.info("Model cache cleared")

    def __repr__(self) -> str:
        """String representation of the detector"""
        architecture = self.config.get("model", {}).get("architecture", "unknown")
        cache_status = "cached" if model_cache.get(self.model_hash) else "not cached"
        return f"ObjectDetector(architecture={architecture}, cache={cache_status}, hash={self.model_hash})"
