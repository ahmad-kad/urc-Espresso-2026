"""
ONNX-compatible Object Detector
Supports .onnx and int8 .onnx models for inference
"""

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from utils.detection_utils import (
    non_max_suppression,
    post_process_yolov8,
    scale_boxes_to_original,
)
from utils.logger_config import get_logger

logger = get_logger(__name__, debug=os.getenv("DEBUG") == "1")


class ONNXDetector:
    """
    ONNX model detector with YOLOv8 output format support

    Supports both FP32 and INT8 quantized ONNX models
    """

    def __init__(self, model_path: str, config: Dict):
        """
        Initialize ONNX detector

        Args:
            model_path: Path to .onnx model file
            config: Configuration dictionary
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. Install with: pip install onnxruntime"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.config = config
        self.model_config = config.get("model", {})
        self.input_size = self.model_config.get("input_size", 416)
        self.conf_threshold = self.model_config.get("confidence_threshold", 0.5)
        self.iou_threshold = self.model_config.get("iou_threshold", 0.4)

        # Load ONNX model
        self._load_model()

        # Thread safety
        self._lock = threading.RLock()
        self._performance_stats = {
            "inference_count": 0,
            "total_inference_time": 0.0,
            "errors": 0,
        }

        logger.info(f"ONNX detector initialized with model: {model_path}")

    def _load_model(self):
        """Load ONNX model with appropriate providers"""
        # Try different provider strategies
        provider_strategies = [
            ["CPUExecutionProvider"],  # Most compatible
            ["CUDAExecutionProvider", "CPUExecutionProvider"],  # GPU if available
        ]

        session = None
        for providers in provider_strategies:
            try:
                session = ort.InferenceSession(
                    str(self.model_path), providers=providers
                )
                actual_providers = session.get_providers()
                logger.info(f"ONNX model loaded with providers: {actual_providers}")
                break
            except Exception as e:
                logger.debug(f"Provider strategy {providers} failed: {e}")
                continue

        if session is None:
            raise RuntimeError(f"Failed to load ONNX model: {self.model_path}")

        self.session = session
        self.input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        # Extract input size from model if not in config
        if len(input_shape) == 4:
            _, _, h, w = input_shape
            if self.input_size != h:
                logger.info(
                    f"Updating input size from config ({self.input_size}) to model ({h})"
                )
                self.input_size = h

        logger.info(f"ONNX model input: {self.input_name}, shape: {input_shape}")

    def predict(self, source: Union[str, np.ndarray], **kwargs) -> List[Dict]:
        """
        Run inference on input image

        Args:
            source: Image path or numpy array (BGR format)
            **kwargs: Additional parameters (conf, iou, etc.)

        Returns:
            List of detection dictionaries with keys: bbox, class_id, confidence
        """
        with self._lock:
            start_time = time.perf_counter()

            try:
                # Load image if path provided
                if isinstance(source, str):
                    image = cv2.imread(source)
                    if image is None:
                        raise ValueError(f"Could not load image: {source}")
                else:
                    image = source.copy()

                original_shape = image.shape[:2]  # (height, width)

                # Preprocess image
                processed = self._preprocess(image)

                # Run inference
                outputs = self.session.run(None, {self.input_name: processed})

                # Post-process outputs
                detections = self._postprocess(outputs[0], original_shape, **kwargs)

                # Update stats
                inference_time = time.perf_counter() - start_time
                self._performance_stats["inference_count"] += 1
                self._performance_stats["total_inference_time"] += inference_time

                return detections

            except Exception as e:
                self._performance_stats["errors"] += 1
                logger.error(f"ONNX inference failed: {e}")
                raise

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ONNX model input

        Args:
            image: Input image in BGR format (H, W, C)

        Returns:
            Preprocessed image tensor (1, C, H, W) in NCHW format, normalized [0, 1]
        """
        # Resize to model input size
        resized = cv2.resize(image, (self.input_size, self.input_size))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Convert HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension: (1, C, H, W)
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def _postprocess(
        self,
        output: np.ndarray,
        original_shape: Tuple[int, int],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Dict]:
        """
        Post-process ONNX model output to extract detections

        Args:
            output: Raw model output
            original_shape: (height, width) of original image
            conf: Confidence threshold override
            iou: IoU threshold override

        Returns:
            List of detection dictionaries
        """
        conf_thresh = conf if conf is not None else self.conf_threshold
        iou_thresh = iou if iou is not None else self.iou_threshold

        # YOLOv8 ONNX output format: (1, num_features, num_predictions)
        # where num_features = 4 + num_classes
        # Format: [x_center, y_center, width, height, class_scores...]

        # Handle different output formats
        if len(output.shape) == 3:
            # Remove batch dimension: (num_features, num_predictions)
            output = output[0]
            num_features, num_predictions = output.shape
        elif len(output.shape) == 2:
            # Format: (num_features, num_predictions) - correct format
            num_features, num_predictions = output.shape
        elif len(output.shape) == 1:
            # Flattened output, try to reshape
            # This shouldn't happen with YOLOv8 but handle it
            logger.warning("Unexpected output shape, attempting to handle")
            return []
        else:
            # Unexpected format
            logger.error(f"Unexpected output shape: {output.shape}")
            return []

        # Extract boxes and scores
        boxes_xywh = output[:4, :].transpose(1, 0)  # (num_predictions, 4)
        class_scores = output[4:, :].transpose(1, 0)  # (num_predictions, num_classes)

        # Get best class for each prediction
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by confidence
        mask = confidences > conf_thresh
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # Convert from xywh to xyxy format
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Scale to input size (boxes are in normalized [0, 1] or input_size coordinates)
        # Check if boxes are normalized
        if np.max(boxes_xyxy) <= 1.0:
            boxes_xyxy *= self.input_size

        # Apply NMS
        indices = non_max_suppression(boxes_xyxy, confidences, iou_thresh)

        if len(indices) == 0:
            return []

        boxes_xyxy = boxes_xyxy[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]

        # Scale boxes to original image size
        boxes_scaled = scale_boxes_to_original(
            boxes_xyxy, original_shape, self.input_size
        )

        # Convert to list of dictionaries
        detections = []
        for i in range(len(boxes_scaled)):
            detections.append(
                {
                    "bbox": boxes_scaled[i].tolist(),  # [x1, y1, x2, y2]
                    "class_id": int(class_ids[i]),
                    "confidence": float(confidences[i]),
                }
            )

        return detections

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self._performance_stats.copy()
        if stats["inference_count"] > 0:
            stats["avg_inference_time"] = (
                stats["total_inference_time"] / stats["inference_count"]
            )
        else:
            stats["avg_inference_time"] = 0.0
        return stats
