"""
ML Inference Component
Handles YOLO/ONNX model inference, separated from ArUco detection
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.models.detector import ObjectDetector
from core.models.onnx_detector import ONNXDetector
from utils.logger_config import get_logger

logger = get_logger(__name__)


class MLInferenceComponent:
    """
    Component for ML model inference (YOLO/ONNX)
    Handles model loading, inference, and result formatting
    """

    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize ML inference component

        Args:
            model_path: Path to model file (.pt or .onnx)
            config: Configuration dictionary
        """
        self.config = config
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._load_model()
        logger.info(f"ML Inference Component initialized: {model_path}")

    def _load_model(self) -> None:
        """Load model based on file extension"""
        model_ext = self.model_path.suffix.lower()

        if model_ext == ".onnx":
            logger.info(f"Loading ONNX model: {self.model_path}")
            self.detector = ONNXDetector(str(self.model_path), self.config)
            self.is_onnx = True
        elif model_ext == ".pt":
            logger.info(f"Loading PyTorch model: {self.model_path}")
            self.config["model"]["pretrained_weights"] = str(self.model_path)
            self.detector = ObjectDetector(self.config)
            self.is_onnx = False
        else:
            raise ValueError(f"Unsupported model format: {model_ext}. Use .pt or .onnx")

    def predict(self, image: np.ndarray) -> Any:
        """
        Run inference on image

        Args:
            image: Input image in BGR format

        Returns:
            Detection results in YOLO format
        """
        model_config = self.config.get("model", {})
        input_size = model_config.get("input_size", 416)
        conf_threshold = model_config.get("confidence_threshold", 0.5)

        if self.is_onnx:
            detections_list = self.detector.predict(image, conf=conf_threshold)
            return self._onnx_to_yolo_format(detections_list)
        else:
            return self.detector.predict(
                image, conf=conf_threshold, imgsz=input_size, verbose=False
            )

    def _onnx_to_yolo_format(self, detections_list: List[Dict]) -> Any:
        """Convert ONNX detector output to YOLO-like format"""

        class MockBoxes:
            def __init__(self, detections):
                self.xyxy = []
                self.conf = []
                self.cls = []
                for det in detections:
                    self.xyxy.append(np.array(det["bbox"], dtype=np.float32))
                    self.conf.append(det["confidence"])
                    self.cls.append(det["class_id"])
                if self.xyxy:
                    self.xyxy = np.array(self.xyxy)
                    self.conf = np.array(self.conf)
                    self.cls = np.array(self.cls)
                else:
                    self.xyxy = np.array([])
                    self.conf = np.array([])
                    self.cls = np.array([])

            def __len__(self):
                return len(self.xyxy)

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        boxes = MockBoxes(detections_list)
        return [MockResult(boxes)]

    def get_class_names(self) -> List[str]:
        """Get class names from model or config"""
        class_names = self.config.get("data", {}).get("classes", [])

        if not class_names and not self.is_onnx:
            try:
                if hasattr(self.detector, "model") and hasattr(
                    self.detector.model, "names"
                ):
                    class_names = list(self.detector.model.names.values())
                elif hasattr(self.detector, "class_names"):
                    class_names = self.detector.class_names
            except:
                pass

        return class_names
