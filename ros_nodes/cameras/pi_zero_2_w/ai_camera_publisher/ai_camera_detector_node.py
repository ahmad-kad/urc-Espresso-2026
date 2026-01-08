#!/usr/bin/env python3
"""
ROS2 Node for Camera AI Inference
Subscribes to camera image stream, runs object detection, and publishes annotated images
Supports .pt, .onnx, and int8 .onnx models
"""

import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# #region agent log
def _debug_log(location, message, data=None, hypothesis_id=None):
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id or "A",
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open("/media/durian/AI/AI/urc-Espresso-2026/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except:
        pass


# #endregion

try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header

    try:
        from vision_msgs.msg import (
            Detection2D,
            Detection2DArray,
            ObjectHypothesisWithPose,
        )

        VISION_MSGS_AVAILABLE = True
    except ImportError:
        # Fallback: use geometry_msgs if vision_msgs not available
        from geometry_msgs.msg import Point, Pose2D
        from std_msgs.msg import Float32MultiArray, Int32MultiArray, String

        VISION_MSGS_AVAILABLE = False
        Detection2D = None  # Will be defined as custom class
        Detection2DArray = None
        ObjectHypothesisWithPose = None
    ROS_AVAILABLE = True
except ImportError as e:
    ROS_AVAILABLE = False
    VISION_MSGS_AVAILABLE = False
    print(
        f"ROS2 not available: {e}. Install with: pip install rclpy sensor-msgs cv-bridge std-msgs vision-msgs"
    )

from core.config.manager import load_config

# Import components
from ros_nodes.components import (
    ArUcoDetectorComponent,
    DetectionMerger,
    ImagePreprocessor,
    MLInferenceComponent,
    MotionTracker,
    TemporalSmoother,
    VisualizationComponent,
)
from utils.logger_config import get_logger

logger = get_logger(__name__)


class CameraDetectorNode(Node):
    """
    ROS2 node that performs object detection on camera stream

    Subscribes to: /camera/image_raw (or configurable)
    Publishes to:
        - /object_detector/detections (Detection2DArray)
        - /object_detector/annotated_image (Image, if enabled)
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize the camera detector node

        Args:
            model_path: Path to model file (.pt, .onnx, or int8 .onnx)
            config_path: Optional path to config YAML file
        """
        # #region agent log
        _debug_log(
            "camera_detector_node.py:58",
            "CameraDetectorNode.__init__ entry",
            {"model_path": str(model_path), "config_path": config_path},
            "A",
        )
        # #endregion

        if not ROS_AVAILABLE:
            # #region agent log
            _debug_log(
                "camera_detector_node.py:66", "ROS_AVAILABLE check failed", {}, "B"
            )
            # #endregion
            raise ImportError("ROS2 dependencies not installed")

        # #region agent log
        _debug_log("camera_detector_node.py:69", "Calling super().__init__", {}, "A")
        # #endregion
        super().__init__("camera_detector_node")

        # Load configuration
        # #region agent log
        _debug_log(
            "camera_detector_node.py:72",
            "Loading config",
            {"config_path": config_path},
            "A",
        )
        # #endregion
        if config_path:
            config_name = config_path.replace(".yaml", "").replace("configs/", "")
            self.config = load_config(config_name)
        else:
            self.config = load_config("robotics")  # Default to robotics config

        # Load class names from data.yaml if available
        data_yaml_path = self.config.get("data", {}).get("yaml_path")
        if data_yaml_path and Path(data_yaml_path).exists():
            try:
                import yaml

                with open(data_yaml_path, "r") as f:
                    data_config = yaml.safe_load(f)
                    if "names" in data_config:
                        class_names = data_config["names"]
                        if isinstance(class_names, list):
                            self.config["data"]["classes"] = class_names
                        elif isinstance(class_names, dict):
                            self.config["data"]["classes"] = [
                                class_names.get(i, f"class_{i}")
                                for i in range(len(class_names))
                            ]
                        # #region agent log
                        _debug_log(
                            "camera_detector_node.py:88",
                            "Loaded class names from data.yaml",
                            {
                                "classes": self.config["data"]["classes"],
                                "num_classes": len(self.config["data"]["classes"]),
                            },
                            "A",
                        )
                        # #endregion
            except Exception as e:
                # #region agent log
                _debug_log(
                    "camera_detector_node.py:92",
                    "Failed to load class names from data.yaml",
                    {"error": str(e)},
                    "A",
                )
                # #endregion
                pass

        # #region agent log
        _debug_log(
            "camera_detector_node.py:78",
            "Config loaded",
            {
                "config_name": self.config.get("project", {}).get("name", "unknown"),
                "classes": self.config.get("data", {}).get("classes", []),
                "conf_threshold": self.config.get("model", {}).get(
                    "confidence_threshold", 0.5
                ),
            },
            "A",
        )
        # #endregion

        ros_config = self.config.get("ros2", {})

        # Initialize components
        # ML Inference Component
        self.ml_inference = MLInferenceComponent(model_path, self.config)

        # Image Preprocessor Component
        self.image_preprocessor = ImagePreprocessor(self.config)

        # Motion Tracker Component (for ArUco)
        self.motion_tracker = MotionTracker(self.config)

        # ArUco Detector Component
        self.aruco_detector = ArUcoDetectorComponent(self.config, self.motion_tracker)
        self.aruco_enabled = self.aruco_detector.enabled

        # Detection Merger Component
        aruco_class_id = getattr(self.aruco_detector, "aruco_class_id", None)
        self.detection_merger = DetectionMerger(
            aruco_class_id if aruco_class_id is not None else 0
        )

        # Temporal Smoother Component
        self.temporal_smoother = TemporalSmoother(self.config, aruco_class_id)

        # Visualization Component
        class_names = self.config.get("data", {}).get("classes", [])
        if not class_names:
            class_names = self.ml_inference.get_class_names()
        self.visualizer = VisualizationComponent(self.config, class_names)

        # CV Bridge for image conversion
        # #region agent log
        _debug_log("camera_detector_node.py:105", "Creating CvBridge", {}, "A")
        # #endregion
        self.bridge = CvBridge()

        # Thread safety
        self._lock = threading.Lock()
        self._processing = False

        # Topics - use config or default to RealSense topic
        input_topic = ros_config.get("input_topic", "/camera/camera/color/image_raw")
        # #region agent log
        _debug_log(
            "camera_detector_node.py:124",
            "Input topic determined",
            {"input_topic": input_topic},
            "D",
        )
        # #endregion
        output_topic = ros_config.get("output_topic", "/object_detector/detections")
        annotated_topic = ros_config.get(
            "annotated_topic", "/object_detector/annotated_image"
        )
        self.publish_annotated = ros_config.get("publish_annotated", True)

        # Subscriber
        # #region agent log
        _debug_log(
            "camera_detector_node.py:130",
            "Creating subscriber",
            {"input_topic": input_topic},
            "D",
        )
        # #endregion
        self.subscription = self.create_subscription(
            Image, input_topic, self.image_callback, 10  # QoS depth
        )
        logger.info(f"Subscribed to: {input_topic}")
        # #region agent log
        _debug_log(
            "camera_detector_node.py:136",
            "Subscriber created",
            {"input_topic": input_topic},
            "D",
        )
        # #endregion

        # Publishers
        if VISION_MSGS_AVAILABLE:
            # #region agent log
            _debug_log(
                "camera_detector_node.py:140",
                "Creating detection publisher",
                {"output_topic": output_topic},
                "A",
            )
            # #endregion
            self.detection_pub = self.create_publisher(
                Detection2DArray, output_topic, 10
            )
            logger.info(f"Publishing detections to: {output_topic}")
        else:
            self.detection_pub = None
            logger.warning(
                "vision_msgs not available - detection messages disabled. Only annotated images will be published."
            )
            # #region agent log
            _debug_log(
                "camera_detector_node.py:147", "vision_msgs not available", {}, "A"
            )
            # #endregion

        if self.publish_annotated:
            # #region agent log
            _debug_log(
                "camera_detector_node.py:151",
                "Creating annotated image publisher",
                {"annotated_topic": annotated_topic},
                "A",
            )
            # #endregion
            self.annotated_pub = self.create_publisher(Image, annotated_topic, 10)
            logger.info(f"Publishing annotated images to: {annotated_topic}")
            # #region agent log
            _debug_log(
                "camera_detector_node.py:156", "Annotated publisher created", {}, "A"
            )
            # #endregion

        # Performance tracking
        self.frame_count = 0
        self.temporal_enabled = self.temporal_smoother.enabled

        self.get_logger().info("Camera Detector Node initialized and ready")
        # #region agent log
        _debug_log(
            "camera_detector_node.py:160",
            "CameraDetectorNode.__init__ complete",
            {
                "input_topic": input_topic,
                "output_topic": output_topic,
                "annotated_topic": annotated_topic,
            },
            "A",
        )
        # #endregion

    def image_callback(self, msg: Image) -> None:
        """
        Callback for incoming camera images

        Args:
            msg: ROS2 Image message
        """
        # #region agent log
        import json
        import time

        log_path = "/media/durian/AI/AI/urc-Espresso-2026/.cursor/debug.log"
        try:
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "IMAGE_STREAM",
                            "location": "camera_detector_node.py:357",
                            "message": "image_callback entry",
                            "data": {
                                "frame_count": self.frame_count,
                                "processing": self._processing,
                                "msg_width": msg.width,
                                "msg_height": msg.height,
                                "publish_annotated": self.publish_annotated,
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except:
            pass
        _debug_log(
            "camera_detector_node.py:162",
            "image_callback entry",
            {
                "frame_count": self.frame_count,
                "processing": self._processing,
                "msg_width": msg.width,
                "msg_height": msg.height,
            },
            "E",
        )
        # #endregion

        # Skip if already processing (simple frame dropping for performance)
        if self._processing:
            # #region agent log
            _debug_log(
                "camera_detector_node.py:170",
                "Skipping frame - already processing",
                {},
                "E",
            )
            # #endregion
            return

        with self._lock:
            self._processing = True

        try:
            import time as time_module

            frame_start_time = time_module.perf_counter()

            # Convert ROS image to OpenCV format
            # #region agent log
            _debug_log(
                "camera_detector_node.py:178",
                "Converting ROS image to OpenCV",
                {"encoding": msg.encoding},
                "F",
            )
            # #endregion
            convert_start = time_module.perf_counter()
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            convert_time = (time_module.perf_counter() - convert_start) * 1000
            # #region agent log
            _debug_log(
                "camera_detector_node.py:180",
                "Image converted",
                {"cv_image_shape": cv_image.shape if cv_image is not None else None},
                "F",
            )
            # #endregion

            # Detect ArUco tags FIRST on original image (before preprocessing)
            aruco_detections = []
            aruco_time = 0.0
            if self.aruco_enabled:
                aruco_start = time_module.perf_counter()
                aruco_detections = self.aruco_detector.detect(cv_image)
                aruco_time = (time_module.perf_counter() - aruco_start) * 1000
                if aruco_detections:
                    logger.info(f"Detected {len(aruco_detections)} ArUco tag(s)")

            # Apply preprocessing for ML inference
            preprocess_time = 0.0
            preprocess_start = time_module.perf_counter()
            processed_image = self.image_preprocessor.preprocess(cv_image)
            preprocess_time = (time_module.perf_counter() - preprocess_start) * 1000

            # Run ML inference
            yolo_start = time_module.perf_counter()
            ml_results = self.ml_inference.predict(processed_image)
            yolo_time = (time_module.perf_counter() - yolo_start) * 1000

            # Merge ArUco and ML detections
            merge_start = time_module.perf_counter()
            results = self.detection_merger.merge(ml_results, aruco_detections)
            merge_time = (time_module.perf_counter() - merge_start) * 1000

            # Apply temporal smoothing
            temporal_time = 0.0
            smoothed_confidences = None
            if self.temporal_enabled:
                temporal_start = time_module.perf_counter()
                smoothed_confidences = self.temporal_smoother.smooth(results)
                temporal_time = (time_module.perf_counter() - temporal_start) * 1000

            # Process results and publish detections (if vision_msgs available)
            if self.detection_pub is not None:
                # #region agent log
                _debug_log(
                    "camera_detector_node.py:194",
                    "Processing results for detection pub",
                    {},
                    "H",
                )
                # #endregion
                detections = self._process_results(results, msg.header)
                if detections:  # Only publish if we have detections and vision_msgs
                    detection_array = Detection2DArray()
                    detection_array.header = msg.header
                    detection_array.detections = detections
                    self.detection_pub.publish(detection_array)
                    # #region agent log
                    _debug_log(
                        "camera_detector_node.py:200",
                        "Published detections",
                        {"num_detections": len(detections)},
                        "H",
                    )
                    # #endregion

            # Update persisted detections
            current_time = time.time()
            self.visualizer.update_persisted_detections(results, current_time)

            # Publish annotated image if enabled
            draw_time = 0.0
            publish_time = 0.0
            if self.publish_annotated:
                draw_start = time_module.perf_counter()
                annotated_image = self.visualizer.draw_detections(
                    cv_image, results, smoothed_confidences
                )
                draw_time = (time_module.perf_counter() - draw_start) * 1000

                publish_start = time_module.perf_counter()
                annotated_msg = self.bridge.cv2_to_imgmsg(
                    annotated_image, encoding="bgr8"
                )
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
                publish_time = (time_module.perf_counter() - publish_start) * 1000

            # #region agent log
            total_time = (time_module.perf_counter() - frame_start_time) * 1000
            fps = 1000.0 / total_time if total_time > 0 else 0
            try:
                # Extract detection counts
                num_yolo_detections = 0
                num_aruco_detections = len(aruco_detections) if aruco_detections else 0
                num_total_detections = 0
                if hasattr(results, "__iter__") and len(results) > 0:
                    result = results[0]
                    if hasattr(result, "boxes"):
                        num_yolo_detections = len(result.boxes)
                        num_total_detections = num_yolo_detections

                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "PERFORMANCE",
                                "location": "camera_detector_node.py:605",
                                "message": "Frame processing performance",
                                "data": {
                                    "total_time_ms": total_time,
                                    "fps": fps,
                                    "convert_time_ms": convert_time,
                                    "aruco_time_ms": aruco_time,
                                    "preprocess_time_ms": preprocess_time,
                                    "yolo_time_ms": yolo_time,
                                    "merge_time_ms": merge_time,
                                    "temporal_time_ms": temporal_time,
                                    "draw_time_ms": draw_time,
                                    "publish_time_ms": publish_time,
                                    "num_yolo_detections": num_yolo_detections,
                                    "num_aruco_detections": num_aruco_detections,
                                    "num_total_detections": num_total_detections,
                                    "aruco_enabled": self.aruco_enabled,
                                    "temporal_enabled": self.temporal_enabled,
                                    "is_onnx": self.ml_inference.is_onnx,
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except:
                pass
            # #endregion

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(f"Processed {self.frame_count} frames")
                # #region agent log
                _debug_log(
                    "camera_detector_node.py:216",
                    "Frame count milestone",
                    {"frame_count": self.frame_count},
                    "E",
                )
                # #endregion

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            # #region agent log
            import traceback

            try:
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "IMAGE_STREAM",
                                "location": "camera_detector_node.py:549",
                                "message": "Error in image_callback",
                                "data": {
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                    "traceback": traceback.format_exc(),
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except:
                pass
            _debug_log(
                "camera_detector_node.py:220",
                "Error in image_callback",
                {"error": str(e), "error_type": type(e).__name__},
                "E",
            )
            _debug_log(
                "camera_detector_node.py:222",
                "Error traceback",
                {"traceback": traceback.format_exc()},
                "E",
            )
            # #endregion

        finally:
            with self._lock:
                self._processing = False
                # #region agent log
                _debug_log(
                    "camera_detector_node.py:228",
                    "image_callback exit",
                    {"frame_count": self.frame_count},
                    "E",
                )
                # #endregion

    def _process_results(self, results, header: Header):
        """
        Convert YOLO/ONNX results to ROS2 Detection2D messages

        Args:
            results: YOLO prediction results or ONNX detections
            header: ROS message header

        Returns:
            List of Detection2D messages or Detection2DArray
        """
        if not VISION_MSGS_AVAILABLE:
            # Fallback: return empty list if vision_msgs not available
            # We'll publish detections via annotated image only
            logger.warning("vision_msgs not available, skipping detection publishing")
            return []

        detections = []

        # Handle YOLO results format
        if hasattr(results, "__iter__") and len(results) > 0:
            result = results[0]  # First result (batch size 1)

            if hasattr(result, "boxes"):
                boxes = result.boxes

                for i in range(len(boxes.xyxy)):
                    # Get box coordinates
                    if isinstance(boxes.xyxy, np.ndarray):
                        box = boxes.xyxy[i]
                        conf = float(boxes.conf[i])
                        cls = int(boxes.cls[i])
                    else:
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())

                    # Create Detection2D message
                    detection = Detection2D()
                    detection.header = header

                    # Bounding box (center_x, center_y, width, height)
                    x1, y1, x2, y2 = box
                    detection.bbox.center.position.x = float((x1 + x2) / 2.0)
                    detection.bbox.center.position.y = float((y1 + y2) / 2.0)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)

                    # Hypothesis (class and confidence)
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(cls)
                    hypothesis.hypothesis.score = conf
                    detection.results = [hypothesis]

                    detections.append(detection)

        return detections

    def main(args=None):
        """
        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image
        """
        preprocess_config = self.config.get("preprocessing", {})
        if not preprocess_config.get("enabled", False):
            return image

        processed = image.copy()

        # Calculate image statistics for logging
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        # #region agent log
        _debug_log(
            "camera_detector_node.py:520",
            "Image preprocessing stats",
            {"mean_brightness": mean_brightness, "std_brightness": std_brightness},
            "J",
        )
        # #endregion

        # Adaptive brightness normalization
        if preprocess_config.get("adaptive_brightness", False):
            brightness_low = preprocess_config.get("brightness_threshold_low", 30)
            brightness_high = preprocess_config.get("brightness_threshold_high", 220)

            if mean_brightness < brightness_low:
                # Image is too dark - brighten it
                alpha = 1.0 + (brightness_low - mean_brightness) / 255.0
                processed = cv2.convertScaleAbs(
                    processed, alpha=min(alpha, 1.5), beta=10
                )
            elif mean_brightness > brightness_high:
                # Image is too bright - darken it
                alpha = 1.0 - (mean_brightness - brightness_high) / 255.0
                processed = cv2.convertScaleAbs(
                    processed, alpha=max(alpha, 0.7), beta=0
                )

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) - memory intensive
        # Skip if optimizing for edge devices
        edge_config = self.config.get("edge", {})
        if preprocess_config.get("clahe", False) and not edge_config.get(
            "optimize_memory", False
        ):
            clahe = cv2.createCLAHE(
                clipLimit=preprocess_config.get("clahe_clip_limit", 2.0),
                tileGridSize=(
                    preprocess_config.get("clahe_tile_size", 8),
                    preprocess_config.get("clahe_tile_size", 8),
                ),
            )
            # Apply CLAHE to each channel separately for color images
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            processed[:, :, 0] = clahe.apply(processed[:, :, 0])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

        # Gamma correction
        if preprocess_config.get("gamma_correction", False):
            gamma = preprocess_config.get("gamma", 1.2)
            inv_gamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
            ).astype("uint8")
            processed = cv2.LUT(processed, table)

        # #region agent log
        post_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        post_mean = float(np.mean(post_gray))
        _debug_log(
            "camera_detector_node.py:560",
            "Image preprocessing complete",
            {
                "pre_mean_brightness": mean_brightness,
                "post_mean_brightness": post_mean,
                "brightness_change": post_mean - mean_brightness,
            },
            "J",
        )
        # #endregion

        return processed

    def _detect_aruco_tags(self, image: np.ndarray) -> list[dict]:
        """
        Detect ArUco tags in image using cv2.aruco with post-processing enhancements

        Post-processing techniques:
        1. Multi-scale detection (different image scales)
        2. Image enhancement (sharpening, denoising, contrast)
        3. Rejected candidate re-evaluation with relaxed parameters
        4. Multiple detection passes with different parameter sets

        Args:
            image: Input image in BGR format

        Returns:
            List of detection dictionaries with keys: bbox, class_id, confidence, marker_id
        """
        if not self.aruco_enabled:
            return []

        # #region agent log
        import json
        import time

        log_path = "/media/durian/AI/AI/urc-Espresso-2026/.cursor/debug.log"
        try:
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H1,H6,H11,H12",
                            "location": "camera_detector_node.py:906",
                            "message": "ArUco detection entry",
                            "data": {
                                "image_shape": list(image.shape),
                                "aruco_enabled": self.aruco_enabled,
                                "maxMarkerPerimeterRate": getattr(
                                    self.aruco_params, "maxMarkerPerimeterRate", None
                                ),
                                "errorCorrectionRate": getattr(
                                    self.aruco_params, "errorCorrectionRate", None
                                ),
                                "polygonalApproxAccuracyRate": getattr(
                                    self.aruco_params,
                                    "polygonalApproxAccuracyRate",
                                    None,
                                ),
                                "cornerRefinementMethod": str(
                                    getattr(
                                        self.aruco_params,
                                        "cornerRefinementMethod",
                                        None,
                                    )
                                ),
                                "perspectiveRemovePixelPerCell": getattr(
                                    self.aruco_params,
                                    "perspectiveRemovePixelPerCell",
                                    None,
                                ),
                                "perspectiveRemoveIgnoredMarginPerCell": getattr(
                                    self.aruco_params,
                                    "perspectiveRemoveIgnoredMarginPerCell",
                                    None,
                                ),
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except:
            pass
        # #endregion

        detections = []
        scale_factor = 1.0  # Initialize scale factor
        original_height, original_width = image.shape[:2]  # (height, width)
        ids = None  # Initialize for exit log
        rejected = None  # Initialize for exit log

        try:
            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # IMPROVEMENT: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This helps detect large tags where lighting might be uneven across the tag face
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Optional: Downscale large images for faster detection
            # ArUco detection works well on downscaled images and is much faster
            max_dimension = self.config.get("aruco", {}).get("max_detection_size", 1280)
            if max_dimension > 0 and max(gray.shape[:2]) > max_dimension:
                scale_factor = max_dimension / max(gray.shape[:2])
                new_width = int(gray.shape[1] * scale_factor)
                new_height = int(gray.shape[0] * scale_factor)
                gray = cv2.resize(
                    gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )
                logger.debug(
                    f"Downscaled image for ArUco detection: {original_height}x{original_width} -> {new_height}x{new_width} (scale={scale_factor:.3f})"
                )

            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H6",
                                "location": "camera_detector_node.py:935",
                                "message": "Image scaling info",
                                "data": {
                                    "original_size": [original_height, original_width],
                                    "gray_size": list(gray.shape[:2]),
                                    "scale_factor": scale_factor,
                                    "max_dimension": max_dimension,
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except:
                pass
            # #endregion

            # POST-PROCESSING: Multi-scale detection and image enhancement
            # OPTIMIZATION: Multi-dictionary only on Pass 1, single dictionary on subsequent passes
            all_detection_ids = set()  # Track unique marker IDs to avoid duplicates

            # Pass 1: Original image with motion-adaptive parameters
            # MULTI-DICTIONARY: Try all dictionary sizes (4x4, 5x5, 6x6, 7x7) - ONLY ON FIRST PASS
            # Use motion-adaptive parameters if enabled
            detection_params = (
                self._get_motion_adaptive_params()
                if self.motion_adaptive
                else self.aruco_params
            )
            corners, ids, rejected = self._detect_aruco_multi_dict(
                gray, detection_params
            )
            if ids is not None and len(ids) > 0:
                for i, marker_id in enumerate(ids.flatten()):
                    all_detection_ids.add(int(marker_id))
            else:
                corners, ids = None, None

            # OPTIMIZATION: Early exit - if Pass 1 found markers, skip most enhancement passes
            # Only do additional passes if Pass 1 found very few markers
            found_markers = len(ids) if ids is not None else 0
            skip_enhancement_passes = found_markers >= 2  # Skip if we found 2+ markers

            # Pass 2: Enhanced image (sharpen, denoise, contrast) - ONLY if Pass 1 found few/no markers
            # SINGLE DICTIONARY: Use primary dictionary (4x4) for speed
            if not skip_enhancement_passes:
                enhanced_gray = self._enhance_image_for_aruco(gray)
                corners2, ids2, rejected2 = self._detect_aruco_single_pass(
                    enhanced_gray, self.aruco_params
                )

                # Pass 2b: Additional enhancement - histogram equalization - ONLY if Pass 2 found nothing
                if ids2 is None or len(ids2) == 0:
                    hist_eq_gray = cv2.equalizeHist(gray)
                    corners2b, ids2b, rejected2b = self._detect_aruco_single_pass(
                        hist_eq_gray, self.aruco_params
                    )
                    if ids2b is not None and len(ids2b) > 0:
                        for i, marker_id in enumerate(ids2b.flatten()):
                            if int(marker_id) not in all_detection_ids:
                                if ids is None:
                                    ids = ids2b
                                    corners = corners2b
                                else:
                                    ids = np.concatenate([ids, ids2b])
                                    corners = np.concatenate([corners, corners2b])
                                all_detection_ids.add(int(marker_id))

                if ids2 is not None and len(ids2) > 0:
                    for i, marker_id in enumerate(ids2.flatten()):
                        if int(marker_id) not in all_detection_ids:
                            # Merge with original detections
                            if ids is None:
                                ids = ids2
                                corners = corners2
                            else:
                                ids = np.concatenate([ids, ids2])
                                corners = np.concatenate([corners, corners2])
                            all_detection_ids.add(int(marker_id))

            # Pass 3: Upscaled image (2x) for small tags (10mm, 20mm)
            # SINGLE DICTIONARY: Use primary dictionary for speed
            # ONLY if we found few markers and image is small enough
            if (ids is None or len(ids) < 3) and max(gray.shape[:2]) < 800:
                upscaled_gray = cv2.resize(
                    gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
                )
                upscale_params = self._copy_aruco_params(self.aruco_params)
                upscale_params.minMarkerPerimeterRate = (
                    self.aruco_params.minMarkerPerimeterRate * 2.0
                )  # Scale threshold
                corners3, ids3, rejected3 = self._detect_aruco_single_pass(
                    upscaled_gray, upscale_params
                )
                if ids3 is not None and len(ids3) > 0:
                    # Scale corners back to original size
                    for i in range(len(corners3)):
                        corners3[i][0] = corners3[i][0] / 2.0
                    for i, marker_id in enumerate(ids3.flatten()):
                        if int(marker_id) not in all_detection_ids:
                            if ids is None:
                                ids = ids3
                                corners = corners3
                            else:
                                ids = np.concatenate([ids, ids3])
                                corners = np.concatenate([corners, corners3])
                            all_detection_ids.add(int(marker_id))

            # Pass 4: Downscaled for LARGE markers (200mm) - OPTIMIZED: only one scale, only if needed
            # SINGLE DICTIONARY: Use primary dictionary for speed
            # ONLY if we found few markers and image is large enough
            if (ids is None or len(ids) < 3) and max(gray.shape[:2]) > 400:
                # OPTIMIZATION: Only try one downscale factor (0.67) instead of 3
                downscale_factor = 0.67
                downscaled_gray = cv2.resize(
                    gray,
                    None,
                    fx=downscale_factor,
                    fy=downscale_factor,
                    interpolation=cv2.INTER_AREA,
                )
                downscale_params = self._copy_aruco_params(self.aruco_params)
                # Adjust threshold proportionally to scale
                downscale_params.minMarkerPerimeterRate = (
                    self.aruco_params.minMarkerPerimeterRate * downscale_factor
                )
                # For large markers, allow higher error correction
                downscale_params.errorCorrectionRate = min(
                    0.9, self.aruco_params.errorCorrectionRate + 0.1
                )
                corners5, ids5, rejected5 = self._detect_aruco_single_pass(
                    downscaled_gray, downscale_params
                )
                if ids5 is not None and len(ids5) > 0:
                    # Scale corners back to original size
                    scale_back = 1.0 / downscale_factor
                    for i in range(len(corners5)):
                        corners5[i][0] = corners5[i][0] * scale_back
                    for i, marker_id in enumerate(ids5.flatten()):
                        if int(marker_id) not in all_detection_ids:
                            if ids is None:
                                ids = ids5
                                corners = corners5
                            else:
                                ids = np.concatenate([ids, ids5])
                                corners = np.concatenate([corners, corners5])
                            all_detection_ids.add(int(marker_id))

            # Pass 5: Re-evaluate rejected candidates - OPTIMIZED: only if needed, limit candidates
            # SINGLE DICTIONARY: Use primary dictionary for speed
            # ONLY if we found few markers and have rejected candidates
            if (
                rejected is not None
                and len(rejected) > 0
                and (ids is None or len(ids) < 3)
            ):
                relaxed_params = self._copy_aruco_params(self.aruco_params)
                relaxed_params.errorCorrectionRate = 0.9  # Extremely relaxed
                relaxed_params.minMarkerPerimeterRate = 0.015  # Very relaxed
                relaxed_params.polygonalApproxAccuracyRate = 0.08  # More relaxed
                # OPTIMIZATION: Only process 5 rejected candidates instead of 20
                for rej_candidate in rejected[:5]:  # Reduced from 20 for performance
                    try:
                        # Create a small image patch around the rejected candidate
                        rej_corners = rej_candidate[0]
                        x_coords = [p[0] for p in rej_corners]
                        y_coords = [p[1] for p in rej_corners]
                        x_min, x_max = int(max(0, min(x_coords) - 10)), int(
                            min(gray.shape[1], max(x_coords) + 10)
                        )
                        y_min, y_max = int(max(0, min(y_coords) - 10)), int(
                            min(gray.shape[0], max(y_coords) + 10)
                        )
                        patch = gray[y_min:y_max, x_min:x_max]
                        if patch.size > 0 and min(patch.shape) > 20:
                            # SINGLE DICTIONARY: Use primary dictionary for speed
                            corners4, ids4, _ = self._detect_aruco_single_pass(
                                patch, relaxed_params
                            )
                            if ids4 is not None and len(ids4) > 0:
                                # Adjust corners to original image coordinates
                                for i in range(len(corners4)):
                                    corners4[i][0][:, 0] += x_min
                                    corners4[i][0][:, 1] += y_min
                                for i, marker_id in enumerate(ids4.flatten()):
                                    if int(marker_id) not in all_detection_ids:
                                        if ids is None:
                                            ids = ids4
                                            corners = corners4
                                        else:
                                            ids = np.concatenate([ids, ids4])
                                            corners = np.concatenate(
                                                [corners, corners4]
                                            )
                                        all_detection_ids.add(int(marker_id))
                    except:
                        pass

            # #region agent log
            try:
                num_detected = len(ids) if ids is not None else 0
                num_rejected = len(rejected) if rejected is not None else 0
                detected_ids = ids.flatten().tolist() if ids is not None else []
                # Log rejected marker sizes for H3, H7
                rejected_sizes = []
                if rejected is not None and len(rejected) > 0:
                    for rej in rejected[:5]:  # Log first 5 rejected
                        if len(rej) > 0 and len(rej[0]) >= 4:
                            rej_corners = rej[0]
                            rej_x = [p[0] for p in rej_corners]
                            rej_y = [p[1] for p in rej_corners]
                            rej_width = max(rej_x) - min(rej_x)
                            rej_height = max(rej_y) - min(rej_y)
                            rej_perimeter = 2 * (rej_width + rej_height)
                            rej_perimeter_rate = rej_perimeter / max(gray.shape[:2])
                            rejected_sizes.append(
                                {
                                    "width": float(rej_width),
                                    "height": float(rej_height),
                                    "perimeter_rate": float(rej_perimeter_rate),
                                }
                            )
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H2,H3,H7,H9",
                                "location": "camera_detector_node.py:950",
                                "message": "ArUco detection results",
                                "data": {
                                    "num_detected": num_detected,
                                    "num_rejected": num_rejected,
                                    "detected_ids": detected_ids,
                                    "gray_size": list(gray.shape[:2]),
                                    "rejected_sizes": rejected_sizes,
                                    "minMarkerPerimeterRate": getattr(
                                        self.aruco_params,
                                        "minMarkerPerimeterRate",
                                        None,
                                    ),
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except:
                pass
            # #endregion

            if ids is not None and len(ids) > 0:
                for i, marker_id in enumerate(ids.flatten()):
                    # Get corner points
                    corner_points = corners[i][0]  # Shape: (4, 2)

                    # Scale coordinates back to original image size if downscaled
                    if scale_factor < 1.0:
                        corner_points = corner_points / scale_factor

                    # RELIABILITY FIX: Use unified filtering function for consistent filtering
                    # This ensures ALL markers go through the same filter logic
                    (
                        should_accept,
                        width,
                        height,
                        bbox_area,
                        aspect_ratio_check,
                        filter_reason,
                    ) = self._filter_marker(
                        corner_points,
                        marker_id,
                        original_width,
                        original_height,
                        image,
                        log_path,
                    )

                    if not should_accept:
                        continue  # Marker filtered out

                    # Get bounding box coordinates from filter function (recalculate for consistency)
                    x_coords = corner_points[:, 0]
                    y_coords = corner_points[:, 1]
                    x_min = float(np.min(x_coords))
                    y_min = float(np.min(y_coords))
                    x_max = float(np.max(x_coords))
                    y_max = float(np.max(y_coords))

                    # Clamp to image bounds
                    x_min = max(0, min(x_min, original_width - 1))
                    y_min = max(0, min(y_min, original_height - 1))
                    x_max = max(0, min(x_max, original_width - 1))
                    y_max = max(0, min(y_max, original_height - 1))

                    # MOTION HANDLING: Validate position against prediction (Kalman filter)
                    bbox = [x_min, y_min, x_max, y_max]
                    if not self._validate_detection_position(marker_id, bbox):
                        # #region agent log
                        try:
                            with open(log_path, "a") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "MOTION_VALIDATION",
                                            "location": "camera_detector_node.py:1405",
                                            "message": "Marker rejected by position validation",
                                            "data": {
                                                "marker_id": int(marker_id),
                                                "bbox": bbox,
                                            },
                                            "timestamp": int(time.time() * 1000),
                                        }
                                    )
                                    + "\n"
                                )
                        except:
                            pass
                        # #endregion
                        continue  # Position doesn't match prediction

                    # Calculate area_ratio for confidence calculation
                    image_area = image.shape[0] * image.shape[1]
                    area_ratio = bbox_area / image_area

                    # MOTION HANDLING: Check multi-frame consensus
                    if not self._check_marker_consensus(
                        marker_id, bbox, 0.8
                    ):  # Use base confidence for consensus
                        # #region agent log
                        try:
                            with open(log_path, "a") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "CONSENSUS",
                                            "location": "camera_detector_node.py:1425",
                                            "message": "Marker rejected by consensus check",
                                            "data": {
                                                "marker_id": int(marker_id),
                                                "bbox": bbox,
                                                "history_size": len(
                                                    self.marker_detection_history.get(
                                                        marker_id, []
                                                    )
                                                ),
                                            },
                                            "timestamp": int(time.time() * 1000),
                                        }
                                    )
                                    + "\n"
                                )
                        except:
                            pass
                        # #endregion
                        continue  # No consensus yet

                    # Use bbox_area from filter for consistency
                    area = bbox_area

                    # Calculate perspective distortion metrics for angle detection (H11-H15)
                    # Measure how "square" the marker appears (angled markers will be more distorted)
                    # Calculate side lengths and angles to detect perspective distortion
                    try:
                        # Get 4 corners in order: top-left, top-right, bottom-right, bottom-left
                        # Calculate distances between adjacent corners
                        p0, p1, p2, p3 = (
                            corner_points[0],
                            corner_points[1],
                            corner_points[2],
                            corner_points[3],
                        )
                        side1 = np.linalg.norm(p1 - p0)  # Top edge
                        side2 = np.linalg.norm(p2 - p1)  # Right edge
                        side3 = np.linalg.norm(p3 - p2)  # Bottom edge
                        side4 = np.linalg.norm(p0 - p3)  # Left edge

                        # Calculate angles at corners (using dot product)
                        def angle_between_vectors(v1, v2):
                            cos_angle = np.dot(v1, v2) / (
                                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
                            )
                            return (
                                np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                            )

                        v1 = p1 - p0  # Top edge vector
                        v2 = p3 - p0  # Left edge vector
                        corner_angle_0 = angle_between_vectors(v1, v2)

                        # Measure how far from 90 degrees (perfect square)
                        angle_deviation = abs(corner_angle_0 - 90.0)

                        # Calculate aspect ratio of sides (distorted markers have unequal opposite sides)
                        side_ratio_1 = max(side1, side3) / (min(side1, side3) + 1e-6)
                        side_ratio_2 = max(side2, side4) / (min(side2, side4) + 1e-6)
                        max_side_ratio = max(side_ratio_1, side_ratio_2)

                        # Estimate viewing angle: high angle_deviation or side_ratio indicates angled view
                        is_angled = angle_deviation > 10.0 or max_side_ratio > 1.2
                    except:
                        angle_deviation = 0.0
                        max_side_ratio = 1.0
                        is_angled = False

                    # Additional size-based filtering (already filtered by area and aspect ratio above)
                    # But keep these checks for very small markers that might slip through
                    min_pixel_size = 8  # Minimum pixel size
                    min_area_ratio = 0.00003  # 0.003% of image

                    filter_reason = None
                    if width < min_pixel_size:
                        filter_reason = f"width_too_small_{width:.1f}<{min_pixel_size}"
                    elif height < min_pixel_size:
                        filter_reason = (
                            f"height_too_small_{height:.1f}<{min_pixel_size}"
                        )
                    elif area_ratio < min_area_ratio:
                        filter_reason = f"area_ratio_too_small_{area_ratio:.6f}<{min_area_ratio:.6f}"

                    if filter_reason:
                        # #region agent log
                        try:
                            with open(log_path, "a") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "sessionId": "debug-session",
                                            "runId": "run1",
                                            "hypothesisId": "H3,H5",
                                            "location": "camera_detector_node.py:1480",
                                            "message": "Filtered marker (size)",
                                            "data": {
                                                "marker_id": int(marker_id),
                                                "width": width,
                                                "height": height,
                                                "area_ratio": area_ratio,
                                                "reason": filter_reason,
                                                "min_pixel_size": min_pixel_size,
                                                "min_area_ratio": min_area_ratio,
                                            },
                                            "timestamp": int(time.time() * 1000),
                                        }
                                    )
                                    + "\n"
                                )
                        except:
                            pass
                        # #endregion
                        continue  # Skip this detection - too small to be valid

                    # Calculate confidence based on marker size and shape
                    # ArUco detection is very reliable - if cv2.aruco detects a marker, it's almost certainly correct
                    # Use high base confidence and only slightly adjust for edge cases

                    # #region agent log
                    try:
                        # Calculate perimeter for H7 hypothesis
                        perimeter = 2 * (width + height)
                        perimeter_rate = perimeter / max(
                            original_width, original_height
                        )
                        min_perimeter_rate = getattr(
                            self.aruco_params, "minMarkerPerimeterRate", None
                        )
                        max_perimeter_rate = getattr(
                            self.aruco_params, "maxMarkerPerimeterRate", None
                        )
                        corner_refinement = getattr(
                            self.aruco_params, "cornerRefinementMethod", None
                        )
                        perspective_pixel = getattr(
                            self.aruco_params, "perspectiveRemovePixelPerCell", None
                        )
                        error_correction = getattr(
                            self.aruco_params, "errorCorrectionRate", None
                        )
                        with open(log_path, "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H4,H7,H8,H10,H11,H12,H13",
                                        "location": "camera_detector_node.py:970",
                                        "message": "Marker detection details",
                                        "data": {
                                            "marker_id": int(marker_id),
                                            "bbox": [x_min, y_min, x_max, y_max],
                                            "width": width,
                                            "height": height,
                                            "area": area,
                                            "area_ratio": area_ratio,
                                            "perimeter": perimeter,
                                            "perimeter_rate": perimeter_rate,
                                            "min_perimeter_rate": min_perimeter_rate,
                                            "max_perimeter_rate": max_perimeter_rate,
                                            "perimeter_rate_valid": (
                                                min_perimeter_rate is None
                                                or perimeter_rate >= min_perimeter_rate
                                            )
                                            and (
                                                max_perimeter_rate is None
                                                or perimeter_rate <= max_perimeter_rate
                                            ),
                                            "image_size": [
                                                original_height,
                                                original_width,
                                            ],
                                            "scale_factor": scale_factor,
                                            "near_edge": x_min < 10
                                            or y_min < 10
                                            or x_max > original_width - 10
                                            or y_max > original_height - 10,
                                            "is_angled": is_angled,
                                            "angle_deviation": angle_deviation,
                                            "max_side_ratio": max_side_ratio,
                                            "corner_refinement": (
                                                str(corner_refinement)
                                                if corner_refinement is not None
                                                else None
                                            ),
                                            "perspective_pixel": perspective_pixel,
                                            "error_correction_rate": error_correction,
                                        },
                                        "timestamp": int(time.time() * 1000),
                                    }
                                )
                                + "\n"
                            )
                    except:
                        pass
                    # #endregion

                    # Base confidence: ArUco detection is deterministic and accurate
                    # Large markers (10-50cm) should get maximum confidence
                    # Start with high confidence (0.85-0.99) since detection itself is reliable
                    if (
                        area_ratio > 0.1
                    ):  # Marker is >10% of image - very large marker (10-50cm), maximum confidence
                        base_confidence = 0.99
                    elif (
                        area_ratio > 0.05
                    ):  # Marker is >5% of image - large marker, very confident
                        base_confidence = 0.98
                    elif area_ratio > 0.01:  # Marker is >1% of image - very confident
                        base_confidence = 0.95
                    elif area_ratio > 0.002:  # Marker is >0.2% of image - confident
                        base_confidence = 0.90
                    elif (
                        area_ratio > 0.0005
                    ):  # Marker is >0.05% of image - still confident
                        base_confidence = 0.85
                    else:  # Very small marker - still reliable but slightly less confident
                        base_confidence = 0.80

                    # Aspect ratio: penalize distorted markers (ArUco markers should be roughly square)
                    aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                    if aspect_ratio > 2.0:
                        # Distorted - reduce confidence (filter should catch these, but penalize if they pass)
                        aspect_penalty = 0.85
                    elif aspect_ratio > 1.5:
                        # Moderately distorted - slight reduction
                        aspect_penalty = 0.95
                    else:
                        # Normal aspect ratio - no penalty
                        aspect_penalty = 1.0

                    confidence = base_confidence * aspect_penalty

                    # Ensure minimum confidence for detected ArUco tags
                    # If cv2.aruco detected it, it's reliable - minimum 0.75 confidence
                    # Large markers should get maximum confidence (0.99)
                    confidence = max(0.75, min(0.99, confidence))

                    # #region agent log
                    try:
                        with open(log_path, "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H4",
                                        "location": "camera_detector_node.py:1020",
                                        "message": "Confidence calculation",
                                        "data": {
                                            "marker_id": int(marker_id),
                                            "base_confidence": base_confidence,
                                            "aspect_penalty": aspect_penalty,
                                            "final_confidence": confidence,
                                            "area_ratio": area_ratio,
                                            "aspect_ratio": aspect_ratio,
                                        },
                                        "timestamp": int(time.time() * 1000),
                                    }
                                )
                                + "\n"
                            )
                    except:
                        pass
                    # #endregion

                    # MOTION HANDLING: Update Kalman filter tracker
                    self._update_marker_tracker(marker_id, [x_min, y_min, x_max, y_max])

                    # MOTION HANDLING: Update Kalman filter tracker
                    bbox = [x_min, y_min, x_max, y_max]
                    self._update_marker_tracker(marker_id, bbox)

                    detections.append(
                        {
                            "bbox": bbox,
                            "class_id": self.aruco_class_id,
                            "confidence": confidence,
                            "marker_id": int(marker_id),
                        }
                    )

                    # #region agent log
                    try:
                        with open(log_path, "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H4,H6",
                                        "location": "camera_detector_node.py:1040",
                                        "message": "Marker accepted",
                                        "data": {
                                            "marker_id": int(marker_id),
                                            "width": width,
                                            "height": height,
                                            "area_ratio": area_ratio,
                                            "confidence": confidence,
                                            "total_accepted_so_far": len(detections),
                                        },
                                        "timestamp": int(time.time() * 1000),
                                    }
                                )
                                + "\n"
                            )
                    except:
                        pass
                    # #endregion

                    logger.info(
                        f"Detected ArUco tag: ID={marker_id}, bbox=[{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}], "
                        f"size={width:.1f}x{height:.1f}, area_ratio={area_ratio:.6f}, conf={confidence:.3f}"
                    )

        except Exception as e:
            logger.error(f"ArUco detection failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H9",
                                "location": "camera_detector_node.py:1035",
                                "message": "ArUco detection error",
                                "data": {
                                    "error": str(e),
                                    "traceback": traceback.format_exc(),
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except:
                pass
            # #endregion

        # MOTION HANDLING: Apply confidence decay for markers not seen recently
        detections = self._apply_confidence_decay(detections)

        # #region agent log
        try:
            num_detected_by_cv2 = len(ids) if ids is not None else 0
            num_filtered = (
                num_detected_by_cv2 - len(detections) if ids is not None else 0
            )

            # Track all IDs detected by cv2 (before filtering)
            all_cv2_ids = []
            if ids is not None and len(ids) > 0:
                all_cv2_ids = [int(marker_id) for marker_id in ids.flatten()]

            # Track all IDs that passed filters
            accepted_marker_ids = [d.get("marker_id") for d in detections]

            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "ID_TRACKING",
                            "location": "camera_detector_node.py:1040",
                            "message": "ArUco detection exit summary",
                            "data": {
                                "num_detected_by_cv2": num_detected_by_cv2,
                                "num_filtered_out": num_filtered,
                                "num_final_detections": len(detections),
                                "filter_rate": (
                                    num_filtered / num_detected_by_cv2
                                    if num_detected_by_cv2 > 0
                                    else 0
                                ),
                                "detection_ids": accepted_marker_ids,
                                "all_cv2_ids": all_cv2_ids,  # All IDs detected by cv2 before filtering
                                "confidences": [
                                    d.get("confidence") for d in detections
                                ],
                                "unique_cv2_ids": sorted(list(set(all_cv2_ids))),
                                "unique_accepted_ids": sorted(
                                    list(set(accepted_marker_ids))
                                ),
                                "ids_filtered_out": sorted(
                                    list(set(all_cv2_ids) - set(accepted_marker_ids))
                                ),
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except:
            pass
        # #endregion

        return detections

    def _copy_aruco_params(
        self, source: cv2.aruco.DetectorParameters
    ) -> cv2.aruco.DetectorParameters:
        """
        Copy ArUco detector parameters (DetectorParameters doesn't support __dict__)

        Args:
            source: Source DetectorParameters object

        Returns:
            New DetectorParameters object with copied values
        """
        dest = cv2.aruco.DetectorParameters()
        # Copy all attributes that are settable
        param_attrs = [
            "adaptiveThreshWinSizeMin",
            "adaptiveThreshWinSizeMax",
            "adaptiveThreshWinSizeStep",
            "adaptiveThreshConstant",
            "minMarkerPerimeterRate",
            "maxMarkerPerimeterRate",
            "polygonalApproxAccuracyRate",
            "minCornerDistanceRate",
            "minDistanceToBorder",
            "minMarkerDistanceRate",
            "cornerRefinementMethod",
            "cornerRefinementWinSize",
            "cornerRefinementMaxIterations",
            "cornerRefinementMinAccuracy",
            "markerBorderBits",
            "perspectiveRemovePixelPerCell",
            "perspectiveRemoveIgnoredMarginPerCell",
            "maxErroneousBitsInBorderRate",
            "minOtsuStdDev",
            "errorCorrectionRate",
        ]
        for attr in param_attrs:
            if hasattr(source, attr):
                try:
                    value = getattr(source, attr)
                    setattr(dest, attr, value)
                except (AttributeError, TypeError) as e:
                    # Skip attributes that can't be copied
                    logger.debug(f"Could not copy ArUco parameter {attr}: {e}")
                    pass
        return dest

    def _init_marker_tracker(self, marker_id: int, bbox: List[float]) -> None:
        """
        Initialize Kalman filter for a marker

        Args:
            marker_id: Marker ID
            bbox: Bounding box [x_min, y_min, x_max, y_max]
        """
        kf = cv2.KalmanFilter(4, 2)  # 4 state (x, y, vx, vy), 2 measurement (x, y)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        kf.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)

        # Initialize with current position
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        kf.statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32)
        kf.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32)

        self.marker_trackers[marker_id] = kf

    def _predict_marker_position(self, marker_id: int) -> Optional[Tuple[float, float]]:
        """
        Predict next marker position using Kalman filter

        Args:
            marker_id: Marker ID

        Returns:
            Predicted (x, y) position or None if no tracker
        """
        if not self.kalman_tracking or marker_id not in self.marker_trackers:
            return None

        kf = self.marker_trackers[marker_id]
        prediction = kf.predict()
        return (float(prediction[0]), float(prediction[1]))

    def _update_marker_tracker(self, marker_id: int, bbox: List[float]) -> None:
        """
        Update Kalman filter with new measurement

        Args:
            marker_id: Marker ID
            bbox: Bounding box [x_min, y_min, x_max, y_max]
        """
        if not self.kalman_tracking:
            return

        if marker_id not in self.marker_trackers:
            self._init_marker_tracker(marker_id, bbox)
            return

        kf = self.marker_trackers[marker_id]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        measurement = np.array([[center_x], [center_y]], dtype=np.float32)
        kf.correct(measurement)

        # Update velocity estimate
        if marker_id in self.marker_last_seen:
            prev_center = self.marker_trackers[marker_id].statePost[:2]
            dt = time.time() - self.marker_last_seen[marker_id]
            if dt > 0:
                velocity = (
                    np.sqrt(
                        (center_x - prev_center[0]) ** 2
                        + (center_y - prev_center[1]) ** 2
                    )
                    / dt
                )
                self.marker_velocities[marker_id] = velocity

        self.marker_last_seen[marker_id] = time.time()

    def _validate_detection_position(self, marker_id: int, bbox: List[float]) -> bool:
        """
        Validate detection position against prediction

        RELIABILITY FIX: Made more lenient for moving objects
        - Increase distance thresholds
        - Allow larger deviations for fast motion
        - Only validate if we have good tracking history

        Args:
            marker_id: Marker ID
            bbox: Bounding box [x_min, y_min, x_max, y_max]

        Returns:
            True if position is valid (within reasonable distance of prediction)
        """
        if not self.kalman_tracking:
            return True  # No tracking, accept all

        predicted = self._predict_marker_position(marker_id)
        if predicted is None:
            return True  # No prediction, accept (new marker)

        # RELIABILITY FIX: Only validate if we have sufficient tracking history
        # New markers or markers with little history should be accepted
        if marker_id not in self.marker_last_seen:
            return True  # New marker, accept

        # Check how long we've been tracking this marker
        tracking_duration = time.time() - self.marker_last_seen.get(marker_id, 0)
        if tracking_duration < 0.1:  # Just started tracking (<100ms)
            return True  # Too new, accept without validation

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        distance = np.sqrt(
            (center_x - predicted[0]) ** 2 + (center_y - predicted[1]) ** 2
        )

        # RELIABILITY FIX: Increase distance thresholds for moving objects
        # Fast motion: 200px, Slow motion: 150px (increased from 100/50)
        velocity = self.marker_velocities.get(marker_id, 0)
        max_distance = 200 if velocity > 2.0 else 150  # Increased thresholds

        return distance < max_distance

    def _check_marker_consensus(
        self, marker_id: int, bbox: List[float], confidence: float
    ) -> bool:
        """
        Check if marker has been detected consistently across frames

        RELIABILITY FIX: Made more lenient for moving objects
        - Allow first detection (don't require consensus for new markers)
        - Increase position tolerance for moving objects
        - Only require consensus for markers already in history

        Args:
            marker_id: Marker ID
            bbox: Bounding box [x_min, y_min, x_max, y_max]
            confidence: Detection confidence

        Returns:
            True if marker has consensus or is new detection
        """
        if self.consensus_frames <= 1:
            return True  # Consensus disabled

        current_time = time.time()

        # Add current detection
        self.marker_detection_history[marker_id].append(
            {"bbox": bbox, "confidence": confidence, "timestamp": current_time}
        )

        # Remove old detections (>1 second)
        self.marker_detection_history[marker_id] = [
            d
            for d in self.marker_detection_history[marker_id]
            if current_time - d["timestamp"] < 1.0
        ]

        history_size = len(self.marker_detection_history[marker_id])

        # RELIABILITY FIX: Allow first detection (new markers)
        # Only require consensus if marker has been seen before
        if history_size == 1:
            return True  # First detection - accept immediately

        # Require consensus across frames for existing markers
        if history_size >= self.consensus_frames:
            # Check if detections are consistent (similar position)
            recent = self.marker_detection_history[marker_id][-self.consensus_frames :]
            centers_x = [(d["bbox"][0] + d["bbox"][2]) / 2 for d in recent]
            centers_y = [(d["bbox"][1] + d["bbox"][3]) / 2 for d in recent]

            # RELIABILITY FIX: Increase tolerance for moving objects (50px -> 100px)
            # Moving objects can have larger position variation
            position_tolerance = 100  # Increased from 50px for moving objects
            if (
                max(centers_x) - min(centers_x) < position_tolerance
                and max(centers_y) - min(centers_y) < position_tolerance
            ):
                return True

        # If we have some history but not enough for consensus, still allow if high confidence
        # This prevents rejecting valid detections that are just starting to be tracked
        if history_size >= 1 and confidence > 0.9:
            return True  # High confidence detection, allow even without full consensus

        return False

    def _apply_confidence_decay(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply confidence decay for markers not seen recently

        Args:
            detections: List of detection dictionaries

        Returns:
            Detections with decayed confidence
        """
        if self.confidence_decay_rate <= 0:
            return detections

        current_time = time.time()

        for det in detections:
            marker_id = det.get("marker_id")
            if marker_id is not None:
                if marker_id in self.marker_last_seen:
                    time_since_seen = current_time - self.marker_last_seen[marker_id]
                    if time_since_seen > 0.1:  # Not seen in this frame
                        # Decay confidence
                        decay = self.confidence_decay_rate * time_since_seen
                        det["confidence"] = max(0.5, det["confidence"] * (1.0 - decay))
                else:
                    self.marker_last_seen[marker_id] = current_time

        return detections

    def _get_motion_adaptive_params(self) -> cv2.aruco.DetectorParameters:
        """
        Get parameters optimized for current motion level

        Returns:
            DetectorParameters adjusted for motion
        """
        if not self.motion_adaptive:
            return self.aruco_params

        # Estimate average motion velocity
        if self.marker_velocities:
            avg_velocity = sum(self.marker_velocities.values()) / len(
                self.marker_velocities
            )
        else:
            avg_velocity = 0.0

        params = self._copy_aruco_params(self.aruco_params)

        if avg_velocity > 2.0:  # Fast motion
            # More relaxed for motion blur
            params.errorCorrectionRate = 0.9
            params.polygonalApproxAccuracyRate = 0.05  # More lenient
        else:  # Slow/stationary
            # Stricter for better accuracy
            params.errorCorrectionRate = 0.8
            params.polygonalApproxAccuracyRate = 0.03

        return params

    def _filter_marker(
        self,
        corner_points: np.ndarray,
        marker_id: int,
        original_width: int,
        original_height: int,
        image: np.ndarray,
        log_path: str,
    ) -> tuple:
        """
        Unified filtering function for all markers - ensures consistent filtering

        Args:
            corner_points: Marker corner points (4, 2)
            marker_id: Marker ID
            original_width: Original image width
            original_height: Original image height
            image: Original image
            log_path: Path to log file

        Returns:
            Tuple of (should_accept: bool, width: float, height: float, area: float, aspect_ratio: float, filter_reason: str)
        """
        # Calculate bounding box
        x_coords = corner_points[:, 0]
        y_coords = corner_points[:, 1]
        x_min = float(np.min(x_coords))
        y_min = float(np.min(y_coords))
        x_max = float(np.max(x_coords))
        y_max = float(np.max(y_coords))

        # Clamp to image bounds
        x_min = max(0, min(x_min, original_width - 1))
        y_min = max(0, min(y_min, original_height - 1))
        x_max = max(0, min(x_max, original_width - 1))
        y_max = max(0, min(y_max, original_height - 1))

        # Calculate dimensions
        width = x_max - x_min
        height = y_max - y_min
        bbox_area = width * height
        image_area = image.shape[0] * image.shape[1]
        area_ratio = bbox_area / image_area
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)

        # Apply filters
        filter_reason = None

        # RELIABILITY FIX: Further relaxed area filter to catch valid markers
        # EVIDENCE: 7 IDs (11, 13, 17, 24, 28, 34, 46) still filtered after 300 px² threshold
        #           IDs 11, 13, 28 have areas 327-372 px² but still filtered (may be size checks)
        #           ID 46 has 292 px² - just below 300, so lower to 250 px²
        #           IDs 17, 24, 34 have areas < 300, so need lower threshold
        # Area filter: must be at least 250 px² (further relaxed from 300)
        if bbox_area < 250:
            filter_reason = f"area_too_small_{bbox_area:.1f}<300"
            try:
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "AREA_FILTER",
                                "location": "camera_detector_node.py:_filter_marker",
                                "message": "Area filtered marker",
                                "data": {
                                    "marker_id": int(marker_id),
                                    "bbox_area": float(bbox_area),
                                    "width": width,
                                    "height": height,
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except:
                pass
            return (False, width, height, bbox_area, aspect_ratio, filter_reason)

        # RELIABILITY FIX: Tighten aspect ratio filter to reduce false positives
        # EVIDENCE: User reports false positives with incorrect aspect ratios
        #           ArUco markers should be roughly square (1.0), allow up to 2.0 for angled views
        #           Anything > 2.0 is likely a false positive (rectangular objects, not square markers)
        # Aspect ratio filter: must be <= 2.0 (tightened from 3.0 to reduce false positives)
        # DEBUG: Log all markers to verify filter is working
        if aspect_ratio > 2.0:
            filter_reason = f"aspect_ratio_extreme_{aspect_ratio:.2f}>2.0"
            try:
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "ASPECT_FILTER",
                                "location": "camera_detector_node.py:_filter_marker",
                                "message": "Aspect ratio filtered marker",
                                "data": {
                                    "marker_id": int(marker_id),
                                    "aspect_ratio": aspect_ratio,
                                    "width": width,
                                    "height": height,
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except:
                pass
            return (False, width, height, bbox_area, aspect_ratio, filter_reason)

        # Additional size checks (relaxed for reliability)
        # EVIDENCE: IDs 11, 13, 28 have area > 300 but still filtered - may be failing size checks
        #           Relax these checks to allow valid markers
        min_pixel_size = 6  # Relaxed from 8 (for very small but valid tags)
        min_area_ratio = (
            0.00002  # Relaxed from 0.00003 (for small tags in large images)
        )

        if width < min_pixel_size:
            filter_reason = f"width_too_small_{width:.1f}<{min_pixel_size}"
        elif height < min_pixel_size:
            filter_reason = f"height_too_small_{height:.1f}<{min_pixel_size}"
        elif area_ratio < min_area_ratio:
            filter_reason = (
                f"area_ratio_too_small_{area_ratio:.6f}<{min_area_ratio:.6f}"
            )

        if filter_reason:
            try:
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "SIZE_FILTER",
                                "location": "camera_detector_node.py:_filter_marker",
                                "message": "Size filtered marker",
                                "data": {
                                    "marker_id": int(marker_id),
                                    "width": width,
                                    "height": height,
                                    "area_ratio": area_ratio,
                                    "reason": filter_reason,
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except:
                pass
            return (False, width, height, bbox_area, aspect_ratio, filter_reason)

        # Marker passes all filters
        # DEBUG: Log all accepted markers to verify filter is working correctly
        try:
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "FILTER_PASS",
                            "location": "camera_detector_node.py:_filter_marker",
                            "message": "Marker passed all filters",
                            "data": {
                                "marker_id": int(marker_id),
                                "width": width,
                                "height": height,
                                "bbox_area": bbox_area,
                                "aspect_ratio": aspect_ratio,
                                "area_ratio": area_ratio,
                            },
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except:
            pass

        return (True, width, height, bbox_area, aspect_ratio, None)

    def _detect_aruco_single_pass(
        self,
        gray: np.ndarray,
        params: cv2.aruco.DetectorParameters,
        aruco_dict: Optional[Any] = None,
    ) -> tuple:
        """
        Single-pass ArUco detection with given parameters

        Args:
            gray: Grayscale image
            params: Detector parameters
            aruco_dict: ArUco dictionary to use (defaults to self.aruco_dict)

        Returns:
            Tuple of (corners, ids, rejected)
        """
        if aruco_dict is None:
            aruco_dict = self.aruco_dict

        if self.use_aruco_detector:
            # Create temporary detector with custom parameters
            temp_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            return temp_detector.detectMarkers(gray)
        else:
            return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    def _detect_aruco_multi_dict(
        self, gray: np.ndarray, params: cv2.aruco.DetectorParameters
    ) -> tuple:
        """
        Detect ArUco markers using all configured dictionaries (4x4, 5x5, 6x6, 7x7)
        and merge results with unique ID encoding

        Args:
            gray: Grayscale image
            params: Detector parameters

        Returns:
            Tuple of (corners, ids, rejected) with merged results from all dictionaries
        """
        use_multi_dict = hasattr(self, "aruco_dicts") and len(self.aruco_dicts) > 1

        if not use_multi_dict:
            # Single dictionary mode - use original behavior
            return self._detect_aruco_single_pass(gray, params)

        # Multi-dictionary mode - try all dictionaries
        all_corners = []
        all_ids = []
        all_rejected = []

        # Dictionary ID offsets to make IDs unique across dictionaries
        dict_offsets = {
            "DICT_4X4_50": 0,
            "DICT_4X4_100": 0,
            "DICT_4X4_250": 0,
            "DICT_4X4_1000": 0,
            "DICT_5X5_50": 10000,
            "DICT_5X5_100": 10000,
            "DICT_6X6_50": 20000,
            "DICT_6X6_100": 20000,
            "DICT_7X7_50": 30000,
            "DICT_7X7_100": 30000,
        }

        for dict_name, aruco_dict in self.aruco_dicts.items():
            try:
                dict_corners, dict_ids, dict_rejected = self._detect_aruco_single_pass(
                    gray, params, aruco_dict
                )

                if dict_ids is not None and len(dict_ids) > 0:
                    dict_offset = dict_offsets.get(dict_name, 0)

                    for i, marker_id in enumerate(dict_ids.flatten()):
                        # Encode dictionary type into marker ID
                        # Format: original_id + dict_offset
                        unique_id = int(marker_id) + dict_offset
                        all_corners.append(dict_corners[i])
                        all_ids.append(unique_id)

                if dict_rejected is not None and len(dict_rejected) > 0:
                    all_rejected.extend(dict_rejected)
            except Exception as e:
                logger.debug(f"Error detecting with {dict_name}: {e}")

        # Convert to numpy arrays
        if all_ids:
            ids = np.array(all_ids).reshape(-1, 1)
            corners = all_corners
        else:
            ids = None
            corners = None

        if not all_rejected:
            rejected = None
        else:
            rejected = all_rejected

        return corners, ids, rejected

    def _deblur_for_motion(self, gray: np.ndarray) -> np.ndarray:
        """
        Deblur image to handle motion blur from moving objects

        Args:
            gray: Input grayscale image

        Returns:
            Deblurred image
        """
        # Light sharpening kernel for motion blur
        kernel = (
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.3
        )  # Light sharpening to avoid artifacts
        deblurred = cv2.filter2D(gray, -1, kernel)

        # Bilateral filter to reduce noise while preserving edges
        deblurred = cv2.bilateralFilter(deblurred, 5, 50, 50)

        return deblurred

    def _enhance_image_for_aruco(self, gray: np.ndarray) -> np.ndarray:
        """
        Enhance image for better ArUco detection

        Techniques:
        - Motion blur deblurring (if enabled)
        - Unsharp masking (sharpening)
        - Bilateral filter (denoising while preserving edges)
        - Contrast enhancement (CLAHE)

        Args:
            gray: Input grayscale image

        Returns:
            Enhanced grayscale image
        """
        enhanced = gray.copy()

        # 0. Motion blur deblurring (if enabled)
        if self.motion_blur_deblur:
            enhanced = self._deblur_for_motion(enhanced)

        # 1. Denoise with bilateral filter (preserves edges better than Gaussian)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

        # 2. Apply CLAHE for contrast enhancement (lightweight)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)

        # 3. Unsharp masking for sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        # 4. Normalize to ensure good contrast
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        return enhanced

    def _aruco_to_yolo_format(self, aruco_detections: List[Dict]) -> Any:
        """
        Convert ArUco detections to YOLO-like format for merging with YOLO results

        Args:
            aruco_detections: List of ArUco detection dictionaries

        Returns:
            YOLO-like result object (MockResult with MockBoxes)
        """
        if not aruco_detections:
            return None

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
                """Return number of detections"""
                return len(self.xyxy)

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        boxes = MockBoxes(aruco_detections)
        return MockResult(boxes)

    def _merge_detection_results(self, yolo_results: Any, aruco_result: Any) -> Any:
        """
        Merge YOLO and ArUco detection results

        Args:
            yolo_results: YOLO detection results
            aruco_result: ArUco detection results (MockResult)

        Returns:
            Merged results in YOLO format
        """
        if aruco_result is None or len(aruco_result.boxes.xyxy) == 0:
            return yolo_results

        # Extract YOLO detections
        yolo_boxes = []
        yolo_confs = []
        yolo_clses = []

        if yolo_results and hasattr(yolo_results, "__iter__") and len(yolo_results) > 0:
            yolo_result = yolo_results[0]
            if hasattr(yolo_result, "boxes") and len(yolo_result.boxes.xyxy) > 0:
                boxes = yolo_result.boxes
                for i in range(len(boxes.xyxy)):
                    try:
                        if isinstance(boxes.xyxy, np.ndarray):
                            yolo_boxes.append(boxes.xyxy[i].copy())
                            yolo_confs.append(float(boxes.conf[i]))
                            yolo_clses.append(int(boxes.cls[i]))
                        else:
                            yolo_boxes.append(boxes.xyxy[i].cpu().numpy().copy())
                            yolo_confs.append(float(boxes.conf[i].cpu().numpy()))
                            yolo_clses.append(int(boxes.cls[i].cpu().numpy()))
                    except Exception as e:
                        logger.debug(f"Error extracting YOLO detection {i}: {e}")
                        continue

        # Add ArUco detections
        aruco_boxes = aruco_result.boxes.xyxy
        aruco_confs = aruco_result.boxes.conf
        aruco_clses = aruco_result.boxes.cls

        # Combine all detections
        all_boxes = yolo_boxes + [aruco_boxes[i] for i in range(len(aruco_boxes))]
        all_confs = yolo_confs + [aruco_confs[i] for i in range(len(aruco_confs))]
        all_clses = yolo_clses + [aruco_clses[i] for i in range(len(aruco_clses))]

        # If no detections at all, return empty result in YOLO format
        if not all_boxes:

            class MockBoxes:
                def __init__(self):
                    self.xyxy = np.array([])
                    self.conf = np.array([])
                    self.cls = np.array([])

                def __len__(self):
                    """Return number of detections"""
                    return 0

            class MockResult:
                def __init__(self, boxes):
                    self.boxes = boxes

            return [MockResult(MockBoxes())]

        # Create merged result
        class MockBoxes:
            def __init__(self, boxes, confs, clses):
                self.xyxy = np.array(boxes) if boxes else np.array([])
                self.conf = np.array(confs) if confs else np.array([])
                self.cls = np.array(clses) if clses else np.array([])

            def __len__(self):
                """Return number of detections"""
                return len(self.xyxy)

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        merged_boxes = MockBoxes(all_boxes, all_confs, all_clses)
        merged_result = MockResult(merged_boxes)

        return [merged_result]

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _onnx_to_yolo_format(self, detections_list: List[Dict]) -> Any:
        """Convert ONNX detector output to YOLO-like format for visualization"""

        class MockBoxes:
            def __init__(self, detections):
                self.xyxy = []
                self.conf = []
                self.cls = []
                for det in detections:
                    self.xyxy.append(np.array(det["bbox"], dtype=np.float32))
                    self.conf.append(det["confidence"])
                    self.cls.append(det["class_id"])
                self.xyxy = np.array(self.xyxy)
                self.conf = np.array(self.conf)
                self.cls = np.array(self.cls)

            def __len__(self):
                """Return number of detections"""
                return len(self.xyxy)

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        boxes = MockBoxes(detections_list)
        return [MockResult(boxes)]

    def _process_results(self, results, header: Header):
        """
        Convert YOLO/ONNX results to ROS2 Detection2D messages

        Args:
            results: YOLO prediction results or ONNX detections
            header: ROS message header

        Returns:
            List of Detection2D messages or Detection2DArray
        """
        if not VISION_MSGS_AVAILABLE:
            # Fallback: return empty list if vision_msgs not available
            # We'll publish detections via annotated image only
            logger.warning("vision_msgs not available, skipping detection publishing")
            return []

        detections = []

        # Handle YOLO results format
        if hasattr(results, "__iter__") and len(results) > 0:
            result = results[0]  # First result (batch size 1)

            if hasattr(result, "boxes"):
                boxes = result.boxes

                for i in range(len(boxes.xyxy)):
                    # Get box coordinates
                    if isinstance(boxes.xyxy, np.ndarray):
                        box = boxes.xyxy[i]
                        conf = float(boxes.conf[i])
                        cls = int(boxes.cls[i])
                    else:
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())

                    # Create Detection2D message
                    detection = Detection2D()
                    detection.header = header

                    # Bounding box (center_x, center_y, width, height)
                    x1, y1, x2, y2 = box
                    detection.bbox.center.position.x = float((x1 + x2) / 2.0)
                    detection.bbox.center.position.y = float((y1 + y2) / 2.0)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)

                    # Hypothesis (class and confidence)
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(cls)
                    hypothesis.hypothesis.score = conf
                    detection.results = [hypothesis]

                    detections.append(detection)

        return detections


def main(args=None):
    """Main entry point for ROS2 node"""
    import argparse
    import os
    import yaml

    parser = argparse.ArgumentParser(description="ROS2 Camera Detector Node")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model file (.pt, .onnx, or int8 .onnx)",
    )
    parser.add_argument(
        "--config", type=str, help="Path to config YAML file"
    )

    # Parse ROS2 args and custom args
    if ROS_AVAILABLE:
        ros_args = rclpy.utilities.remove_ros_args(args)
    else:
        ros_args = args
    parsed_args, unknown = parser.parse_known_args(ros_args)

    # Auto-load config if not provided
    if not parsed_args.config:
        # Try to find the AI camera config automatically
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
        config_path = os.path.join(project_root, "ros_nodes", "configs", "pi_zero_ai", "camera_config_ai.yaml")

        if os.path.exists(config_path):
            parsed_args.config = config_path
            print(f"Auto-loading config from: {config_path}")
        else:
            print(f"Warning: AI config not found at {config_path}")

    # Load config and set defaults
    if parsed_args.config and os.path.exists(parsed_args.config):
        with open(parsed_args.config, 'r') as f:
            config = yaml.safe_load(f)

        if not parsed_args.model and 'inference' in config:
            model_path = config['inference'].get('model')
            if model_path:
                # Convert relative path to absolute
                if not os.path.isabs(model_path):
                    config_dir = os.path.dirname(parsed_args.config)
                    project_root = os.path.abspath(os.path.join(config_dir, "../../../.."))
                    parsed_args.model = os.path.join(project_root, model_path)
                else:
                    parsed_args.model = model_path

                print(f"Auto-loading model: {parsed_args.model}")

    # Validate required arguments
    if not parsed_args.model:
        parser.error("--model is required (or valid config file with model path)")

    args = parsed_args

    # Parse ROS2 args and custom args
    if ROS_AVAILABLE:
        ros_args = rclpy.utilities.remove_ros_args(args)
    else:
        ros_args = args
    args, unknown = parser.parse_known_args(ros_args)

    if not ROS_AVAILABLE:
        print("ERROR: ROS2 dependencies not installed")
        print(
            "Install with: pip install rclpy sensor-msgs cv-bridge std-msgs vision-msgs"
        )
        sys.exit(1)

    # #region agent log
    _debug_log(
        "camera_detector_node.py:395",
        "main: initializing rclpy",
        {"args": str(unknown)[:100]},
        "A",
    )
    # #endregion
    rclpy.init(args=unknown)
    # #region agent log
    _debug_log("camera_detector_node.py:397", "main: rclpy initialized", {}, "A")
    # #endregion

    try:
        # #region agent log
        _debug_log(
            "camera_detector_node.py:400",
            "main: creating CameraDetectorNode",
            {"model": args.model, "config": args.config},
            "A",
        )
        # #endregion
        node = CameraDetectorNode(args.model, args.config)
        # #region agent log
        _debug_log(
            "camera_detector_node.py:402", "main: node created, starting spin", {}, "A"
        )
        # #endregion
        rclpy.spin(node)
    except KeyboardInterrupt:
        # #region agent log
        _debug_log("camera_detector_node.py:405", "main: KeyboardInterrupt", {}, "A")
        # #endregion
        pass
    except Exception as e:
        logger.error(f"Node error: {e}")
        # #region agent log
        _debug_log(
            "camera_detector_node.py:409",
            "main: exception caught",
            {"error": str(e), "error_type": type(e).__name__},
            "A",
        )
        import traceback

        _debug_log(
            "camera_detector_node.py:411",
            "main: exception traceback",
            {"traceback": traceback.format_exc()},
            "A",
        )
        # #endregion
    finally:
        if "node" in locals():
            # #region agent log
            _debug_log("camera_detector_node.py:415", "main: destroying node", {}, "A")
            # #endregion
            node.destroy_node()
        # #region agent log
        _debug_log("camera_detector_node.py:418", "main: shutting down rclpy", {}, "A")
        # #endregion
        rclpy.shutdown()


if __name__ == "__main__":
    main()
