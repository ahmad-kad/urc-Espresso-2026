#!/usr/bin/env python3
"""
ROS2 Node for YOLO object detection with bounding box publishing
General-purpose real-time object detection
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
import cv_bridge
import cv2
import numpy as np
import torch
import time
from pathlib import Path
from ultralytics import YOLO
import logging
from temporal_filter import TemporalConfidenceFilter


class YOLODetectorNode(Node):
    """
    ROS2 node for real-time YOLO object detection
    Publishes detection results and annotated images
    """

    def __init__(self, model_path: str = None):
        super().__init__('yolo_detector')

        # Parameters
        self.declare_parameter('model_path', model_path or 'models/yolov8s.pt')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.4)
        self.declare_parameter('max_detections', 20)
        self.declare_parameter('input_image_topic', '/camera/image_raw')
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('annotated_image_topic', '/yolo_detector/annotated_image')

        # Temporal filtering parameters (optimized for desert environments)
        self.declare_parameter('enable_temporal_filter', True)
        self.declare_parameter('temporal_alpha', 0.4)  # EMA smoothing factor
        self.declare_parameter('temporal_min_frames', 5)  # Frames needed for boost
        self.declare_parameter('temporal_boost_factor', 0.25)  # Max confidence boost
        self.declare_parameter('desert_mode', True)  # Enable desert optimizations
        self.declare_parameter('dust_tolerance', 0.3)  # Dust interference tolerance
        self.declare_parameter('lighting_adaptation', True)  # Adapt to extreme lighting

        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.max_detections = self.get_parameter('max_detections').value
        self.input_topic = self.get_parameter('input_image_topic').value
        self.publish_annotated = self.get_parameter('publish_annotated').value
        self.annotated_topic = self.get_parameter('annotated_image_topic').value

        # Get temporal filtering parameters
        self.enable_temporal_filter = self.get_parameter('enable_temporal_filter').value
        self.temporal_alpha = self.get_parameter('temporal_alpha').value
        self.temporal_min_frames = self.get_parameter('temporal_min_frames').value
        self.temporal_boost_factor = self.get_parameter('temporal_boost_factor').value
        self.desert_mode = self.get_parameter('desert_mode').value
        self.dust_tolerance = self.get_parameter('dust_tolerance').value
        self.lighting_adaptation = self.get_parameter('lighting_adaptation').value

        # Class names for hammer detection
        self.class_names = ['BrickHammer', 'ArUcoTag', 'Bottle', 'BrickHammer_duplicate', 'OrangeHammer']

        # Class-specific confidence thresholds (higher for safety-critical objects)
        self.class_thresholds = {
            'BrickHammer': 0.6,
            'ArUcoTag': 0.7,      # Higher threshold for ArUco tags
            'Bottle': 0.5,
            'OrangeHammer': 0.6,
            'BrickHammer_duplicate': 0.6
        }

        # Initialize model
        self._load_model()

        # Initialize temporal confidence filter
        if self.enable_temporal_filter:
            self.temporal_filter = TemporalConfidenceFilter(
                alpha=self.temporal_alpha,
                min_frames=self.temporal_min_frames,
                confidence_boost_factor=self.temporal_boost_factor,
                desert_mode=self.desert_mode,
                dust_tolerance=self.dust_tolerance,
                lighting_adaptation=self.lighting_adaptation
            )
            self.get_logger().info("Temporal confidence filter enabled with "
                                 f"alpha={self.temporal_alpha}, min_frames={self.temporal_min_frames}, "
                                 f"desert_mode={self.desert_mode}")
        else:
            self.temporal_filter = None

        # CV Bridge
        self.bridge = cv_bridge.CvBridge()

        # QoS Profile for real-time performance
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/yolo_detector/detections',
            qos_profile
        )

        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image,
                self.annotated_topic,
                qos_profile
            )

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            qos_profile
        )

        # Performance tracking
        self.inference_times = []
        self.frame_count = 0

        self.get_logger().info(f"YOLO Detector initialized with model: {self.model_path}")
        self.get_logger().info(f"Subscribed to: {self.input_topic}")
        self.get_logger().info(f"Publishing detections to: /yolo_detector/detections")

        if self.enable_temporal_filter:
            self.get_logger().info("Temporal confidence filtering: ENABLED")
        else:
            self.get_logger().info("Temporal confidence filtering: DISABLED")

        if self.publish_annotated:
            self.get_logger().info(f"Publishing annotated images to: {self.annotated_topic}")

    def _load_model(self):
        """Load YOLO model with optimizations"""
        try:
            self.model = YOLO(self.model_path)

            # Model optimizations for real-time inference
            self.model.eval()

            # Disable gradient computation
            torch.set_grad_enabled(False)

            # CPU optimizations
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(4)  # Raspberry Pi 4 cores

            # Warm up the model
            dummy_input = torch.randn(1, 3, 416, 416)
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(dummy_input)

            self.get_logger().info("Model loaded and optimized successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

    def image_callback(self, msg: Image):
        """Process incoming images and publish detections"""

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run inference
            start_time = time.time()
            detections, inference_time = self._run_inference(cv_image)
            self.inference_times.append(inference_time)

            # Apply temporal confidence filtering
            if self.temporal_filter is not None:
                detections = self.temporal_filter.update_confidence(detections, msg.header.stamp.sec)

            # Publish detections
            self._publish_detections(detections, msg.header)

            # Publish annotated image if enabled
            if self.publish_annotated:
                annotated_image = self._draw_detections(cv_image, detections)
                self._publish_annotated_image(annotated_image, msg.header)

            # Performance logging
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # Log every 30 frames
                avg_time = np.mean(self.inference_times[-30:])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(".2f")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def _run_inference(self, image: np.ndarray) -> tuple:
        """Run YOLO inference with optimized settings"""

        start_time = time.time()

        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            imgsz=416,
            verbose=False,
            device='cpu'
        )

        inference_time = time.time() - start_time

        # Process results
        detections = []
        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    class_name = self.class_names[int(cls)]

                    # Apply class-specific threshold
                    threshold = self.class_thresholds.get(class_name, self.conf_threshold)

                    if conf >= threshold:
                        detection = {
                            'bbox': box.tolist(),  # [x1, y1, x2, y2]
                            'confidence': float(conf),
                            'class_id': int(cls),
                            'class_name': class_name
                        }
                        detections.append(detection)

        return detections, inference_time

    def _publish_detections(self, detections: list, header: Header):
        """Publish detections as Detection2DArray message"""

        detection_array = Detection2DArray()
        detection_array.header = header
        detection_array.header.frame_id = 'camera_frame'

        for det in detections:
            detection = Detection2D()
            detection.header = header

            # Bounding box
            bbox = BoundingBox2D()
            bbox.center.x = (det['bbox'][0] + det['bbox'][2]) / 2.0
            bbox.center.y = (det['bbox'][1] + det['bbox'][3]) / 2.0
            bbox.size_x = det['bbox'][2] - det['bbox'][0]
            bbox.size_y = det['bbox'][3] - det['bbox'][1]
            detection.bbox = bbox

            # Object hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(det['class_id'])
            hypothesis.hypothesis.score = det['confidence']
            detection.results.append(hypothesis)

            detection_array.detections.append(detection)

        self.detection_pub.publish(detection_array)

    def _draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on image"""

        display_img = image.copy()

        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']

            # Draw rectangle
            cv2.rectangle(
                display_img,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2
            )

            # Draw label
            label = ".2f"
            cv2.putText(
                display_img,
                label,
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return display_img

    def _publish_annotated_image(self, image: np.ndarray, header: Header):
        """Publish annotated image"""

        try:
            ros_image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            ros_image.header = header
            self.annotated_pub.publish(ros_image)

        except Exception as e:
            self.get_logger().error(f"Error publishing annotated image: {e}")


def main(args=None):
    rclpy.init(args=args)

    # Check for model path argument
    model_path = None
    if len(args) > 1:
        model_path = args[1]

    node = YOLODetectorNode(model_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
