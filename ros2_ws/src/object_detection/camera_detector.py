#!/usr/bin/env python3
"""
ROS2 Node for camera-based YOLO object detection
Handles camera capture and publishes detections for robotics applications
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header
import cv_bridge
import cv2
import numpy as np
import torch
import time
from pathlib import Path
from ultralytics import YOLO
import threading
import queue
from temporal_filter import TemporalConfidenceFilter


class RaspberryPiCameraDetector(Node):
    """
    ROS2 node for Raspberry Pi camera with YOLO detection
    Optimized for real-time robotics hammer/bottle/ArUco detection
    """

    def __init__(self, model_path: str = None):
        super().__init__('rpi_camera_detector')

        # Parameters
        self.declare_parameter('model_path', model_path or 'models/yolov8s.pt')
        self.declare_parameter('camera_device', 0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('publish_camera_info', True)

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
        self.camera_device = self.get_parameter('camera_device').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.publish_annotated = self.get_parameter('publish_annotated').value
        self.publish_camera_info = self.get_parameter('publish_camera_info').value

        # Get temporal filtering parameters
        self.enable_temporal_filter = self.get_parameter('enable_temporal_filter').value
        self.temporal_alpha = self.get_parameter('temporal_alpha').value
        self.temporal_min_frames = self.get_parameter('temporal_min_frames').value
        self.temporal_boost_factor = self.get_parameter('temporal_boost_factor').value
        self.desert_mode = self.get_parameter('desert_mode').value
        self.dust_tolerance = self.get_parameter('dust_tolerance').value
        self.lighting_adaptation = self.get_parameter('lighting_adaptation').value

        # Class names
        self.class_names = ['BrickHammer', 'ArUcoTag', 'Bottle', 'BrickHammer_duplicate', 'OrangeHammer']

        # Class-specific thresholds
        self.class_thresholds = {
            'BrickHammer': 0.6,
            'ArUcoTag': 0.7,
            'Bottle': 0.5,
            'OrangeHammer': 0.6,
            'BrickHammer_duplicate': 0.6
        }

        # Camera and model initialization
        self.cap = None
        self.model = None
        self.bridge = cv_bridge.CvBridge()

        # Threading for camera capture
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        self.running = False

        # Performance tracking
        self.inference_times = []
        self.frame_count = 0

        # Initialize camera and model
        self._initialize_camera()
        self._initialize_model()

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

        # QoS Profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', qos_profile)
        self.detection_pub = self.create_publisher(Detection2DArray, '/yolo_detector/detections', qos_profile)

        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(Image, '/yolo_detector/annotated_image', qos_profile)

        if self.publish_camera_info:
            self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', qos_profile)

        # Timer for processing
        self.timer = self.create_timer(1.0 / self.fps, self.process_frame)

        self.get_logger().info("Raspberry Pi Camera Detector initialized")
        self.get_logger().info(f"Camera: {self.width}x{self.height} @ {self.fps} FPS")
        self.get_logger().info(f"Model: {self.model_path}")

        if self.enable_temporal_filter:
            self.get_logger().info("Temporal confidence filtering: ENABLED")
        else:
            self.get_logger().info("Temporal confidence filtering: DISABLED")

    def _initialize_camera(self):
        """Initialize Raspberry Pi camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_device)

            if not self.cap.isOpened():
                self.get_logger().error(f"Failed to open camera device {self.camera_device}")
                return

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            self.get_logger().info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")

            # Start capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()

        except Exception as e:
            self.get_logger().error(f"Camera initialization failed: {e}")
            raise

    def _initialize_model(self):
        """Initialize YOLO model with RPi optimizations"""
        try:
            self.model = YOLO(self.model_path)
            self.model.eval()

            # RPi optimizations
            torch.set_grad_enabled(False)
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(4)

            # Warm up
            dummy_input = torch.randn(1, 3, 416, 416)
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(dummy_input)

            self.get_logger().info("Model loaded and optimized for RPi")

        except Exception as e:
            self.get_logger().error(f"Model initialization failed: {e}")
            raise

    def _capture_loop(self):
        """Camera capture loop running in separate thread"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Add to queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Remove old frame if queue is full
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                else:
                    self.get_logger().warning("Failed to capture frame")

            except Exception as e:
                self.get_logger().error(f"Camera capture error: {e}")

            # Small delay to prevent busy waiting
            time.sleep(0.01)

    def process_frame(self):
        """Process frame from camera and publish results"""
        try:
            # Get latest frame
            if self.frame_queue.empty():
                return

            frame = self.frame_queue.get()

            # Create timestamp
            now = self.get_clock().now()
            timestamp = now.to_msg()

            # Publish raw image
            self._publish_image(frame, timestamp)

            # Run inference
            start_time = time.time()
            detections, inference_time = self._run_inference(frame)
            self.inference_times.append(inference_time)

            # Apply temporal confidence filtering
            if self.temporal_filter is not None:
                detections = self.temporal_filter.update_confidence(detections, timestamp)

            # Publish detections
            self._publish_detections(detections, timestamp)

            # Publish annotated image
            if self.publish_annotated:
                annotated = self._draw_detections(frame, detections)
                self._publish_annotated_image(annotated, timestamp)

            # Publish camera info
            if self.publish_camera_info:
                self._publish_camera_info(timestamp)

            # Performance logging
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                avg_time = np.mean(self.inference_times[-30:])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(".2f")

        except Exception as e:
            self.get_logger().error(f"Frame processing error: {e}")

    def _run_inference(self, image: np.ndarray) -> tuple:
        """Run optimized inference"""
        start_time = time.time()

        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=0.4,
            max_det=20,
            imgsz=416,
            verbose=False,
            device='cpu'
        )

        inference_time = time.time() - start_time

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    class_name = self.class_names[int(cls)]
                    threshold = self.class_thresholds.get(class_name, self.conf_threshold)

                    if conf >= threshold:
                        detection = {
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class_id': int(cls),
                            'class_name': class_name
                        }
                        detections.append(detection)

        return detections, inference_time

    def _publish_image(self, image: np.ndarray, timestamp):
        """Publish raw camera image"""
        try:
            ros_image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            ros_image.header.stamp = timestamp
            ros_image.header.frame_id = 'camera_frame'
            self.image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Image publish error: {e}")

    def _publish_detections(self, detections: list, timestamp):
        """Publish detections as Detection2DArray"""
        from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose

        detection_array = Detection2DArray()
        detection_array.header.stamp = timestamp
        detection_array.header.frame_id = 'camera_frame'

        for det in detections:
            detection = Detection2D()
            detection.header = detection_array.header

            bbox = BoundingBox2D()
            bbox.center.x = (det['bbox'][0] + det['bbox'][2]) / 2.0
            bbox.center.y = (det['bbox'][1] + det['bbox'][3]) / 2.0
            bbox.size_x = det['bbox'][2] - det['bbox'][0]
            bbox.size_y = det['bbox'][3] - det['bbox'][1]
            detection.bbox = bbox

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(det['class_id'])
            hypothesis.hypothesis.score = det['confidence']
            detection.results.append(hypothesis)

            detection_array.detections.append(detection)

        self.detection_pub.publish(detection_array)

    def _draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw detections on image"""
        display_img = image.copy()

        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']

            cv2.rectangle(display_img, (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

            label = ".2f"
            cv2.putText(display_img, label, (int(bbox[0]), int(bbox[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return display_img

    def _publish_annotated_image(self, image: np.ndarray, timestamp):
        """Publish annotated image"""
        try:
            ros_image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            ros_image.header.stamp = timestamp
            ros_image.header.frame_id = 'camera_frame'
            self.annotated_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Annotated image publish error: {e}")

    def _publish_camera_info(self, timestamp):
        """Publish camera info"""
        try:
            camera_info = CameraInfo()
            camera_info.header.stamp = timestamp
            camera_info.header.frame_id = 'camera_frame'
            camera_info.width = self.width
            camera_info.height = self.height

            # Basic camera matrix (can be calibrated for better accuracy)
            camera_info.k = [self.width, 0, self.width/2,
                           0, self.height, self.height/2,
                           0, 0, 1]

            self.camera_info_pub.publish(camera_info)
        except Exception as e:
            self.get_logger().error(f"Camera info publish error: {e}")

    def destroy_node(self):
        """Clean shutdown"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    model_path = None
    if args and len(args) > 1:
        model_path = args[1]

    node = RaspberryPiCameraDetector(model_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
