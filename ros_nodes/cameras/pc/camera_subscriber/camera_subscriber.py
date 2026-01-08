#!/usr/bin/env python3
"""
Multi-Camera Subscriber Node for PC
Subscribes to multiple camera streams and displays them in a grid layout
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image
    from vision_msgs.msg import Detection2DArray

    ROS_AVAILABLE = True
except ImportError as e:
    ROS_AVAILABLE = False
    print(f"ROS2 not available: {e}")


class CameraSubscriber(Node):
    """Multi-camera subscriber node for PC"""

    def __init__(self, config_file: str):
        """
        Initialize multi-camera subscriber

        Args:
            config_file: Path to YAML configuration file
        """
        if not ROS_AVAILABLE:
            raise ImportError("ROS2 dependencies not installed")

        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        ros_config = self.config.get('ros2', {})
        node_name = ros_config.get('node_name', 'camera_subscriber_master')

        super().__init__(node_name)

        self.bridge = CvBridge()
        self.camera_frames: Dict[str, np.ndarray] = {}
        self.camera_timestamps: Dict[str, float] = {}
        self.last_display_time = time.time()

        # Setup QoS for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Subscribe to camera streams
        subscriptions_config = self.config.get('subscriptions', {})
        camera_configs = subscriptions_config.get('cameras', [])

        self.camera_subscriptions = []
        for camera_config in camera_configs:
            camera_name = camera_config['name']
            topic = camera_config['topic']
            topic_type = camera_config['type']

            if topic_type == 'sensor_msgs/CompressedImage':
                subscription = self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda msg, name=camera_name: self.compressed_image_callback(msg, name),
                    qos_profile
                )
            else:  # sensor_msgs/Image
                subscription = self.create_subscription(
                    Image,
                    topic,
                    lambda msg, name=camera_name: self.image_callback(msg, name),
                    qos_profile
                )

            self.camera_subscriptions.append(subscription)
            self.get_logger().info(f"Subscribed to {camera_name} on {topic}")

        # Subscribe to inference results
        inference_configs = subscriptions_config.get('inference', [])
        self.inference_subscriptions = []
        self.latest_detections: Optional[Detection2DArray] = None

        for inference_config in inference_configs:
            topic = inference_config['topic']
            topic_type = inference_config['type']

            if topic_type == 'vision_msgs/Detection2DArray':
                subscription = self.create_subscription(
                    Detection2DArray,
                    topic,
                    self.detection_callback,
                    qos_profile
                )
                self.inference_subscriptions.append(subscription)
                self.get_logger().info(f"Subscribed to detections on {topic}")

        # Display settings
        display_config = self.config.get('display', {})
        self.display_enabled = display_config.get('enabled', True)
        self.window_name = display_config.get('window_name', 'Multi-Camera View')
        self.layout = display_config.get('layout', 'grid')
        self.show_fps = display_config.get('show_fps', True)
        self.show_inference_overlay = display_config.get('show_inference_overlay', True)

        # Monitoring settings
        monitoring_config = self.config.get('monitoring', {})
        self.monitoring_enabled = monitoring_config.get('enabled', True)
        self.log_fps = monitoring_config.get('log_fps', True)
        self.alert_on_disconnect = monitoring_config.get('alert_on_disconnect', True)
        self.reconnect_timeout = monitoring_config.get('reconnect_timeout', 5)

        # FPS tracking
        self.fps_counters: Dict[str, List[float]] = {}
        self.fps_window = 30  # frames to average over

        # Create display timer
        if self.display_enabled:
            self.create_timer(0.1, self.display_callback)  # 10 FPS display update

        self.get_logger().info(f"Multi-camera subscriber initialized with {len(camera_configs)} cameras")

    def compressed_image_callback(self, msg: CompressedImage, camera_name: str):
        """Callback for compressed image messages"""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                self.camera_frames[camera_name] = frame
                self.camera_timestamps[camera_name] = time.time()
                self.update_fps_counter(camera_name)

        except Exception as e:
            self.get_logger().error(f"Error processing compressed image from {camera_name}: {e}")

    def image_callback(self, msg: Image, camera_name: str):
        """Callback for raw image messages"""
        try:
            # Convert ROS image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.camera_frames[camera_name] = frame
            self.camera_timestamps[camera_name] = time.time()
            self.update_fps_counter(camera_name)

        except Exception as e:
            self.get_logger().error(f"Error processing image from {camera_name}: {e}")

    def detection_callback(self, msg: Detection2DArray):
        """Callback for detection results"""
        self.latest_detections = msg

    def update_fps_counter(self, camera_name: str):
        """Update FPS counter for a camera"""
        current_time = time.time()

        if camera_name not in self.fps_counters:
            self.fps_counters[camera_name] = []

        self.fps_counters[camera_name].append(current_time)

        # Keep only recent timestamps
        cutoff_time = current_time - 1.0  # Last second
        self.fps_counters[camera_name] = [
            t for t in self.fps_counters[camera_name] if t > cutoff_time
        ]

    def get_fps(self, camera_name: str) -> float:
        """Get current FPS for a camera"""
        if camera_name not in self.fps_counters:
            return 0.0

        timestamps = self.fps_counters[camera_name]
        if len(timestamps) < 2:
            return 0.0

        # Calculate FPS from timestamp differences
        time_span = timestamps[-1] - timestamps[0]
        if time_span > 0:
            return len(timestamps) / time_span
        return 0.0

    def check_camera_connectivity(self) -> Dict[str, bool]:
        """Check which cameras are still connected"""
        current_time = time.time()
        connected = {}

        for camera_name in self.camera_frames.keys():
            last_time = self.camera_timestamps.get(camera_name, 0)
            connected[camera_name] = (current_time - last_time) < self.reconnect_timeout

        return connected

    def create_display_grid(self) -> Optional[np.ndarray]:
        """Create a grid display of all camera feeds"""
        if not self.camera_frames:
            return None

        camera_names = list(self.camera_frames.keys())
        num_cameras = len(camera_names)

        if num_cameras == 0:
            return None

        # Determine grid layout
        if self.layout == 'grid':
            if num_cameras == 1:
                cols, rows = 1, 1
            elif num_cameras == 2:
                cols, rows = 2, 1
            elif num_cameras <= 4:
                cols, rows = 2, 2
            elif num_cameras <= 6:
                cols, rows = 3, 2
            elif num_cameras <= 9:
                cols, rows = 3, 3
            else:
                cols, rows = 4, (num_cameras + 3) // 4
        elif self.layout == 'horizontal':
            cols, rows = num_cameras, 1
        else:  # vertical
            cols, rows = 1, num_cameras

        # Get frame dimensions (assume all frames are similar size)
        sample_frame = list(self.camera_frames.values())[0]
        frame_h, frame_w = sample_frame.shape[:2]

        # Calculate resized dimensions to fit grid
        grid_w = frame_w * cols
        grid_h = frame_h * rows

        # Create grid canvas
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        # Place each camera feed in the grid
        for i, camera_name in enumerate(camera_names):
            if camera_name not in self.camera_frames:
                continue

            frame = self.camera_frames[camera_name].copy()

            # Resize frame to fit grid cell
            cell_w = frame_w
            cell_h = frame_h
            if frame.shape[1] != cell_w or frame.shape[0] != cell_h:
                frame = cv2.resize(frame, (cell_w, cell_h))

            # Calculate position in grid
            col = i % cols
            row = i // cols

            x = col * cell_w
            y = row * cell_h

            # Place frame in grid
            grid[y:y+cell_h, x:x+cell_w] = frame

            # Add camera label and FPS
            if self.show_fps:
                fps = self.get_fps(camera_name)
                label = f"{camera_name}: {fps:.1f} FPS"
                cv2.putText(grid, label, (x + 10, y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add inference overlay if enabled and this is the AI camera
            if self.show_inference_overlay and camera_name == 'ai_camera' and self.latest_detections:
                self.add_inference_overlay(grid, x, y, cell_w, cell_h)

        return grid

    def add_inference_overlay(self, grid: np.ndarray, x: int, y: int, w: int, h: int):
        """Add inference overlay to the AI camera feed"""
        if not self.latest_detections:
            return

        # Scale detection coordinates to grid cell
        scale_x = w / 640.0  # Assume original detection size
        scale_y = h / 480.0

        for detection in self.latest_detections.detections:
            # Get bounding box
            bbox = detection.bbox
            center_x = bbox.center.position.x * scale_x + x
            center_y = bbox.center.position.y * scale_y + y
            size_x = bbox.size_x * scale_x
            size_y = bbox.size_y * scale_y

            # Draw bounding box
            x1 = int(center_x - size_x / 2)
            y1 = int(center_y - size_y / 2)
            x2 = int(center_x + size_x / 2)
            y2 = int(center_y + size_y / 2)

            cv2.rectangle(grid, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add class label and confidence
            if detection.results:
                result = detection.results[0]  # Take highest confidence result
                class_name = result.hypothesis.class_id
                confidence = result.hypothesis.score

                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(grid, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def display_callback(self):
        """Timer callback to update display"""
        if not self.display_enabled:
            return

        # Create grid display
        grid = self.create_display_grid()

        if grid is not None:
            cv2.imshow(self.window_name, grid)
            cv2.waitKey(1)

        # Log FPS periodically
        current_time = time.time()
        if self.log_fps and (current_time - self.last_display_time) > 5.0:
            self.last_display_time = current_time

            connectivity = self.check_camera_connectivity()
            for camera_name in self.camera_frames.keys():
                fps = self.get_fps(camera_name)
                connected = connectivity.get(camera_name, False)
                status = "CONNECTED" if connected else "DISCONNECTED"

                self.get_logger().info(".1f")

                if self.alert_on_disconnect and not connected:
                    self.get_logger().warn(f"Camera {camera_name} appears disconnected!")

    def destroy_node(self):
        """Cleanup"""
        if self.display_enabled:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ROS2 Multi-Camera Subscriber")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pc/camera_subscriber_config.yaml",
        help="Path to configuration YAML file",
    )

    if ROS_AVAILABLE:
        ros_args = rclpy.utilities.remove_ros_args(args)
    else:
        ros_args = args
    args, unknown = parser.parse_known_args(ros_args)

    if not ROS_AVAILABLE:
        print("ERROR: ROS2 dependencies not installed")
        print("Install with: pip install rclpy sensor-msgs cv-bridge vision-msgs")
        sys.exit(1)

    rclpy.init(args=unknown)

    try:
        node = CameraSubscriber(args.config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if "node" in locals():
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass  # Ignore shutdown errors


if __name__ == "__main__":
    main()
