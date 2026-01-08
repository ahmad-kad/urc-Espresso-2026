#!/usr/bin/env python3
"""
ROS2 Camera Publisher Node for Pi Zero 2W
Supports v2.0 and v2.1 Raspberry Pi cameras with automatic calibration loading
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    ROS_AVAILABLE = True
except ImportError as e:
    ROS_AVAILABLE = False
    print(f"ROS2 not available: {e}")

# Try to import RealSense
try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


class CameraPublisher(Node):
    """Simple camera publisher node"""

    def __init__(
        self,
        source: str,
        topic: str = "/camera/image_raw",
        fps: float = 30.0,
        test_mode: bool = False,
        jpeg_quality: int = None,
        grayscale: bool = False,
        resolution: str = "full",
        start_time: float = 0.0,
        target_frame_size_kb: int = None,
        calibration_file: str = None,
        enable_undistortion: bool = False,
    ):
        """
        Initialize camera publisher

        Args:
            source: Camera device index (e.g., '0') or video file path
            topic: ROS topic to publish to
            fps: Target FPS for publishing
            test_mode: If True, generate test pattern if camera fails
            jpeg_quality: JPEG quality (1-100), None for raw/uncompressed
            grayscale: If True, convert to grayscale
            resolution: 'full' (1920x1080), 'half' (1280x720), 'quarter' (640x480)
            start_time: Start time in seconds for video files (default: 0.0)
            target_frame_size_kb: Target frame size in KB after compression (adaptive JPEG quality)
            calibration_file: Path to JSON calibration file for undistortion (optional)
            enable_undistortion: Enable lens distortion correction using calibration file
        """
        if not ROS_AVAILABLE:
            raise ImportError("ROS2 dependencies not installed")

        super().__init__("camera_publisher")

        self.bridge = CvBridge()
        self.timer_period = 1.0 / fps
        self.test_mode = test_mode
        self.use_test_pattern = False

        # Compression settings
        self.jpeg_quality = jpeg_quality
        self.target_frame_size_kb = target_frame_size_kb
        self.grayscale = grayscale
        self.resolution = resolution

        # Resolution mapping - Camera Module 2 maximum capabilities
        resolution_map = {"full": (1920, 1080), "half": (1280, 720), "quarter": (640, 480)}
        self.target_size = resolution_map.get(resolution.lower(), (1920, 1080))

        # Calibration settings for undistortion
        self.calibration_file = calibration_file
        self.enable_undistortion = enable_undistortion
        self.camera_matrix = None
        self.dist_coeffs = None
        self.undistort_maps = None

        # Load calibration if enabled
        if self.enable_undistortion and self.calibration_file:
            self._load_camera_calibration()

        # Create appropriate publisher
        if jpeg_quality is not None or target_frame_size_kb is not None:
            self.publisher = self.create_publisher(CompressedImage, topic, 10)
            self.use_compression = True
        else:
            self.publisher = self.create_publisher(Image, topic, 10)
            self.use_compression = False

        # Open video source
        self.cap = None
        self.realsense_pipeline = None

        if source.isdigit():
            device_idx = int(source)

            # Try RealSense first if available
            if REALSENSE_AVAILABLE:
                try:
                    self.get_logger().info("Attempting to use RealSense camera...")
                    pipeline = rs.pipeline()
                    config = rs.config()

                    # Try to enable color stream
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

                    # If device index specified, try to use it
                    if device_idx > 0:
                        # Get available devices
                        ctx = rs.context()
                        devices = ctx.query_devices()
                        if device_idx < len(devices):
                            config.enable_device(
                                devices[device_idx].get_info(
                                    rs.camera_info.serial_number
                                )
                            )

                    pipeline.start(config)
                    self.realsense_pipeline = pipeline
                    self.get_logger().info("RealSense camera initialized successfully")
                except Exception as e:
                    self.get_logger().warn(
                        f"RealSense failed: {e}, trying standard OpenCV..."
                    )
                    self.realsense_pipeline = None

            # Try standard OpenCV if RealSense didn't work
            if self.realsense_pipeline is None:
                camera_found = False  # Initialize before use
                # Try device path first (for RealSense and other USB cameras)
                import os

                device_path = f"/dev/video{device_idx}"
                if os.path.exists(device_path):
                    self.get_logger().info(f"Trying device path: {device_path}")
                    self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
                    if self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret and frame is not None and frame.size > 0:
                            self.get_logger().info(
                                f"✓ Camera opened via device path: {device_path}"
                            )
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_size[0])
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_size[1])
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            camera_found = True
                        else:
                            self.cap.release()
                            self.cap = None

                # Try different backends by index
                if not camera_found:
                    backends = [
                        (cv2.CAP_V4L2, "V4L2"),
                        (cv2.CAP_ANY, "ANY"),
                    ]

                    for backend, name in backends:
                        self.get_logger().info(
                            f"Trying camera {device_idx} with {name} backend..."
                        )
                        self.cap = cv2.VideoCapture(device_idx, backend)

                        if self.cap.isOpened():
                            # Set some properties for better compatibility
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_size[0])
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_size[1])
                            self.cap.set(cv2.CAP_PROP_FPS, 30)

                            ret, frame = self.cap.read()
                            if ret and frame is not None and frame.size > 0:
                                self.get_logger().info(
                                    f"✓ Camera {device_idx} working with {name} backend"
                                )
                                camera_found = True
                                break
                            else:
                                self.cap.release()
                                self.cap = None

                    # If still no camera, try other device paths and indices
                    if not camera_found:
                        self.get_logger().warn(
                            f"Failed to open camera {device_idx}, trying other devices..."
                        )
                        # Try device paths first
                        for idx in range(10):
                            if idx == device_idx:
                                continue
                            device_path = f"/dev/video{idx}"
                            if os.path.exists(device_path):
                                test_cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
                                if test_cap.isOpened():
                                    test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_size[0])
                                    test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_size[1])
                                    ret, frame = test_cap.read()
                                    if ret and frame is not None and frame.size > 0:
                                        self.get_logger().info(
                                            f"Found working camera at {device_path}"
                                        )
                                        self.cap = test_cap
                                        camera_found = True
                                        break
                                    test_cap.release()

                        # Then try indices
                        if not camera_found:
                            for idx in range(10):
                                if idx == device_idx:
                                    continue
                                for backend, name in backends:
                                    test_cap = cv2.VideoCapture(idx, backend)
                                    if test_cap.isOpened():
                                        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                        ret, frame = test_cap.read()
                                        if ret and frame is not None and frame.size > 0:
                                            self.get_logger().info(
                                                f"Found working camera at index {idx} with {name} backend"
                                            )
                                            self.cap = test_cap
                                            camera_found = True
                                            break
                                    if camera_found:
                                        break
                                    test_cap.release()
                                if camera_found:
                                    break
        else:
            # Try as file path
            self.cap = cv2.VideoCapture(source)
            if self.cap.isOpened():
                self._is_file = True

        # Check if we got a working source
        if self.realsense_pipeline is None and (
            self.cap is None or not self.cap.isOpened()
        ):
            if test_mode:
                self.get_logger().warn("No camera available, using test pattern")
                self.use_test_pattern = True
            else:
                raise RuntimeError(
                    f"Failed to open video source: {source}\n"
                    f"Try: --test-mode to use test pattern"
                )
        elif self.realsense_pipeline is None:
            # Verify we can read a frame (for OpenCV)
            ret, frame = self.cap.read()
            if not ret:
                if test_mode:
                    self.get_logger().warn(
                        "Camera opened but can't read frames, using test pattern"
                    )
                    self.cap.release()
                    self.use_test_pattern = True
                else:
                    self.cap.release()
                    raise RuntimeError(
                        f"Camera opened but cannot read frames. Try: --test-mode"
                    )
            else:
                # Set start position if it's a file
                if not str(source).isdigit():
                    if start_time > 0:
                        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                        if video_fps > 0:
                            start_frame = int(start_time * video_fps)
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                            self.get_logger().info(
                                f"Starting video at {start_time}s (frame {start_frame})"
                            )
                        else:
                            self.get_logger().warn(
                                "Could not determine video FPS, starting from beginning"
                            )
                    else:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Create timer for publishing
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Log configuration
        config_parts = []
        if self.resolution != "full":
            config_parts.append(f"res={self.resolution}")
        if self.grayscale:
            config_parts.append("grayscale")
        if self.target_frame_size_kb is not None:
            config_parts.append(f"target={self.target_frame_size_kb}KB")
        elif self.jpeg_quality is not None:
            config_parts.append(f"jpeg={self.jpeg_quality}")
        config_str = f" ({', '.join(config_parts)})" if config_parts else ""

        if self.use_test_pattern:
            self.get_logger().info(
                f"Camera publisher initialized: TEST PATTERN -> {topic}{config_str}"
            )
        else:
            self.get_logger().info(
                f"Camera publisher initialized: {source} -> {topic}{config_str}"
            )

    def _load_camera_calibration(self):
        """Load camera calibration parameters for undistortion"""
        try:
            import json
            import os

            # Find calibration file path relative to project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
            calib_path = os.path.join(project_root, "ros_nodes", "calibrations", self.calibration_file)

            if not os.path.exists(calib_path):
                self.get_logger().error(f"Calibration file not found: {calib_path}")
                self.enable_undistortion = False
                return

            # Load calibration data
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)

            self.camera_matrix = np.array(calib_data['camera_matrix'])
            # Handle nested distortion coefficients (some JSON files have [[values]])
            dist_coeffs = calib_data['distortion_coefficients']
            if isinstance(dist_coeffs[0], list):
                dist_coeffs = dist_coeffs[0]  # Flatten if double-nested
            self.dist_coeffs = np.array(dist_coeffs)

            # Create undistortion maps
            h, w = self.target_size[1], self.target_size[0]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )

            self.undistort_maps = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None, new_camera_matrix,
                (w, h), cv2.CV_32FC1
            )

            self.get_logger().info(f"Loaded calibration from {self.calibration_file}")
            self.get_logger().info(f"Image size: {w}x{h}, Reprojection error: {calib_data.get('reprojection_error', 'N/A')}")

        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")
            self.enable_undistortion = False

    def timer_callback(self):
        """Timer callback to publish frames"""
        if self.use_test_pattern:
            frame = self._generate_test_pattern()
        elif self.realsense_pipeline is not None:
            # Use RealSense
            frames = self.realsense_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                self.get_logger().warn("Failed to get RealSense frame")
                return
            frame = np.asanyarray(color_frame.get_data())
        else:
            # Use OpenCV
            ret, frame = self.cap.read()

            if not ret:
                # If reading from file and reached end, loop
                if hasattr(self, "_is_file") and self._is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

                if not ret:
                    self.get_logger().warn("Failed to read frame")
                    return

        try:
            # Apply preprocessing
            frame = self._preprocess_frame(frame)

            if self.use_compression:
                # JPEG compression
                if self.target_frame_size_kb is not None:
                    # Use target frame size compression (adaptive quality)
                    jpeg_bytes = self._compress_to_target_size(
                        frame, self.target_frame_size_kb
                    )

                    msg = CompressedImage()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "camera"
                    msg.format = "jpeg"
                    msg.data = jpeg_bytes
                else:
                    # Use fixed JPEG quality
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    _, jpeg_data = cv2.imencode(".jpg", frame, encode_params)

                    msg = CompressedImage()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "camera"
                    msg.format = "jpeg"
                    msg.data = jpeg_data.tobytes()
            else:
                # Raw image
                encoding = "mono8" if self.grayscale else "bgr8"
                msg = self.bridge.cv2_to_imgmsg(frame, encoding=encoding)
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "camera"

            self.publisher.publish(msg)
            self.get_logger().debug("Published frame")

        except Exception as e:
            self.get_logger().error(f"Error publishing frame: {e}")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing: resize and/or grayscale conversion"""
        # Resize if needed
        current_h, current_w = frame.shape[:2]
        target_w, target_h = self.target_size

        if (current_w, current_h) != (target_w, target_h):
            frame = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        # Apply undistortion if enabled
        if self.enable_undistortion and self.undistort_maps is not None:
            frame = cv2.remap(frame, self.undistort_maps[0], self.undistort_maps[1], cv2.INTER_LINEAR)

        # Convert to grayscale if requested
        if self.grayscale:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def _compress_to_target_size(self, frame: np.ndarray, target_size_kb: int) -> bytes:
        """
        Compress frame to target size (in KB) by adjusting JPEG quality

        Args:
            frame: Image frame to compress
            target_size_kb: Target size in KB

        Returns:
            Compressed JPEG data as bytes
        """
        target_size_bytes = target_size_kb * 1024

        # Binary search for optimal quality
        min_quality = 1
        max_quality = 100
        best_quality = 50
        best_data = None
        best_diff = float("inf")

        # Try up to 10 iterations
        for _ in range(10):
            quality = (min_quality + max_quality) // 2
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, jpeg_data = cv2.imencode(".jpg", frame, encode_params)
            current_size = len(jpeg_data)

            diff = abs(current_size - target_size_bytes)
            if diff < best_diff:
                best_diff = diff
                best_quality = quality
                best_data = jpeg_data

            # If we're within 5% of target, good enough
            if diff < target_size_bytes * 0.05:
                break

            # Adjust search range
            if current_size < target_size_bytes:
                min_quality = quality + 1
            else:
                max_quality = quality - 1

            # If range collapsed, use best found
            if min_quality > max_quality:
                break

        return best_data.tobytes()

    def _generate_test_pattern(self):
        """Generate a test pattern image"""
        # Create test pattern at target resolution
        target_w, target_h = self.target_size
        frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Get current time for animation
        t = time.time()

        # More efficient animated gradient using vectorized operations
        y_coords, x_coords = np.mgrid[0:target_h, 0:target_w]
        scale = max(target_w, target_h) / max(self.target_size[0], self.target_size[1])  # Scale relative to target resolution
        frame[:, :, 0] = (127 + 127 * np.sin(x_coords / (50 * scale) + t)).astype(
            np.uint8
        )
        frame[:, :, 1] = (127 + 127 * np.sin(y_coords / (50 * scale) + t)).astype(
            np.uint8
        )
        frame[:, :, 2] = (
            127 + 127 * np.sin((x_coords + y_coords) / (70 * scale) + t)
        ).astype(np.uint8)

        # Add some shapes for testing detection (scaled)
        rect_size = int(150 * scale)
        circle_radius = int(80 * scale)
        font_scale = 0.7 * scale
        thickness = max(1, int(2 * scale))

        cv2.rectangle(
            frame,
            (int(50 * scale), int(50 * scale)),
            (int(50 * scale) + rect_size, int(50 * scale) + rect_size),
            (0, 255, 0),
            thickness,
        )
        cv2.circle(
            frame,
            (int(400 * scale), int(200 * scale)),
            circle_radius,
            (255, 0, 0),
            thickness,
        )
        cv2.putText(
            frame,
            "TEST PATTERN",
            (int(100 * scale), int(300 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )
        cv2.putText(
            frame,
            f"Time: {t:.1f}",
            (int(100 * scale), int(330 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.7,
            (255, 255, 255),
            max(1, int(thickness * 0.7)),
        )

        return frame

    def destroy_node(self):
        """Cleanup"""
        if hasattr(self, "realsense_pipeline") and self.realsense_pipeline is not None:
            self.realsense_pipeline.stop()
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ROS2 Camera Publisher")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera device index (e.g., 0) or video file path",
    )
    parser.add_argument(
        "--topic", type=str, default="/camera/image_raw", help="ROS topic to publish to"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Target FPS for publishing"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use test pattern if camera unavailable",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="JPEG quality (1-100), None for raw/uncompressed",
    )
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale")
    parser.add_argument(
        "--resolution",
        type=str,
        default="full",
        choices=["full", "half", "quarter"],
        help="Resolution: full (1920x1080), half (1280x720), quarter (640x480)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time in seconds for video files (default: 0.0)",
    )
    parser.add_argument(
        "--target-frame-size-kb",
        type=int,
        default=None,
        help="Target frame size in KB after compression (adaptive JPEG quality, overrides --jpeg-quality)",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Path to JSON calibration file for undistortion (relative to ros_nodes/calibrations/)",
    )
    parser.add_argument(
        "--enable-undistortion",
        action="store_true",
        help="Enable lens distortion correction using calibration file",
    )
    parser.add_argument(
        "--pi-number",
        type=int,
        help="Pi number (1-3) for automatic configuration loading",
    )
    parser.add_argument(
        "--camera-type",
        type=str,
        choices=["v20", "v21"],
        help="Camera type for automatic configuration loading",
    )

    if ROS_AVAILABLE:
        ros_args = rclpy.utilities.remove_ros_args(args)
    else:
        ros_args = args
    args, unknown = parser.parse_known_args(ros_args)

    # Auto-configure based on pi-number and camera-type
    if args.pi_number and args.camera_type:
        import os
        import yaml

        # Determine config file path
        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "configs", f"pi_zero_{args.camera_type}"
        )
        config_file = os.path.join(config_dir, f"camera_config_pi{args.pi_number}.yaml")

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Override arguments with config values
            if 'ros2' in config:
                ros2_config = config['ros2']
                if not args.topic:
                    args.topic = ros2_config.get('topic', args.topic)
                if not args.fps:
                    args.fps = ros2_config.get('fps', args.fps)

            if 'compression' in config:
                comp_config = config['compression']
                if args.jpeg_quality is None:
                    args.jpeg_quality = comp_config.get('jpeg_quality')
                if not args.resolution or args.resolution == "full":
                    args.resolution = comp_config.get('resolution', args.resolution)
                if args.target_frame_size_kb is None:
                    args.target_frame_size_kb = comp_config.get('target_frame_size_kb')

            if 'camera' in config:
                cam_config = config['camera']
                if not args.calibration_file:
                    args.calibration_file = cam_config.get('calibration_file')
                if not args.enable_undistortion:
                    args.enable_undistortion = cam_config.get('enable_undistortion', False)

            print(f"Auto-configured from {config_file}")
            print(f"  Topic: {args.topic}")
            print(f"  Calibration: {args.calibration_file}")
            print(f"  Undistortion: {args.enable_undistortion}")
        else:
            print(f"Warning: Config file not found: {config_file}")
            print("Using default settings")

    if not ROS_AVAILABLE:
        print("ERROR: ROS2 dependencies not installed")
        print("Install with: pip install rclpy sensor-msgs cv-bridge")
        sys.exit(1)

    rclpy.init(args=unknown)

    try:
        node = CameraPublisher(
            args.source,
            args.topic,
            args.fps,
            args.test_mode,
            jpeg_quality=args.jpeg_quality,
            grayscale=args.grayscale,
            resolution=args.resolution,
            start_time=args.start_time,
            target_frame_size_kb=args.target_frame_size_kb,
            calibration_file=args.calibration_file,
            enable_undistortion=args.enable_undistortion,
        )
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
            pass  # Ignore shutdown errors (context may already be shut down)


if __name__ == "__main__":
    main()
