#!/usr/bin/env python3
"""
ROS2 Camera Streaming Benchmark
Compares performance between Zenoh and default middleware
Measures FPS, latency, bandwidth, and dropped frames
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

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
    print("Install with: pip install rclpy sensor-msgs cv-bridge")


class CameraBenchmarkSubscriber(Node):
    """Subscriber for benchmarking camera performance"""

    def __init__(
        self,
        topic: str = "/camera/image_raw",
        duration: float = 30.0,
        expected_fps: float = 30.0,
    ):
        super().__init__("camera_benchmark_subscriber")

        self.bridge = CvBridge()
        self.topic = topic
        self.duration = duration
        self.expected_fps = expected_fps
        self.start_time = None
        self.end_time = None
        self.message_type = None  # Will be set when first message arrives

        # Metrics
        self.frame_timestamps: List[float] = []
        self.frame_sizes: List[int] = []
        self.latencies: deque = deque(maxlen=100)  # Rolling window for latency
        self.last_receive_time = None
        self.frame_count = 0
        self.dropped_frames = 0

        # Create subscriber with high QoS for benchmarking
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Try to detect message type by checking topic info first
        self.image_subscription = None
        self.compressed_subscription = None
        detected_type = None

        # Try to get topic type info (if topic already exists)
        try:
            import subprocess

            result = subprocess.run(
                ["ros2", "topic", "info", topic],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if "Type: sensor_msgs/msg/CompressedImage" in result.stdout:
                detected_type = "CompressedImage"
            elif "Type: sensor_msgs/msg/Image" in result.stdout:
                detected_type = "Image"
        except Exception:
            # Topic might not exist yet, we'll try to detect from publisher args or try both
            pass

        # Create subscription - prefer CompressedImage if detected, otherwise try both
        subscription_created = False

        # If we detected CompressedImage, use it
        if detected_type == "CompressedImage":
            try:
                self.compressed_subscription = self.create_subscription(
                    CompressedImage, topic, self.compressed_image_callback, qos_profile
                )
                subscription_created = True
                self.get_logger().info(
                    f"Subscribed to CompressedImage on {topic} (detected from topic info)"
                )
            except Exception as e:
                self.get_logger().warn(f"CompressedImage subscription failed: {e}")

        # If we detected Image, use it
        elif detected_type == "Image":
            try:
                self.image_subscription = self.create_subscription(
                    Image, topic, self.image_callback, qos_profile
                )
                subscription_created = True
                self.get_logger().info(
                    f"Subscribed to Image on {topic} (detected from topic info)"
                )
            except Exception as e:
                self.get_logger().warn(f"Image subscription failed: {e}")

        # If no detection, try CompressedImage first (more common with compression benchmarks)
        # then fall back to Image
        if not subscription_created:
            # Try CompressedImage first (common for compression tests)
            try:
                self.compressed_subscription = self.create_subscription(
                    CompressedImage, topic, self.compressed_image_callback, qos_profile
                )
                subscription_created = True
                self.get_logger().info(
                    f"Subscribed to CompressedImage on {topic} (default, will auto-detect Image if needed)"
                )
            except Exception as e:
                error_str = str(e)
                # Check if error is about incompatible type
                if "incompatible type" in error_str.lower() or "Image" in error_str:
                    # Topic is Image, try that
                    self.get_logger().info(
                        "Topic uses Image, switching subscription..."
                    )
                    try:
                        self.image_subscription = self.create_subscription(
                            Image, topic, self.image_callback, qos_profile
                        )
                        subscription_created = True
                        self.get_logger().info(f"Subscribed to Image on {topic}")
                    except Exception as e2:
                        self.get_logger().error(f"Failed to subscribe to Image: {e2}")
                        raise
                else:
                    # Different error, try Image as fallback
                    self.get_logger().warn(
                        f"CompressedImage subscription failed: {e}, trying Image..."
                    )
                    try:
                        self.image_subscription = self.create_subscription(
                            Image, topic, self.image_callback, qos_profile
                        )
                        subscription_created = True
                        self.get_logger().info(f"Subscribed to Image on {topic}")
                    except Exception as e2:
                        self.get_logger().error(
                            f"Failed to subscribe to either type: {e2}"
                        )
                        raise

        if not subscription_created:
            raise RuntimeError(
                "Failed to create subscription to either Image or CompressedImage"
            )

        rmw = os.environ.get("RMW_IMPLEMENTATION", "default")
        self.get_logger().info(f"Benchmark subscriber created for: {topic}")
        self.get_logger().info(f"Duration: {duration}s")
        self.get_logger().info(f"Expected FPS: {expected_fps}")
        self.get_logger().info(f"RMW Implementation: {rmw}")

    def _process_message(self, msg, data_size: int):
        """Common processing for both Image and CompressedImage messages"""
        current_time = time.time()

        if self.start_time is None:
            self.start_time = current_time
            self.last_receive_time = current_time
            self.get_logger().info("Benchmark started - first message received")
            # Don't return - count the first message too

        # Calculate latency (time since message was stamped)
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        latency = current_time - msg_time
        self.latencies.append(latency)

        # Track frame rate and detect dropped frames by interval analysis
        if self.last_receive_time:
            frame_interval = current_time - self.last_receive_time
            expected_interval = 1.0 / self.expected_fps  # Expected time between frames

            # If gap is more than 1.5x expected interval, likely dropped frame(s)
            if frame_interval > (expected_interval * 1.5):
                # Estimate number of dropped frames based on interval
                estimated_dropped = int((frame_interval / expected_interval) - 1)
                self.dropped_frames += max(
                    1, estimated_dropped
                )  # At least 1 frame dropped

        self.frame_timestamps.append(current_time)
        self.frame_sizes.append(data_size)
        self.frame_count += 1
        self.last_receive_time = current_time

        # Check if duration reached
        elapsed = current_time - self.start_time
        if elapsed >= self.duration:
            self.end_time = current_time
            self.get_logger().info(
                f"Benchmark complete: {self.frame_count} frames in {elapsed:.2f}s"
            )

    def image_callback(self, msg: Image):
        """Callback for uncompressed Image messages"""
        # If we subscribed to wrong type, destroy wrong subscription
        if self.compressed_subscription is not None:
            self.get_logger().warn(
                "Received Image but subscribed to CompressedImage - destroying wrong subscription"
            )
            self.destroy_subscription(self.compressed_subscription)
            self.compressed_subscription = None

        if self.message_type is None:
            self.message_type = "Image"
            self.get_logger().info("Detected Image (uncompressed) messages")
        self._process_message(msg, len(msg.data))

    def compressed_image_callback(self, msg: CompressedImage):
        """Callback for CompressedImage messages"""
        # If we subscribed to wrong type, destroy wrong subscription
        if self.image_subscription is not None:
            self.get_logger().warn(
                "Received CompressedImage but subscribed to Image - destroying wrong subscription"
            )
            self.destroy_subscription(self.image_subscription)
            self.image_subscription = None

        if self.message_type is None:
            self.message_type = "CompressedImage"
            self.get_logger().info(f"Detected CompressedImage ({msg.format}) messages")
        self._process_message(msg, len(msg.data))

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return benchmark metrics"""
        if not self.frame_timestamps or self.start_time is None:
            return {}

        total_time = (self.end_time or time.time()) - self.start_time
        fps = self.frame_count / total_time if total_time > 0 else 0

        # Calculate frame intervals
        intervals = []
        for i in range(1, len(self.frame_timestamps)):
            intervals.append(self.frame_timestamps[i] - self.frame_timestamps[i - 1])

        avg_interval = statistics.mean(intervals) if intervals else 0
        std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
        min_interval = min(intervals) if intervals else 0
        max_interval = max(intervals) if intervals else 0

        # Latency statistics
        avg_latency = statistics.mean(self.latencies) if self.latencies else 0
        max_latency = max(self.latencies) if self.latencies else 0
        min_latency = min(self.latencies) if self.latencies else 0
        p95_latency = (
            statistics.quantiles(self.latencies, n=20)[18]
            if len(self.latencies) > 20
            else 0
        )
        p99_latency = (
            statistics.quantiles(self.latencies, n=100)[98]
            if len(self.latencies) > 100
            else 0
        )

        # Bandwidth
        total_bytes = sum(self.frame_sizes)
        bandwidth_mbps = (total_bytes * 8) / (total_time * 1e6) if total_time > 0 else 0
        avg_frame_size = statistics.mean(self.frame_sizes) if self.frame_sizes else 0
        max_frame_size = max(self.frame_sizes) if self.frame_sizes else 0
        min_frame_size = min(self.frame_sizes) if self.frame_sizes else 0

        return {
            "total_frames": self.frame_count,
            "dropped_frames": self.dropped_frames,
            "total_time_seconds": total_time,
            "fps": fps,
            "avg_frame_interval_ms": avg_interval * 1000,
            "std_frame_interval_ms": std_interval * 1000,
            "min_frame_interval_ms": min_interval * 1000,
            "max_frame_interval_ms": max_interval * 1000,
            "avg_latency_ms": avg_latency * 1000,
            "min_latency_ms": min_latency * 1000,
            "max_latency_ms": max_latency * 1000,
            "p95_latency_ms": p95_latency * 1000,
            "p99_latency_ms": p99_latency * 1000,
            "bandwidth_mbps": bandwidth_mbps,
            "avg_frame_size_kb": avg_frame_size / 1024,
            "min_frame_size_kb": min_frame_size / 1024,
            "max_frame_size_kb": max_frame_size / 1024,
            "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", "default"),
            "message_type": self.message_type or "unknown",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def run_benchmark(
    topic: str = "/camera/image_raw", duration: float = 30.0, expected_fps: float = 30.0
) -> Dict[str, Any]:
    """Run camera streaming benchmark"""
    if not ROS_AVAILABLE:
        raise RuntimeError("ROS2 not available")

    rclpy.init()

    try:
        node = CameraBenchmarkSubscriber(topic, duration, expected_fps)

        # Spin until duration reached
        start = time.time()
        last_message_time = start
        no_message_warnings = 0
        max_warnings = 3

        while time.time() - start < duration + 5:  # Add 5s buffer for startup
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.end_time is not None:
                break
            # Check if we're receiving messages
            if node.frame_count > 0:
                last_message_time = time.time()
                no_message_warnings = 0  # Reset warning counter
            # If no messages for 5 seconds, try to switch subscription proactively
            elif time.time() - last_message_time > 5 and time.time() - start > 5:
                no_message_warnings += 1
                if no_message_warnings <= max_warnings:
                    node.get_logger().warn(
                        f"No messages received for 5 seconds (warning {no_message_warnings}/{max_warnings}). Attempting subscription switch..."
                    )
                    # Try to switch subscription proactively
                    # If subscribed to CompressedImage but no messages, try Image
                    if node.compressed_subscription and not node.image_subscription:
                        try:
                            node.get_logger().info(
                                "Switching from CompressedImage to Image subscription..."
                            )
                            node.destroy_subscription(node.compressed_subscription)
                            from rclpy.qos import (
                                HistoryPolicy,
                                QoSProfile,
                                ReliabilityPolicy,
                            )

                            qos_profile = QoSProfile(
                                reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST,
                                depth=10,
                            )
                            node.image_subscription = node.create_subscription(
                                Image, node.topic, node.image_callback, qos_profile
                            )
                            node.compressed_subscription = None
                            node.get_logger().info(
                                "Switched to Image subscription - waiting for messages..."
                            )
                            last_message_time = time.time()  # Reset timer after switch
                        except Exception as e:
                            node.get_logger().error(
                                f"Failed to switch to Image subscription: {e}"
                            )
                    # If subscribed to Image but no messages, try CompressedImage
                    elif node.image_subscription and not node.compressed_subscription:
                        try:
                            node.get_logger().info(
                                "Switching from Image to CompressedImage subscription..."
                            )
                            node.destroy_subscription(node.image_subscription)
                            from rclpy.qos import (
                                HistoryPolicy,
                                QoSProfile,
                                ReliabilityPolicy,
                            )

                            qos_profile = QoSProfile(
                                reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST,
                                depth=10,
                            )
                            node.compressed_subscription = node.create_subscription(
                                CompressedImage,
                                node.topic,
                                node.compressed_image_callback,
                                qos_profile,
                            )
                            node.image_subscription = None
                            node.get_logger().info(
                                "Switched to CompressedImage subscription - waiting for messages..."
                            )
                            last_message_time = time.time()  # Reset timer after switch
                        except Exception as e:
                            node.get_logger().error(
                                f"Failed to switch to CompressedImage subscription: {e}"
                            )
                last_message_time = time.time()  # Reset to avoid spam

        metrics = node.get_metrics()

        # Additional check: if we got start_time but no frames, we only got 1 message
        if node.start_time and node.frame_count == 0:
            node.get_logger().error(
                "Only received 1 message. Check middleware compatibility and topic name."
            )
        node.destroy_node()

        return metrics

    finally:
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="ROS2 Camera Streaming Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with default settings (30s, /camera/image_raw)
  python3 benchmark_camera_streaming.py

  # Custom duration and topic
  python3 benchmark_camera_streaming.py --topic /camera/image_raw --duration 60

  # Save results to specific file (overwrite)
  python3 benchmark_camera_streaming.py --output results/default.json

  # Append results to existing file
  python3 benchmark_camera_streaming.py --output results/default.json --append
        """,
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/camera/image_raw",
        help="Camera topic to benchmark (default: /camera/image_raw)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Benchmark duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../output/benchmarking/results/benchmark_results.json",
        help="Output file for results (default: ../output/benchmarking/results/benchmark_results.json)",
    )
    parser.add_argument(
        "--expected-fps",
        type=float,
        default=30.0,
        help="Expected FPS for dropped frame detection (default: 30.0)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append results to existing file instead of overwriting",
    )

    args = parser.parse_args()

    rmw = os.environ.get("RMW_IMPLEMENTATION", "default")
    print(f"\n{'='*70}")
    print(f"ROS2 Camera Streaming Benchmark")
    print(f"{'='*70}")
    print(f"RMW Implementation: {rmw}")
    print(f"Topic: {args.topic}")
    print(f"Duration: {args.duration}s")
    print(f"Expected FPS: {args.expected_fps}")
    print(f"{'='*70}\n")

    try:
        print("Waiting for messages on topic...")
        print("Make sure camera_publisher.py is running in another terminal!")
        print("")

        metrics = run_benchmark(args.topic, args.duration, args.expected_fps)

        if metrics:
            print("\n" + "=" * 70)
            print("BENCHMARK RESULTS")
            print("=" * 70)
            print(f"RMW Implementation: {metrics['rmw_implementation']}")
            print(f"Timestamp: {metrics['timestamp']}")
            print(f"\nFrame Statistics:")
            print(f"  Total Frames: {metrics['total_frames']}")
            print(f"  Dropped Frames: {metrics['dropped_frames']}")
            print(f"  FPS: {metrics['fps']:.2f}")
            print(
                f"  Frame Interval: {metrics['avg_frame_interval_ms']:.2f} ± {metrics['std_frame_interval_ms']:.2f} ms"
            )
            print(
                f"  Interval Range: {metrics['min_frame_interval_ms']:.2f} - {metrics['max_frame_interval_ms']:.2f} ms"
            )
            print(f"\nLatency Statistics:")
            print(f"  Average: {metrics['avg_latency_ms']:.2f} ms")
            print(f"  Min: {metrics['min_latency_ms']:.2f} ms")
            print(f"  Max: {metrics['max_latency_ms']:.2f} ms")
            print(f"  P95: {metrics['p95_latency_ms']:.2f} ms")
            print(f"  P99: {metrics['p99_latency_ms']:.2f} ms")
            print(f"\nBandwidth Statistics:")
            print(f"  Average: {metrics['bandwidth_mbps']:.2f} Mbps")
            print(
                f"  Frame Size: {metrics['avg_frame_size_kb']:.2f} KB (range: {metrics['min_frame_size_kb']:.2f} - {metrics['max_frame_size_kb']:.2f} KB)"
            )
            print("=" * 70)

            # Save results
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if args.append:
                # Append mode: load existing results and add new entry
                if output_path.exists():
                    try:
                        with open(output_path, "r") as f:
                            existing_data = json.load(f)

                        # Handle both single dict and list formats
                        if isinstance(existing_data, list):
                            results_list = existing_data
                        elif isinstance(existing_data, dict):
                            # Convert single result to list format
                            results_list = [existing_data]
                        else:
                            results_list = []
                    except (json.JSONDecodeError, ValueError):
                        # File exists but is invalid JSON, start fresh
                        results_list = []
                else:
                    results_list = []

                # Add new result
                results_list.append(metrics)

                # Save as list
                with open(output_path, "w") as f:
                    json.dump(results_list, f, indent=2)

                print(f"\nResults appended to: {output_path.absolute()}")
                print(f"Total entries: {len(results_list)}")
            else:
                # Overwrite mode: save single result
                with open(output_path, "w") as f:
                    json.dump(metrics, f, indent=2)

                print(f"\nResults saved to: {output_path.absolute()}")
        else:
            print("ERROR: No metrics collected")
            print(
                "Make sure camera_publisher.py is running and publishing to the topic"
            )
            print("\nTroubleshooting:")
            print("  1. Check both publisher and subscriber use the same middleware:")
            print("     - Publisher: Check RMW_IMPLEMENTATION in publisher terminal")
            print("     - Subscriber: Check RMW_IMPLEMENTATION above")
            print("  2. Verify topic exists: ros2 topic list | grep /camera/image_raw")
            print("  3. Check topic info: ros2 topic info /camera/image_raw")
            print(
                "  4. Verify publisher is actually publishing: ros2 topic hz /camera/image_raw"
            )
            print("  5. Check message type matches (Image vs CompressedImage)")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
