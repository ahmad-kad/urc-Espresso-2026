#!/usr/bin/env python3
"""
ROS2 Data Packet Benchmark Subscriber
Subscribes to data packets and measures performance metrics
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import ByteMultiArray

    ROS_AVAILABLE = True
except ImportError as e:
    ROS_AVAILABLE = False
    print(f"ROS2 not available: {e}")
    print("Install with: pip install rclpy")
    sys.exit(1)


class DataPacketBenchmarkSubscriber(Node):
    """Subscriber for benchmarking data packet performance"""

    def __init__(
        self,
        topic: str = "/data_packets",
        duration: float = 30.0,
        expected_fps: float = 30.0,
    ):
        super().__init__("data_packet_benchmark_subscriber")

        self.topic = topic
        self.duration = duration
        self.expected_fps = expected_fps
        self.start_time = None
        self.end_time = None

        # Metrics
        self.packet_timestamps: List[float] = []
        self.packet_sizes: List[int] = []
        self.latencies: deque = deque(maxlen=100)  # Rolling window for latency
        self.last_receive_time = None
        self.packet_count = 0
        self.dropped_packets = 0
        self.expected_packet_interval = 1.0 / expected_fps

        # Create subscriber with high QoS for benchmarking
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.subscription = self.create_subscription(
            ByteMultiArray, topic, self.packet_callback, qos_profile
        )

        self.get_logger().info(f"Subscribed to {topic}, waiting for packets...")

    def packet_callback(self, msg: ByteMultiArray):
        """Callback for received data packets"""
        current_time = time.time()

        # Initialize start time on first packet
        if self.start_time is None:
            self.start_time = current_time
            self.last_receive_time = current_time
            self.get_logger().info("First packet received, starting benchmark...")
            return

        # Check if we've exceeded duration
        elapsed = current_time - self.start_time
        if elapsed >= self.duration:
            if self.end_time is None:
                self.end_time = current_time
                self.get_logger().info(f"Benchmark duration ({self.duration}s) reached")
            return

        # Calculate latency (inter-packet interval)
        if self.last_receive_time is not None:
            interval = (current_time - self.last_receive_time) * 1000  # Convert to ms
            self.latencies.append(interval)

            # Detect dropped packets (intervals significantly longer than expected)
            expected_interval_ms = self.expected_packet_interval * 1000
            if interval > expected_interval_ms * 1.5:  # 50% tolerance
                dropped = int((interval - expected_interval_ms) / expected_interval_ms)
                self.dropped_packets += max(
                    0, dropped - 1
                )  # -1 because we count the current packet

        # Record metrics
        packet_size = len(msg.data)
        self.packet_timestamps.append(current_time)
        self.packet_sizes.append(packet_size)
        self.packet_count += 1
        self.last_receive_time = current_time

        # Log progress every 100 packets
        if self.packet_count % 100 == 0:
            self.get_logger().info(f"Received {self.packet_count} packets...")

    def get_results(self) -> Dict[str, Any]:
        """Calculate and return benchmark results"""
        if not self.packet_timestamps or len(self.packet_timestamps) < 2:
            return {
                "error": "Insufficient data collected",
                "packet_count": self.packet_count,
            }

        # Calculate time span
        if self.end_time is None:
            self.end_time = time.time()

        total_time = self.end_time - self.start_time

        # Calculate FPS
        fps = self.packet_count / total_time if total_time > 0 else 0

        # Calculate latency statistics
        if self.latencies:
            latencies_list = list(self.latencies)
            avg_latency_ms = statistics.mean(latencies_list)
            min_latency_ms = min(latencies_list)
            max_latency_ms = max(latencies_list)

            # Calculate percentiles
            sorted_latencies = sorted(latencies_list)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            p95_latency_ms = (
                sorted_latencies[p95_idx]
                if p95_idx < len(sorted_latencies)
                else sorted_latencies[-1]
            )
            p99_latency_ms = (
                sorted_latencies[p99_idx]
                if p99_idx < len(sorted_latencies)
                else sorted_latencies[-1]
            )

            # Jitter (standard deviation)
            jitter_ms = (
                statistics.stdev(latencies_list) if len(latencies_list) > 1 else 0
            )
        else:
            avg_latency_ms = 0
            min_latency_ms = 0
            max_latency_ms = 0
            p95_latency_ms = 0
            p99_latency_ms = 0
            jitter_ms = 0

        # Calculate packet size statistics
        if self.packet_sizes:
            avg_packet_size_bytes = statistics.mean(self.packet_sizes)
            min_packet_size_bytes = min(self.packet_sizes)
            max_packet_size_bytes = max(self.packet_sizes)
            avg_packet_size_kb = avg_packet_size_bytes / 1024
        else:
            avg_packet_size_bytes = 0
            min_packet_size_bytes = 0
            max_packet_size_bytes = 0
            avg_packet_size_kb = 0

        # Calculate bandwidth
        # Bandwidth = (packet_size_bytes * fps * 8) / (1024 * 1024) Mbps
        bandwidth_mbps = (
            (avg_packet_size_bytes * fps * 8) / (1024 * 1024) if fps > 0 else 0
        )

        # Get middleware info
        rmw_implementation = os.environ.get("RMW_IMPLEMENTATION", "unknown")

        return {
            "total_packets": self.packet_count,
            "dropped_packets": self.dropped_packets,
            "total_time_seconds": total_time,
            "fps": fps,
            "avg_packet_interval_ms": avg_latency_ms,
            "std_packet_interval_ms": jitter_ms,
            "min_packet_interval_ms": min_latency_ms,
            "max_packet_interval_ms": max_latency_ms,
            "avg_latency_ms": avg_latency_ms,
            "min_latency_ms": min_latency_ms,
            "max_latency_ms": max_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "bandwidth_mbps": bandwidth_mbps,
            "avg_packet_size_bytes": avg_packet_size_bytes,
            "avg_packet_size_kb": avg_packet_size_kb,
            "min_packet_size_bytes": min_packet_size_bytes,
            "max_packet_size_bytes": max_packet_size_bytes,
            "rmw_implementation": rmw_implementation,
            "message_type": "ByteMultiArray",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def run_benchmark(
    topic: str = "/data_packets", duration: float = 30.0, expected_fps: float = 30.0
) -> Dict[str, Any]:
    """
    Run data packet benchmark

    Args:
        topic: ROS topic to subscribe to
        duration: Benchmark duration in seconds
        expected_fps: Expected packets per second

    Returns:
        Dictionary with benchmark results
    """
    if not ROS_AVAILABLE:
        raise ImportError("ROS2 dependencies not installed")

    rclpy.init()

    try:
        node = DataPacketBenchmarkSubscriber(topic, duration, expected_fps)

        # Spin until duration is reached
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.start_time is not None:
                elapsed = time.time() - node.start_time
                if elapsed >= duration:
                    break

            # Safety timeout
            if time.time() - start_time > duration + 10:
                break

        results = node.get_results()
        return results

    finally:
        if rclpy.ok():
            rclpy.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ROS2 Data Packet Benchmark")
    parser.add_argument(
        "--topic", type=str, default="/data_packets", help="ROS topic to subscribe to"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Benchmark duration in seconds"
    )
    parser.add_argument(
        "--expected-fps", type=float, default=30.0, help="Expected packets per second"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path"
    )
    parser.add_argument(
        "--append", action="store_true", help="Append results to existing file"
    )

    if ROS_AVAILABLE:
        ros_args = rclpy.utilities.remove_ros_args(sys.argv)
    else:
        ros_args = sys.argv
    args, unknown = parser.parse_known_args(ros_args)

    if not ROS_AVAILABLE:
        print("ERROR: ROS2 dependencies not installed")
        print("Install with: pip install rclpy")
        sys.exit(1)

    print(f"Starting data packet benchmark...")
    print(f"  Topic: {args.topic}")
    print(f"  Duration: {args.duration}s")
    print(f"  Expected FPS: {args.expected_fps}")
    print("")

    results = run_benchmark(args.topic, args.duration, args.expected_fps)

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Packets: {results.get('total_packets', 0)}")
    print(f"Dropped Packets: {results.get('dropped_packets', 0)}")
    print(f"FPS: {results.get('fps', 0):.2f}")
    print(f"Bandwidth: {results.get('bandwidth_mbps', 0):.2f} Mbps")
    print(f"Avg Packet Size: {results.get('avg_packet_size_kb', 0):.2f} KB")
    print(f"Avg Latency: {results.get('avg_latency_ms', 0):.2f} ms")
    print(f"P95 Latency: {results.get('p95_latency_ms', 0):.2f} ms")
    print(f"Jitter: {results.get('std_packet_interval_ms', 0):.2f} ms")
    print(f"Middleware: {results.get('rmw_implementation', 'unknown')}")
    print("=" * 80)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.append and output_path.exists():
            try:
                with open(output_path) as f:
                    existing_data = json.load(f)
                if isinstance(existing_data, list):
                    existing_data.append(results)
                    data_to_save = existing_data
                else:
                    data_to_save = [existing_data, results]
            except (json.JSONDecodeError, ValueError):
                data_to_save = [results]
        else:
            data_to_save = results

        with open(output_path, "w") as f:
            json.dump(data_to_save, f, indent=2)

        print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
