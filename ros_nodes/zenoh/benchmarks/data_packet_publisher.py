#!/usr/bin/env python3
"""
ROS2 Data Packet Publisher
Publishes arbitrary-sized data packets for network benchmarking
"""

import argparse
import sys
import time
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import ByteMultiArray, UInt8MultiArray

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ERROR: ROS2 dependencies not installed")
    print("Install with: pip install rclpy")
    sys.exit(1)


class DataPacketPublisher(Node):
    """Publishes data packets of specified size"""

    def __init__(
        self, packet_size_kb: int, topic: str = "/data_packets", fps: float = 30.0
    ):
        """
        Initialize data packet publisher

        Args:
            packet_size_kb: Size of each packet in KB
            topic: ROS topic to publish to
            fps: Target FPS for publishing
        """
        super().__init__("data_packet_publisher")

        self.packet_size_bytes = packet_size_kb * 1024
        self.timer_period = 1.0 / fps
        self.packet_count = 0

        # Create publisher
        self.publisher = self.create_publisher(ByteMultiArray, topic, 10)

        # Generate packet data (random bytes for realistic testing)
        import random

        self.packet_data = bytes(
            random.randint(0, 255) for _ in range(self.packet_size_bytes)
        )

        # Create timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(
            f"Data packet publisher initialized: {packet_size_kb} KB packets "
            f"-> {topic} at {fps} FPS"
        )

    def timer_callback(self):
        """Publish a data packet"""
        try:
            msg = ByteMultiArray()
            msg.data = list(self.packet_data)

            self.publisher.publish(msg)
            self.packet_count += 1

            if self.packet_count % 100 == 0:
                self.get_logger().info(f"Published {self.packet_count} packets")

        except Exception as e:
            self.get_logger().error(f"Error publishing packet: {e}")


def main(args=None):
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ROS2 Data Packet Publisher")
    parser.add_argument(
        "--packet-size-kb",
        type=int,
        required=True,
        help="Size of each packet in KB (e.g., 128, 256, 512, 1024, 2048)",
    )
    parser.add_argument(
        "--topic", type=str, default="/data_packets", help="ROS topic to publish to"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Target FPS for publishing"
    )

    if ROS_AVAILABLE:
        ros_args = rclpy.utilities.remove_ros_args(args)
    else:
        ros_args = args
    args, unknown = parser.parse_known_args(ros_args)

    if not ROS_AVAILABLE:
        print("ERROR: ROS2 dependencies not installed")
        print("Install with: pip install rclpy")
        sys.exit(1)

    # Validate packet size
    valid_sizes = [128, 256, 512, 1024, 2048]
    if args.packet_size_kb not in valid_sizes:
        print(f"ERROR: Packet size must be one of: {valid_sizes}")
        sys.exit(1)

    rclpy.init(args=unknown)

    try:
        node = DataPacketPublisher(args.packet_size_kb, args.topic, args.fps)

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
