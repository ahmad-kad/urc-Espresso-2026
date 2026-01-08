#!/usr/bin/env python3
"""
RealSense ROS2 Bridge
Subscribes to RealSense camera topics and republishes to standard topic for detector
"""

import argparse
import sys

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    ROS_AVAILABLE = True
except ImportError as e:
    ROS_AVAILABLE = False
    print(f"ROS2 not available: {e}")


class RealSenseBridge(Node):
    """Bridge from RealSense topics to standard camera topic"""

    def __init__(
        self,
        input_topic: str = "/camera/color/image_raw",
        output_topic: str = "/camera/image_raw",
    ):
        """
        Initialize RealSense bridge

        Args:
            input_topic: RealSense image topic to subscribe to
            output_topic: Topic to republish to (for detector)
        """
        if not ROS_AVAILABLE:
            raise ImportError("ROS2 dependencies not installed")

        super().__init__("realsense_bridge")

        # Subscriber
        self.subscription = self.create_subscription(
            Image, input_topic, self.image_callback, 10
        )
        self.get_logger().info(f"Subscribed to: {input_topic}")

        # Publisher
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.get_logger().info(f"Publishing to: {output_topic}")

        self.frame_count = 0

    def image_callback(self, msg: Image):
        """Callback for incoming RealSense images"""
        # Simply republish with updated header
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(f"Bridged {self.frame_count} frames")


def main(args=None):
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RealSense ROS2 Bridge")
    parser.add_argument(
        "--input",
        type=str,
        default="/camera/color/image_raw",
        help="RealSense input topic",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/camera/image_raw",
        help="Output topic for detector",
    )

    if ROS_AVAILABLE:
        ros_args = rclpy.utilities.remove_ros_args(args)
    else:
        ros_args = args
    args, unknown = parser.parse_known_args(ros_args)

    if not ROS_AVAILABLE:
        print("ERROR: ROS2 dependencies not installed")
        sys.exit(1)

    rclpy.init(args=unknown)

    try:
        node = RealSenseBridge(args.input, args.output)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
