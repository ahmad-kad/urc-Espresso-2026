#!/usr/bin/env python3
"""
ROS2 Listener for YOLO Detection Service
Remote testing script to monitor detection alerts
Usage: python3 ros2_listener.py [topic_name]
"""

import sys
import time
from datetime import datetime

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Error: ROS2 not available. Install ROS2 Humble first.")
    print("Source: source /opt/ros/humble/setup.bash")
    sys.exit(1)

# Class names matching the detector
CLASS_NAMES = ["ArUcoTag", "Bottle", "BrickHammer", "OrangeHammer", "USB-A", "USB-C"]


class DetectionListener(Node):
    def __init__(self, topic_name="/yolo_detector/detections"):
        super().__init__("yolo_detection_listener")
        self.topic_name = topic_name
        self.detection_count = 0

        # Subscribe to detection topic
        self.subscription = self.create_subscription(
            Float32MultiArray, topic_name, self.detection_callback, 10
        )

        self.get_logger().info(f"Listening to topic: {topic_name}")
        self.get_logger().info("Press Ctrl+C to stop")

    def detection_callback(self, msg):
        """Process incoming detection messages"""
        self.detection_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Parse detection data
        # Format: [class_id, confidence, x1, y1, x2, y2, ...]
        data = msg.data

        if len(data) == 0:
            return

        # Each detection is 6 values
        num_detections = len(data) // 6

        print(
            f"\n[{timestamp}] Detection #{self.detection_count} - {num_detections} object(s) detected:"
        )
        print("-" * 60)

        for i in range(num_detections):
            idx = i * 6
            class_id = int(data[idx])
            confidence = data[idx + 1]
            x1, y1, x2, y2 = data[idx + 2], data[idx + 3], data[idx + 4], data[idx + 5]

            class_name = (
                CLASS_NAMES[class_id]
                if 0 <= class_id < len(CLASS_NAMES)
                else f"Unknown({class_id})"
            )

            print(
                f"  {i+1}. {class_name:15s} | Confidence: {confidence:.2f} | "
                f"Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
            )

        print("-" * 60)


class AlertMessageListener(Node):
    def __init__(self, topic_name="/yolo_detector/alert_messages"):
        super().__init__("yolo_alert_listener")
        self.topic_name = topic_name
        self.alert_count = 0

        self.subscription = self.create_subscription(
            String, topic_name, self.alert_callback, 10
        )

        self.get_logger().info(f"Listening to alert messages: {topic_name}")
        self.get_logger().info("Press Ctrl+C to stop")

    def alert_callback(self, msg):
        """Process incoming alert messages"""
        self.alert_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Alert #{self.alert_count}: {msg.data}")


def main():
    """Main function"""
    if not ROS2_AVAILABLE:
        return

    # Parse command line arguments
    topic_name = sys.argv[1] if len(sys.argv) > 1 else "/yolo_detector/detections"

    print("=" * 60)
    print("YOLO Detection Service - ROS2 Listener")
    print("=" * 60)
    print(f"Topic: {topic_name}")
    print(f"Available topics:")
    print("  - /yolo_detector/detections (all detections)")
    print("  - /yolo_detector/alerts/low_confidence")
    print("  - /yolo_detector/alerts/medium_confidence")
    print("  - /yolo_detector/alerts/high_confidence")
    print("  - /yolo_detector/alerts/critical")
    print("  - /yolo_detector/alert_messages (string messages)")
    print("=" * 60)
    print()

    # Initialize ROS2
    rclpy.init()

    # Create appropriate listener based on topic
    if "alert_messages" in topic_name:
        listener = AlertMessageListener(topic_name)
    else:
        listener = DetectionListener(topic_name)

    try:
        # Spin node
        rclpy.spin(listener)
    except KeyboardInterrupt:
        print("\n\nShutting down listener...")
        print(
            f"Total messages received: {listener.detection_count if hasattr(listener, 'detection_count') else listener.alert_count}"
        )
    finally:
        listener.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
