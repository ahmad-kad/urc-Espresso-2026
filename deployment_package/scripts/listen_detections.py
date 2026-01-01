#!/usr/bin/env python3
"""
ROS2 Detection Listener for Remote Testing
Subscribes to YOLO detector topics and displays detections
"""

import json
import sys
from datetime import datetime

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Error: ROS2 not available. Install ROS2 Humble first.")
    sys.exit(1)

# Class names matching the detector
CLASS_NAMES = ["ArUcoTag", "Bottle", "BrickHammer", "OrangeHammer", "USB-A", "USB-C"]


class DetectionListener(Node):
    """ROS2 node to listen to YOLO detection topics"""

    def __init__(self):
        super().__init__("detection_listener")

        # Subscribe to all detection topics
        self.sub_detections = self.create_subscription(
            Float32MultiArray, "/yolo_detector/detections", self.detections_callback, 10
        )

        self.sub_low = self.create_subscription(
            Float32MultiArray,
            "/yolo_detector/alerts/low_confidence",
            lambda msg: self.confidence_callback(msg, "LOW"),
            10,
        )

        self.sub_medium = self.create_subscription(
            Float32MultiArray,
            "/yolo_detector/alerts/medium_confidence",
            lambda msg: self.confidence_callback(msg, "MEDIUM"),
            10,
        )

        self.sub_high = self.create_subscription(
            Float32MultiArray,
            "/yolo_detector/alerts/high_confidence",
            lambda msg: self.confidence_callback(msg, "HIGH"),
            10,
        )

        self.sub_critical = self.create_subscription(
            Float32MultiArray,
            "/yolo_detector/alerts/critical",
            lambda msg: self.confidence_callback(msg, "CRITICAL"),
            10,
        )

        self.sub_messages = self.create_subscription(
            String, "/yolo_detector/alert_messages", self.message_callback, 10
        )

        self.detection_count = 0
        self.get_logger().info("Detection listener started")
        self.get_logger().info("Listening to topics:")
        self.get_logger().info("  - /yolo_detector/detections")
        self.get_logger().info("  - /yolo_detector/alerts/low_confidence")
        self.get_logger().info("  - /yolo_detector/alerts/medium_confidence")
        self.get_logger().info("  - /yolo_detector/alerts/high_confidence")
        self.get_logger().info("  - /yolo_detector/alerts/critical")
        self.get_logger().info("  - /yolo_detector/alert_messages")

    def parse_detections(self, msg):
        """Parse Float32MultiArray message into detections"""
        if len(msg.data) == 0:
            return []

        detections = []
        # Format: [class_id, confidence, x1, y1, x2, y2, ...]
        for i in range(0, len(msg.data), 6):
            if i + 5 < len(msg.data):
                class_id = int(msg.data[i])
                confidence = float(msg.data[i + 1])
                x1 = float(msg.data[i + 2])
                y1 = float(msg.data[i + 3])
                x2 = float(msg.data[i + 4])
                y2 = float(msg.data[i + 5])

                class_name = (
                    CLASS_NAMES[class_id]
                    if 0 <= class_id < len(CLASS_NAMES)
                    else f"Unknown({class_id})"
                )

                detections.append(
                    {
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": confidence,
                        "box": [x1, y1, x2, y2],
                    }
                )

        return detections

    def detections_callback(self, msg):
        """Callback for all detections topic"""
        detections = self.parse_detections(msg)
        self.detection_count += 1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"\n[{timestamp}] Detection #{self.detection_count}: {len(detections)} objects"
        )

        if detections:
            for det in detections:
                print(
                    f"  - {det['class']}: {det['confidence']:.2f} confidence "
                    f"box=[{det['box'][0]:.1f}, {det['box'][1]:.1f}, {det['box'][2]:.1f}, {det['box'][3]:.1f}]"
                )
        else:
            print("  No detections")

    def confidence_callback(self, msg, level):
        """Callback for confidence-level specific topics"""
        detections = self.parse_detections(msg)

        if detections:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\n[{timestamp}] {level} CONFIDENCE ALERT: {len(detections)} objects"
            )

            for det in detections:
                print(f"  - {det['class']}: {det['confidence']:.3f}")

    def message_callback(self, msg):
        """Callback for alert messages"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] ALERT MESSAGE: {msg.data}")


def main():
    """Main function"""
    if not ROS2_AVAILABLE:
        print("ROS2 is required but not available.")
        print("Install ROS2 Humble: https://docs.ros.org/en/humble/Installation.html")
        sys.exit(1)

    print("=" * 60)
    print("YOLO Detection Listener - Remote Testing")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print("=" * 60)

    rclpy.init()

    try:
        listener = DetectionListener()
        rclpy.spin(listener)
    except KeyboardInterrupt:
        print("\n\nShutting down listener...")
    finally:
        listener.destroy_node()
        rclpy.shutdown()
        print("Listener stopped")


if __name__ == "__main__":
    main()
