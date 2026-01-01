#!/usr/bin/env python3
"""
ROS2 Alert Manager for YOLO Detection System
Publishes alerts via ROS2 topics with different confidence levels
"""

import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: ROS2 not available. Alerts will not be published.")


class AlertManager(Node):
    def __init__(self, config_file="alert_config.json"):
        """Initialize ROS2 alert manager with configuration"""
        if ROS2_AVAILABLE:
            super().__init__("yolo_alert_manager")
        else:
            # Fallback for non-ROS2 environments
            pass

        self.load_config(config_file)
        self.detection_history = deque(maxlen=100)
        self.alert_cooldowns = {}
        self.alert_log = []

        # Setup logging
        self.logger = logging.getLogger("AlertManager")

        # Initialize ROS2 publishers if available
        if ROS2_AVAILABLE:
            self._init_ros2_publishers()

    def _init_ros2_publishers(self):
        """Initialize ROS2 publishers for different confidence levels"""
        if not ROS2_AVAILABLE:
            return

        # Create publishers for different confidence level alerts
        # Using Float32MultiArray to send detection data: [class_id, confidence, x1, y1, x2, y2, ...]
        self.pub_low_confidence = self.create_publisher(
            Float32MultiArray, "/yolo_detector/alerts/low_confidence", 10
        )
        self.pub_medium_confidence = self.create_publisher(
            Float32MultiArray, "/yolo_detector/alerts/medium_confidence", 10
        )
        self.pub_high_confidence = self.create_publisher(
            Float32MultiArray, "/yolo_detector/alerts/high_confidence", 10
        )
        self.pub_critical = self.create_publisher(
            Float32MultiArray, "/yolo_detector/alerts/critical", 10
        )

        # Also publish all detections on a general topic
        self.pub_detections = self.create_publisher(
            Float32MultiArray, "/yolo_detector/detections", 10
        )

        # Publish alert messages as strings
        self.pub_alert_messages = self.create_publisher(
            String, "/yolo_detector/alert_messages", 10
        )

        self.get_logger().info("ROS2 alert publishers initialized")

    def load_config(self, config_file):
        """Load alert configuration"""
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
            self.logger.info(f"Loaded alert configuration from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load alert config: {e}")
            # Default config with confidence level thresholds
            self.config = {
                "confidence_levels": {
                    "low": 0.3,
                    "medium": 0.5,
                    "high": 0.7,
                    "critical": 0.9,
                },
                "alert_rules": [],
            }

    def should_trigger_alert(self, detections):
        """Determine if alert should be triggered based on rules"""
        current_time = time.time()

        for rule in self.config["alert_rules"]:
            rule_name = rule["name"]

            # Check cooldown
            if rule_name in self.alert_cooldowns:
                if current_time - self.alert_cooldowns[rule_name] < rule["cooldown"]:
                    continue

            # Check rule conditions
            if self._check_rule_conditions(detections, rule):
                self.alert_cooldowns[rule_name] = current_time
                return True, rule

        return False, None

    def _check_rule_conditions(self, detections, rule):
        """Check if detection matches rule conditions"""
        rule_type = rule["type"]

        if rule_type == "object_detected":
            # Alert when specific object is detected
            target_classes = rule["target_classes"]
            min_confidence = rule.get("min_confidence", 0.5)

            for det in detections:
                if (
                    det["class"] in target_classes
                    and det["confidence"] >= min_confidence
                ):
                    return True

        elif rule_type == "object_count":
            # Alert when object count exceeds threshold
            target_classes = rule["target_classes"]
            threshold = rule["threshold"]
            condition = rule.get("condition", "greater_than")

            count = sum(1 for det in detections if det["class"] in target_classes)
            if condition == "greater_than" and count > threshold:
                return True
            elif condition == "less_than" and count < threshold:
                return True

        elif rule_type == "object_persistence":
            # Alert when object persists for duration
            target_classes = rule["target_classes"]
            duration = rule["duration"]

            self.detection_history.append(
                {"timestamp": time.time(), "detections": detections}
            )

            # Check if object persisted
            recent_history = [
                h
                for h in self.detection_history
                if time.time() - h["timestamp"] <= duration
            ]

            if len(recent_history) < duration:
                return False

            # Check all recent frames have the target object
            for hist in recent_history:
                has_target = any(
                    det["class"] in target_classes for det in hist["detections"]
                )
                if not has_target:
                    return False

            return True

        elif rule_type == "zone_intrusion":
            # Alert when object enters defined zone
            target_classes = rule["target_classes"]
            zone = rule["zone"]  # [x1, y1, x2, y2]

            for det in detections:
                if det["class"] not in target_classes:
                    continue

                # Check if bounding box overlaps with zone
                box = det["box"]
                if self._box_in_zone(box, zone):
                    return True

        return False

    def _box_in_zone(self, box, zone):
        """Check if bounding box intersects with zone"""
        # Box format: [x1, y1, x2, y2]
        # Zone format: [x1, y1, x2, y2]
        return not (
            box[2] < zone[0] or box[0] > zone[2] or box[3] < zone[1] or box[1] > zone[3]
        )

    def send_alert(self, rule, detections, frame=None):
        """Publish ROS2 alert based on confidence levels"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_message = f"[{timestamp}] {rule['message']}"

        # Log alert
        alert_record = {
            "timestamp": timestamp,
            "rule": rule["name"],
            "message": rule["message"],
            "detections": len(detections),
            "classes": [det["class"] for det in detections],
        }
        self.alert_log.append(alert_record)

        # Publish ROS2 alerts grouped by confidence level
        if ROS2_AVAILABLE:
            self._publish_ros2_alerts(detections)

        # Log to file
        self._log_alert(alert_message, detections)

        self.logger.info(f"Alert sent: {alert_message}")

    def _publish_ros2_alerts(self, detections):
        """Publish detections to ROS2 topics based on confidence levels"""
        if not ROS2_AVAILABLE or not hasattr(self, "pub_detections"):
            return

        # Get confidence thresholds from config
        conf_levels = self.config.get(
            "confidence_levels",
            {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9},
        )

        # Group detections by confidence level
        low_conf_detections = []
        medium_conf_detections = []
        high_conf_detections = []
        critical_detections = []

        for det in detections:
            conf = det["confidence"]
            if conf >= conf_levels.get("critical", 0.9):
                critical_detections.append(det)
            elif conf >= conf_levels.get("high", 0.7):
                high_conf_detections.append(det)
            elif conf >= conf_levels.get("medium", 0.5):
                medium_conf_detections.append(det)
            elif conf >= conf_levels.get("low", 0.3):
                low_conf_detections.append(det)

        # Publish to appropriate topics
        if low_conf_detections:
            self._publish_detection_array(
                self.pub_low_confidence, low_conf_detections, None
            )

        if medium_conf_detections:
            self._publish_detection_array(
                self.pub_medium_confidence, medium_conf_detections, None
            )

        if high_conf_detections:
            self._publish_detection_array(
                self.pub_high_confidence, high_conf_detections, None
            )

        if critical_detections:
            self._publish_detection_array(self.pub_critical, critical_detections, None)

        # Always publish all detections to general topic
        all_detections = (
            low_conf_detections
            + medium_conf_detections
            + high_conf_detections
            + critical_detections
        )
        if all_detections:
            self._publish_detection_array(self.pub_detections, all_detections, None)

        # Publish alert message as string
        if all_detections:
            alert_str = String()
            alert_str.data = f"Detections: {len(all_detections)} objects"
            self.pub_alert_messages.publish(alert_str)

    def _publish_detection_array(self, publisher, detections, timestamp):
        """Convert detections to ROS2 Float32MultiArray message and publish"""
        if not ROS2_AVAILABLE:
            return

        # Create class name to ID mapping
        class_names = self.config.get(
            "class_names",
            ["ArUcoTag", "Bottle", "BrickHammer", "OrangeHammer", "USB-A", "USB-C"],
        )
        class_to_id = {name: idx for idx, name in enumerate(class_names)}

        # Pack detections as flat array: [class_id, confidence, x1, y1, x2, y2, ...]
        data = []
        for det in detections:
            class_id = class_to_id.get(det["class"], -1)
            box = det["box"]  # [x1, y1, x2, y2]
            data.extend(
                [
                    float(class_id),
                    float(det["confidence"]),
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ]
            )

        msg = Float32MultiArray()
        msg.data = data

        # Set layout: each detection is 6 values (class_id, conf, x1, y1, x2, y2)
        if len(detections) > 0:
            dim = MultiArrayDimension()
            dim.label = "detections"
            dim.size = len(detections)
            dim.stride = 6 * len(detections)
            msg.layout.dim.append(dim)

            dim2 = MultiArrayDimension()
            dim2.label = "values_per_detection"
            dim2.size = 6
            dim2.stride = 6
            msg.layout.dim.append(dim2)

        publisher.publish(msg)

    def _log_alert(self, message, detections):
        """Log alert to file"""
        try:
            log_file = Path("alerts.log")
            with open(log_file, "a") as f:
                f.write(f"{message}\n")
                for det in detections:
                    f.write(f"  - {det['class']}: {det['confidence']:.2f}\n")
                f.write("\n")
            self.logger.info("Alert logged to file")
        except Exception as e:
            self.logger.error(f"Failed to log alert: {e}")

    def save_alert_log(self, filename="alert_history.json"):
        """Save alert history to file"""
        try:
            with open(filename, "w") as f:
                json.dump(self.alert_log, f, indent=2)
            self.logger.info(f"Alert history saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save alert history: {e}")

    def get_alert_stats(self):
        """Get alert statistics"""
        if not self.alert_log:
            return {"total_alerts": 0, "alerts_by_rule": {}, "recent_alerts": []}

        alerts_by_rule = {}
        for alert in self.alert_log:
            rule = alert["rule"]
            alerts_by_rule[rule] = alerts_by_rule.get(rule, 0) + 1

        recent_alerts = self.alert_log[-10:]  # Last 10 alerts

        return {
            "total_alerts": len(self.alert_log),
            "alerts_by_rule": alerts_by_rule,
            "recent_alerts": recent_alerts,
        }
