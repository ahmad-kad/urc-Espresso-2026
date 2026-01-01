#!/usr/bin/env python3
"""
Production YOLO Detection Service with Alerts
Includes: inference, alerts, logging, metrics, and health monitoring
"""

import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from alert_manager import ROS2_AVAILABLE, AlertManager
from picamera2 import Picamera2

# Initialize ROS2 if available
if ROS2_AVAILABLE:
    import rclpy

    rclpy.init()

# Configuration
CONFIG_FILE = "service_config.json"


class DetectionService:
    def __init__(self, config_file):
        """Initialize detection service"""
        self.load_config(config_file)
        self.setup_logging()
        self.running = False
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "start_time": None,
            "fps_history": [],
            "alerts_sent": 0,
        }

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def load_config(self, config_file):
        """Load service configuration"""
        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)
            self.logger.info(f"Loaded service configuration from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load service config: {e}")
            self.config = self._get_default_config()

    def _get_default_config(self):
        """Get default configuration"""
        return {
            "model": {
                "path": "models/best.onnx",
                "input_size": 224,
                "conf_threshold": 0.3,
                "iou_threshold": 0.4,
                "class_names": [
                    "ArUcoTag",
                    "Bottle",
                    "BrickHammer",
                    "OrangeHammer",
                    "USB-A",
                    "USB-C",
                ],
            },
            "camera": {"device": 0, "width": 224, "height": 224},
            "alerts": {"config_file": "alert_config.json"},
            "logging": {"directory": "logs", "stats_interval": 300},
            "storage": {
                "save_snapshots": True,
                "snapshot_directory": "snapshots",
                "stats_file": "stats.json",
            },
        }

    def setup_logging(self):
        """Configure logging"""
        log_dir = Path(self.config["logging"]["directory"])
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "detector.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("DetectionService")

    def initialize_camera(self):
        """Initialize Pi camera"""
        try:
            self.logger.info("Initializing camera...")
            self.picam2 = Picamera2()

            input_size = self.config["model"]["input_size"]
            config = self.picam2.create_preview_configuration(
                main={"size": (input_size, input_size), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()

            self.logger.info("Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise

    def initialize_model(self):
        """Initialize ONNX model"""
        try:
            self.logger.info("Loading ONNX model...")
            import onnxruntime as ort

            model_path = self.config["model"]["path"]
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def initialize_alerts(self):
        """Initialize alert manager"""
        try:
            self.logger.info("Initializing alert manager...")
            alert_config = self.config["alerts"]["config_file"]
            self.alert_manager = AlertManager(alert_config)

            # Pass class names to alert manager config
            if hasattr(self.alert_manager, "config"):
                self.alert_manager.config["class_names"] = self.config["model"][
                    "class_names"
                ]

            self.logger.info("Alert manager initialized")
            if ROS2_AVAILABLE:
                self.logger.info("ROS2 alerts enabled")
        except Exception as e:
            self.logger.error(f"Failed to initialize alert manager: {e}")
            raise

    def post_process_yolov8(self, output, conf_threshold, iou_threshold):
        """Post-process YOLO output"""
        output = output[0]
        output = output.transpose(1, 0)

        boxes = output[:, :4]
        class_scores = output[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return [], [], []

        # Convert xywh to xyxy
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold
        )
        if len(indices) > 0:
            # NMSBoxes returns numpy array, ensure it's 1D
            indices = (
                np.array(indices).flatten()
                if isinstance(indices, np.ndarray)
                else np.array(indices).ravel()
            )
            return boxes[indices], confidences[indices], class_ids[indices]

        return [], [], []

    def process_frame(self, frame):
        """Process single frame"""
        # Preprocess
        input_size = self.config["model"]["input_size"]
        img_resized = cv2.resize(frame, (input_size, input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = img_rgb.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, 0)

        # Inference
        start_time = time.time()
        results = self.session.run([self.output_name], {self.input_name: img_tensor})
        inference_time = time.time() - start_time

        # Post-process
        conf_threshold = self.config["model"]["conf_threshold"]
        iou_threshold = self.config["model"]["iou_threshold"]
        boxes, confidences, class_ids = self.post_process_yolov8(
            results[0], conf_threshold, iou_threshold
        )

        return boxes, confidences, class_ids, inference_time

    def save_snapshot(self, frame, detections):
        """Save detection snapshot"""
        if not self.config["storage"]["save_snapshots"]:
            return

        try:
            snapshot_dir = Path(self.config["storage"]["snapshot_directory"])
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = snapshot_dir / f"detection_{timestamp}.jpg"

            cv2.imwrite(str(filename), frame)
            self.logger.debug(f"Snapshot saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")

    def update_stats(self, num_detections, inference_time):
        """Update service statistics"""
        self.stats["total_frames"] += 1
        self.stats["total_detections"] += num_detections

        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.stats["fps_history"].append(fps)

        # Keep only last 100 FPS measurements
        if len(self.stats["fps_history"]) > 100:
            self.stats["fps_history"].pop(0)

    def log_stats(self):
        """Log service statistics"""
        uptime = time.time() - self.stats["start_time"]
        avg_fps = np.mean(self.stats["fps_history"]) if self.stats["fps_history"] else 0

        self.logger.info(
            f"Stats - Frames: {self.stats['total_frames']}, "
            f"Detections: {self.stats['total_detections']}, "
            f"Alerts: {self.stats['alerts_sent']}, "
            f"Avg FPS: {avg_fps:.2f}, "
            f"Uptime: {uptime:.0f}s"
        )

    def run(self):
        """Main service loop"""
        try:
            self.logger.info("Starting Detection Service...")

            # Initialize components
            self.initialize_camera()
            self.initialize_model()
            self.initialize_alerts()

            self.running = True
            self.stats["start_time"] = time.time()

            self.logger.info("Service running - press Ctrl+C to stop")

            frame_count = 0
            log_interval = self.config["logging"]["stats_interval"]

            while self.running:
                try:
                    # Spin ROS2 node if available
                    if ROS2_AVAILABLE and hasattr(self.alert_manager, "spin_once"):
                        self.alert_manager.spin_once(timeout_sec=0.001)

                    # Capture frame
                    frame = self.picam2.capture_array()

                    # Process frame
                    boxes, confidences, class_ids, inference_time = self.process_frame(
                        frame
                    )

                    # Update statistics
                    self.update_stats(len(boxes), inference_time)

                    # Check alerts and publish ROS2 messages
                    if len(boxes) > 0:
                        class_names = self.config["model"]["class_names"]
                        detections = [
                            {
                                "class": class_names[int(cls_id)],
                                "confidence": float(conf),
                                "box": box.tolist() if hasattr(box, "tolist") else box,
                            }
                            for box, conf, cls_id in zip(boxes, confidences, class_ids)
                        ]

                        # Always publish detections to ROS2 (grouped by confidence)
                        if ROS2_AVAILABLE:
                            self.alert_manager._publish_ros2_alerts(detections)

                        # Check for rule-based alerts
                        should_alert, rule = self.alert_manager.should_trigger_alert(
                            detections
                        )
                        if should_alert:
                            self.alert_manager.send_alert(rule, detections, frame)
                            self.save_snapshot(frame, detections)
                            self.stats["alerts_sent"] += 1

                    # Log stats periodically
                    frame_count += 1
                    if frame_count % log_interval == 0:
                        self.log_stats()

                    # Small delay to prevent CPU overload
                    time.sleep(0.001)

                except Exception as e:
                    self.logger.error(f"Frame processing error: {e}")
                    time.sleep(1)  # Brief pause before retrying

        except Exception as e:
            self.logger.error(f"Service error: {e}", exc_info=True)
            raise

        finally:
            self.shutdown()

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def shutdown(self):
        """Clean shutdown"""
        self.logger.info("Shutting down service...")

        if hasattr(self, "picam2"):
            try:
                self.picam2.stop()
            except:
                pass

        # Shutdown ROS2 if initialized
        if ROS2_AVAILABLE and hasattr(self, "alert_manager"):
            try:
                self.alert_manager.destroy_node()
            except:
                pass
            try:
                rclpy.shutdown()
            except:
                pass

        # Save final stats
        try:
            stats_file = Path(self.config["storage"]["stats_file"])
            with open(stats_file, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save stats: {e}")

        # Save alert history
        if hasattr(self, "alert_manager"):
            try:
                self.alert_manager.save_alert_log()
            except Exception as e:
                self.logger.error(f"Failed to save alert history: {e}")

        self.logger.info("Service stopped")


if __name__ == "__main__":
    service = DetectionService(CONFIG_FILE)
    service.run()
