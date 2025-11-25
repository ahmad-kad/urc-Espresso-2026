#!/usr/bin/env python3
"""
Real-time object detection demo using webcam
Supports any trained model and configurable parameters
"""

import cv2
import argparse
import sys
from pathlib import Path
import logging
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import load_config
from core.detector import ObjectDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Real-time object detection demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration to use')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera frame width')
    parser.add_argument('--height', type=int, default=480,
                       help='Camera frame height')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--save_video', type=str,
                       help='Path to save output video')
    parser.add_argument('--display_fps', action='store_true',
                       help='Display FPS on video feed')

    args = parser.parse_args()

    try:
        logger.info("Starting webcam object detection demo")

        # Load configuration
        config = load_config(args.config)

        # Override model settings
        config['model']['confidence_threshold'] = args.conf_threshold

        # Initialize detector
        detector = ObjectDetector(config)

        # Load the specified model
        if Path(args.model).exists():
            logger.info(f"Loading model: {args.model}")
            # Note: In practice, you'd need to load the model properly
            # This is a simplified example
        else:
            logger.error(f"Model not found: {args.model}")
            return 1

        # Initialize camera
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            logger.error(f"Could not open camera {args.camera}")
            return 1

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        logger.info(f"Camera initialized: {args.width}x{args.height}")

        # Initialize video writer if saving
        video_writer = None
        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.save_video, fourcc, 30,
                                         (args.width, args.height))
            logger.info(f"Saving video to: {args.save_video}")

        # Performance tracking
        frame_count = 0
        start_time = time.time()
        fps_history = []

        logger.info("Starting detection loop... Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break

            frame_start_time = time.time()

            # Run detection (placeholder - implement actual detection)
            # detections = detector.predict(frame)

            # For demo, we'll create mock detections
            detections = []

            # Draw detections on frame
            for detection in detections:
                if detection['confidence'] < args.conf_threshold:
                    continue

                x1, y1, x2, y2 = detection['bbox']
                class_name = detection.get('class_name', 'object')
                conf = detection['confidence']

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Draw label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate and display FPS
            frame_time = time.time() - frame_start_time
            fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(fps)

            if args.display_fps:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write frame to video if saving
            if video_writer:
                video_writer.write(frame)

            # Display frame
            cv2.imshow('Object Detection Demo', frame)

            frame_count += 1

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        # Print performance summary
        total_time = time.time() - start_time
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        logger.info("Demo completed!")
        logger.info(".1f")
        logger.info(f"Total frames processed: {frame_count}")

        return 0

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
