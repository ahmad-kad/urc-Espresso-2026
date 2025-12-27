#!/usr/bin/env python3
"""
Real-time Object Detection Stream for Raspberry Pi 5
Streams camera feed and highlights detected objects using ONNX models
"""

import sys
import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Class names for the espresso dataset
CLASS_NAMES = ['ArUcoTag', 'Bottle', 'BrickHammer', 'OrangeHammer', 'USB-A', 'USB-C']
CLASS_COLORS = [
    (0, 255, 0),      # ArUcoTag - Green
    (255, 0, 0),      # Bottle - Blue
    (0, 0, 255),      # BrickHammer - Red
    (255, 165, 0),    # OrangeHammer - Orange
    (255, 0, 255),    # USB-A - Magenta
    (0, 255, 255),    # USB-C - Cyan
]


class ONNXDetector:
    """ONNX model detector for real-time inference"""
    
    def __init__(self, model_path: str, input_size: int = 224, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize ONNX detector
        
        Args:
            model_path: Path to ONNX model file
            input_size: Model input size (224, 192, or 160)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        logger.info(f"Loading ONNX model: {model_path}")
        logger.info(f"Input size: {input_size}x{input_size}")
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output details
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Model input: {input_name}, shape: {input_shape}")
        
        # Get output names
        self.output_names = [output.name for output in self.session.get_outputs()]
        logger.info(f"Model outputs: {self.output_names}")
        
        # Warm up the model
        self._warmup()
        
    def _warmup(self):
        """Warm up the model with dummy input"""
        dummy_input = np.random.randn(1, 3, self.input_size, self.input_size).astype(np.float32)
        _ = self.session.run(self.output_names, {self.input_name: dummy_input})
        logger.info("Model warmed up")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for model input
        
        Returns:
            preprocessed_image: Preprocessed image tensor
            scale: Scale factor for coordinates
            original_shape: (height, width) of original image
        """
        original_shape = image.shape[:2]  # (height, width)
        h, w = original_shape
        
        # Calculate scale to maintain aspect ratio
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize with aspect ratio
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, scale, original_shape
    
    def postprocess(self, outputs: List[np.ndarray], scale: float, original_shape: Tuple[int, int]) -> List[Dict]:
        """
        Postprocess model outputs to get detections
        
        Returns:
            List of detections with format: [x1, y1, x2, y2, confidence, class_id]
        """
        # YOLOv8 output format: (batch, num_detections, 84) where 84 = 4 (bbox) + 80 (classes)
        # For our 6 classes: (batch, num_detections, 10) where 10 = 4 (bbox) + 6 (classes)
        
        predictions = outputs[0]  # Shape: (1, num_detections, 4+num_classes)
        
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension
        
        detections = []
        
        # Extract boxes and scores
        boxes = predictions[:, :4]  # (num_detections, 4)
        scores = predictions[:, 4:]  # (num_detections, num_classes)
        
        # Get class predictions
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Scale boxes back to original image size
        boxes[:, [0, 2]] *= (original_shape[1] / self.input_size)  # x coordinates
        boxes[:, [1, 3]] *= (original_shape[0] / self.input_size)  # y coordinates
        
        # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes = np.column_stack([x1, y1, x2, y2])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'bbox': boxes[i],
                    'confidence': float(confidences[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': CLASS_NAMES[int(class_ids[i])] if int(class_ids[i]) < len(CLASS_NAMES) else f'Class_{int(class_ids[i])}'
                })
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on image"""
        # Preprocess
        tensor, scale, original_shape = self.preprocess(image)
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        inference_time = time.time() - start_time
        
        # Postprocess
        detections = self.postprocess(outputs, scale, original_shape)
        
        return detections, inference_time


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    result_image = image.copy()
    
    for det in detections:
        bbox = det['bbox']
        confidence = det['confidence']
        class_id = det['class_id']
        class_name = det['class_name']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color for this class
        color = CLASS_COLORS[class_id] if class_id < len(CLASS_COLORS) else (255, 255, 255)
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_name}: {confidence:.2f}"
        
        # Get label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            result_image,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            result_image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return result_image


def get_camera(camera_index: int = 0, width: int = 640, height: int = 480):
    """Initialize camera (supports both Pi camera and USB cameras)"""
    # Try to open camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_index}")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    return cap


def main():
    parser = argparse.ArgumentParser(description='Real-time object detection on Raspberry Pi 5')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--input-size', type=int, default=224,
                       help='Model input size (default: 224)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Camera height (default: 480)')
    parser.add_argument('--fps-display', action='store_true',
                       help='Display FPS counter')
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {args.model}")
        return 1
    
    logger.info("=" * 80)
    logger.info("RASPBERRY PI 5 OBJECT DETECTION STREAM")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Input size: {args.input_size}x{args.input_size}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"IoU threshold: {args.iou}")
    logger.info("=" * 80)
    
    # Initialize detector
    try:
        detector = ONNXDetector(
            str(model_path),
            input_size=args.input_size,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1
    
    # Initialize camera
    cap = get_camera(args.camera, args.width, args.height)
    if cap is None:
        return 1
    
    # FPS tracking
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0
    
    logger.info("\nStarting detection stream...")
    logger.info("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break
            
            # Run detection
            detections, inference_time = detector.detect(frame)
            
            # Draw detections
            result_frame = draw_detections(frame, detections)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Draw FPS and info
            if args.fps_display:
                info_text = f"FPS: {fps_display:.1f} | Inference: {inference_time*1000:.1f}ms | Detections: {len(detections)}"
                cv2.putText(
                    result_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            
            # Display frame
            cv2.imshow('Object Detection', result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, result_frame)
                logger.info(f"Saved frame to: {filename}")
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released, cleanup complete")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
