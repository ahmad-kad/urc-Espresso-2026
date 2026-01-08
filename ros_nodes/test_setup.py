#!/usr/bin/env python3
"""
Test script to verify ROS2 camera detector setup
Tests model loading, inference, and ROS2 connectivity
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config.manager import load_config
from core.models.detector import ObjectDetector
from core.models.onnx_detector import ONNXDetector
from utils.logger_config import get_logger

logger = get_logger(__name__)


def test_model_loading(model_path: str, config_path: str = None):
    """Test if model can be loaded"""
    print(f"\n{'='*60}")
    print("TEST 1: Model Loading")
    print(f"{'='*60}")

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ FAILED: Model file not found: {model_path}")
        return False

    print(f"✓ Model file exists: {model_path}")

    # Load config
    if config_path:
        config_name = config_path.replace(".yaml", "").replace("configs/", "")
        config = load_config(config_name)
    else:
        config = load_config("robotics")

    print(f"✓ Config loaded: {config.get('project', {}).get('name', 'default')}")

    # Try loading model
    try:
        model_ext = model_path.suffix.lower()
        if model_ext == ".onnx":
            print("Loading ONNX model...")
            detector = ONNXDetector(str(model_path), config)
            print("✓ ONNX model loaded successfully")
        elif model_ext == ".pt":
            print("Loading PyTorch model...")
            config["model"]["pretrained_weights"] = str(model_path)
            detector = ObjectDetector(config)
            print("✓ PyTorch model loaded successfully")
        else:
            print(f"❌ FAILED: Unsupported format: {model_ext}")
            return False

        return True, detector

    except Exception as e:
        print(f"❌ FAILED: Error loading model: {e}")
        return False, None


def test_inference(detector, test_image_path: str = None):
    """Test inference on a test image"""
    print(f"\n{'='*60}")
    print("TEST 2: Inference")
    print(f"{'='*60}")

    # Create test image if not provided
    if test_image_path and Path(test_image_path).exists():
        image = cv2.imread(test_image_path)
        print(f"✓ Loaded test image: {test_image_path}")
    else:
        # Create dummy image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("✓ Created dummy test image (640x480)")

    try:
        # Run inference
        print("Running inference...")
        if isinstance(detector, ONNXDetector):
            results = detector.predict(image)
            print(f"✓ Inference completed: {len(results)} detections")
            if results:
                print(
                    f"  First detection: class={results[0]['class_id']}, "
                    f"conf={results[0]['confidence']:.2f}"
                )
        else:
            results = detector.predict(image, verbose=False)
            print("✓ Inference completed")
            if hasattr(results, "__iter__") and len(results) > 0:
                result = results[0]
                if hasattr(result, "boxes") and len(result.boxes) > 0:
                    print(f"  Detections: {len(result.boxes)}")

        return True

    except Exception as e:
        print(f"❌ FAILED: Inference error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ros2_imports():
    """Test if ROS2 dependencies are available"""
    print(f"\n{'='*60}")
    print("TEST 3: ROS2 Dependencies")
    print(f"{'='*60}")

    try:
        import rclpy

        print("✓ rclpy imported")
    except ImportError as e:
        print(f"❌ FAILED: rclpy not available: {e}")
        return False

    try:
        from sensor_msgs.msg import Image

        print("✓ sensor_msgs imported")
    except ImportError as e:
        print(f"❌ FAILED: sensor_msgs not available: {e}")
        return False

    try:
        from cv_bridge import CvBridge

        print("✓ cv_bridge imported")
    except ImportError as e:
        print(f"❌ FAILED: cv_bridge not available: {e}")
        return False

    try:
        from vision_msgs.msg import Detection2D, Detection2DArray

        print("✓ vision_msgs imported")
    except ImportError as e:
        print(f"⚠ WARNING: vision_msgs not available: {e}")
        print("  (This is optional - annotated images will still work)")
        # Don't fail the test, vision_msgs is optional
        return True

    return True


def test_ros2_node_import():
    """Test if ROS node can be imported"""
    print(f"\n{'='*60}")
    print("TEST 4: ROS Node Import")
    print(f"{'='*60}")

    try:
        from ros_nodes.camera_detector_node import CameraDetectorNode

        print("✓ CameraDetectorNode can be imported")
        return True
    except Exception as e:
        print(f"❌ FAILED: Cannot import CameraDetectorNode: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(description="Test ROS2 camera detector setup")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model file (.pt, .onnx)"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML file"
    )
    parser.add_argument(
        "--test-image", type=str, default=None, help="Path to test image (optional)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ROS2 Camera Detector Setup Test")
    print("=" * 60)

    results = {}

    # Test 1: Model loading
    success, detector = test_model_loading(args.model, args.config)
    results["model_loading"] = success
    if not success:
        print("\n❌ Model loading failed. Cannot continue.")
        return 1

    # Test 2: Inference
    results["inference"] = test_inference(detector, args.test_image)

    # Test 3: ROS2 dependencies
    results["ros2_deps"] = test_ros2_imports()

    # Test 4: ROS node import
    results["ros_node"] = test_ros2_node_import()

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    if all_passed:
        print("\n✓ All tests passed! Setup is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
