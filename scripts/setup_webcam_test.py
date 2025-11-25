#!/usr/bin/env python3
"""
Setup and environment check for webcam testing
Verifies dependencies and provides setup instructions
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def check_camera():
    """Check if camera is accessible"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print("✓ Camera (device 0) is accessible")
                return True
            else:
                print("✗ Camera found but cannot capture frames")
                return False
        else:
            print("✗ Camera (device 0) is not accessible")
            return False
    except Exception as e:
        print(f"✗ Camera check failed: {e}")
        return False

def check_models():
    """Check for available trained models"""
    model_paths = [
        "runs/baseline/yolov8s_baseline_hammer_detection/weights/best.pt",
        "runs/cbam/yolov8s_cbam_hammer_detection/weights/best.pt"
    ]

    found_models = []
    for path in model_paths:
        if Path(path).exists():
            found_models.append(path)
            print(f"✓ Found trained model: {path}")

    if not found_models:
        print("ℹ No trained models found - will use default YOLOv8s")
        print("   Train models with: python scripts/train.py --data_yaml data/data.yaml")

    return found_models

def main():
    print("="*60)
    print("Hammer Detection Webcam Test Setup Check")
    print("="*60)

    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✓ Python {python_version.major}.{python_version.minor} is compatible")
    else:
        print(f"✗ Python {python_version.major}.{python_version.minor} - requires Python 3.8+")
        return

    print("\nChecking dependencies...")

    # Check required packages
    required_packages = [
        ("torch", "torch"),
        ("cv2", "cv2"),
        ("numpy", "numpy"),
        ("ultralytics", "ultralytics"),
        ("pathlib", "pathlib"),
        ("collections", "collections"),
    ]

    missing_packages = []
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)

    print("\nChecking camera access...")
    camera_ok = check_camera()

    print("\nChecking for trained models...")
    available_models = check_models()

    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)

    all_good = True

    if missing_packages:
        all_good = False
        print("❌ MISSING PACKAGES:")
        for package in missing_packages:
            if package == "cv2":
                print(f"   pip install opencv-python")
            else:
                print(f"   pip install {package}")
        print()

    if not camera_ok:
        all_good = False
        print("❌ CAMERA ISSUES:")
        print("   - Check camera permissions")
        print("   - Try different camera device: python webcam_test.py --camera 1")
        print("   - Ensure no other apps are using the camera")
        print()

    if all_good:
        print("✅ SETUP COMPLETE - Ready to test!")
        print("\nQuick start commands:")
        print("  python webcam_test.py                           # Test with default model")
        if available_models:
            print(f"  python webcam_test.py --model {available_models[0]}  # Test trained model")
        print("  python webcam_test.py --help                    # Show all options")
    else:
        print("❌ SETUP INCOMPLETE - Please fix issues above")

    print("\nFor detailed instructions, see: WEBCAM_TEST_README.md")

if __name__ == "__main__":
    main()
