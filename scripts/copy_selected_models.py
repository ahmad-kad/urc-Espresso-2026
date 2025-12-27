#!/usr/bin/env python3
"""
Copy selected models to deployment folder
"""

import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_onnx_model(model_name_patterns, search_dir="output/onnx"):
    """Find ONNX model files matching the patterns"""
    search_path = Path(search_dir)
    found_files = []
    
    if not search_path.exists():
        logger.warning(f"Search directory {search_dir} does not exist")
        # Try to find in current directory or model directories
        search_path = Path(".")
    
    # Search for ONNX files
    for pattern in model_name_patterns:
        # Try different naming patterns
        patterns_to_try = [
            f"*{pattern}*.onnx",
            f"{pattern}*.onnx",
            f"*{pattern}_*.onnx",
            f"*{pattern}_224.onnx",
        ]
        
        for pat in patterns_to_try:
            for file_path in search_path.rglob(pat):
                if file_path.is_file() and file_path.suffix == '.onnx':
                    found_files.append(file_path)
                    logger.info(f"Found: {file_path}")
                    break
    
    return found_files


def copy_models_to_deployment(models_to_copy, deployment_dir="output/deployment_models"):
    """Copy selected models to deployment directory"""
    deployment_path = Path(deployment_dir)
    deployment_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("COPYING SELECTED MODELS TO DEPLOYMENT FOLDER")
    logger.info("=" * 80)
    
    # Model name patterns to search for
    model_patterns = [
        "yolov8s_cbam_confidence_cbam",
        "yolov8m_confidence",
        "efficientnet_confidence",
        "yolov8s_confidence",
        "mobilenet_224",
        "yolov8n_224"
    ]
    
    logger.info(f"\nSearching for models: {', '.join(model_patterns)}")
    
    # Search in multiple locations
    search_dirs = ["output/onnx", "output/models", "."]
    
    all_found = []
    for search_dir in search_dirs:
        found = find_onnx_model(model_patterns, search_dir)
        all_found.extend(found)
    
    # Remove duplicates
    unique_files = list(set(all_found))
    
    if not unique_files:
        logger.error("No ONNX model files found!")
        logger.info("Trying to find PyTorch models instead...")
        
        # Try to find PyTorch models and convert them
        pytorch_patterns = [
            "yolov8s_cbam_confidence_cbam",
            "yolov8m_confidence",
            "efficientnet_confidence",
            "yolov8s_confidence",
            "mobilenet_224",
            "yolov8n_224"
        ]
        
        pytorch_files = []
        for pattern in pytorch_patterns:
            for search_dir in ["output/models", "."]:
                search_path = Path(search_dir)
                for file_path in search_path.rglob(f"*{pattern}*/weights/best.pt"):
                    if file_path.is_file():
                        pytorch_files.append(file_path)
                        logger.info(f"Found PyTorch model: {file_path}")
        
        if pytorch_files:
            logger.info(f"\nFound {len(pytorch_files)} PyTorch models")
            logger.info("Copying PyTorch models (you may need to convert to ONNX later)")
            for pt_file in pytorch_files:
                dest_file = deployment_path / pt_file.name
                shutil.copy2(pt_file, dest_file)
                logger.info(f"Copied: {pt_file.name} -> {dest_file}")
        
        return len(pytorch_files)
    
    logger.info(f"\nFound {len(unique_files)} ONNX model files")
    
    # Copy files
    copied_count = 0
    for src_file in unique_files:
        dest_file = deployment_path / src_file.name
        
        try:
            shutil.copy2(src_file, dest_file)
            logger.info(f"✅ Copied: {src_file.name}")
            logger.info(f"   From: {src_file}")
            logger.info(f"   To: {dest_file}")
            copied_count += 1
        except Exception as e:
            logger.error(f"❌ Failed to copy {src_file.name}: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"COPY COMPLETE: {copied_count}/{len(unique_files)} models copied")
    logger.info(f"Deployment folder: {deployment_path.absolute()}")
    logger.info("=" * 80)
    
    return copied_count


def main():
    """Main function"""
    models_to_copy = [
        "yolov8s_cbam_confidence_cbam",
        "yolov8m_confidence",
        "efficientnet_confidence",
        "yolov8s_confidence",
        "mobilenet_224",
        "yolov8n_224"
    ]
    
    copied = copy_models_to_deployment(models_to_copy)
    
    if copied > 0:
        return 0
    else:
        logger.error("No models were copied!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
