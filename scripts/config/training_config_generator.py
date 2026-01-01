#!/usr/bin/env python3
"""
Training configuration generator for YOLO models
Generates optimized training configurations for different architectures
"""

from pathlib import Path
from typing import Dict, List

import yaml


class ConfidenceTrainingConfig:
    """Generate confidence-optimized training configurations for all architectures"""

    def __init__(self):
        self.base_config = {
            "task": "detect",
            "mode": "train",
            "data": "enhanced_dataset/data.yaml",
            "epochs": 100,
            "patience": 20,
            "batch": 16,
            "imgsz": 416,
            "save": True,
            "save_period": 10,
            "cache": "disk",
            "device": "0",
            "workers": 4,
            "project": "output",
            "name": "confidence",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",
            "verbose": True,
            "seed": 42,
            "deterministic": True,
            "single_cls": False,
            "rect": False,
            "cos_lr": True,
            "close_mosaic": 15,
            "resume": False,
            "amp": True,
            "fraction": 1.0,
            "profile": False,
            "freeze": None,
            "multi_scale": True,
            "compile": False,
            "overlap_mask": True,
            "mask_ratio": 4,
            "dropout": 0.1,
            "val": True,
            "split": "val",
            "save_json": False,
            "conf": 0.1,
            "iou": 0.7,
            "max_det": 300,
            "half": False,
            "dnn": False,
            "plots": True,
        }

    def generate_yolov8s_config(self) -> Dict:
        """Generate YOLOv8s confidence configuration"""
        config = self.base_config.copy()
        config.update(
            {
                "model": "models/yolov8s.pt",
                "name": "yolov8s_confidence",
                "batch": 16,
                "imgsz": 416,
                "lr0": 0.001,
                "lrf": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "warmup_epochs": 3.0,
                "warmup_momentum": 0.8,
                "warmup_bias_lr": 0.1,
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
                "pose": 12.0,
                "kobj": 1.0,
                "nbs": 64,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
            }
        )
        return config

    def generate_yolov8m_config(self) -> Dict:
        """Generate YOLOv8m confidence configuration"""
        config = self.base_config.copy()
        config.update(
            {
                "model": "models/yolov8m.pt",
                "name": "yolov8m_confidence",
                "batch": 12,
                "imgsz": 416,
                "lr0": 0.001,
                "lrf": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "warmup_epochs": 3.0,
                "warmup_momentum": 0.8,
                "warmup_bias_lr": 0.1,
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
                "pose": 12.0,
                "kobj": 1.0,
                "nbs": 64,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
            }
        )
        return config

    def generate_yolov8l_config(self) -> Dict:
        """Generate YOLOv8l confidence configuration"""
        config = self.base_config.copy()
        config.update(
            {
                "model": "models/yolov8l.pt",
                "name": "yolov8l_confidence",
                "batch": 8,
                "imgsz": 416,
                "lr0": 0.001,
                "lrf": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "warmup_epochs": 3.0,
                "warmup_momentum": 0.8,
                "warmup_bias_lr": 0.1,
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
                "pose": 12.0,
                "kobj": 1.0,
                "nbs": 64,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
            }
        )
        return config

    def generate_all_configs(self, output_dir: str = "configs/training") -> List[str]:
        """Generate all confidence configurations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        configs = {
            "yolov8s_confidence.yaml": self.generate_yolov8s_config(),
            "yolov8m_confidence.yaml": self.generate_yolov8m_config(),
            "yolov8l_confidence.yaml": self.generate_yolov8l_config(),
        }

        for filename, config in configs.items():
            output_path = output_dir / filename
            with open(output_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Generated {output_path}")

        return list(configs.keys())
