"""
Image Preprocessing Component
Handles image enhancement for ML inference
"""

from typing import Any, Dict

import cv2
import numpy as np

from utils.logger_config import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Component for image preprocessing
    Applies lighting normalization, CLAHE, gamma correction, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image preprocessor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocess_config = config.get("preprocessing", {})
        self.edge_config = config.get("edge", {})

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to image

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image
        """
        if not self.preprocess_config.get("enabled", False):
            return image

        processed = image.copy()

        # Adaptive brightness normalization
        if self.preprocess_config.get("adaptive_brightness", False):
            processed = self._apply_adaptive_brightness(processed)

        # CLAHE (skip if optimizing for edge devices)
        if self.preprocess_config.get("clahe", False) and not self.edge_config.get(
            "optimize_memory", False
        ):
            processed = self._apply_clahe(processed)

        # Gamma correction
        if self.preprocess_config.get("gamma_correction", False):
            processed = self._apply_gamma_correction(processed)

        return processed

    def _apply_adaptive_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive brightness normalization"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        brightness_low = self.preprocess_config.get("brightness_threshold_low", 30)
        brightness_high = self.preprocess_config.get("brightness_threshold_high", 220)

        if mean_brightness < brightness_low:
            alpha = 1.0 + (brightness_low - mean_brightness) / 255.0
            image = cv2.convertScaleAbs(image, alpha=min(alpha, 1.5), beta=10)
        elif mean_brightness > brightness_high:
            alpha = 1.0 - (mean_brightness - brightness_high) / 255.0
            image = cv2.convertScaleAbs(image, alpha=max(alpha, 0.7), beta=0)

        return image

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement"""
        clahe = cv2.createCLAHE(
            clipLimit=self.preprocess_config.get("clahe_clip_limit", 2.0),
            tileGridSize=(
                self.preprocess_config.get("clahe_tile_size", 8),
                self.preprocess_config.get("clahe_tile_size", 8),
            ),
        )
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        processed[:, :, 0] = clahe.apply(processed[:, :, 0])
        return cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

    def _apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction"""
        gamma = self.preprocess_config.get("gamma", 1.2)
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)
