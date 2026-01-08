"""
ArUco Detector Component
Handles all ArUco marker detection logic, separated from ML inference
This is a large component that encapsulates all ArUco detection, filtering, and enhancement logic
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger_config import get_logger

logger = get_logger(__name__)


class ArUcoDetectorComponent:
    """
    Component for ArUco marker detection
    Handles dictionary setup, parameter configuration, multi-pass detection, filtering, and motion tracking integration
    """

    def __init__(self, config: Dict[str, Any], motion_tracker=None):
        """
        Initialize ArUco detector component

        Args:
            config: Configuration dictionary
            motion_tracker: Optional MotionTracker instance for motion handling
        """
        self.config = config
        self.motion_tracker = motion_tracker
        self.aruco_config = config.get("aruco", {})
        self.enabled = self.aruco_config.get("enabled", True)

        if not self.enabled:
            logger.info("ArUco detection disabled in config")
            return

        # Motion handling configuration
        self.motion_blur_deblur = self.aruco_config.get("motion_blur_deblur", True)
        self.motion_adaptive = self.aruco_config.get("motion_adaptive", True)

        # Setup dictionaries and parameters
        self._setup_dictionaries()
        self._setup_parameters()
        self._setup_detector_api()

        # Get ArUcoTag class ID
        self._setup_class_id()

        logger.info(
            f"ArUco Detector Component initialized: {len(self.aruco_dicts)} dictionaries, class_id={self.aruco_class_id}"
        )

    def _setup_dictionaries(self) -> None:
        """Setup ArUco dictionaries (single or multi-dictionary mode)"""
        use_multi_dict = self.aruco_config.get("multi_dictionary", True)

        dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        }

        if use_multi_dict:
            self.aruco_dicts = {}
            dict_types_to_try = [
                "DICT_4X4_50",
                "DICT_5X5_50",
                "DICT_6X6_50",
                "DICT_7X7_50",
            ]

            for dict_type in dict_types_to_try:
                if dict_type in dict_map:
                    dict_id = dict_map[dict_type]
                    try:
                        if hasattr(cv2.aruco, "getPredefinedDictionary"):
                            self.aruco_dicts[dict_type] = (
                                cv2.aruco.getPredefinedDictionary(dict_id)
                            )
                        else:
                            self.aruco_dicts[dict_type] = cv2.aruco.Dictionary_get(
                                dict_id
                            )
                        logger.info(f"Loaded ArUco dictionary: {dict_type}")
                    except Exception as e:
                        logger.warning(f"Failed to load dictionary {dict_type}: {e}")

            self.aruco_dict = self.aruco_dicts.get(
                "DICT_4X4_50",
                list(self.aruco_dicts.values())[0] if self.aruco_dicts else None,
            )
        else:
            aruco_dict_type = self.aruco_config.get("dictionary", "DICT_4X4_50")
            aruco_dict_id = dict_map.get(aruco_dict_type, cv2.aruco.DICT_4X4_50)

            if hasattr(cv2.aruco, "getPredefinedDictionary"):
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
            else:
                self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)

            self.aruco_dicts = {aruco_dict_type: self.aruco_dict}

    def _setup_parameters(self) -> None:
        """Setup ArUco detector parameters optimized for wide dynamic range"""
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Adaptive threshold parameters
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 45
        self.aruco_params.adaptiveThreshWinSizeStep = 3
        self.aruco_params.adaptiveThreshConstant = 7

        # Corner refinement
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1

        # Detection parameters
        self.aruco_params.minMarkerPerimeterRate = 0.01
        self.aruco_params.maxMarkerPerimeterRate = 10.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.03
        self.aruco_params.errorCorrectionRate = 0.8
        self.aruco_params.minDistanceToBorder = 3
        self.aruco_params.markerBorderBits = 1

        # Perspective removal
        self.aruco_params.perspectiveRemovePixelPerCell = 8
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.15

        # Additional parameters
        if hasattr(self.aruco_params, "minOtsuStdDev"):
            self.aruco_params.minOtsuStdDev = 3.0
        if hasattr(self.aruco_params, "maxErroneousBitsInBorderRate"):
            self.aruco_params.maxErroneousBitsInBorderRate = 0.5

    def _setup_detector_api(self) -> None:
        """Setup ArUco detector API (OpenCV 4.7+ or legacy)"""
        self.use_aruco_detector = hasattr(cv2.aruco, "ArucoDetector")
        if self.use_aruco_detector:
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, self.aruco_params
            )
            logger.info("Using OpenCV 4.7+ ArucoDetector API")
        else:
            self.aruco_detector = None
            logger.info("Using legacy OpenCV ArUco API")

    def _setup_class_id(self) -> None:
        """Setup ArUcoTag class ID, adding to class list if needed"""
        class_names = self.config.get("data", {}).get("classes", [])

        if "ArUcoTag" not in class_names:
            class_names.append("ArUcoTag")
            self.config["data"]["classes"] = class_names
            self.config["data"]["num_classes"] = len(class_names)
            logger.info(f"Added 'ArUcoTag' to class list. New classes: {class_names}")

        try:
            self.aruco_class_id = class_names.index("ArUcoTag")
            logger.info(
                f"ArUcoTag class_id: {self.aruco_class_id} (out of {len(class_names)} classes)"
            )
        except ValueError:
            self.aruco_class_id = len(class_names) - 1
            logger.warning(
                f"ArUcoTag not found in class names after adding, using class_id={self.aruco_class_id}"
            )

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect ArUco tags in image

        Args:
            image: Input image in BGR format

        Returns:
            List of detection dictionaries with keys: bbox, class_id, confidence, marker_id
        """
        if not self.enabled:
            return []

        detections = []
        original_height, original_width = image.shape[:2]

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Optional downscaling for large images
            scale_factor = 1.0
            max_dimension = self.aruco_config.get("max_detection_size", 1280)
            if max_dimension > 0 and max(gray.shape[:2]) > max_dimension:
                scale_factor = max_dimension / max(gray.shape[:2])
                new_width = int(gray.shape[1] * scale_factor)
                new_height = int(gray.shape[0] * scale_factor)
                gray = cv2.resize(
                    gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )

            # Multi-pass detection
            all_detection_ids = set()
            corners, ids, rejected = self._detect_multi_pass(gray, all_detection_ids)

            # Process detected markers
            if ids is not None and len(ids) > 0:
                for i, marker_id in enumerate(ids.flatten()):
                    corner_points = corners[i][0]

                    # Scale coordinates back if downscaled
                    if scale_factor < 1.0:
                        corner_points = corner_points / scale_factor

                    # Filter marker
                    (
                        should_accept,
                        width,
                        height,
                        bbox_area,
                        aspect_ratio,
                        filter_reason,
                    ) = self._filter_marker(
                        corner_points, marker_id, original_width, original_height, image
                    )

                    if not should_accept:
                        continue

                    # Get bounding box
                    x_coords = corner_points[:, 0]
                    y_coords = corner_points[:, 1]
                    x_min = float(np.min(x_coords))
                    y_min = float(np.min(y_coords))
                    x_max = float(np.max(x_coords))
                    y_max = float(np.max(y_coords))

                    # Clamp to image bounds
                    x_min = max(0, min(x_min, original_width - 1))
                    y_min = max(0, min(y_min, original_height - 1))
                    x_max = max(0, min(x_max, original_width - 1))
                    y_max = max(0, min(y_max, original_height - 1))

                    bbox = [x_min, y_min, x_max, y_max]

                    # Motion handling: validate position and consensus
                    if self.motion_tracker:
                        if not self.motion_tracker.validate_position(marker_id, bbox):
                            continue
                        if not self.motion_tracker.check_consensus(
                            marker_id, bbox, 0.8
                        ):
                            continue
                        self.motion_tracker.update_tracker(marker_id, bbox)

                    # Calculate confidence
                    image_area = image.shape[0] * image.shape[1]
                    area_ratio = bbox_area / image_area
                    confidence = self._calculate_confidence(area_ratio, aspect_ratio)

                    detections.append(
                        {
                            "bbox": bbox,
                            "class_id": self.aruco_class_id,
                            "confidence": confidence,
                            "marker_id": int(marker_id),
                        }
                    )

                    logger.info(
                        f"Detected ArUco tag: ID={marker_id}, bbox={bbox}, conf={confidence:.3f}"
                    )

        except Exception as e:
            logger.error(f"ArUco detection failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())

        # Apply confidence decay if motion tracker available
        if self.motion_tracker:
            detections = self.motion_tracker.apply_confidence_decay(detections)

        return detections

    def _detect_multi_pass(self, gray: np.ndarray, all_detection_ids: set) -> Tuple:
        """Multi-pass detection with optimization"""
        # Pass 1: Multi-dictionary detection
        detection_params = (
            self._get_motion_adaptive_params()
            if self.motion_adaptive
            else self.aruco_params
        )
        corners, ids, rejected = self._detect_multi_dict(gray, detection_params)

        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                all_detection_ids.add(int(marker_id))
        else:
            corners, ids = None, None

        found_markers = len(ids) if ids is not None else 0
        skip_enhancement = found_markers >= 2

        # Pass 2: Enhanced image (if needed)
        if not skip_enhancement:
            enhanced_gray = self._enhance_image(gray)
            corners2, ids2, _ = self._detect_single_pass(
                enhanced_gray, self.aruco_params
            )

            if ids2 is not None and len(ids2) > 0:
                for i, marker_id in enumerate(ids2.flatten()):
                    if int(marker_id) not in all_detection_ids:
                        if ids is None:
                            ids, corners = ids2, corners2
                        else:
                            ids = np.concatenate([ids, ids2])
                            corners = np.concatenate([corners, corners2])
                        all_detection_ids.add(int(marker_id))

        return corners, ids, rejected

    def _detect_single_pass(
        self,
        gray: np.ndarray,
        params: cv2.aruco.DetectorParameters,
        aruco_dict: Optional[Any] = None,
    ) -> Tuple:
        """Single-pass ArUco detection"""
        if aruco_dict is None:
            aruco_dict = self.aruco_dict

        if self.use_aruco_detector:
            temp_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            return temp_detector.detectMarkers(gray)
        else:
            return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    def _detect_multi_dict(
        self, gray: np.ndarray, params: cv2.aruco.DetectorParameters
    ) -> Tuple:
        """Detect using all configured dictionaries"""
        if len(self.aruco_dicts) <= 1:
            return self._detect_single_pass(gray, params)

        all_corners = []
        all_ids = []
        all_rejected = []

        dict_offsets = {
            "DICT_4X4_50": 0,
            "DICT_4X4_100": 0,
            "DICT_4X4_250": 0,
            "DICT_4X4_1000": 0,
            "DICT_5X5_50": 10000,
            "DICT_5X5_100": 10000,
            "DICT_6X6_50": 20000,
            "DICT_6X6_100": 20000,
            "DICT_7X7_50": 30000,
            "DICT_7X7_100": 30000,
        }

        for dict_name, aruco_dict in self.aruco_dicts.items():
            try:
                dict_corners, dict_ids, dict_rejected = self._detect_single_pass(
                    gray, params, aruco_dict
                )

                if dict_ids is not None and len(dict_ids) > 0:
                    dict_offset = dict_offsets.get(dict_name, 0)
                    for i, marker_id in enumerate(dict_ids.flatten()):
                        unique_id = int(marker_id) + dict_offset
                        all_corners.append(dict_corners[i])
                        all_ids.append(unique_id)

                if dict_rejected is not None and len(dict_rejected) > 0:
                    all_rejected.extend(dict_rejected)
            except Exception as e:
                logger.debug(f"Error detecting with {dict_name}: {e}")

        if all_ids:
            ids = np.array(all_ids).reshape(-1, 1)
            corners = all_corners
        else:
            ids, corners = None, None

        rejected = all_rejected if all_rejected else None
        return corners, ids, rejected

    def _enhance_image(self, gray: np.ndarray) -> np.ndarray:
        """Enhance image for better ArUco detection"""
        enhanced = gray.copy()

        if self.motion_blur_deblur:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.3
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        return enhanced

    def _filter_marker(
        self,
        corner_points: np.ndarray,
        marker_id: int,
        original_width: int,
        original_height: int,
        image: np.ndarray,
    ) -> Tuple:
        """Unified filtering function for markers"""
        x_coords = corner_points[:, 0]
        y_coords = corner_points[:, 1]
        x_min = float(np.min(x_coords))
        y_min = float(np.min(y_coords))
        x_max = float(np.max(x_coords))
        y_max = float(np.max(y_coords))

        x_min = max(0, min(x_min, original_width - 1))
        y_min = max(0, min(y_min, original_height - 1))
        x_max = max(0, min(x_max, original_width - 1))
        y_max = max(0, min(y_max, original_height - 1))

        width = x_max - x_min
        height = y_max - y_min
        bbox_area = width * height
        image_area = image.shape[0] * image.shape[1]
        area_ratio = bbox_area / image_area
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)

        # Apply filters
        if bbox_area < 250:
            return (False, width, height, bbox_area, aspect_ratio, "area_too_small")
        if aspect_ratio > 2.0:
            return (
                False,
                width,
                height,
                bbox_area,
                aspect_ratio,
                "aspect_ratio_extreme",
            )
        if width < 6 or height < 6:
            return (False, width, height, bbox_area, aspect_ratio, "size_too_small")
        if area_ratio < 0.00002:
            return (
                False,
                width,
                height,
                bbox_area,
                aspect_ratio,
                "area_ratio_too_small",
            )

        return (True, width, height, bbox_area, aspect_ratio, None)

    def _calculate_confidence(self, area_ratio: float, aspect_ratio: float) -> float:
        """Calculate confidence based on marker size and shape"""
        if area_ratio > 0.1:
            base_confidence = 0.99
        elif area_ratio > 0.05:
            base_confidence = 0.98
        elif area_ratio > 0.01:
            base_confidence = 0.95
        elif area_ratio > 0.002:
            base_confidence = 0.90
        elif area_ratio > 0.0005:
            base_confidence = 0.85
        else:
            base_confidence = 0.80

        if aspect_ratio > 2.0:
            aspect_penalty = 0.85
        elif aspect_ratio > 1.5:
            aspect_penalty = 0.95
        else:
            aspect_penalty = 1.0

        confidence = base_confidence * aspect_penalty
        return max(0.75, min(0.99, confidence))

    def _get_motion_adaptive_params(self) -> cv2.aruco.DetectorParameters:
        """Get parameters optimized for current motion level"""
        if not self.motion_adaptive or not self.motion_tracker:
            return self.aruco_params

        params = self._copy_params(self.aruco_params)
        avg_velocity = (
            sum(self.motion_tracker.marker_velocities.values())
            / len(self.motion_tracker.marker_velocities)
            if self.motion_tracker.marker_velocities
            else 0.0
        )

        if avg_velocity > 2.0:
            params.errorCorrectionRate = 0.9
            params.polygonalApproxAccuracyRate = 0.05
        else:
            params.errorCorrectionRate = 0.8
            params.polygonalApproxAccuracyRate = 0.03

        return params

    def _copy_params(
        self, source: cv2.aruco.DetectorParameters
    ) -> cv2.aruco.DetectorParameters:
        """Copy ArUco detector parameters"""
        dest = cv2.aruco.DetectorParameters()
        param_attrs = [
            "adaptiveThreshWinSizeMin",
            "adaptiveThreshWinSizeMax",
            "adaptiveThreshWinSizeStep",
            "adaptiveThreshConstant",
            "minMarkerPerimeterRate",
            "maxMarkerPerimeterRate",
            "polygonalApproxAccuracyRate",
            "minCornerDistanceRate",
            "minDistanceToBorder",
            "minMarkerDistanceRate",
            "cornerRefinementMethod",
            "cornerRefinementWinSize",
            "cornerRefinementMaxIterations",
            "cornerRefinementMinAccuracy",
            "markerBorderBits",
            "perspectiveRemovePixelPerCell",
            "perspectiveRemoveIgnoredMarginPerCell",
            "maxErroneousBitsInBorderRate",
            "minOtsuStdDev",
            "errorCorrectionRate",
        ]
        for attr in param_attrs:
            if hasattr(source, attr):
                try:
                    setattr(dest, attr, getattr(source, attr))
                except (AttributeError, TypeError):
                    pass
        return dest

