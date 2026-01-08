"""
Utility modules for YOLO AI Camera project
"""

# Data utilities
from .data_utils import (
    get_image_paths,
    get_image_paths_from_dir,
    load_data_config,
    load_yolo_labels,
)

# Detection utilities
from .detection_utils import (
    draw_detections,
    filter_boxes_by_class,
    non_max_suppression,
    post_process_yolov8,
    scale_boxes_to_original,
    xywh2xyxy,
)

# Metrics
from .metrics import calculate_iou

# Output utilities
from .output_utils import (
    format_metrics_table,
    print_section_header,
    save_benchmark_results,
    save_debug_info,
    save_evaluation_results,
)

# Visualization
from .visualization import (
    create_detection_mosaic,
    plot_confusion_matrix,
    plot_performance_comparison,
    plot_prediction_results,
    plot_realtime_performance,
    plot_training_history,
)

__version__ = "0.1.0"

# Explicitly define what's available from this module
__all__ = [
    # Data utilities
    "load_data_config",
    "get_image_paths",
    "get_image_paths_from_dir",
    "load_yolo_labels",
    # Detection utilities
    "xywh2xyxy",
    "non_max_suppression",
    "post_process_yolov8",
    "scale_boxes_to_original",
    "filter_boxes_by_class",
    "draw_detections",
    # Metrics
    "calculate_iou",
    # Output utilities
    "save_benchmark_results",
    "save_evaluation_results",
    "save_debug_info",
    "format_metrics_table",
    "print_section_header",
    # Visualization
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_prediction_results",
    "plot_performance_comparison",
    "create_detection_mosaic",
    "plot_realtime_performance",
]
