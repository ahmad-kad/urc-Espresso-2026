"""
Visualization and plotting utilities for object detection
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_training_history(logs_path: str, output_path: Optional[str] = None) -> str:
    """
    Plot training history from YOLO logs
    """

    try:
        # Read training results CSV
        results_csv = Path(logs_path) / "results.csv"
        if not results_csv.exists():
            logger.warning(f"Training results not found: {results_csv}")
            return ""

        df = pd.read_csv(results_csv)

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Training History", fontsize=16)

        # Plot metrics
        metrics = [
            ("epoch", "Epoch"),
            ("train/box_loss", "Box Loss"),
            ("train/cls_loss", "Classification Loss"),
            ("val/box_loss", "Validation Box Loss"),
            ("val/cls_loss", "Validation Classification Loss"),
            ("metrics/mAP50(B)", "mAP@50"),
        ]

        for i, (col, title) in enumerate(metrics):
            ax = axes.flat[i]
            if col in df.columns:
                ax.plot(df["epoch"], df[col], marker="o", markersize=2)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Training history plot saved: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    except Exception as e:
        logger.error(f"Could not plot training history: {str(e)}")
        return ""


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    normalize: bool = True,
) -> str:
    """
    Plot confusion matrix
    """

    try:
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Confusion matrix saved: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    except Exception as e:
        logger.error(f"Could not plot confusion matrix: {str(e)}")
        return ""


def plot_prediction_results(
    image: np.ndarray,
    detections: List[Dict],
    class_names: List[str],
    output_path: Optional[str] = None,
    conf_threshold: float = 0.5,
) -> str:
    """
    Plot detection results on image
    """

    try:
        img = image.copy()

        # Generate colors for different classes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))

        for detection in detections:
            if detection["confidence"] < conf_threshold:
                continue

            # Extract bounding box
            x1, y1, x2, y2 = detection["bbox"]
            class_id = int(detection["class"])
            conf = detection["confidence"]

            # Get class name and color
            class_name = (
                class_names[class_id]
                if class_id < len(class_names)
                else f"class_{class_id}"
            )
            color = tuple(int(c * 255) for c in colors[class_id][:3])

            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Draw background rectangle for text
            cv2.rectangle(
                img,
                (int(x1), int(y1) - text_height - 5),
                (int(x1) + text_width, int(y1)),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            logger.info(f"Detection results saved: {output_path}")
            return output_path
        else:
            # Display image
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
            return ""

    except Exception as e:
        logger.error(f"Could not plot detection results: {str(e)}")
        return ""


def plot_performance_comparison(
    models_results: Dict[str, Dict], output_path: Optional[str] = None
) -> str:
    """
    Plot performance comparison between models
    """

    try:
        model_names = list(models_results.keys())
        metrics = ["mAP@50", "mAP@50:95", "Precision", "Recall"]

        # Extract metrics
        metric_data = {}
        for metric in metrics:
            metric_data[metric] = []
            for model_name in model_names:
                if "metrics" in models_results[model_name]:
                    value = models_results[model_name]["metrics"].get(
                        metric.replace("@", ""), 0
                    )
                    metric_data[metric].append(value)
                else:
                    metric_data[metric].append(0)

        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Model Performance Comparison", fontsize=16)

        for i, metric in enumerate(metrics):
            ax = axes.flat[i]
            bars = ax.bar(
                range(len(model_names)),
                metric_data[metric],
                color=plt.cm.Set3(np.linspace(0, 1, len(model_names))),
            )
            ax.set_title(metric)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    ".3f",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Performance comparison saved: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    except Exception as e:
        logger.error(f"Could not plot performance comparison: {str(e)}")
        return ""


def create_detection_mosaic(
    images: List[np.ndarray],
    detections: List[List[Dict]],
    class_names: List[str],
    grid_size: Tuple[int, int] = (2, 3),
    output_path: Optional[str] = None,
) -> str:
    """
    Create a mosaic of detection results
    """

    try:
        rows, cols = grid_size
        num_images = min(len(images), rows * cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(num_images):
            ax = axes[i]
            img = images[i].copy()

            # Draw detections
            if i < len(detections) and detections[i]:
                for det in detections[i]:
                    x1, y1, x2, y2 = det["bbox"]
                    class_id = int(det["class"])
                    conf = det["confidence"]

                    # Simple rectangle for mosaic
                    rect = plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
                    )
                    ax.add_patch(rect)

                    # Add label
                    class_name = (
                        class_names[class_id]
                        if class_id < len(class_names)
                        else f"class_{class_id}"
                    )
                    ax.text(
                        x1,
                        y1 - 5,
                        f"{class_name}: {conf:.2f}",
                        color="red",
                        fontsize=8,
                        backgroundcolor="white",
                    )

            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title(f"Image {i+1}")

        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Detection mosaic saved: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    except Exception as e:
        logger.error(f"Could not create detection mosaic: {str(e)}")
        return ""


def plot_realtime_performance(
    fps_data: List[float], output_path: Optional[str] = None
) -> str:
    """
    Plot real-time performance metrics
    """

    try:
        plt.figure(figsize=(10, 6))

        # Plot FPS over time
        plt.subplot(1, 2, 1)
        plt.plot(fps_data, marker="o", markersize=2)
        plt.title("FPS Over Time")
        plt.xlabel("Frame")
        plt.ylabel("FPS")
        plt.grid(True, alpha=0.3)

        # Plot FPS distribution
        plt.subplot(1, 2, 2)
        plt.hist(fps_data, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        plt.axvline(np.mean(fps_data), color="red", linestyle="--", label=".1f")
        plt.title("FPS Distribution")
        plt.xlabel("FPS")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Real-time performance plot saved: {output_path}")
            return output_path
        else:
            plt.show()
            return ""

    except Exception as e:
        logger.error(f"Could not plot real-time performance: {str(e)}")
        return ""
