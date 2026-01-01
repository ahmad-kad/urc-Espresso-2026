#!/usr/bin/env python3
"""
Consolidated validation and plotting utilities
Combines functionality from create_class_grid.py, create_detailed_class_grid.py,
show_class_samples.py, analyze_per_class_performance.py, create_model_class_matrix.py,
create_per_class_comparison.py, recall_comparison.py, and create_comparison_tables.py
"""

import logging
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class DatasetVisualizer:
    """Create visualizations of dataset samples and class distributions"""

    def __init__(self, data_path="balanced_data"):
        self.data_path = Path(data_path)
        self.train_images = self.data_path / "train" / "images"
        self.train_labels = self.data_path / "train" / "labels"

        # Class names mapping
        self.class_names = {
            0: "ArUcoTag",
            1: "Bottle",
            2: "BrickHammer",
            3: "OrangeHammer",
        }

    def create_class_grid(
        self, output_path="../output/class_samples_grid.jpg", grid_size=(5, 1)
    ):
        """Create a grid image showing one sample from each class"""

        # Select one representative image for each class
        selected_images = {}

        # Get samples for each class
        for class_id in range(len(self.class_names)):
            class_samples = []

            for label_file in self.train_labels.glob("*.txt"):
                try:
                    with open(label_file, "r") as f:
                        content = f.read().strip()
                        if content and content.split()[0] == str(class_id):
                            image_name = label_file.stem
                            image_path = self.train_images / f"{image_name}.jpg"
                            if image_path.exists() and "_aug_" not in image_name:
                                class_samples.append((image_path, image_name))
                except Exception as e:
                    logger.warning(f"Error processing class {class_id}: {e}")
                    continue

            if class_samples:
                # Select a random representative sample
                selected_images[class_id] = random.choice(class_samples)

        print("Selected representative images:")
        for class_id, (image_path, image_name) in selected_images.items():
            print(f"  Class {class_id} ({self.class_names[class_id]}): {image_name}")

        # Create grid layout
        grid_cols, grid_rows = grid_size
        image_size = (200, 200)  # Size for each thumbnail
        padding = 10
        title_height = 40

        # Calculate total grid size
        grid_width = grid_cols * (image_size[0] + padding) + padding
        grid_height = grid_rows * (image_size[1] + title_height + padding) + padding

        # Create blank canvas
        grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
        draw = ImageDraw.Draw(grid_image)

        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception as e:
            logger.debug(f"Could not load arial font: {e}")
            font = ImageFont.load_default()

        # Add images to grid
        for col, class_id in enumerate(range(len(self.class_names))):
            if class_id in selected_images:
                image_path, image_name = selected_images[class_id]

                # Load and resize image
                try:
                    img = Image.open(image_path)
                    img.thumbnail(image_size, Image.Resampling.LANCZOS)

                    # Create a white background for the thumbnail
                    thumb_bg = Image.new("RGB", image_size, (255, 255, 255))
                    # Center the image on the white background
                    x_offset = (image_size[0] - img.size[0]) // 2
                    y_offset = (image_size[1] - img.size[1]) // 2
                    thumb_bg.paste(img, (x_offset, y_offset))

                    # Calculate position
                    x_pos = col * (image_size[0] + padding) + padding
                    y_pos = padding

                    # Paste thumbnail
                    grid_image.paste(thumb_bg, (x_pos, y_pos))

                    # Add title
                    title = f"{self.class_names[class_id]}\n(Class {class_id})"
                    bbox = draw.textbbox((0, 0), title, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_x = x_pos + (image_size[0] - text_width) // 2
                    text_y = y_pos + image_size[1] + 5

                    draw.text(
                        (text_x, text_y),
                        title,
                        fill=(0, 0, 0),
                        font=font,
                        align="center",
                    )

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    # Draw placeholder
                    x_pos = col * (image_size[0] + padding) + padding
                    y_pos = padding
                    draw.rectangle(
                        [x_pos, y_pos, x_pos + image_size[0], y_pos + image_size[1]],
                        fill=(200, 200, 200),
                        outline=(0, 0, 0),
                    )
                    draw.text(
                        (x_pos + 10, y_pos + image_size[1] // 2),
                        f"No image\n{self.class_names[class_id]}",
                        fill=(0, 0, 0),
                        font=font,
                    )

        # Add main title
        title_font = (
            ImageFont.truetype("arial.ttf", 24)
            if font != ImageFont.load_default()
            else font
        )
        main_title = "Dataset Class Samples - One From Each Class"
        bbox = draw.textbbox((0, 0), main_title, font=title_font)
        text_width = bbox[2] - bbox[0]
        text_x = (grid_width - text_width) // 2
        draw.text(
            (text_x, grid_height - 30), main_title, fill=(0, 0, 0), font=title_font
        )

        # Save the grid image
        grid_image.save(output_path)
        print(f"\nGrid image saved as: {output_path}")
        print(f"Image size: {grid_width}x{grid_height} pixels")

        return output_path

    def show_class_samples(self, output_path="../output/detailed_class_samples.jpg"):
        """Create detailed visualization showing multiple samples per class"""

        # Collect samples for each class
        class_samples = defaultdict(list)

        for label_file in self.train_labels.glob("*.txt"):
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                image_name = label_file.stem
                                image_path = self.train_images / f"{image_name}.jpg"
                                if image_path.exists() and "_aug_" not in image_name:
                                    class_samples[class_id].append(
                                        (image_path, image_name)
                                    )
            except Exception as e:
                logger.warning(f"Error processing class {class_id}: {e}")
                continue

        # Create detailed grid
        samples_per_class = 5
        image_size = (150, 150)
        padding = 15
        class_title_height = 30
        sample_title_height = 20

        # Calculate grid dimensions
        classes_count = len(self.class_names)
        grid_cols = samples_per_class
        grid_rows = classes_count

        grid_width = grid_cols * (image_size[0] + padding) + padding
        grid_height = (
            grid_rows
            * (image_size[1] + class_title_height + sample_title_height + padding)
            + padding
        )

        # Create canvas
        grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
        draw = ImageDraw.Draw(grid_image)

        try:
            class_font = ImageFont.truetype("arial.ttf", 18)
            sample_font = ImageFont.truetype("arial.ttf", 12)
        except Exception as e:
            logger.debug(f"Could not load arial fonts: {e}")
            class_font = ImageFont.load_default()
            sample_font = ImageFont.load_default()

        # Add samples to grid
        for row, class_id in enumerate(sorted(self.class_names.keys())):
            samples = class_samples[class_id][:samples_per_class]

            # Add class title
            class_title = f"Class {class_id}: {self.class_names[class_id]} ({len(class_samples[class_id])} total)"
            title_y = (
                row
                * (image_size[1] + class_title_height + sample_title_height + padding)
                + padding
            )
            draw.text((padding, title_y), class_title, fill=(0, 0, 0), font=class_font)

            # Add samples
            for col, (image_path, image_name) in enumerate(samples):
                try:
                    img = Image.open(image_path)
                    img.thumbnail(image_size, Image.Resampling.LANCZOS)

                    # Create white background
                    thumb_bg = Image.new("RGB", image_size, (255, 255, 255))
                    x_offset = (image_size[0] - img.size[0]) // 2
                    y_offset = (image_size[1] - img.size[1]) // 2
                    thumb_bg.paste(img, (x_offset, y_offset))

                    # Position
                    x_pos = col * (image_size[0] + padding) + padding
                    y_pos = title_y + class_title_height + 5

                    # Paste image
                    grid_image.paste(thumb_bg, (x_pos, y_pos))

                    # Add sample name
                    sample_name = (
                        image_name[:20] + "..." if len(image_name) > 20 else image_name
                    )
                    name_bbox = draw.textbbox((0, 0), sample_name, font=sample_font)
                    name_width = name_bbox[2] - name_bbox[0]
                    name_x = x_pos + (image_size[0] - name_width) // 2
                    name_y = y_pos + image_size[1] + 2

                    draw.text(
                        (name_x, name_y), sample_name, fill=(0, 0, 0), font=sample_font
                    )

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        # Save
        grid_image.save(output_path)
        print(f"Detailed class samples saved as: {output_path}")

        return output_path


class PerformanceAnalyzer:
    """Analyze and visualize model performance metrics"""

    def __init__(self):
        self.class_names = {
            0: "ArUcoTag",
            1: "Bottle",
            2: "BrickHammer",
            3: "OrangeHammer",
        }

    def create_per_class_analysis(self, performance_data=None):
        """Create detailed per-class performance analysis"""

        if performance_data is None:
            # Default data structure for demonstration
            performance_data = {
                "YOLOv8s Baseline": {
                    "ArUcoTag": {
                        "Precision": 0.814,
                        "Recall": 0.617,
                        "mAP50": 0.628,
                        "mAP50-95": 0.422,
                    },
                    "Bottle": {
                        "Precision": 0.892,
                        "Recall": 0.878,
                        "mAP50": 0.897,
                        "mAP50-95": 0.627,
                    },
                    "BrickHammer": {
                        "Precision": 0.977,
                        "Recall": 0.825,
                        "mAP50": 0.927,
                        "mAP50-95": 0.732,
                    },
                    "OrangeHammer": {
                        "Precision": 0.872,
                        "Recall": 0.975,
                        "mAP50": 0.983,
                        "mAP50-95": 0.798,
                    },
                }
            }

        print("=" * 80)
        print("PER-CLASS PERFORMANCE ANALYSIS")
        print("=" * 80)
        print()

        # Create summary table
        summary_data = []
        for model_name, class_data in performance_data.items():
            for class_name, metrics in class_data.items():
                row = {
                    "Model": model_name,
                    "Class": class_name,
                    "Precision": metrics.get("Precision", 0),
                    "Recall": metrics.get("Recall", 0),
                    "mAP50": metrics.get("mAP50", 0),
                    "mAP50-95": metrics.get("mAP50-95", 0),
                }
                summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Create visualizations
        self._create_performance_plots(df)

        return df

    def _create_performance_plots(self, df):
        """Create performance comparison plots"""

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (15, 10)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Precision comparison
        precision_pivot = df.pivot(index="Class", columns="Model", values="Precision")
        precision_pivot.plot(kind="bar", ax=axes[0, 0])
        axes[0, 0].set_title("Precision by Class and Model")
        axes[0, 0].set_ylabel("Precision")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Recall comparison
        recall_pivot = df.pivot(index="Class", columns="Model", values="Recall")
        recall_pivot.plot(kind="bar", ax=axes[0, 1])
        axes[0, 1].set_title("Recall by Class and Model")
        axes[0, 1].set_ylabel("Recall")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # mAP50 comparison
        map50_pivot = df.pivot(index="Class", columns="Model", values="mAP50")
        map50_pivot.plot(kind="bar", ax=axes[1, 0])
        axes[1, 0].set_title("mAP50 by Class and Model")
        axes[1, 0].set_ylabel("mAP50")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # mAP50-95 comparison
        map5095_pivot = df.pivot(index="Class", columns="Model", values="mAP50-95")
        map5095_pivot.plot(kind="bar", ax=axes[1, 1])
        axes[1, 1].set_title("mAP50-95 by Class and Model")
        axes[1, 1].set_ylabel("mAP50-95")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            "../output/per_class_performance_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            "Performance plots saved to: ../output/per_class_performance_comparison.png"
        )

    def create_model_comparison_tables(
        self, results_path="../output/benchmark_results"
    ):
        """Create comprehensive model comparison tables"""

        results_path = Path(results_path)

        # Read performance data
        try:
            performance_df = pd.read_csv(results_path / "performance_report.csv")
        except FileNotFoundError:
            print("Performance report not found, creating sample data")
            # Create sample data for demonstration
            performance_df = pd.DataFrame(
                {
                    "Model": ["YOLOv8s", "YOLOv8m"],
                    "mAP50": [0.85, 0.87, 0.82, 0.83],
                    "mAP50-95": [0.65, 0.67, 0.62, 0.63],
                    "Precision": [0.88, 0.89, 0.85, 0.86],
                    "Recall": [0.82, 0.84, 0.79, 0.80],
                }
            )

        print("=" * 80)
        print("MODEL COMPARISON TABLES")
        print("=" * 80)
        print()

        # Overall performance table
        print("OVERALL PERFORMANCE:")
        print(performance_df.to_string(index=False))
        print()

        # Save to file
        performance_df.to_csv(
            results_path / "model_comparison_overall.csv", index=False
        )

        # Create detailed tables
        self._create_detailed_comparison_tables(performance_df, results_path)

        return performance_df

    def _create_detailed_comparison_tables(self, df, output_path):
        """Create detailed comparison tables"""

        # Sort by mAP50
        sorted_df = df.sort_values("mAP50", ascending=False)

        print("MODELS RANKED BY mAP50:")
        for idx, row in sorted_df.iterrows():
            print(".3f")
        print()

        # Performance summary
        print("PERFORMANCE SUMMARY:")
        print(
            f"Best mAP50: {sorted_df.iloc[0]['Model']} ({sorted_df.iloc[0]['mAP50']:.3f})"
        )
        print(
            f"Best mAP50-95: {sorted_df.loc[sorted_df['mAP50-95'].idxmax()]['Model']} ({sorted_df['mAP50-95'].max():.3f})"
        )
        print(
            f"Best Precision: {sorted_df.loc[sorted_df['Precision'].idxmax()]['Model']} ({sorted_df['Precision'].max():.3f})"
        )
        print(
            f"Best Recall: {sorted_df.loc[sorted_df['Recall'].idxmax()]['Model']} ({sorted_df['Recall'].max():.3f})"
        )
        print()

        # Save summary
        with open(output_path / "performance_comparison_table.txt", "w") as f:
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("=" * 50 + "\n\n")
            f.write("Overall Performance:\n")
            f.write(df.to_string(index=False))
            f.write("\n\nRanked by mAP50:\n")
            for idx, row in sorted_df.iterrows():
                f.write(".3f")
            f.write("\n\nPerformance Summary:\n")
            f.write(
                f"Best mAP50: {sorted_df.iloc[0]['Model']} ({sorted_df.iloc[0]['mAP50']:.3f})\n"
            )
            f.write(
                f"Best mAP50-95: {sorted_df.loc[sorted_df['mAP50-95'].idxmax()]['Model']} ({sorted_df['mAP50-95'].max():.3f})\n"
            )
            f.write(
                f"Best Precision: {sorted_df.loc[sorted_df['Precision'].idxmax()]['Model']} ({sorted_df['Precision'].max():.3f})\n"
            )
            f.write(
                f"Best Recall: {sorted_df.loc[sorted_df['Recall'].idxmax()]['Model']} ({sorted_df['Recall'].max():.3f})\n"
            )


def main():
    """Main function for validation utilities"""
    import argparse

    parser = argparse.ArgumentParser(description="Validation and plotting utilities")
    parser.add_argument(
        "--action",
        choices=[
            "class_grid",
            "detailed_samples",
            "analyze_performance",
            "comparison_tables",
        ],
        required=True,
        help="Action to perform",
    )
    parser.add_argument("--data_path", default="balanced_data", help="Path to dataset")
    parser.add_argument("--output_path", help="Output path for visualizations")
    parser.add_argument(
        "--results_path",
        default="../output/benchmark_results",
        help="Path to results data",
    )

    args = parser.parse_args()

    if args.action == "class_grid":
        visualizer = DatasetVisualizer(args.data_path)
        output_path = args.output_path or "../output/class_samples_grid.jpg"
        visualizer.create_class_grid(output_path)

    elif args.action == "detailed_samples":
        visualizer = DatasetVisualizer(args.data_path)
        output_path = args.output_path or "../output/detailed_class_samples.jpg"
        visualizer.show_class_samples(output_path)

    elif args.action == "analyze_performance":
        analyzer = PerformanceAnalyzer()
        analyzer.create_per_class_analysis()

    elif args.action == "comparison_tables":
        analyzer = PerformanceAnalyzer()
        analyzer.create_model_comparison_tables(args.results_path)


if __name__ == "__main__":
    main()
