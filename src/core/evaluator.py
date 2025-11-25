"""
Model evaluation and performance analysis tools
"""

import os
import cv2
import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import logging

from .detector import ObjectDetector

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison
    """

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config.get('output', {}).get('results', 'output/results'))

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(self, model_path: str, data_yaml: str, model_name: str = "model") -> Dict:
        """
        Evaluate a single model comprehensively

        Args:
            model_path: Path to trained model
            data_yaml: Path to data configuration
            model_name: Name identifier for the model

        Returns:
            Evaluation results dictionary
        """

        logger.info(f"Evaluating {model_name} model: {model_path}")

        try:
            # Load model
            detector = ObjectDetector(self.config)
            detector.model = torch.load(model_path) if model_path.endswith('.pt') else detector.model

            # Run validation
            metrics = detector.val(data_yaml, split='test')

            results = {
                'model_name': model_name,
                'model_path': model_path,
                'metrics': {
                    'map50': float(metrics.box.map50),
                    'map50_95': float(metrics.box.map),
                    'precision': float(metrics.box.mp),
                    'recall': float(metrics.box.mr)
                },
                'per_class_metrics': {}
            }

            # Per-class metrics if available
            if hasattr(metrics.box, 'maps'):
                results['per_class_metrics'] = {
                    f'class_{i}': float(metrics.box.maps[i]) for i in range(len(metrics.box.maps))
                }

            logger.info(f"{model_name} Results:")
            logger.info(".4f")
            logger.info(".4f")

            return results

        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'error': str(e)
            }

    def compare_models(self, model_paths: Dict[str, str], data_yaml: str) -> Dict:
        """
        Compare multiple models

        Args:
            model_paths: Dict of model_name -> model_path
            data_yaml: Path to data configuration

        Returns:
            Comparison results
        """

        logger.info("Comparing model performance...")

        results = {}
        for model_name, model_path in model_paths.items():
            results[model_name] = self.evaluate_model(model_path, data_yaml, model_name)

        # Create comparison plots if we have successful results
        successful_results = {k: v for k, v in results.items() if 'metrics' in v}

        if len(successful_results) > 1:
            self._create_comparison_plots(successful_results)
            self._create_performance_report(successful_results)

        return results

    def test_realtime_performance(self, model_path: str, test_images: List[str],
                                num_runs: int = 100) -> Dict:
        """
        Test real-time inference performance

        Args:
            model_path: Path to model
            test_images: List of test image paths
            num_runs: Number of inference runs for averaging

        Returns:
            Performance metrics
        """

        logger.info("Testing real-time performance...")

        try:
            detector = ObjectDetector(self.config)
            detector.model = torch.load(model_path) if model_path.endswith('.pt') else detector.model

            inference_times = []

            for i in range(min(num_runs, len(test_images))):
                image_path = test_images[i % len(test_images)]

                # Load and preprocess image
                image = cv2.imread(image_path)
                if image is None:
                    continue

                # Time inference
                start_time = time.time()
                results = detector.predict(image)
                inference_time = time.time() - start_time

                inference_times.append(inference_time * 1000)  # Convert to ms

            if inference_times:
                avg_time = np.mean(inference_times)
                std_time = np.std(inference_times)
                fps = 1000 / avg_time

                results = {
                    'average_inference_time_ms': avg_time,
                    'std_inference_time_ms': std_time,
                    'fps': fps,
                    'min_time_ms': min(inference_times),
                    'max_time_ms': max(inference_times)
                }

                logger.info(".2f")
                logger.info(".1f")

                return results
            else:
                return {'error': 'No valid test images found'}

        except Exception as e:
            logger.error(f"Real-time performance test failed: {str(e)}")
            return {'error': str(e)}

    def test_small_object_detection(self, model_path: str, test_images: List[str],
                                  min_area_threshold: int = 32*32) -> Dict:
        """
        Test performance on small objects

        Args:
            model_path: Path to model
            test_images: Test images
            min_area_threshold: Minimum bounding box area for "small" objects

        Returns:
            Small object detection metrics
        """

        logger.info("Testing small object detection performance...")

        try:
            detector = ObjectDetector(self.config)
            detector.model = torch.load(model_path) if model_path.endswith('.pt') else detector.model

            small_objects_found = 0
            total_small_objects = 0
            detections = []

            for image_path in test_images[:50]:  # Limit for performance
                image = cv2.imread(image_path)
                if image is None:
                    continue

                results = detector.predict(image)

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)

                        detection = {
                            'area': area,
                            'confidence': float(box.conf[0]),
                            'class': int(box.cls[0])
                        }
                        detections.append(detection)

                        if area < min_area_threshold:
                            total_small_objects += 1
                            if detection['confidence'] > self.config.get('model', {}).get('confidence_threshold', 0.5):
                                small_objects_found += 1

            if total_small_objects > 0:
                small_object_accuracy = small_objects_found / total_small_objects
            else:
                small_object_accuracy = 0.0

            results = {
                'small_object_accuracy': small_object_accuracy,
                'total_small_objects': total_small_objects,
                'small_objects_detected': small_objects_found,
                'average_small_object_confidence': np.mean([d['confidence'] for d in detections if d['area'] < min_area_threshold]) if detections else 0.0
            }

            logger.info(".1%")

            return results

        except Exception as e:
            logger.error(f"Small object detection test failed: {str(e)}")
            return {'error': str(e)}

    def _create_comparison_plots(self, results: Dict):
        """Create performance comparison plots"""
        try:
            models = list(results.keys())
            map50_scores = [results[m]['metrics']['map50'] for m in models]
            map50_95_scores = [results[m]['metrics']['map50_95'] for m in models]

            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            x = range(len(models))
            ax1.bar(x, map50_scores, color='skyblue', alpha=0.8)
            ax1.set_title('mAP@50 Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45)
            ax1.set_ylabel('mAP@50')

            ax2.bar(x, map50_95_scores, color='lightgreen', alpha=0.8)
            ax2.set_title('mAP@50:95 Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, rotation=45)
            ax2.set_ylabel('mAP@50:95')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Performance comparison plot saved")

        except Exception as e:
            logger.warning(f"Could not create comparison plots: {str(e)}")

    def _create_performance_report(self, results: Dict):
        """Create detailed performance report"""
        try:
            report_data = []
            for model_name, result in results.items():
                if 'metrics' in result:
                    report_data.append({
                        'Model': model_name,
                        'mAP@50': ".4f",
                        'mAP@50:95': ".4f",
                        'Precision': ".4f",
                        'Recall': ".4f"
                    })

            df = pd.DataFrame(report_data)
            report_path = self.output_dir / 'performance_report.csv'
            df.to_csv(report_path, index=False)

            logger.info(f"Performance report saved to: {report_path}")

            # Also create a formatted table for display
            self._create_performance_table(df)

        except Exception as e:
            logger.warning(f"Could not create performance report: {str(e)}")

    def _create_performance_table(self, df: pd.DataFrame):
        """Create a formatted performance comparison table"""
        try:
            # Create a nice formatted table
            table_path = self.output_dir / 'performance_comparison_table.txt'
            with open(table_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("MODEL PERFORMANCE COMPARISON TABLE\n")
                f.write("="*80 + "\n\n")

                # Write the dataframe as formatted text
                f.write(df.to_string(index=False, float_format='%.4f'))
                f.write("\n\n")

                # Add interpretation
                f.write("INTERPRETATION:\n")
                f.write("- Higher mAP@50: Better object detection accuracy\n")
                f.write("- Higher mAP@50:95: Better across different IoU thresholds\n")
                f.write("- Higher Precision: Fewer false positives\n")
                f.write("- Higher Recall: Fewer false negatives\n\n")

                # Find best models
                if not df.empty:
                    best_map50 = df.loc[df['mAP@50'].idxmax(), 'Model']
                    best_precision = df.loc[df['Precision'].idxmax(), 'Model']
                    best_recall = df.loc[df['Recall'].idxmax(), 'Model']

                    f.write("BEST PERFORMERS:\n")
                    f.write(f"- Highest mAP@50: {best_map50}\n")
                    f.write(f"- Highest Precision: {best_precision}\n")
                    f.write(f"- Highest Recall: {best_recall}\n")

            logger.info(f"Performance table saved to: {table_path}")

        except Exception as e:
            logger.warning(f"Could not create performance table: {str(e)}")

    def create_training_history_comparison(self, model_logs_paths: Dict[str, str]):
        """
        Create accuracy over time comparison plot for all models

        Args:
            model_logs_paths: Dict of model_name -> path to training logs/results.csv
        """
        try:
            plt.figure(figsize=(15, 10))

            colors = ['blue', 'red', 'green', 'orange']
            model_names = list(model_logs_paths.keys())

            # Plot mAP@50 over time for each model
            for i, (model_name, logs_path) in enumerate(model_logs_paths.items()):
                results_csv = Path(logs_path) / 'results.csv'
                if results_csv.exists():
                    df = pd.read_csv(results_csv)
                    if 'metrics/mAP50(B)' in df.columns:
                        color = colors[i % len(colors)]
                        plt.plot(df['epoch'], df['metrics/mAP50(B)'],
                               label=model_name, color=color, linewidth=2, marker='o', markersize=4)

            plt.title('Model Accuracy Comparison Over Training Time', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('mAP@50', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = self.output_dir / 'accuracy_over_time_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Training history comparison plot saved: {plot_path}")

        except Exception as e:
            logger.warning(f"Could not create training history comparison: {str(e)}")

    def create_comprehensive_model_comparison(self, results: Dict, training_logs: Dict[str, str] = None):
        """
        Create comprehensive model comparison with plots and tables

        Args:
            results: Evaluation results from model comparison
            training_logs: Optional dict of model_name -> logs path for training history
        """
        logger.info("Creating comprehensive model comparison...")

        # Create performance comparison plots
        self.compare_models_plot(results)

        # Create accuracy over time plot if training logs provided
        if training_logs:
            self.create_training_history_comparison(training_logs)

        # Create performance report and table
        self._create_performance_report(results)

        # Create additional visualizations
        self._create_model_ranking_plot(results)

        logger.info("Comprehensive model comparison completed")

    def _create_model_ranking_plot(self, results: Dict):
        """Create a radar/spider plot showing model strengths"""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from math import pi

            # Extract metrics for successful models
            successful_results = {k: v for k, v in results.items() if 'metrics' in v}

            if len(successful_results) < 3:
                logger.info("Need at least 3 models for radar plot")
                return

            # Metrics to plot
            metrics = ['mAP@50', 'Precision', 'Recall']
            model_names = list(successful_results.keys())

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

            # Calculate angles
            angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
            angles += angles[:1]  # Close the plot

            # Plot each model
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, model_name in enumerate(model_names):
                values = []
                for metric in metrics:
                    if 'metrics' in successful_results[model_name]:
                        value = successful_results[model_name]['metrics'].get(metric.replace('@', ''), 0)
                        values.append(value)
                    else:
                        values.append(0)

                values += values[:1]  # Close the plot

                ax.plot(angles, values, 'o-', linewidth=2, label=model_name,
                       color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Comparison (Radar)', size=16, fontweight='bold')
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)

            plot_path = self.output_dir / 'model_performance_radar.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', bbox_to_anchor='tight')
            plt.close()

            logger.info(f"Model ranking radar plot saved: {plot_path}")

        except Exception as e:
            logger.warning(f"Could not create radar plot: {str(e)}")

    def generate_comprehensive_report(self, model_paths: Dict[str, str],
                                    data_yaml: str, test_images: List[str]) -> Dict:
        """
        Generate comprehensive evaluation report

        Args:
            model_paths: Dict of model_name -> model_path
            data_yaml: Data configuration path
            test_images: List of test image paths

        Returns:
            Complete evaluation results
        """

        logger.info("Generating comprehensive evaluation report...")

        report = {
            'model_comparison': self.compare_models(model_paths, data_yaml),
            'realtime_performance': {},
            'small_object_detection': {}
        }

        # Test real-time performance for each model
        for model_name, model_path in model_paths.items():
            report['realtime_performance'][model_name] = self.test_realtime_performance(
                model_path, test_images
            )

        # Test small object detection for each model
        for model_name, model_path in model_paths.items():
            report['small_object_detection'][model_name] = self.test_small_object_detection(
                model_path, test_images
            )

        # Save complete report
        import json
        report_path = self.output_dir / 'comprehensive_evaluation.json'
        with open(report_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_report = json.dumps(report, indent=2, default=str)
            f.write(json_report)

        logger.info(f"Comprehensive report saved to: {report_path}")

        return report
