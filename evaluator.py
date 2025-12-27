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

from detector import ObjectDetector

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
        Evaluate a single model comprehensively with robotics-specific metrics

        Args:
            model_path: Path to trained model
            data_yaml: Path to data configuration
            model_name: Name identifier for the model

        Returns:
            Evaluation results dictionary
        """

        logger.info(f"Evaluating {model_name} model: {model_path}")

        try:
            import yaml

            # Load class names from the dataset config for accurate reporting
            class_names: List[str] = []
            try:
                with open(data_yaml, "r") as f:
                    data_cfg = yaml.safe_load(f)
                    names = data_cfg.get("names", [])
                    if isinstance(names, dict):
                        # YOLO may store class names as {id: name}
                        class_names = [name for _, name in sorted(names.items(), key=lambda kv: kv[0])]
                    elif isinstance(names, list):
                        class_names = names
            except Exception as e:
                logger.warning(f"Could not load class names from {data_yaml}: {e}")
            # Load model using YOLO directly for more control
            from ultralytics import YOLO
            model = YOLO(model_path)

            # Run validation with detailed metrics
            metrics = model.val(data=data_yaml, split='test', save_json=True, plots=True)

            # Basic metrics
            core_metrics = {
                'map50': float(metrics.box.map50),
                'map50_95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'f1': 2 * float(metrics.box.mp) * float(metrics.box.mr) / (float(metrics.box.mp) + float(metrics.box.mr)) if (float(metrics.box.mp) + float(metrics.box.mr)) > 0 else 0
            }

            results = {
                'model_name': model_name,
                'model_path': model_path,
                # Keep "metrics" aligned with downstream plotting functions
                'metrics': core_metrics,
                # Backwards-compatible key
                'basic_metrics': core_metrics,
                'per_class_metrics': {}
            }

            # Per-class metrics if available
            if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
                results['per_class_metrics'] = {}
                for i in range(len(metrics.box.maps)):
                    class_label = class_names[i] if i < len(class_names) else f"class_{i}"
                    try:
                        class_result = metrics.box.class_result(i) if hasattr(metrics.box, 'class_result') else [0.0, 0.0, 0.0]
                        results['per_class_metrics'][class_label] = {
                            'map50': float(metrics.box.maps[i]),
                            'precision': float(class_result[0]),
                            'recall': float(class_result[1]),
                            'f1': float(class_result[2])
                        }
                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Could not get metrics for class {class_label}: {e}")
                        results['per_class_metrics'][class_label] = {
                            'map50': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'f1': 0.0
                        }

            # Add robotics-specific metrics
            results['robotics_metrics'] = self._evaluate_robotics_metrics(model, data_yaml)

            logger.info(f"{model_name} Results:")
            logger.info(f"mAP50: {core_metrics['map50']:.4f}")
            logger.info(f"mAP50-95: {core_metrics['map50_95']:.4f}")
            logger.info(f"Precision: {core_metrics['precision']:.4f}")
            logger.info(f"Recall: {core_metrics['recall']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'error': str(e)
            }

    def _evaluate_robotics_metrics(self, model, data_yaml: str) -> Dict:
        """
        Evaluate robotics-specific metrics

        Args:
            model: YOLO model instance
            data_yaml: Data configuration path

        Returns:
            Robotics-specific metrics
        """
        try:
            robotics_metrics = {
                'small_object_detection': self._evaluate_small_object_performance(model, data_yaml),
                'distance_performance': self._evaluate_distance_performance(model, data_yaml),
                'occlusion_robustness': self._evaluate_occlusion_robustness(model, data_yaml),
                'lighting_robustness': self._evaluate_lighting_robustness(model, data_yaml),
                'temporal_consistency': 0.0,  # Would need video sequences for this
                'inference_efficiency': self._evaluate_inference_efficiency(model)
            }

            return robotics_metrics

        except Exception as e:
            logger.warning(f"Could not evaluate robotics metrics: {str(e)}")
            return {}

    def _evaluate_small_object_performance(self, model, data_yaml: str) -> Dict:
        """Evaluate performance on small objects (< 32x32 pixels)"""
        try:
            # Run validation and analyze results
            results = model.val(data=data_yaml, split='test')

            small_objects = []
            all_objects = []

            # This would need access to ground truth and predictions
            # For now, return placeholder metrics
            return {
                'small_object_map': 0.0,  # mAP for objects < 32x32
                'small_object_recall': 0.0,
                'small_object_precision': 0.0,
                'min_detection_size': 16  # Minimum detectable object size
            }

        except Exception as e:
            logger.warning(f"Small object evaluation failed: {str(e)}")
            return {}

    def _evaluate_distance_performance(self, model, data_yaml: str) -> Dict:
        """Evaluate performance across different distance ranges"""
        # Distance ranges for robotics applications
        distance_ranges = {
            'close': (0, 2),      # 0-2 meters
            'medium': (2, 5),     # 2-5 meters
            'far': (5, 10),       # 5-10 meters
            'very_far': (10, 15)  # 10-15 meters
        }

        distance_metrics = {}
        for range_name, (min_dist, max_dist) in distance_ranges.items():
            # This would analyze performance based on object sizes as distance proxies
            # For now, return placeholder
            distance_metrics[range_name] = {
                'map50': 0.0,
                'recall': 0.0,
                'precision': 0.0
            }

        return distance_metrics

    def _evaluate_occlusion_robustness(self, model, data_yaml: str) -> float:
        """Evaluate robustness to object occlusion"""
        # This would require occluded test data
        # For now, return placeholder
        return 0.0

    def _evaluate_lighting_robustness(self, model, data_yaml: str) -> float:
        """Evaluate robustness to different lighting conditions"""
        # This would require test data with various lighting
        # For now, return placeholder
        return 0.0

    def _evaluate_inference_efficiency(self, model) -> Dict:
        """Evaluate inference efficiency metrics"""
        try:
            # Get model parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

            # Estimate memory usage (rough approximation)
            param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32

            # Estimate model size on disk if path is available
            model_size_mb = 0.0
            if hasattr(model, "model") and hasattr(model, "ckpt_path"):
                try:
                    model_size_mb = Path(model.ckpt_path).stat().st_size / (1024 * 1024)
                except Exception:
                    pass

            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'estimated_memory_mb': param_memory_mb,
                'model_size_mb': model_size_mb
            }

        except Exception as e:
            logger.warning(f"Inference efficiency evaluation failed: {str(e)}")
            return {}

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
                        'mAP@50': float(result['metrics'].get('map50', 0.0)),
                        'mAP@50:95': float(result['metrics'].get('map50_95', 0.0)),
                        'Precision': float(result['metrics'].get('precision', 0.0)),
                        'Recall': float(result['metrics'].get('recall', 0.0))
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

    def compare_models_plot(self, results: Dict):
        """Create comparison plots for models (alias for _create_comparison_plots)"""
        self._create_comparison_plots(results)

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
            metrics = ['map50', 'precision', 'recall']
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
                        value = successful_results[model_name]['metrics'].get(metric, 0)
                        values.append(value)
                    else:
                        values.append(0)

                values += values[:1]  # Close the plot

                ax.plot(angles, values, 'o-', linewidth=2, label=model_name,
                       color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(['mAP@50', 'Precision', 'Recall'])
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Comparison (Radar)', size=16, fontweight='bold')
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)

            plot_path = self.output_dir / 'model_performance_radar.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
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
