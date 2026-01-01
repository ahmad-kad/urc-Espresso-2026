#!/usr/bin/env python3
"""
Unified Training Framework
Consolidates all training functionality into a single, maintainable module
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime

import optuna
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.trainer import ModelTrainer
from utils.logger_config import get_logger
from core.config.manager import create_accuracy_training_config
from utils.output_utils import OutputManager

logger = get_logger(__name__, debug=os.getenv("DEBUG") == "1")


class UnifiedTrainer:
    """
    Unified training framework consolidating all training scenarios
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the unified trainer

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.trainer = ModelTrainer(self.config)
        self.output_manager = OutputManager()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        config_file = Path(config_path)
        if not config_file.exists():
            # Try common config locations
            for config_dir in ["configs", "config", "."]:
                test_path = Path(config_dir) / config_file.name
                if test_path.exists():
                    config_file = test_path
                    break

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def train_accuracy_focused(
        self,
        data_path: str,
        classes: List[str],
        model_size: str = "yolov8m",
        img_size: int = 640,
        epochs: int = 200,
        batch_size: int = 8,
        output_dir: str = "output/models",
    ) -> Dict[str, Any]:
        """
        Train model with accuracy-focused configuration

        Args:
            data_path: Path to dataset directory
            classes: List of class names
            model_size: YOLO model size
            img_size: Input image size
            epochs: Number of training epochs
            batch_size: Batch size
            output_dir: Output directory

        Returns:
            Training results
        """
        logger.info("Starting accuracy-focused training")

        # Create data.yaml
        data_yaml = self._create_data_yaml(data_path, classes, output_dir)

        # Create training configuration
        config = create_accuracy_training_config(
            model=model_size,
            input_size=img_size,
            epochs=epochs,
            batch_size=batch_size,
            flat_format=True,
        )

        # Update with data path
        config["data"] = data_yaml

        # Save configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = Path(output_dir) / f"training_config_{timestamp}.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        logger.info(f"Saved training configuration to: {config_path}")

        # Run training
        result = self.trainer.train(
            data_yaml, model_name=f"accuracy_{model_size}_{timestamp}"
        )

        return result

    def train_with_optuna(
        self,
        data_yaml: str,
        model: str = "yolov8m",
        input_size: int = 224,
        epochs: int = 100,
        n_trials: int = 50,
        output_dir: str = "output/optuna",
    ) -> Dict[str, Any]:
        """
        Train model with Optuna hyperparameter optimization

        Args:
            data_yaml: Path to data.yaml file
            model: Model architecture
            input_size: Input image size
            epochs: Number of training epochs per trial
            n_trials: Number of Optuna trials
            output_dir: Output directory for results

        Returns:
            Optuna study results and best configuration
        """
        logger.info(
            f"Starting Optuna hyperparameter optimization with {n_trials} trials"
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create base configuration
        base_config = create_accuracy_training_config(
            model=model, input_size=input_size, epochs=epochs
        )

        # Create Optuna study
        study_name = (
            f"yolo_{model}_{input_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Objective function
        def objective(trial):
            return self._optuna_objective(
                trial, data_yaml, model, input_size, epochs, base_config
            )

        # Run optimization
        study.optimize(
            objective, n_trials=n_trials, timeout=24 * 60 * 60
        )  # 24 hour timeout

        # Save study results
        study_path = output_path / f"{study_name}.db"
        # Note: Optuna automatically saves to SQLite

        # Get best configuration
        best_config = create_accuracy_training_config(
            model=model,
            input_size=input_size,
            epochs=epochs,
            learning_rate=study.best_params.get("lr0", 0.001),
            weight_decay=study.best_params.get("weight_decay", 0.0005),
            box_loss=study.best_params.get("box_loss", 7.5),
            cls_loss=study.best_params.get("cls_loss", 0.5),
            dfl_loss=study.best_params.get("dfl_loss", 1.5),
        )

        # Save best configuration
        config_path = output_path / "best_config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(best_config, f, default_flow_style=False)

        # Train final model with best parameters
        logger.info("Training final model with best hyperparameters")
        final_result = self.trainer.train(
            data_yaml, model_name=f"optuna_best_{model}_{input_size}"
        )

        results = {
            "study_name": study_name,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "config_path": str(config_path),
            "final_training": final_result,
        }

        return results

    def _optuna_objective(
        self,
        trial: optuna.Trial,
        data_yaml: str,
        model: str,
        input_size: int,
        epochs: int,
        base_config: Dict,
    ) -> float:
        """
        Optuna objective function for hyperparameter optimization
        """
        # Suggest hyperparameters
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        box_loss = trial.suggest_float("box_loss", 5.0, 10.0)
        cls_loss = trial.suggest_float("cls_loss", 0.1, 2.0)
        dfl_loss = trial.suggest_float("dfl_loss", 1.0, 3.0)


        try:
            # Train model
            result = self.trainer.train(data_yaml, model_name=f"trial_{trial.number}")

            # Extract validation mAP50 score
            if result and isinstance(result, dict):
                val_results = result.get("results", {})
                if hasattr(val_results, "box") and hasattr(val_results.box, "map50"):
                    score = float(val_results.box.map50)
                else:
                    score = 0.0
            else:
                score = 0.0

            logger.info(f"Trial {trial.number}: mAP50 = {score:.4f}")

            return score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0

    def finalize_optuna_results(
        self, study_path: str, output_dir: str = "output/optuna_final"
    ) -> Dict[str, Any]:
        """
        Finalize Optuna results - save hyperparameters and prepare for deployment

        Args:
            study_path: Path to Optuna study database
            output_dir: Output directory for final results

        Returns:
            Finalized results dictionary
        """
        logger.info("Finalizing Optuna results")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load study
        study = optuna.load_study(
            study_name="yolo_study", storage=f"sqlite:///{study_path}"
        )

        # Extract best hyperparameters
        best_params = study.best_params
        best_score = study.best_value

        # Convert batch_size_log to actual batch_size if present
        if "batch_size_log" in best_params:
            batch_size_log = best_params["batch_size_log"]
            batch_size = 2**batch_size_log
            best_params["batch_size"] = int(batch_size)

        # Create final configuration
        final_config = {
            "optuna_study": {
                "best_mAP50": float(best_score),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "n_trials": len(study.trials),
            },
            "model": {
                "architecture": "yolov8m",  # Should be parameterized
                "input_size": 224,  # Should be parameterized
            },
            "hyperparameters": best_params,
        }

        # Save configuration
        config_path = output_path / "optuna_best_hyperparameters.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(final_config, f, default_flow_style=False)

        # Save as JSON too
        json_config_path = output_path / "optuna_best_hyperparameters.json"
        with open(json_config_path, "w") as f:
            json.dump(final_config, f, indent=2, default=str)

        logger.info(f"Saved Optuna results to: {config_path}")

        return {
            "config_path": str(config_path),
            "json_config_path": str(json_config_path),
            "best_params": best_params,
            "best_score": best_score,
            "study_summary": {
                "n_trials": len(study.trials),
                "best_trial": study.best_trial.number,
            },
        }

    def _create_data_yaml(
        self, data_path: str, classes: List[str], output_dir: str
    ) -> str:
        """Create data.yaml file for YOLO training"""
        data_yaml = {
            "path": str(Path(data_path).absolute()),
            "train": "images",
            "val": "images",
            "names": {i: name for i, name in enumerate(classes)},
            "nc": len(classes),
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        yaml_path = output_path / "data.yaml"

        with open(yaml_path, "w") as f:
            yaml.safe_dump(data_yaml, f, default_flow_style=False)

        logger.info(f"Created data.yaml at: {yaml_path}")
        return str(yaml_path)


def main():
    """Main CLI interface for unified training"""
    parser = argparse.ArgumentParser(description="Unified YOLO Training Framework")

    # Common arguments
    parser.add_argument(
        "--mode",
        choices=["accuracy", "optuna", "finalize"],
        default="accuracy",
        help="Training mode",
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-dir", default="output", help="Output directory")

    # Accuracy-focused training arguments
    parser.add_argument("--data-path", help="Path to dataset directory")
    parser.add_argument("--classes", nargs="+", help="List of class names")

    # Model configuration
    parser.add_argument(
        "--model-size",
        default="yolov8m",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLO model size",
    )
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")

    # Optuna arguments
    parser.add_argument("--data-yaml", help="Path to data.yaml file")
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of Optuna trials"
    )

    # Finalize arguments
    parser.add_argument("--study-path", help="Path to Optuna study database")

    args = parser.parse_args()

    # Initialize trainer
    trainer = UnifiedTrainer(args.config)

    try:
        if args.mode == "accuracy":
            if not args.data_path or not args.classes:
                print("Error: --data-path and --classes required for accuracy mode")
                return 1

            result = trainer.train_accuracy_focused(
                data_path=args.data_path,
                classes=args.classes,
                model_size=args.model_size,
                img_size=args.img_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
            )

        elif args.mode == "optuna":
            if not args.data_yaml:
                print("Error: --data-yaml required for optuna mode")
                return 1

            result = trainer.train_with_optuna(
                data_yaml=args.data_yaml,
                model=args.model_size,
                input_size=args.img_size,
                epochs=args.epochs,
                n_trials=args.n_trials,
                output_dir=args.output_dir,
            )

        elif args.mode == "finalize":
            if not args.study_path:
                print("Error: --study-path required for finalize mode")
                return 1

            result = trainer.finalize_optuna_results(
                study_path=args.study_path, output_dir=args.output_dir
            )

        # Print results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
