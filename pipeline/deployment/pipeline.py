"""
Deployment Pipeline Implementation
Consolidated deployment functionality
"""

from pathlib import Path
from typing import Dict, Any, Optional

from ..base import BasePipeline, PipelineConfig, PipelineResult
from utils.logger_config import get_logger

logger = get_logger(__name__)


class DeploymentPipeline(BasePipeline):
    """
    Unified deployment pipeline
    Handles model packaging and deployment preparation
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize deployment pipeline

        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        self.model_path: Optional[Path] = None
        self.config_data: Optional[Dict[str, Any]] = None
        self.results_data: Optional[Dict[str, Any]] = None

    def set_model(self, model_path: Path) -> None:
        """Set model for deployment"""
        self.model_path = Path(model_path)

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set configuration data"""
        self.config_data = config

    def set_results(self, results: Dict[str, Any]) -> None:
        """Set evaluation results"""
        self.results_data = results

    def validate_config(self) -> bool:
        """
        Validate deployment configuration

        Returns:
            True if configuration is valid
        """
        try:
            if not self.model_path or not self.model_path.exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                return False

            self.logger.info(" Deployment configuration validated")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def run(self) -> PipelineResult:
        """
        Execute deployment pipeline

        Returns:
            PipelineResult with deployment results
        """
        result = PipelineResult(
            success=False, duration=0.0, metrics={}, artifacts=[], errors=[]
        )

        try:
            self.logger.info(" Starting deployment pipeline")

            # Update progress
            self.update_progress(0.2, "Creating deployment package")

            # Create deployment package
            package_path = self._create_deployment_package()

            # Update progress
            self.update_progress(0.8, "Generating deployment documentation")

            # Generate deployment documentation
            docs_path = self._generate_deployment_docs(package_path)

            # Update progress
            self.update_progress(0.9, "Finalizing deployment")

            # Add artifacts
            result.add_artifact(package_path)
            if docs_path:
                result.add_artifact(docs_path)

            result.success = True
            result.metrics = {
                "package_created": True,
                "package_path": str(package_path),
                "model_included": str(self.model_path),
            }

            self.logger.info(" Deployment pipeline completed successfully")

        except Exception as e:
            result.add_error(f"Deployment pipeline error: {e}")
            self.logger.error(f" Deployment pipeline failed: {e}")

        finally:
            self.update_progress(1.0, "Deployment completed")

        return result

    def _create_deployment_package(self) -> Path:
        """
        Create deployment package

        Returns:
            Path to deployment package directory
        """
        import shutil
        import json

        # Create package directory
        package_name = f"{self.config.name}_deployment"
        package_dir = self.config.output_dir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy model file
            if self.model_path:
                model_dest = package_dir / self.model_path.name
                shutil.copy2(self.model_path, model_dest)

            # Save configuration
            if self.config_data:
                config_file = package_dir / "config.json"
                with open(config_file, "w") as f:
                    json.dump(self.config_data, f, indent=2, default=str)

            # Save results
            if self.results_data:
                results_file = package_dir / "evaluation_results.json"
                with open(results_file, "w") as f:
                    json.dump(self.results_data, f, indent=2, default=str)

            # Create deployment script
            self._create_deployment_script(package_dir)

            return package_dir

        except Exception as e:
            self.logger.error(f"Failed to create deployment package: {e}")
            raise

    def _create_deployment_script(self, package_dir: Path) -> None:
        """Create deployment script for the package"""
        script_content = f"""#!/bin/bash
# Deployment script for {self.config.name}
# Generated automatically by deployment pipeline

echo " Deploying {self.config.name}..."

# Add your deployment commands here
# Example:
# sudo cp model.onnx /opt/models/
# sudo systemctl restart yolov8-service

echo " Deployment completed!"
"""

        script_path = package_dir / "deploy.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Make script executable (on Unix-like systems)
        try:
            script_path.chmod(0o755)
        except OSError:
            pass  # Skip on Windows

    def _generate_deployment_docs(self, package_dir: Path) -> Optional[Path]:
        """Generate deployment documentation"""
        try:
            docs_content = f"""# {self.config.name} Deployment Guide

## Overview
This package contains a trained YOLO model ready for deployment.

## Contents
- `model.onnx` - Optimized ONNX model
- `config.json` - Model configuration
- `evaluation_results.json` - Model performance metrics
- `deploy.sh` - Deployment script

## Deployment Instructions
1. Copy the package to your target system
2. Run the deployment script: `./deploy.sh`
3. Update your application configuration to use the new model

## Performance Metrics
{self._format_metrics()}

## Requirements
- ONNX Runtime
- Python 3.8+
- Compatible hardware for inference
"""

            docs_path = package_dir / "DEPLOYMENT_README.md"
            with open(docs_path, "w") as f:
                f.write(docs_content)

            return docs_path

        except Exception as e:
            self.logger.warning(f"Failed to generate deployment docs: {e}")
            return None

    def _format_metrics(self) -> str:
        """Format performance metrics for documentation"""
        if not self.results_data:
            return "No performance metrics available."

        lines = []
        if isinstance(self.results_data, dict):
            for key, value in self.results_data.items():
                lines.append(f"- {key}: {value}")

        return (
            "\n".join(lines)
            if lines
            else "Metrics available in evaluation_results.json"
        )