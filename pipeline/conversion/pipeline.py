"""
Conversion Pipeline Implementation
Consolidated model conversion functionality
"""

from pathlib import Path
from typing import List, Optional

from ..base import BasePipeline, PipelineConfig, PipelineResult
from utils.logger_config import get_logger

logger = get_logger(__name__)


class ConversionPipeline(BasePipeline):
    """
    Unified model conversion pipeline
    Handles conversion between different model formats
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize conversion pipeline

        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        self.input_model: Optional[Path] = None
        self.input_directory: Optional[Path] = None
        self.batch_mode: bool = False
        self.target_formats: List[str] = ["onnx"]
        self.input_size: int = 416
        self.precision: str = "fp32"
        self.opset_version: int = 11
        self.quantize: bool = False
        self.optimize: bool = False

    def set_input_model(self, model_path: Path) -> None:
        """Set single model for conversion"""
        self.input_model = Path(model_path)
        self.batch_mode = False

    def set_input_directory(self, directory: Path) -> None:
        """Set directory for batch conversion"""
        self.input_directory = Path(directory)
        self.batch_mode = True

    def set_batch_mode(self, batch: bool) -> None:
        """Enable/disable batch conversion mode"""
        self.batch_mode = batch

    def set_target_formats(self, formats: List[str]) -> None:
        """Set target conversion formats"""
        self.target_formats = formats

    def set_input_size(self, size: int) -> None:
        """Set model input size"""
        self.input_size = size

    def set_precision(self, precision: str) -> None:
        """Set target precision"""
        self.precision = precision

    def set_opset_version(self, version: int) -> None:
        """Set ONNX opset version"""
        self.opset_version = version

    def set_quantization(self, quantize: bool) -> None:
        """Enable/disable quantization"""
        self.quantize = quantize

    def set_optimization(self, optimize: bool) -> None:
        """Enable/disable graph optimization"""
        self.optimize = optimize

    def validate_config(self) -> bool:
        """
        Validate conversion configuration

        Returns:
            True if configuration is valid
        """
        try:
            # Validate input
            if self.batch_mode:
                if not self.input_directory or not self.input_directory.exists():
                    self.logger.error(f"Input directory not found: {self.input_directory}")
                    return False
            else:
                if not self.input_model or not self.input_model.exists():
                    self.logger.error(f"Input model not found: {self.input_model}")
                    return False

            # Validate target formats
            valid_formats = ["onnx", "tensorrt", "openvino", "tflite"]
            for fmt in self.target_formats:
                if fmt not in valid_formats:
                    self.logger.error(f"Unsupported target format: {fmt}")
                    return False

            # Validate precision
            valid_precisions = ["fp32", "fp16", "int8"]
            if self.precision not in valid_precisions:
                self.logger.error(f"Unsupported precision: {self.precision}")
                return False

            self.logger.info(" Conversion configuration validated")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def run(self) -> PipelineResult:
        """
        Execute conversion pipeline

        Returns:
            PipelineResult with conversion results
        """
        result = PipelineResult(
            success=False, duration=0.0, metrics={}, artifacts=[], errors=[]
        )

        try:
            self.logger.info(" Starting model conversion pipeline")

            # Determine models to convert
            if self.batch_mode:
                models_to_convert = self._find_models_in_directory()
            else:
                models_to_convert = [self.input_model]

            self.logger.info(
                f"Converting {len(models_to_convert)} models to {self.target_formats}"
            )

            # Update progress
            self.update_progress(0.1, "Initializing conversion environment")

            total_conversions = len(models_to_convert) * len(self.target_formats)
            current_conversion = 0
            conversion_results = {}

            for model_path in models_to_convert:
                model_results = {}

                for target_format in self.target_formats:
                    current_conversion += 1
                    progress = 0.1 + (current_conversion / total_conversions) * 0.8
                    self.update_progress(
                        progress,
                        f"Converting {Path(model_path).name} to {target_format}",
                    )

                    try:
                        converted_path = self._convert_model(model_path, target_format)
                        if converted_path:
                            model_results[target_format] = str(converted_path)
                            result.add_artifact(converted_path)
                        else:
                            model_results[target_format] = {"error": "Conversion failed"}

                    except Exception as e:
                        error_msg = f"Failed to convert {model_path} to {target_format}: {e}"
                        model_results[target_format] = {"error": error_msg}
                        result.add_error(error_msg)

                conversion_results[str(model_path)] = model_results

            # Process results
            result.success = True
            result.metrics = {
                "total_models": len(models_to_convert),
                "total_conversions": total_conversions,
                "successful_conversions": len(result.artifacts),
                "conversion_results": conversion_results,
            }

            self.logger.info(" Model conversion pipeline completed successfully")

        except Exception as e:
            result.add_error(f"Conversion pipeline error: {e}")
            self.logger.error(f" Conversion pipeline failed: {e}")

        finally:
            self.update_progress(1.0, "Pipeline execution completed")

        return result

    def _find_models_in_directory(self) -> List[Path]:
        """Find all model files in input directory"""
        if not self.input_directory:
            return []

        model_extensions = {".pt", ".pth", ".onnx"}
        models = []

        for file_path in self.input_directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in model_extensions:
                models.append(file_path)

        return models

    def _convert_model(self, model_path: Path, target_format: str) -> Optional[Path]:
        """
        Convert a single model to target format
        """
        try:
            if target_format == "onnx":
                return self._convert_to_onnx(model_path)
            elif target_format == "tensorrt":
                return self._convert_to_tensorrt(model_path)
            elif target_format == "openvino":
                return self._convert_to_openvino(model_path)
            elif target_format == "tflite":
                return self._convert_to_tflite(model_path)
            else:
                raise ValueError(f"Unsupported target format: {target_format}")

        except Exception as e:
            self.logger.error(f"Model conversion failed: {e}")
            raise

    def _convert_to_onnx(self, model_path: Path) -> Optional[Path]:
        """Convert model to ONNX format"""
        try:
            from scripts.convert_to_onnx import convert_model, convert_onnx_to_int8

            # Create output path
            output_name = f"{model_path.stem}.onnx"
            output_path = self.config.output_dir / output_name

            # Convert to FP32 ONNX
            convert_model(str(model_path), str(output_path), input_size=self.input_size)

            # Apply quantization if requested
            if self.quantize and self.precision == "int8":
                int8_path = output_path.with_name(f"{model_path.stem}_int8.onnx")
                convert_onnx_to_int8(str(output_path), str(int8_path))
                return int8_path

            return output_path

        except Exception as e:
            self.logger.error(f"ONNX conversion failed: {e}")
            raise

    def _convert_to_tensorrt(self, model_path: Path) -> Optional[Path]:
        """Convert model to TensorRT format (placeholder)"""
        self.logger.warning("TensorRT conversion not yet implemented")
        return None

    def _convert_to_openvino(self, model_path: Path) -> Optional[Path]:
        """Convert model to OpenVINO format (placeholder)"""
        self.logger.warning("OpenVINO conversion not yet implemented")
        return None

    def _convert_to_tflite(self, model_path: Path) -> Optional[Path]:
        """Convert model to TFLite format (placeholder)"""
        self.logger.warning("TFLite conversion not yet implemented")
        return None