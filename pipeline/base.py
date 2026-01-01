"""
Unified Pipeline Base Classes
Provides consistent interfaces and error handling across all pipelines
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from utils.logger_config import get_logger


class PipelineStatus(Enum):
    """Pipeline execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineError(Exception):
    """Base pipeline error"""

    pass



@dataclass
class PipelineResult:
    """Standardized pipeline result"""

    success: bool
    duration: float
    metrics: Dict[str, Any]
    artifacts: List[Path]
    errors: List[str]
    status: PipelineStatus = PipelineStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert enum to string
        result["status"] = self.status.value
        # Convert Path objects to strings
        result["artifacts"] = [str(p) for p in self.artifacts]
        return result

    def add_error(self, error: str) -> None:
        """Add an error message"""
        self.errors.append(error)
        if self.success:
            self.success = False
            self.status = PipelineStatus.FAILED

    def add_artifact(self, artifact: Union[str, Path]) -> None:
        """Add an output artifact"""
        if isinstance(artifact, str):
            artifact = Path(artifact)
        self.artifacts.append(artifact)

    def add_metric(self, key: str, value: Any) -> None:
        """Add a performance metric"""
        self.metrics[key] = value


@dataclass
class PipelineConfig:
    """Base pipeline configuration"""

    name: str
    version: str = "1.0.0"
    output_dir: Path = Path("output")
    log_level: str = "INFO"
    dry_run: bool = False

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


class BasePipeline(ABC):
    """
    Abstract base class for all pipelines
    Provides consistent interface, error handling, and monitoring
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_logger(
            self.__class__.__name__, debug=config.log_level.upper() == "DEBUG"
        )
        self._start_time: Optional[float] = None
        self._status = PipelineStatus.PENDING
        self._progress = 0.0

    @property
    def status(self) -> PipelineStatus:
        """Get current pipeline status"""
        return self._status

    @property
    def progress(self) -> float:
        """Get current progress (0.0 to 1.0)"""
        return self._progress

    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update pipeline progress"""
        self._progress = max(0.0, min(1.0, progress))
        if message:
            self.logger.info(f"Progress: {self._progress:.1%} - {message}")
        else:
            self.logger.debug(f"Progress: {self._progress:.1%}")

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate pipeline configuration

        Returns:
            True if configuration is valid
        """
        pass

    @abstractmethod
    def run(self) -> PipelineResult:
        """
        Execute the pipeline

        Returns:
            PipelineResult with execution results
        """
        pass

    def _execute_with_monitoring(self) -> PipelineResult:
        """
        Execute pipeline with monitoring and error handling

        Returns:
            PipelineResult with execution results
        """
        self._start_time = time.time()
        self._status = PipelineStatus.RUNNING

        result = PipelineResult(
            success=False, duration=0.0, metrics={}, artifacts=[], errors=[]
        )

        try:
            self.logger.info(f"Starting pipeline: {self.config.name}")

            # Validate configuration
            if not self.validate_config():
                result.add_error("Configuration validation failed")
                return result

            # Execute pipeline
            result = self.run()
            result.status = PipelineStatus.COMPLETED

            self.logger.info(f"Pipeline completed successfully: {self.config.name}")

        except PipelineError as e:
            self.logger.error(f"Pipeline error: {e}")
            result.add_error(str(e))
            result.status = PipelineStatus.FAILED

        except Exception as e:
            self.logger.error(f"Unexpected pipeline error: {e}")
            result.add_error(f"Unexpected error: {e}")
            result.status = PipelineStatus.FAILED

        finally:
            # Calculate duration
            if self._start_time:
                result.duration = time.time() - self._start_time

            self._status = result.status

            # Log final status
            self.logger.info(
                f"Pipeline finished in {result.duration:.2f}s with status: {result.status.value}"
            )

        return result

    def cancel(self) -> None:
        """Cancel pipeline execution"""
        self._status = PipelineStatus.CANCELLED
        self.logger.info(f"Pipeline cancelled: {self.config.name}")

    def reset(self) -> None:
        """Reset pipeline state"""
        self._status = PipelineStatus.PENDING
        self._progress = 0.0
        self._start_time = None
        self.logger.info(f"Pipeline reset: {self.config.name}")


class ProgressCallback:
    """
    Callback interface for pipeline progress updates
    """

    def on_progress(
        self, pipeline: BasePipeline, progress: float, message: Optional[str] = None
    ) -> None:
        """Called when pipeline progress updates"""
        pass

    def on_status_change(self, pipeline: BasePipeline, status: PipelineStatus) -> None:
        """Called when pipeline status changes"""
        pass

    def on_error(self, pipeline: BasePipeline, error: str) -> None:
        """Called when pipeline encounters an error"""
        pass

    def on_completion(self, pipeline: BasePipeline, result: PipelineResult) -> None:
        """Called when pipeline completes"""
        pass


# Utility functions for pipeline management


def create_pipeline_output_dir(
    pipeline_name: str, base_dir: Path = Path("output")
) -> Path:
    """
    Create standardized output directory for pipeline

    Args:
        pipeline_name: Name of the pipeline
        base_dir: Base output directory

    Returns:
        Pipeline-specific output directory
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / pipeline_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def save_pipeline_result(result: PipelineResult, output_dir: Path) -> Path:
    """
    Save pipeline result to JSON file

    Args:
        result: Pipeline result to save
        output_dir: Output directory

    Returns:
        Path to saved result file
    """
    import json

    result_file = output_dir / "pipeline_result.json"
    with open(result_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    return result_file


def load_pipeline_result(result_file: Path) -> PipelineResult:
    """
    Load pipeline result from JSON file

    Args:
        result_file: Path to result file

    Returns:
        Loaded PipelineResult
    """
    import json

    with open(result_file, "r") as f:
        data = json.load(f)

    # Convert back to PipelineResult
    result = PipelineResult(
        success=data["success"],
        duration=data["duration"],
        metrics=data["metrics"],
        artifacts=[Path(p) for p in data["artifacts"]],
        errors=data["errors"],
        status=PipelineStatus(data["status"]),
    )

    return result
