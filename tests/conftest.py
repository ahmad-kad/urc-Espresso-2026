"""
Pytest configuration and shared fixtures
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors in tests
    format="%(levelname)s: %(message)s",
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing"""
    return {
        "model": {
            "architecture": "yolov8n",
            "input_size": 224,
            "confidence_threshold": 0.5,
            "iou_threshold": 0.4,
            "max_detections": 20,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "device": "cpu",
        },
        "data": {"classes": ["class0", "class1"], "num_classes": 2},
        "project": {"name": "test_project"},
    }


@pytest.fixture
def debug_mode(monkeypatch):
    """Enable debug mode for tests"""
    monkeypatch.setenv("DEBUG", "1")
    yield
    monkeypatch.delenv("DEBUG", raising=False)


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data"""
    return project_root / "tests" / "test_data"


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model path for testing"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir / "test_model.pt")
