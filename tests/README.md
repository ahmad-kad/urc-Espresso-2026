# Testing Framework

## Overview

Comprehensive testing framework with unit, integration, and end-to-end tests.

## Test Structure

```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_detector.py     # ObjectDetector tests
│   ├── test_trainer.py      # ModelTrainer tests
│   ├── test_detection_utils.py  # Detection utilities
│   └── test_metrics.py      # Metrics calculations
│
├── integration/             # Integration tests (component interactions)
│   └── test_training_workflow.py  # Training workflows
│
├── e2e/                     # End-to-end tests (complete pipelines)
│   └── test_complete_pipeline.py  # Full pipeline tests
│
└── utils/                   # Test utilities
    └── test_logging.py     # Logging configuration tests
```

## Quick Start

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific category
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m e2e           # End-to-end tests

# Run in debug mode
DEBUG=1 pytest -v
```

## Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=1
pytest -v
```

Or in Python:
```python
import os
os.environ['DEBUG'] = '1'
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_gpu` - GPU required
- `@pytest.mark.requires_model` - Trained model required
- `@pytest.mark.debug` - Debug mode tests

## Coverage Goals

- **Unit Tests**: >90% coverage
- **Integration Tests**: Critical workflows covered
- **E2E Tests**: Main pipelines validated

## Continuous Integration

Tests run automatically on:
- Push to main/develop
- Pull requests
- Multiple Python versions (3.8-3.11)

