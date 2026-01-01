# Testing Guide

## Overview

This project includes comprehensive testing with unit tests, integration tests, and end-to-end tests.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
│   ├── test_detector.py
│   ├── test_trainer.py
│   ├── test_detection_utils.py
│   └── test_metrics.py
├── integration/      # Integration tests for workflows
│   └── test_training_workflow.py
├── e2e/              # End-to-end tests
│   └── test_complete_pipeline.py
└── utils/            # Test utilities
    └── test_logging.py
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run by Category
```bash
# Unit tests only
pytest tests/unit/ -m unit

# Integration tests
pytest tests/integration/ -m integration

# End-to-end tests
pytest tests/e2e/ -m e2e
```

### Run with Coverage
```bash
pytest --cov=. --cov-report=html
```

### Run in Debug Mode
```bash
DEBUG=1 pytest -v
```

### Run Specific Test
```bash
pytest tests/unit/test_detector.py::TestObjectDetector::test_init_with_yolov8
```

## Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_gpu` - Tests requiring GPU
- `@pytest.mark.requires_model` - Tests requiring trained models
- `@pytest.mark.debug` - Debug mode tests

## Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=1
python your_script.py
```

Or in code:
```python
import os
os.environ['DEBUG'] = '1'
```

## Logging Levels

- **DEBUG**: Detailed diagnostic info (only in debug mode)
- **INFO**: General operational messages (default)
- **WARNING**: Potential issues that don't stop execution
- **ERROR**: Errors that prevent a feature from working
- **CRITICAL**: Serious errors that may stop the program

## Code Quality

### Run Linters
```bash
flake8 .
black --check .
isort --check .
mypy .
```

### Auto-fix Issues
```bash
black .
isort .
```

## Continuous Integration

Tests should pass before merging:
- All unit tests
- All integration tests
- Code coverage > 80%
- No linting errors

