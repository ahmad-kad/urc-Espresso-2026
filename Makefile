.PHONY: test test-unit test-integration test-e2e test-cov lint format quality check

# Testing
test:
	pytest

test-unit:
	pytest tests/unit/ -m unit

test-integration:
	pytest tests/integration/ -m integration

test-e2e:
	pytest tests/e2e/ -m e2e

test-cov:
	pytest --cov=. --cov-report=html:output/testing/coverage --cov-report=term-missing

test-debug:
	DEBUG=1 pytest -v

# Code Quality
lint:
	flake8 . --exclude=tests,venv,env
	black --check .
	isort --check .
	mypy . --ignore-missing-imports

format:
	black .
	isort .

quality: lint test-cov

# Quick check before commit
check: format lint test-unit

