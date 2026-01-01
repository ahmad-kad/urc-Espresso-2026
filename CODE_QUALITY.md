# Code Quality Standards

## Automated Checks

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m e2e
```

### Linting
```bash
# Run all linters
flake8 . --exclude=tests,venv,env
black --check .
isort --check .
mypy . --ignore-missing-imports
```

### Auto-fix
```bash
black .
isort .
```

## Quality Gates

### Before Commit
- [ ] All unit tests pass
- [ ] Code formatted (black, isort)
- [ ] No linting errors
- [ ] Type hints added where appropriate

### Before Merge
- [ ] All tests pass (unit + integration)
- [ ] Coverage > 80%
- [ ] No critical linting errors
- [ ] Documentation updated

## Debug Mode

### Enable Debug Logging
```bash
export DEBUG=1
python your_script.py
```

### Logging Levels
- **DEBUG**: Detailed diagnostics (debug mode only)
- **INFO**: General operations (default)
- **WARNING**: Potential issues
- **ERROR**: Errors preventing features
- **CRITICAL**: Serious errors

## Code Standards

### Style
- Follow PEP 8
- Use Black for formatting
- Use isort for imports
- Type hints for function signatures

### Testing
- Unit tests for all utilities
- Integration tests for workflows
- E2E tests for critical paths
- Mock external dependencies

### Documentation
- Docstrings for all functions
- Type hints in signatures
- README for major features
- Inline comments for complex logic

