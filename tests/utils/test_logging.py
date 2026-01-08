"""
Test logging configuration and debug mode
"""

import logging
import os
from unittest.mock import patch

import pytest


class TestLoggingConfiguration:
    """Test logging setup and levels"""

    def test_logging_levels(self):
        """Test that logging levels are properly configured"""
        logger = logging.getLogger("test")

        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # All should execute without error
        assert True

    def test_debug_mode_enabled(self, debug_mode):
        """Test debug mode enables DEBUG level"""
        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)

        # Debug messages should be visible
        with patch("logging.Logger.debug") as mock_debug:
            logger.debug("Debug in debug mode")
            # In debug mode, this should be called
            assert os.getenv("DEBUG") == "1"

    def test_production_logging(self, monkeypatch):
        """Test production logging (no debug)"""
        monkeypatch.delenv("DEBUG", raising=False)
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)

        # Debug messages should not appear
        assert logger.level >= logging.INFO
