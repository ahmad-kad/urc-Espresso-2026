"""
Centralized logging configuration
Small but meaningful logs with proper levels
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: Optional[str] = None, debug: bool = False, log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup centralized logging configuration

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        debug: Enable debug mode (overrides level to DEBUG)
        log_file: Optional file path for logging

    Returns:
        Configured logger
    """
    # Determine log level
    if debug or os.getenv("DEBUG") == "1":
        log_level = logging.DEBUG
    elif level:
        log_level = getattr(logging, level.upper(), logging.INFO)
    else:
        log_level = logging.INFO

    # Create formatter - small but meaningful
    formatter = logging.Formatter(
        fmt="%(levelname)s [%(name)s]: %(message)s", datefmt="%H:%M:%S"
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler - always present
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler - if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str, debug: bool = False) -> logging.Logger:
    """
    Get a logger with appropriate level

    Args:
        name: Logger name (typically __name__)
        debug: Enable debug mode for this logger

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    if debug or os.getenv("DEBUG") == "1":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger


# Logging levels guide:
# DEBUG: Detailed diagnostic info (only in debug mode)
# INFO: General operational messages (default)
# WARNING: Potential issues that don't stop execution
# ERROR: Errors that prevent a feature from working
# CRITICAL: Serious errors that may stop the program
