"""
Device management utilities
Ensures consistent device handling and prevents thrashing
"""

import os
from typing import Optional

import torch

from utils.logger_config import get_logger

logger = get_logger(__name__, debug=os.getenv("DEBUG") == "1")

# Cache device availability to prevent repeated checks
_cuda_available_cache: Optional[bool] = None


def get_device(preference: Optional[str] = None) -> str:
    """
    Get device with caching to prevent thrashing

    Args:
        preference: Preferred device ('cpu', 'cuda', 'auto', or None)

    Returns:
        Device string ('cpu' or 'cuda')
    """
    global _cuda_available_cache

    if preference == "cpu":
        return "cpu"

    if preference == "cuda":
        # Check if CUDA is actually available
        if _cuda_available_cache is None:
            _cuda_available_cache = torch.cuda.is_available()

        if _cuda_available_cache:
            return "cuda"
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"

    # Auto mode or None - check once and cache
    if _cuda_available_cache is None:
        _cuda_available_cache = torch.cuda.is_available()
        logger.debug(f"CUDA availability cached: {_cuda_available_cache}")

    return "cuda" if _cuda_available_cache else "cpu"


def resolve_device(
    config_device: Optional[str], training_device: Optional[str] = None
) -> str:
    """
    Resolve device with priority: explicit > training > auto

    Args:
        config_device: Device from main config
        training_device: Device from training config (lower priority)

    Returns:
        Resolved device string
    """
    # Priority: explicit config > training config > auto
    if config_device and config_device != "auto":
        return get_device(config_device)

    if training_device and training_device != "auto":
        return get_device(training_device)

    # Default to auto
    return get_device("auto")


def ensure_device_consistency(model_device: str, tensor_device: str) -> bool:
    """
    Check if model and tensor are on same device

    Args:
        model_device: Device of the model
        tensor_device: Device of the tensor

    Returns:
        True if devices match, False otherwise
    """
    if model_device != tensor_device:
        logger.warning(
            f"Device mismatch: model on {model_device}, tensor on {tensor_device}. "
            "This may cause runtime errors."
        )
        return False
    return True


def clear_device_cache():
    """Clear device availability cache (useful for testing)"""
    global _cuda_available_cache
    _cuda_available_cache = None
