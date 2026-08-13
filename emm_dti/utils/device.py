"""
Device management utilities for EMM-DTI.

Handles GPU/CPU device selection and configuration.
"""

import torch
import logging
from typing import Literal

logger = logging.getLogger(__name__)


def get_device(device: str | Literal["cuda", "cpu"] = "cuda") -> torch.device:
    """
    Get PyTorch device instance.

    Args:
        device: Device type ("cuda" or "cpu")

    Returns:
        torch.device instance

    Raises:
        ValueError: If device type is invalid
    """
    if device not in ["cuda", "cpu"]:
        raise ValueError(f"Invalid device: {device}. Must be 'cuda' or 'cpu'")

    if device == "cuda":
        if torch.cuda.is_available():
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
            return torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU")
            return torch.device("cpu")
    else:
        logger.info("Using CPU device")
        return torch.device("cpu")


def print_device_info() -> None:
    """Print detailed device information."""
    logger.info("=" * 60)
    logger.info("Device Information")
    logger.info("=" * 60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            logger.info(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)"
            )

    logger.info(f"CPUs available: {torch.get_num_threads()}")
    logger.info("=" * 60)


def empty_cuda_cache() -> None:
    """Clear CUDA cache to free GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")
