"""
Logging utilities for EMM-DTI.

Provides consistent logging configuration across the project.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str | Path = "results/logs",
    level: str = "INFO",
    name: str = "emm_dti",
) -> logging.Logger:
    """
    Configure logging for EMM-DTI experiments.

    Args:
        log_dir: Directory to save log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        name: Logger name

    Returns:
        Configured logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (rotating)
    log_file = log_dir / "emm_dti.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10_000_000,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log directory: {log_dir}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
