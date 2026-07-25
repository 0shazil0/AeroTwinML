"""Structured logging utility."""

import logging
import sys
from pathlib import Path

from utils.config import get

_LOG_FORMAT = get("logging.format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
_LOG_LEVEL = get("logging.level", "INFO")


def setup_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
