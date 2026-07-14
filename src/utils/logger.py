"""Logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str, log_file: str | Path | None = None, rank: int = 0) -> logging.Logger:
    """Create a console/file logger without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None and rank == 0:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

