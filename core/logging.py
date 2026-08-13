"""Standard logging setup for all agonistes modules."""
from __future__ import annotations

import logging
import sys

from core.config import LOG_LEVEL

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = LOG_LEVEL, log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=_FORMAT,
    )
    return logging.getLogger(name)
