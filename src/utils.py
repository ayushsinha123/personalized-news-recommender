"""
Shared utility helpers: logging setup and JSON I/O.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

from src import config


def get_logger(name: str = "news_recommender") -> logging.Logger:
    """
    Return a configured logger. Safe to call multiple times — handlers
    are only attached once per logger name.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(config.LOG_LEVEL)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = get_logger()


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a dictionary to disk as pretty-printed JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        # numpy scalars / arrays -> native python
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)

    logger.info(f"Saved JSON to {path}")


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)