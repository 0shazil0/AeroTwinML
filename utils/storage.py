"""Storage helpers — read/write Parquet and JSON files."""

import json
from pathlib import Path

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


def save_json(data: dict | list, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug("Saved JSON: %s", filepath)


def load_json(filepath: Path) -> dict | list:
    with open(filepath) as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    logger.debug("Saved Parquet: %s (%d rows)", filepath, len(df))


def load_parquet(filepath: Path) -> pd.DataFrame:
    return pd.read_parquet(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.debug("Saved CSV: %s (%d rows)", filepath, len(df))


def load_csv(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath, parse_dates=True)
