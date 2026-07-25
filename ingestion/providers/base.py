"""Abstract base class for data providers."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.config import get
from utils.logging import get_logger

DATA_DIR = Path(get("storage.data_dir", "./data"))


class BaseProvider(ABC):
    """Each provider must implement fetch_raw, normalize, and validate."""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"provider.{name}")
        self.raw_dir = DATA_DIR / "raw" / name

    @abstractmethod
    def fetch_raw(self) -> Dict[str, Any]:
        """Fetch raw data from the provider API. Returns dict or list."""
        ...

    @abstractmethod
    def normalize(self, raw: Dict[str, Any]) -> pd.DataFrame:
        """Convert raw API response to normalized DataFrame with consistent columns."""
        ...

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges, remove impossible values, flag issues."""
        ...

    def run(self) -> pd.DataFrame:
        """Full pipeline: fetch → normalize → validate → save raw."""
        raw = self.fetch_raw()
        self._save_raw(raw)
        df = self.normalize(raw)
        df = self.validate(df)
        return df

    def _save_raw(self, raw: Dict[str, Any]) -> None:
        now = datetime.now()
        path = (
            self.raw_dir
            / str(now.year)
            / f"{now.month:02d}"
            / f"{now.day:02d}"
            / f"{now.hour:02d}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(path, "w") as f:
            json.dump(raw, f, indent=2, default=str)
        self.logger.info("Saved raw data: %s", path)

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
