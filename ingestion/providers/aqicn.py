"""AQICN provider — fetches observed station-level AQI data."""

import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import requests

from ingestion.providers.base import BaseProvider
from utils.config import get
from utils.time_utils import floor_hour, now_local, utc_to_local


class AQICNProvider(BaseProvider):
    def __init__(self):
        super().__init__("aqicn")
        self.token = os.getenv("AQICN_TOKEN", "")
        self.station = get("providers.aqicn.station_id", "A546205")
        self.base_url = get("providers.aqicn.base_url", "https://api.waqi.info")
        self.timeout = get("providers.aqicn.timeout_seconds", 30)

    def fetch_raw(self) -> Dict[str, Any]:
        """Fetch current station feed from AQICN."""
        url = f"{self.base_url}/feed/{self.station}/"
        params = {"token": self.token}

        self.logger.info("Fetching AQICN data for station %s", self.station)
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            raise RuntimeError(f"AQICN API error: {data.get('data', 'unknown error')}")

        return {"raw": data, "fetched_at": datetime.now().isoformat()}

    def normalize(self, raw: Dict[str, Any]) -> pd.DataFrame:
        """Convert raw AQICN response to normalized DataFrame."""
        data = raw.get("raw", {}).get("data", {})
        iaqi = data.get("iaqi", {})
        forecast = data.get("forecast", {})
        time_info = data.get("time", {})

        record = {
            "timestamp": self._parse_time(time_info),
            "station_name": data.get("city", {}).get("name", self.station),
            "city": data.get("city", {}).get("name", "Hyderabad"),
            "country": data.get("city", {}).get("country", "PK"),
            "aqi": self._safe_float(data.get("aqi")),
            "pm2_5": self._safe_float(iaqi.get("pm25", {}).get("v") if isinstance(iaqi.get("pm25"), dict) else iaqi.get("pm25")),
            "pm10": self._safe_float(iaqi.get("pm10", {}).get("v") if isinstance(iaqi.get("pm10"), dict) else iaqi.get("pm10")),
            "no2": self._safe_float(iaqi.get("no2", {}).get("v") if isinstance(iaqi.get("no2"), dict) else iaqi.get("no2")),
            "o3": self._safe_float(iaqi.get("o3", {}).get("v") if isinstance(iaqi.get("o3"), dict) else iaqi.get("o3")),
            "so2": self._safe_float(iaqi.get("so2", {}).get("v") if isinstance(iaqi.get("so2"), dict) else iaqi.get("so2")),
            "co": self._safe_float(iaqi.get("co", {}).get("v") if isinstance(iaqi.get("co"), dict) else iaqi.get("co")),
            "dominant_pollutant": data.get("dominentpol", ""),
            "source": "aqicn",
        }

        df = pd.DataFrame([record])
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate value ranges."""
        if df.empty:
            return df

        checks = {
            "aqi": (0, 999),
            "pm2_5": (0, 1000),
            "pm10": (0, 1000),
            "no2": (0, 500),
            "o3": (0, 500),
            "so2": (0, 500),
            "co": (0, 200),
        }

        for col, (lo, hi) in checks.items():
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask & (df[col] < lo), col] = None
                df.loc[mask & (df[col] > hi), col] = None

        return df

    def _parse_time(self, time_info: Dict[str, Any]) -> datetime:
        iso = time_info.get("iso")
        if iso:
            return floor_hour(utc_to_local(pd.Timestamp(iso).to_pydatetime()))

        s = time_info.get("s")
        if s:
            try:
                return floor_hour(utc_to_local(pd.Timestamp(s).to_pydatetime()))
            except Exception:
                pass

        return floor_hour(now_local())
