"""Open-Meteo provider — fetches air quality and weather data."""

import time
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import requests

from ingestion.providers.base import BaseProvider
from utils.config import get
from utils.time_utils import floor_hour, now_local, utc_to_local


class OpenMeteoProvider(BaseProvider):
    def __init__(self):
        super().__init__("open_meteo")
        self.lat = get("city.latitude", 25.396)
        self.lon = get("city.longitude", 68.357)
        self.tz = get("city.timezone", "Asia/Karachi")
        self.air_quality_url = get(
            "providers.open_meteo.base_url",
            "https://air-quality-api.open-meteo.com/v1/air-quality",
        )
        self.weather_url = get(
            "providers.open_meteo.weather_url",
            "https://api.open-meteo.com/v1/forecast",
        )
        self.timeout = get("providers.open_meteo.timeout_seconds", 30)

    def fetch_raw(self) -> Dict[str, Any]:
        """Fetch current air quality + weather from Open-Meteo."""
        aq_params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
            "hourly": "us_aqi,pm2_5,pm10",
            "timezone": self.tz,
            "forecast_days": 4,  # enough for 72h horizon
        }
        weather_params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,rain,cloud_cover",
            "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover",
            "timezone": self.tz,
            "forecast_days": 4,
        }

        self.logger.info("Fetching air quality data from Open-Meteo")
        aq_resp = requests.get(self.air_quality_url, params=aq_params, timeout=self.timeout)
        aq_resp.raise_for_status()
        aq_data = aq_resp.json()

        time.sleep(0.3)  # rate limiting

        self.logger.info("Fetching weather data from Open-Meteo")
        w_resp = requests.get(self.weather_url, params=weather_params, timeout=self.timeout)
        w_resp.raise_for_status()
        w_data = w_resp.json()

        return {"air_quality": aq_data, "weather": w_data, "fetched_at": datetime.now().isoformat()}

    def normalize(self, raw: Dict[str, Any]) -> pd.DataFrame:
        """Convert raw Open-Meteo response to normalized DataFrame."""
        aq = raw.get("air_quality", {})
        w = raw.get("weather", {})

        # Normalize current values into a single row
        current = {}
        if aq.get("current"):
            current.update(aq["current"])
        if w.get("current"):
            current.update(w["current"])

        # Also parse hourly forecast for future features
        hourly_records = []
        aq_hourly = aq.get("hourly", {})
        w_hourly = w.get("hourly", {})

        aq_times = aq_hourly.get("time", [])
        w_times = w_hourly.get("time", [])

        if aq_times:
            for i, t in enumerate(aq_times):
                dt = utc_to_local(pd.Timestamp(t).to_pydatetime())
                record = {
                    "timestamp": floor_hour(dt),
                    "us_aqi": aq_hourly.get("us_aqi", [None] * len(aq_times))[i],
                    "pm2_5": aq_hourly.get("pm2_5", [None] * len(aq_times))[i],
                    "pm10": aq_hourly.get("pm10", [None] * len(aq_times))[i],
                }
                hourly_records.append(record)

        # Merge weather hourly data
        if w_times:
            for i, t in enumerate(w_times):
                dt = utc_to_local(pd.Timestamp(t).to_pydatetime())
                dt_floor = floor_hour(dt)
                # Find matching record or create new one
                match = next((r for r in hourly_records if r["timestamp"] == dt_floor), None)
                if match:
                    match.update(
                        {
                            "temperature_2m": w_hourly.get("temperature_2m", [None] * len(w_times))[i],
                            "relative_humidity_2m": w_hourly.get("relative_humidity_2m", [None] * len(w_times))[i],
                            "dew_point_2m": w_hourly.get("dew_point_2m", [None] * len(w_times))[i],
                            "pressure_msl": w_hourly.get("pressure_msl", [None] * len(w_times))[i],
                            "wind_speed_10m": w_hourly.get("wind_speed_10m", [None] * len(w_times))[i],
                            "wind_direction_10m": w_hourly.get("wind_direction_10m", [None] * len(w_times))[i],
                            "precipitation": w_hourly.get("precipitation", [None] * len(w_times))[i],
                            "cloud_cover": w_hourly.get("cloud_cover", [None] * len(w_times))[i],
                        }
                    )
                else:
                    hourly_records.append(
                        {
                            "timestamp": dt_floor,
                            "temperature_2m": w_hourly.get("temperature_2m", [None] * len(w_times))[i],
                            "relative_humidity_2m": w_hourly.get("relative_humidity_2m", [None] * len(w_times))[i],
                            "dew_point_2m": w_hourly.get("dew_point_2m", [None] * len(w_times))[i],
                            "pressure_msl": w_hourly.get("pressure_msl", [None] * len(w_times))[i],
                            "wind_speed_10m": w_hourly.get("wind_speed_10m", [None] * len(w_times))[i],
                            "wind_direction_10m": w_hourly.get("wind_direction_10m", [None] * len(w_times))[i],
                            "precipitation": w_hourly.get("precipitation", [None] * len(w_times))[i],
                            "cloud_cover": w_hourly.get("cloud_cover", [None] * len(w_times))[i],
                        }
                    )

        df = pd.DataFrame(hourly_records) if hourly_records else pd.DataFrame([current])

        if "timestamp" not in df.columns:
            df["timestamp"] = floor_hour(now_local())

        df["source"] = "open_meteo"
        df["latitude"] = self.lat
        df["longitude"] = self.lon
        df["fetched_at"] = raw.get("fetched_at", datetime.now().isoformat())

        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate value ranges and flag issues."""
        if df.empty:
            self.logger.warning("Empty DataFrame from Open-Meteo")
            return df

        checks = {
            "relative_humidity_2m": (0, 100),
            "wind_speed_10m": (0, None),
            "precipitation": (0, None),
            "cloud_cover": (0, 100),
            "pm2_5": (0, None),
            "pm10": (0, None),
            "us_aqi": (0, None),
            "dew_point_2m": (-50, 60),
            "pressure_msl": (800, 1100),
        }

        for col, (lo, hi) in checks.items():
            if col in df.columns:
                mask = df[col].notna()
                if lo is not None:
                    df.loc[mask & (df[col] < lo), col] = None
                if hi is not None:
                    df.loc[mask & (df[col] > hi), col] = None

        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp")

        return df

    def fetch_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical hourly air quality + weather for backfill."""
        aq_params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
            "timezone": self.tz,
            "start_date": start_date,
            "end_date": end_date,
        }
        weather_params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover",
            "timezone": self.tz,
            "start_date": start_date,
            "end_date": end_date,
        }

        self.logger.info("Fetching historical Open-Meteo data: %s → %s", start_date, end_date)
        aq_resp = requests.get(self.air_quality_url, params=aq_params, timeout=self.timeout)
        aq_resp.raise_for_status()
        aq_data = aq_resp.json()

        time.sleep(0.5)
        w_resp = requests.get(self.weather_url, params=weather_params, timeout=self.timeout)
        w_resp.raise_for_status()
        w_data = w_resp.json()

        raw = {"air_quality": aq_data, "weather": w_data, "fetched_at": datetime.now().isoformat()}
        return self.normalize(raw)
