"""AQI route handlers."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from backend.schemas import (
    AQIReading,
    APIResponse,
    ForecastPoint,
    ForecastResponse,
    HistoricalDataPoint,
    WeatherSnapshot,
)
from utils.aqi_utils import classify_aqi, category_advice, category_color
from utils.config import get
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

DATA_DIR = Path(get("storage.data_dir", "./data"))


def _load_latest_forecast() -> Optional[Dict]:
    path = DATA_DIR / "processed" / "predictions" / "forecast_latest.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _load_merged_data(hours: int = 168) -> List[Dict]:
    """Load recent historical data from merged table."""
    path = DATA_DIR / "processed" / "merged_hourly" / "merged_latest.parquet"
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            cutoff = datetime.now() - timedelta(hours=hours)
            df = df[df["timestamp"] >= cutoff]
        return df.tail(hours * 2).to_dict(orient="records")
    except Exception:
        return []


@router.get("/aqi/current", response_model=APIResponse)
async def get_current_aqi():
    """Get current AQI reading with weather snapshot."""
    forecast = _load_latest_forecast()

    if forecast:
        current_aqi = forecast.get("current_aqi", 0)
        category = classify_aqi(current_aqi)

        # Build weather snapshot
        weather = None
        merged = _load_merged_data(hours=1)
        if merged:
            latest = merged[-1]
            weather = WeatherSnapshot(
                temperature=latest.get("temperature_2m"),
                humidity=latest.get("relative_humidity_2m"),
                pressure=latest.get("pressure_msl"),
                wind_speed=latest.get("wind_speed_10m"),
                wind_direction=latest.get("wind_direction_10m"),
                precipitation=latest.get("precipitation"),
                cloud_cover=latest.get("cloud_cover"),
            )

        data = {
            "aqi": current_aqi,
            "category": category.value,
            "category_color": category_color(category),
            "health_advice": category_advice(category),
            "dominant_pollutant": latest.get("dominant_pollutant", "PM2.5") if merged else "PM2.5",
            "pm2_5": latest.get("pm2_5") if merged else None,
            "pm10": latest.get("pm10") if merged else None,
            "weather": weather.model_dump() if weather else None,
            "updated_at": forecast.get("timestamp"),
        }
    else:
        data = {
            "aqi": 0,
            "category": "Unknown",
            "category_color": "#808080",
            "health_advice": "No data available yet. Run the pipeline first.",
            "dominant_pollutant": None,
            "weather": None,
            "updated_at": None,
        }

    return APIResponse(data=data, meta={"city": get("city.name", "Hyderabad")})


@router.get("/aqi/forecast", response_model=APIResponse)
async def get_forecast():
    """Get 72-hour AQI forecast."""
    forecast = _load_latest_forecast()

    if not forecast:
        return APIResponse(
            data={"message": "No forecast available. Run the pipeline first."},
            status="no_data",
        )

    fc = forecast.get("forecast", {})
    data = {
        "current_aqi": forecast.get("current_aqi", 0),
        "forecast": {
            "24h": fc.get("24h", {}),
            "48h": fc.get("48h", {}),
            "72h": fc.get("72h", {}),
        },
        "model_info": forecast.get("model_info", {}),
        "timestamp": forecast.get("timestamp"),
    }

    return APIResponse(data=data)


@router.get("/aqi/history", response_model=APIResponse)
async def get_history(
    hours: int = Query(default=168, ge=1, le=720, description="Hours of history to return"),
):
    """Get historical AQI data."""
    records = _load_merged_data(hours)

    if not records:
        return APIResponse(data=[], status="no_data")

    history = []
    for r in records:
        ts = r.get("timestamp")
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()

        aqi_val = r.get("aqi") or r.get("us_aqi")
        history.append(
            {
                "timestamp": ts,
                "aqi": aqi_val,
                "pm2_5": r.get("pm2_5"),
                "pm10": r.get("pm10"),
                "temperature": r.get("temperature_2m"),
                "humidity": r.get("relative_humidity_2m"),
                "wind_speed": r.get("wind_speed_10m"),
            }
        )

    return APIResponse(data=history, meta={"hours": hours, "count": len(history)})


@router.get("/aqi/pollutants", response_model=APIResponse)
async def get_pollutant_breakdown():
    """Get current pollutant breakdown."""
    merged = _load_merged_data(hours=1)

    if not merged:
        return APIResponse(data={}, status="no_data")

    latest = merged[-1]
    data = {
        "pm2_5": latest.get("pm2_5"),
        "pm10": latest.get("pm10"),
        "no2": latest.get("no2"),
        "o3": latest.get("o3"),
        "so2": latest.get("so2"),
        "co": latest.get("co"),
        "dominant_pollutant": latest.get("dominant_pollutant", "PM2.5"),
        "timestamp": latest.get("timestamp"),
    }

    return APIResponse(data=data)
