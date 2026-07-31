"""Hourly pipeline — serverless ingestion, features → Hopsworks FS, inference, alerts.

Runs on GitHub Actions every hour. No persistent server needed.
Features are written to Hopsworks Feature Store for training availability.
Inference uses model from Hopsworks Model Registry (with local fallback).
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from feature_store.feature_builder import FeatureBuilder
from feature_store.hopsworks_client import (
    write_feature_group,
    is_available as hopsworks_available,
)
from ingestion.orchestrator import IngestionOrchestrator
from models.inference import InferenceEngine
from utils.aqi_utils import is_alert_level
from utils.config import get
from utils.logging import setup_logger
from utils.storage import save_json, save_parquet, load_parquet
from utils.time_utils import format_iso, now_local

logger = setup_logger("hourly_pipeline")

DATA_DIR = Path(get("storage.data_dir", "./data"))
PREDICTIONS_DIR = DATA_DIR / "processed" / "predictions"
MERGED_DIR = DATA_DIR / "processed" / "merged_hourly"
ALERTS_FILE = DATA_DIR / "raw" / "logs" / "alerts.jsonl"
STATUS_FILE = DATA_DIR / "processed" / "pipeline_status.json"


def run_hourly_pipeline() -> dict:
    status = {
        "pipeline": "hourly",
        "started_at": format_iso(now_local()),
        "steps": {},
        "success": False,
        "backend": "hopsworks" if hopsworks_available() else "local",
    }

    try:
        # Step 1: Ingestion (Open-Meteo + OpenAQ + AQICN)
        logger.info("=== Step 1: Data Ingestion ===")
        orchestrator = IngestionOrchestrator()
        merged = orchestrator.run_full_cycle()
        status["steps"]["ingestion"] = {
            "status": "ok",
            "rows_fetched": len(merged),
        }
        logger.info("Ingestion complete: %d rows", len(merged))

        if merged.empty:
            status["steps"]["ingestion"]["status"] = "empty"
            status["completed_at"] = format_iso(now_local())
            _save_status(status)
            return status

        # Step 2: Feature Engineering
        logger.info("=== Step 2: Feature Engineering ===")
        builder = FeatureBuilder(merged)
        featured = builder.build_all()

        # Write features to Hopsworks Feature Store (serverless)
        if hopsworks_available():
            write_feature_group(
                name="aqi_features",
                df=featured,
                version=1,
                description="Hourly AQI features: time, lags, rolling, weather, interactions, targets",
                primary_key=["timestamp"],
                online_enabled=True,
            )
            logger.info("Features written to Hopsworks FS")

        # Also save locally as fallback
        feature_path = PREDICTIONS_DIR / f"features_{now_local().strftime('%Y%m%d_%H')}.parquet"
        save_parquet(featured, feature_path)

        status["steps"]["features"] = {
            "status": "ok",
            "features_generated": len(featured.columns),
            "rows": len(featured),
        }

        # Step 3: Inference (model from Hopsworks MR or local fallback)
        logger.info("=== Step 3: Inference ===")
        engine = InferenceEngine()
        forecast = engine.predict(featured)

        # Embed weather + pollutant data from the latest merged row
        _embed_weather_and_pollutants(forecast, merged)

        # Embed 7-day history for dashboard (avoids needing parquet on deployment)
        _embed_history(forecast)

        forecast_path = PREDICTIONS_DIR / f"forecast_{now_local().strftime('%Y%m%d_%H')}.json"
        save_json(forecast, forecast_path)
        save_json(forecast, PREDICTIONS_DIR / "forecast_latest.json")

        status["steps"]["inference"] = {
            "status": "ok",
            "current_aqi": forecast.get("current_aqi", 0),
            "forecast_24h": forecast.get("forecast", {}).get("24h", {}).get("aqi"),
        }

        # Step 4: Alerts
        logger.info("=== Step 4: Alerts Check ===")
        alerts = _check_alerts(forecast)
        if alerts:
            _save_alerts(alerts)
            status["steps"]["alerts"] = {"status": "triggered", "alerts": alerts}
            logger.warning("ALERTS TRIGGERED: %s", alerts)
        else:
            status["steps"]["alerts"] = {"status": "ok", "alerts": []}

        # Step 5: Quality
        logger.info("=== Step 5: Quality Check ===")
        quality = _quality_check(merged)
        quality_path = DATA_DIR / "quality" / f"quality_{now_local().strftime('%Y%m%d_%H')}.json"
        save_json(quality, quality_path)
        status["steps"]["quality"] = {"status": "ok", "issues": quality.get("issues_count", 0)}

        status["success"] = True

    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        status["success"] = False
        status["error"] = str(e)

    status["completed_at"] = format_iso(now_local())
    _save_status(status)
    return status


def _check_alerts(forecast: dict) -> list:
    alerts = []
    current = forecast.get("current_aqi", 0)
    if is_alert_level(current):
        alerts.append({
            "type": "current_aqi",
            "aqi": current,
            "level": "hazardous" if current >= 300 else "very_unhealthy",
            "timestamp": forecast.get("timestamp"),
        })
    for horizon, data in forecast.get("forecast", {}).items():
        if data.get("alert"):
            alerts.append({
                "type": f"forecast_{horizon}",
                "aqi": data.get("aqi"),
                "category": data.get("category"),
                "horizon": horizon,
                "timestamp": forecast.get("timestamp"),
            })
    return alerts


def _save_alerts(alerts: list):
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_FILE, "a") as f:
        for alert in alerts:
            f.write(json.dumps(alert) + "\n")


def _quality_check(df) -> dict:
    issues = []
    if df.empty:
        issues.append("empty_dataset")
    for col in ["aqi", "pm2_5", "pm10"]:
        if col in df.columns and df[col].notna().sum() == 0:
            issues.append(f"missing_{col}")
    return {
        "timestamp": format_iso(now_local()),
        "rows": len(df),
        "columns": list(df.columns),
        "issues": issues,
        "issues_count": len(issues),
    }


def _save_status(status: dict):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_json(status, STATUS_FILE)


def _to_json_safe(v):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if pd.notna(v) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if pd.isna(v):
        return None
    # Ensure strings that look like numbers are converted to numbers
    if isinstance(v, str):
        try:
            return float(v)
        except (ValueError, TypeError):
            return v
    return v


def _embed_weather_and_pollutants(forecast: dict, merged: pd.DataFrame):
    """Extract weather + pollutant data from the latest merged row and embed in forecast."""
    if merged.empty:
        return

    latest = merged.iloc[-1]

    forecast["weather"] = {
        "temperature": _to_json_safe(latest.get("temperature_2m")),
        "humidity": _to_json_safe(latest.get("relative_humidity_2m")),
        "pressure": _to_json_safe(latest.get("pressure_msl")),
        "wind_speed": _to_json_safe(latest.get("wind_speed_10m")),
        "wind_direction": _to_json_safe(latest.get("wind_direction_10m")),
        "precipitation": _to_json_safe(latest.get("precipitation")),
        "cloud_cover": _to_json_safe(latest.get("cloud_cover")),
    }

    forecast["pollutants"] = {
        "pm2_5": _to_json_safe(latest.get("pm2_5")),
        "pm10": _to_json_safe(latest.get("pm10")),
        "no2": _to_json_safe(latest.get("no2")),
        "o3": _to_json_safe(latest.get("o3")),
        "so2": _to_json_safe(latest.get("so2")),
        "co": _to_json_safe(latest.get("co")),
    }

    forecast["station"] = str(latest.get("station_name", "OpenAQ/4889110"))
    logger.info("Embedded weather + pollutant data in forecast JSON")


def _embed_history(forecast: dict, hours: int = 168):
    """Load merged parquet and embed last N hours as compact history array.

    This lets the dashboard show historical trends without needing the parquet file
    on the deployment platform.
    """
    merged_path = MERGED_DIR / "merged_latest.parquet"
    try:
        if not merged_path.exists():
            return

        df = pd.read_parquet(merged_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Use tz-aware cutoff to match tz-aware parquet timestamps
            cutoff = pd.Timestamp.now(tz="UTC") - timedelta(hours=hours)
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
            df = df[df["timestamp"] >= cutoff]

        # Only keep columns the dashboard needs
        keep_cols = [
            "timestamp", "aqi", "pm2_5", "pm10", "no2", "o3", "so2", "co",
            "temperature_2m", "relative_humidity_2m", "pressure_msl",
            "wind_speed_10m", "wind_direction_10m", "precipitation", "cloud_cover",
            "om_forecast_aqi", "dominant_pollutant",
        ]
        available = [c for c in keep_cols if c in df.columns]
        df = df[available].tail(hours * 2)

        # Convert to JSON-safe records
        records = []
        for _, row in df.iterrows():
            rec = {}
            for col in available:
                rec[col] = _to_json_safe(row[col])
            records.append(rec)

        forecast["history"] = records
        logger.info("Embedded %d history rows in forecast JSON", len(records))

    except Exception as e:
        logger.warning("Could not embed history: %s", e)


if __name__ == "__main__":
    result = run_hourly_pipeline()
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["success"] else 1)
