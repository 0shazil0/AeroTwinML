"""Flask dashboard application — standalone, no FastAPI dependency needed.

Reads forecast and history directly from data files produced by GitHub Actions.
Designed for deployment on Render / Streamlit Cloud / any free Python host.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path — needed when running from dashboards/ subdir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template
import os
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from utils.aqi_utils import classify_aqi, category_advice, category_color, is_alert_level
from utils.config import get
from utils.logging import get_logger

logger = get_logger(__name__)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
FORECAST_PATH = DATA_DIR / "processed" / "predictions" / "forecast_latest.json"
MERGED_PATH = DATA_DIR / "processed" / "merged_hourly" / "merged_latest.parquet"
PIPELINE_PATH = DATA_DIR / "processed" / "pipeline_status.json"
DAILY_PATH = DATA_DIR / "processed" / "daily_status.json"
ALERTS_PATH = DATA_DIR / "raw" / "logs" / "alerts.jsonl"


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _load_forecast() -> dict:
    return _load_json(FORECAST_PATH)


def _load_merged(hours: int = 168) -> list:
    try:
        df = pd.read_parquet(MERGED_PATH)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            cutoff = datetime.now() - timedelta(hours=hours)
            df = df[df["timestamp"] >= cutoff]
        # Convert to JSON-safe format
        records = df.tail(hours * 2).to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if isinstance(v, (np.integer,)):
                    r[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    r[k] = float(v) if pd.notna(v) else None
                elif isinstance(v, pd.Timestamp):
                    r[k] = v.isoformat()
                elif pd.isna(v):
                    r[k] = None
        return records
    except (FileNotFoundError, Exception):
        return []


def _load_alerts(limit: int = 10) -> list:
    alerts = []
    if ALERTS_PATH.exists():
        with open(ALERTS_PATH) as f:
            lines = f.readlines()
        for line in lines[-limit:]:
            try:
                alerts.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        alerts.reverse()
    return alerts


# ─── Routes ───────────────────────────────────────────────────

@app.route("/")
def home():
    forecast = _load_forecast()
    merged = _load_merged(hours=24)
    alerts = _load_alerts(3)

    current_aqi = forecast.get("current_aqi", 0)
    category = classify_aqi(current_aqi)

    aqi_data = {
        "aqi": current_aqi,
        "category": category.value,
        "category_color": category_color(category),
        "health_advice": category_advice(category),
        "dominant_pollutant": "PM2.5",
        "pm2_5": None,
        "pm10": None,
        "updated_at": forecast.get("timestamp"),
    }

    # Weather from forecast JSON (embedded by daily pipeline) or from merged parquet
    weather = forecast.get("weather")
    if weather is None:
        merged = _load_merged(hours=1)
        if merged:
            latest = merged[-1]
            weather = {
                "temperature": latest.get("temperature_2m"),
                "humidity": latest.get("relative_humidity_2m"),
                "pressure": latest.get("pressure_msl"),
                "wind_speed": latest.get("wind_speed_10m"),
                "wind_direction": latest.get("wind_direction_10m"),
                "precipitation": latest.get("precipitation"),
                "cloud_cover": latest.get("cloud_cover"),
            }
            aqi_data["pm2_5"] = latest.get("pm2_5")
            aqi_data["pm10"] = latest.get("pm10")
            aqi_data["dominant_pollutant"] = latest.get("dominant_pollutant", "PM2.5")

    # Pollutant data from forecast JSON if available
    forecast_pollutants = forecast.get("pollutants", {})
    if forecast_pollutants:
        aqi_data["pm2_5"] = forecast_pollutants.get("pm2_5") or aqi_data.get("pm2_5")
        aqi_data["pm10"] = forecast_pollutants.get("pm10") or aqi_data.get("pm10")

    aqi_data["weather"] = weather

    # Check alerts
    if current_aqi >= 200 and not any(
        a.get("type") == "current_aqi" for a in alerts[-3:] if alerts
    ):
        alerts.insert(0, {
            "type": "current_aqi",
            "aqi": current_aqi,
            "level": "hazardous" if current_aqi >= 300 else "very_unhealthy",
            "timestamp": forecast.get("timestamp"),
        })

    return render_template(
        "index.html",
        aqi=aqi_data,
        forecast=forecast,
        alerts=alerts,
        city=get("city.name", "Hyderabad"),
    )


@app.route("/forecast")
def forecast():
    forecast = _load_forecast()
    history = _load_merged(hours=168)
    merged = _load_merged(hours=1)
    pollutants = {}
    if merged:
        latest = merged[-1]
        pollutants = {
            "pm2_5": latest.get("pm2_5"),
            "pm10": latest.get("pm10"),
            "no2": latest.get("no2"),
            "o3": latest.get("o3"),
            "so2": latest.get("so2"),
            "co": latest.get("co"),
            "dominant_pollutant": latest.get("dominant_pollutant", "PM2.5"),
        }

    return render_template(
        "forecast.html",
        forecast=forecast,
        history=history,
        pollutants=pollutants,
        city=get("city.name", "Hyderabad"),
    )


@app.route("/analytics")
def analytics():
    history = _load_merged(hours=720)
    return render_template(
        "analytics.html",
        history=history,
        city=get("city.name", "Hyderabad"),
    )


@app.route("/explainability")
def explainability():
    # Correlation-based explanation as fallback
    merged = _load_merged(hours=168)
    explanation = {"top_drivers": [], "natural_language": "", "method": "none"}

    if merged:
        df = pd.DataFrame(merged)
        aqi_col = "aqi" if "aqi" in df.columns else "om_forecast_aqi"
        if aqi_col in df.columns:
            numeric = df.select_dtypes(include=[np.number])
            if aqi_col in numeric.columns and len(numeric) > 1:
                corr = numeric.corr()[aqi_col].dropna().drop(aqi_col, errors="ignore")
                drivers = []
                for feat, val in corr.abs().sort_values(ascending=False).head(10).items():
                    drivers.append({
                        "feature": feat,
                        "importance": round(abs(val), 4),
                        "correlation": round(corr[feat], 4),
                        "direction": "positive" if corr[feat] > 0 else "negative",
                    })
                top = drivers[0] if drivers else None
                nl = ""
                if top:
                    direction = "increase" if top["direction"] == "positive" else "decrease"
                    nl = f"AQI shows the strongest correlation with {top['feature'].replace('_',' ')} (r={top['correlation']:.3f}, {direction})."
                explanation = {
                    "top_drivers": drivers,
                    "natural_language": nl,
                    "global_importance": drivers,
                    "method": "correlation",
                }

    return render_template(
        "explainability.html",
        explanation=explanation,
        importance={"features": explanation.get("global_importance", []), "method": explanation.get("method")},
        city=get("city.name", "Hyderabad"),
    )


@app.route("/pipeline")
def pipeline():
    hourly = _load_json(PIPELINE_PATH)
    daily = _load_json(DAILY_PATH)

    # Data freshness
    freshness = "no_data"
    if MERGED_PATH.exists():
        try:
            df = pd.read_parquet(MERGED_PATH)
            if "timestamp" in df.columns and len(df) > 0:
                latest_ts = pd.to_datetime(df["timestamp"]).max().isoformat()
                freshness = latest_ts
        except Exception:
            pass

    return render_template(
        "pipeline.html",
        pipeline={
            "hourly_pipeline": hourly,
            "daily_pipeline": daily,
            "data_freshness": {
                "latest_timestamp": freshness,
                "status": "healthy" if freshness != "no_data" else "no_data",
            },
        },
        city=get("city.name", "Hyderabad"),
    )


@app.route("/data-sources")
def data_sources():
    sources = {
        "providers": [
            {
                "name": "Open-Meteo",
                "role": "Weather features + forecast inputs",
                "description": "Open-Meteo provides free weather data including temperature, humidity, wind speed, precipitation, and air quality indices. Used as the primary source for future-facing meteorological features that drive AQI predictions.",
                "url": "https://open-meteo.com/",
                "data_used": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl", "wind_speed_10m", "wind_direction_10m", "precipitation", "cloud_cover"],
                "used_for": "training features, inference features",
            },
            {
                "name": "OpenAQ",
                "role": "Primary observed AQI — ground truth labels (2+ years history)",
                "description": "OpenAQ provides 2+ years of observed hourly AQI and pollutant measurements from station 4889110 in Hyderabad. This is the measured reality layer used for supervised training labels and evaluation targets.",
                "url": "https://openaq.org/",
                "station": "4889110",
                "city": "Hyderabad, Pakistan",
                "data_used": ["Observed PM2.5", "PM10", "NO₂", "O₃", "SO₂", "CO"],
                "used_for": "training labels, evaluation targets, validation",
            },
            {
                "name": "AQICN",
                "role": "Secondary observed AQI — live data only",
                "description": "AQICN provides real-time observed AQI from station A546205. Used as a secondary live data source.",
                "url": "https://aqicn.org/",
                "station": "A546205",
                "data_used": ["Observed AQI", "PM2.5", "PM10", "NO₂", "O₃", "SO₂", "CO"],
                "used_for": "secondary labels, live validation",
            },
        ],
        "merge_strategy": {
            "description": "Data merged on hourly timestamps (Asia/Karachi). Label priority: OpenAQ > AQICN > Open-Meteo forecast. Open-Meteo contributes weather features (predictors). Observed sources provide AQI labels (targets). The model learns: weather(t) → AQI(t+N).",
            "merge_key": "timestamp (hourly, Asia/Karachi)",
            "target_construction": "AQI at t+24h, t+48h, t+72h shifted from observed values",
        },
        "fallback_logic": "1. Try OpenAQ observed station data for labels. 2. Try AQICN as secondary. 3. Use Open-Meteo weather regardless. 4. Skip rows without labels from supervised training. 5. Continue inference with latest Open-Meteo features.",
    }
    return render_template(
        "data_sources.html",
        sources=sources,
        city=get("city.name", "Hyderabad"),
    )


@app.route("/digital-twin")
def digital_twin():
    forecast = _load_forecast()
    merged = _load_merged(hours=1)
    pollutants = {}
    if merged:
        latest = merged[-1]
        pollutants = {
            "pm2_5": latest.get("pm2_5"),
            "pm10": latest.get("pm10"),
            "dominant_pollutant": latest.get("dominant_pollutant", "PM2.5"),
        }
    return render_template(
        "digital_twin.html",
        forecast=forecast,
        pollutants=pollutants,
        city=get("city.name", "Hyderabad"),
    )


@app.route("/settings")
def settings():
    return render_template("settings.html", city=get("city.name", "Hyderabad"))


if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", 5000)))
    app.run(host="0.0.0.0", port=port, debug=False)
