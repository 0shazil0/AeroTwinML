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


@app.errorhandler(500)
def internal_error(e):
    logger.error("Internal server error: %s", e)
    return render_template("index.html",
        aqi={"aqi": None, "category": "No Data", "category_color": "#666",
             "health_advice": "Dashboard encountered an error. Data will refresh on next pipeline run.",
             "weather": None, "pm2_5": None, "pm10": None, "dominant_pollutant": "--", "updated_at": None},
        forecast={}, alerts=[], city=get("city.name", "Hyderabad"),
        cities=[], current_city=None), 200


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _load_forecast() -> dict:
    return _load_json(FORECAST_PATH)


def _get_available_cities(forecast: dict) -> list:
    """Get list of available cities from forecast JSON."""
    cities_dict = forecast.get("cities", {})
    if cities_dict:
        return list(cities_dict.keys())
    city = forecast.get("city")
    if city:
        return [city]
    return [get("city.name", "Hyderabad")]


def _get_city_forecast(city: str = None) -> dict:
    """Get forecast for a specific city. Falls back to primary forecast."""
    forecast = _load_forecast()
    if not city:
        return forecast
    cities_dict = forecast.get("cities", {})
    if city in cities_dict:
        return cities_dict[city]
    if forecast.get("city") == city:
        return forecast
    return forecast


def _resolve_pollutants(forecast: dict, merged_list: list = None, current_aqi: float = 0) -> dict:
    """Guarantees realistic, non-null pollutant values for UI components."""
    forecast_pollutants = forecast.get("pollutants") or {}
    pm2_5 = forecast_pollutants.get("pm2_5")
    pm10 = forecast_pollutants.get("pm10")
    no2 = forecast_pollutants.get("no2")
    o3 = forecast_pollutants.get("o3")
    so2 = forecast_pollutants.get("so2")
    co = forecast_pollutants.get("co")

    # Search backwards in merged history for any non-null readings
    if merged_list:
        for row in reversed(merged_list):
            if pm2_5 is None and row.get("pm2_5") is not None:
                pm2_5 = row.get("pm2_5")
            if pm10 is None and row.get("pm10") is not None:
                pm10 = row.get("pm10")
            if no2 is None and row.get("no2") is not None:
                no2 = row.get("no2")
            if o3 is None and row.get("o3") is not None:
                o3 = row.get("o3")
            if so2 is None and row.get("so2") is not None:
                so2 = row.get("so2")
            if co is None and row.get("co") is not None:
                co = row.get("co")

    # Derive PM2.5 from current AQI if still missing
    if pm2_5 is None and current_aqi is not None and float(current_aqi or 0) > 0:
        pm2_5 = round((float(current_aqi) * 35.4) / 100, 1)

    # Derive PM10 from PM2.5 if missing
    if pm10 is None and pm2_5 is not None:
        pm10 = round(float(pm2_5) * 1.6, 1)

    dominant = (
        forecast.get("dominant_pollutant")
        or forecast_pollutants.get("dominant_pollutant")
        or "PM2.5"
    )

    return {
        "pm2_5": pm2_5,
        "pm10": pm10,
        "no2": no2,
        "o3": o3,
        "so2": so2,
        "co": co,
        "dominant_pollutant": dominant,
    }


def _load_history_from_forecast(hours: int = 168, city: str = None) -> list:
    """Load history from forecast JSON's embedded history array.

    This is the primary source when parquet is unavailable (e.g. on deployment).
    """
    try:
        forecast = _get_city_forecast(city) if city else _load_forecast()
        history = forecast.get("history", [])
        if not history and city:
            # Fall back to primary history if city-specific history is absent
            history = _load_forecast().get("history", [])
        if not history:
            return []

        # Filter to requested time window — use pd.Timestamp for timezone-safe comparison
        cutoff = pd.Timestamp.now(tz="UTC") - timedelta(hours=hours)
        filtered = []
        for r in history:
            rec = dict(r)
            if "temperature" not in rec or rec["temperature"] is None:
                rec["temperature"] = rec.get("temperature_2m")
            if "humidity" not in rec or rec["humidity"] is None:
                rec["humidity"] = rec.get("relative_humidity_2m")
            if "aqi" not in rec or rec["aqi"] is None:
                rec["aqi"] = rec.get("om_forecast_aqi")
            if rec.get("pm2_5") is None and rec.get("aqi") is not None and float(rec.get("aqi") or 0) > 0:
                rec["pm2_5"] = round((float(rec["aqi"]) * 35.4) / 100, 1)
            if rec.get("pm10") is None and rec.get("pm2_5") is not None:
                rec["pm10"] = round(float(rec["pm2_5"]) * 1.6, 1)

            ts = rec.get("timestamp")
            if ts:
                try:
                    dt = pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts, tz="UTC")
                    if dt >= cutoff:
                        filtered.append(rec)
                except Exception:
                    filtered.append(rec)
            else:
                filtered.append(rec)
        return filtered if filtered else [dict(r) for r in history]
    except Exception:
        return []


def _load_merged(hours: int = 168, city: str = None) -> list:
    """Load history data — tries parquet first, falls back to forecast JSON."""
    try:
        df = pd.read_parquet(MERGED_PATH)
        if city and "city" in df.columns:
            df = df[df["city"] == city]
        if "timestamp" in df.columns and not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            cutoff = pd.Timestamp.now(tz="UTC") - timedelta(hours=hours)
            # Ensure timestamps are tz-aware for comparison
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
            df = df[df["timestamp"] >= cutoff]
        if df.empty:
            return _load_history_from_forecast(hours, city=city)
        # Convert to JSON-safe format
        records = df.tail(hours * 2).to_dict(orient="records")
        for r in records:
            for k, v in list(r.items()):
                if isinstance(v, (np.integer,)):
                    r[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    r[k] = float(v) if pd.notna(v) else None
                elif isinstance(v, pd.Timestamp):
                    r[k] = v.isoformat()
                elif pd.isna(v):
                    r[k] = None
            if "temperature" not in r or r["temperature"] is None:
                r["temperature"] = r.get("temperature_2m")
            if "humidity" not in r or r["humidity"] is None:
                r["humidity"] = r.get("relative_humidity_2m")
            if "aqi" not in r or r["aqi"] is None:
                r["aqi"] = r.get("om_forecast_aqi")
            if r.get("pm2_5") is None and r.get("aqi") is not None and float(r.get("aqi") or 0) > 0:
                r["pm2_5"] = round((float(r["aqi"]) * 35.4) / 100, 1)
            if r.get("pm10") is None and r.get("pm2_5") is not None:
                r["pm10"] = round(float(r["pm2_5"]) * 1.6, 1)
        return records
    except (FileNotFoundError, Exception):
        # Fallback: read from embedded history in forecast JSON
        return _load_history_from_forecast(hours, city=city)


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
    from flask import request as flask_request
    forecast_all = _load_forecast()
    available_cities = _get_available_cities(forecast_all)
    selected_city = flask_request.args.get("city") or (available_cities[0] if available_cities else None)
    forecast = _get_city_forecast(selected_city)

    merged = _load_merged(hours=24, city=selected_city)
    alerts = _load_alerts(3)

    current_aqi = forecast.get("current_aqi", 0)
    category = classify_aqi(current_aqi)

    pollutant_data = _resolve_pollutants(forecast, merged, current_aqi)

    aqi_data = {
        "aqi": current_aqi,
        "category": category.value,
        "category_color": category_color(category),
        "health_advice": category_advice(category),
        "dominant_pollutant": pollutant_data["dominant_pollutant"],
        "pm2_5": pollutant_data["pm2_5"],
        "pm10": pollutant_data["pm10"],
        "updated_at": forecast.get("timestamp"),
    }

    # Weather from forecast JSON or latest merged row
    weather = forecast.get("weather")
    if weather is None and merged:
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
        city=selected_city or get("city.name", "Hyderabad"),
        cities=available_cities,
        current_city=selected_city,
    )


@app.route("/forecast")
def forecast():
    from flask import request as flask_request
    forecast_all = _load_forecast()
    available_cities = _get_available_cities(forecast_all)
    selected_city = flask_request.args.get("city") or (available_cities[0] if available_cities else None)
    fc = _get_city_forecast(selected_city)

    history = _load_merged(hours=168, city=selected_city)
    merged = _load_merged(hours=1, city=selected_city)
    current_aqi = fc.get("current_aqi", 0)
    pollutants = _resolve_pollutants(fc, merged, current_aqi)

    return render_template(
        "forecast.html",
        forecast=fc,
        history=history,
        pollutants=pollutants,
        city=selected_city or get("city.name", "Hyderabad"),
        cities=available_cities,
        current_city=selected_city,
    )


@app.route("/analytics")
def analytics():
    from flask import request as flask_request
    forecast_all = _load_forecast()
    available_cities = _get_available_cities(forecast_all)
    selected_city = flask_request.args.get("city") or (available_cities[0] if available_cities else None)

    history = _load_merged(hours=720, city=selected_city)
    return render_template(
        "analytics.html",
        history=history,
        city=selected_city or get("city.name", "Hyderabad"),
        cities=available_cities,
        current_city=selected_city,
    )


@app.route("/explainability")
def explainability():
    from flask import request as flask_request
    forecast_all = _load_forecast()
    available_cities = _get_available_cities(forecast_all)
    selected_city = flask_request.args.get("city") or (available_cities[0] if available_cities else None)
    forecast = _get_city_forecast(selected_city)
    merged = _load_merged(hours=168, city=selected_city)
    explanation = {"top_drivers": [], "natural_language": "", "method": "none"}

    # Build a DataFrame from whatever data is available
    df = None
    if merged:
        try:
            df = pd.DataFrame(merged)
        except Exception:
            df = None

    if df is not None and not df.empty and len(df) > 1:
        # We have enough history data — compute correlation-based drivers
        aqi_col = "aqi" if "aqi" in df.columns else "om_forecast_aqi"
        if aqi_col in df.columns:
            numeric = df.select_dtypes(include=[np.number])
            if aqi_col in numeric.columns:
                try:
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
                except Exception:
                    pass

    # Fallback: generate drivers from current forecast data (pollutants + weather)
    if not explanation["top_drivers"] and forecast:
        drivers = []
        current_aqi = forecast.get("current_aqi", 0) or 0

        # Pollutant drivers — show relative contribution to AQI
        pollutants = forecast.get("pollutants", {})
        pollutant_max = {"pm2_5": 75, "pm10": 150, "no2": 200, "o3": 200, "so2": 250, "co": 10000}
        for key, max_val in pollutant_max.items():
            val = pollutants.get(key)
            if val is not None:
                try:
                    val = float(val)
                    importance = min(val / max_val, 1.0)
                    drivers.append({
                        "feature": key,
                        "importance": round(importance, 4),
                        "direction": "positive",
                    })
                except (ValueError, TypeError):
                    pass

        # Weather drivers — show current conditions
        weather = forecast.get("weather", {})
        weather_factors = {
            "temperature": (float(weather.get("temperature", 0) or 0), "positive" if float(weather.get("temperature", 0) or 0) > 30 else "negative"),
            "humidity": (float(weather.get("humidity", 0) or 0) / 100, "positive" if float(weather.get("humidity", 0) or 0) > 60 else "negative"),
            "wind_speed": (float(weather.get("wind_speed", 0) or 0) / 20, "negative"),
        }
        for feat, (val, direction) in weather_factors.items():
            if val > 0:
                drivers.append({
                    "feature": feat,
                    "importance": round(min(abs(val), 1.0), 4),
                    "direction": direction,
                })

        drivers.sort(key=lambda d: d["importance"], reverse=True)

        if drivers:
            top = drivers[0]
            dominant = pollutants.get("dominant_pollutant") or "PM2.5"
            nl = (f"Current AQI is {current_aqi:.0f}. "
                  f"Dominant pollutant: {dominant}. "
                  f"Strongest factor: {top['feature'].replace('_', ' ')}.")
            explanation = {
                "top_drivers": drivers[:10],
                "natural_language": nl,
                "global_importance": drivers[:10],
                "method": "current_snapshot",
            }

    return render_template(
        "explainability.html",
        explanation=explanation,
        importance={"features": explanation.get("global_importance", []), "method": explanation.get("method")},
        city=selected_city or get("city.name", "Hyderabad"),
        cities=available_cities,
        current_city=selected_city,
    )


@app.route("/pipeline")
def pipeline():
    from flask import request as flask_request
    forecast_all = _load_forecast()
    available_cities = _get_available_cities(forecast_all)
    selected_city = flask_request.args.get("city") or (available_cities[0] if available_cities else None)

    hourly = _load_json(PIPELINE_PATH)
    daily = _load_json(DAILY_PATH)

    # Data freshness — try parquet first, fall back to forecast JSON timestamp
    freshness = "no_data"
    if MERGED_PATH.exists():
        try:
            df = pd.read_parquet(MERGED_PATH)
            if "timestamp" in df.columns and len(df) > 0:
                latest_ts = pd.to_datetime(df["timestamp"]).max().isoformat()
                freshness = latest_ts
        except Exception:
            pass

    if freshness == "no_data":
        ts = forecast_all.get("timestamp")
        if ts:
            freshness = ts

    # Load training metrics
    METRICS_PATH = DATA_DIR / "processed" / "training_metrics.json"
    training_metrics = _load_json(METRICS_PATH)

    return render_template(
        "pipeline.html",
        pipeline={
            "hourly_pipeline": hourly,
            "daily_pipeline": daily,
            "data_freshness": {
                "latest_timestamp": freshness,
                "status": "healthy" if freshness != "no_data" else "no_data",
            },
            "training_metrics": training_metrics,
        },
        city=selected_city or get("city.name", "Hyderabad"),
        cities=available_cities,
        current_city=selected_city,
    )


@app.route("/data-sources")
def data_sources():
    from flask import request as flask_request
    forecast_all = _load_forecast()
    available_cities = _get_available_cities(forecast_all)
    selected_city = flask_request.args.get("city") or (available_cities[0] if available_cities else None)
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
        city=selected_city or get("city.name", "Hyderabad"),
        cities=available_cities,
        current_city=selected_city,
    )


@app.route("/digital-twin")
def digital_twin():
    from flask import request as flask_request
    forecast_all = _load_forecast()
    available_cities = _get_available_cities(forecast_all)
    selected_city = flask_request.args.get("city") or (available_cities[0] if available_cities else None)
    forecast = _get_city_forecast(selected_city)

    merged = _load_merged(hours=1, city=selected_city)
    current_aqi = forecast.get("current_aqi", 0)
    pollutants = _resolve_pollutants(forecast, merged, current_aqi)

    return render_template(
        "digital_twin.html",
        forecast=forecast,
        pollutants=pollutants,
        city=selected_city or get("city.name", "Hyderabad"),
        cities=available_cities,
        current_city=selected_city,
    )


@app.route("/settings")
def settings():
    from flask import request as flask_request
    forecast_all = _load_forecast()
    available_cities = _get_available_cities(forecast_all)
    selected_city = flask_request.args.get("city") or (available_cities[0] if available_cities else None)
    return render_template("settings.html",
        city=selected_city or get("city.name", "Hyderabad"),
        cities=available_cities,
        current_city=selected_city,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", 5000)))
    app.run(host="0.0.0.0", port=port, debug=False)
