"""Flask dashboard application — serves the AQI Predictor UI."""

from flask import Flask, render_template
import os
import requests

from utils.config import get
from utils.logging import get_logger

logger = get_logger(__name__)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def api_get(endpoint: str):
    """Helper to fetch data from the FastAPI backend."""
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "data": {}}
    except Exception as e:
        logger.error("API call failed: %s → %s", endpoint, e)
        return {"status": "error", "data": {}, "meta": {"error": str(e)}}


@app.route("/")
def home():
    """Home dashboard — current AQI, weather snapshot, alerts."""
    aqi_data = api_get("/api/v1/aqi/current")
    forecast_data = api_get("/api/v1/aqi/forecast")
    alerts_data = api_get("/api/v1/alerts?limit=3")

    return render_template(
        "index.html",
        aqi=aqi_data.get("data", {}),
        forecast=forecast_data.get("data", {}),
        alerts=alerts_data.get("data", []),
        city=get("city.name", "Hyderabad"),
    )


@app.route("/forecast")
def forecast():
    """72-hour forecast page."""
    forecast_data = api_get("/api/v1/aqi/forecast")
    history_data = api_get("/api/v1/aqi/history?hours=168")
    pollutants_data = api_get("/api/v1/aqi/pollutants")

    return render_template(
        "forecast.html",
        forecast=forecast_data.get("data", {}),
        history=history_data.get("data", []),
        pollutants=pollutants_data.get("data", {}),
        city=get("city.name", "Hyderabad"),
    )


@app.route("/analytics")
def analytics():
    """Analytics page — historical trends, correlations."""
    history_data = api_get("/api/v1/aqi/history?hours=720")

    return render_template(
        "analytics.html",
        history=history_data.get("data", []),
        city=get("city.name", "Hyderabad"),
    )


@app.route("/explainability")
def explainability():
    """Explainability page — SHAP/LIME feature importance."""
    explain_data = api_get("/api/v1/explain/latest")
    importance_data = api_get("/api/v1/explain/feature-importance")

    return render_template(
        "explainability.html",
        explanation=explain_data.get("data", {}),
        importance=importance_data.get("data", {}),
        city=get("city.name", "Hyderabad"),
    )


@app.route("/pipeline")
def pipeline():
    """Pipeline status page."""
    pipeline_data = api_get("/api/v1/pipeline/status")

    return render_template(
        "pipeline.html",
        pipeline=pipeline_data.get("data", {}),
        city=get("city.name", "Hyderabad"),
    )


@app.route("/data-sources")
def data_sources():
    """Data sources page."""
    sources_data = api_get("/api/v1/data-sources")

    return render_template(
        "data_sources.html",
        sources=sources_data.get("data", {}),
        city=get("city.name", "Hyderabad"),
    )


@app.route("/digital-twin")
def digital_twin():
    """Digital Twin page — 2.5D view of Hyderabad."""
    forecast_data = api_get("/api/v1/aqi/forecast")
    pollutants_data = api_get("/api/v1/aqi/pollutants")

    return render_template(
        "digital_twin.html",
        forecast=forecast_data.get("data", {}),
        pollutants=pollutants_data.get("data", {}),
        city=get("city.name", "Hyderabad"),
    )


@app.route("/settings")
def settings():
    return render_template("settings.html", city=get("city.name", "Hyderabad"))


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
