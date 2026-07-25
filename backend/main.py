"""FastAPI application — main entry point for the AQI Predictor API."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import aqi, alerts, explain, pipeline, sources
from utils.config import get
from utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown events."""
    logger.info("Starting AQI Predictor API")
    # Ensure data directories exist
    data_dir = Path(get("storage.data_dir", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down AQI Predictor API")


app = FastAPI(
    title="Pearls AQI Predictor",
    description="Real-time AQI forecasting for Hyderabad, Pakistan — 72-hour predictions using dual-source data (Open-Meteo + AQICN)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=get("api.cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(aqi.router, prefix="/api/v1", tags=["AQI"])
app.include_router(alerts.router, prefix="/api/v1", tags=["Alerts"])
app.include_router(explain.router, prefix="/api/v1", tags=["Explainability"])
app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])
app.include_router(sources.router, prefix="/api/v1", tags=["Data Sources"])


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {
        "status": "healthy",
        "service": "aqi-predictor-api",
        "version": "1.0.0",
        "city": get("city.name", "Hyderabad"),
    }


@app.get("/api/v1/status", tags=["System"])
async def system_status() -> Dict:
    """Overall system status including current AQI and pipeline health."""
    import json

    status = {"city": get("city.name", "Hyderabad"), "timezone": get("city.timezone", "Asia/Karachi")}

    # Load latest forecast
    forecast_path = Path(get("storage.data_dir", "./data")) / "processed" / "predictions" / "forecast_latest.json"
    try:
        with open(forecast_path) as f:
            forecast = json.load(f)
        status["current_aqi"] = forecast.get("current_aqi")
        status["forecast_24h"] = forecast.get("forecast", {}).get("24h", {})
        status["last_updated"] = forecast.get("timestamp")
    except FileNotFoundError:
        status["current_aqi"] = None
        status["forecast_24h"] = None
        status["last_updated"] = None

    # Load pipeline status
    pipeline_path = Path(get("storage.data_dir", "./data")) / "processed" / "pipeline_status.json"
    try:
        with open(pipeline_path) as f:
            pipeline_status = json.load(f)
        status["pipeline"] = {
            "success": pipeline_status.get("success"),
            "completed_at": pipeline_status.get("completed_at"),
        }
    except FileNotFoundError:
        status["pipeline"] = None

    return status
