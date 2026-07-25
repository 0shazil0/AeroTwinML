"""API response schemas using Pydantic."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AQIReading(BaseModel):
    aqi: float
    category: str
    dominant_pollutant: Optional[str] = None
    pm2_5: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    o3: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None


class WeatherSnapshot(BaseModel):
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    precipitation: Optional[float] = None
    cloud_cover: Optional[float] = None


class ForecastPoint(BaseModel):
    aqi: float
    category: str
    alert: bool
    dominant_pollutant: Optional[str] = None


class ForecastResponse(BaseModel):
    current_aqi: float
    current_category: str
    weather: Optional[WeatherSnapshot] = None
    forecast_24h: ForecastPoint
    forecast_48h: ForecastPoint
    forecast_72h: ForecastPoint
    timestamp: str
    model_info: Optional[Dict[str, Any]] = None


class HistoricalDataPoint(BaseModel):
    timestamp: str
    aqi: Optional[float] = None
    pm2_5: Optional[float] = None
    pm10: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None


class AlertRecord(BaseModel):
    type: str
    aqi: float
    level: Optional[str] = None
    category: Optional[str] = None
    horizon: Optional[str] = None
    timestamp: Optional[str] = None


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    direction: str = "neutral"  # positive, negative, neutral


class ExplanationResponse(BaseModel):
    top_drivers: List[FeatureImportance]
    natural_language: str
    global_importance: List[FeatureImportance]
    timestamp: str


class PipelineStep(BaseModel):
    status: str
    detail: Optional[str] = None
    rows: Optional[int] = None


class PipelineStatus(BaseModel):
    pipeline: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    success: bool
    steps: Dict[str, Any] = {}


class APIResponse(BaseModel):
    status: str = "ok"
    data: Any
    meta: Dict[str, Any] = {}
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
