"""Data sources route handlers."""

from fastapi import APIRouter

from backend.schemas import APIResponse
from utils.config import get

router = APIRouter()


@router.get("/data-sources", response_model=APIResponse)
async def get_data_sources():
    """Get information about data sources and their roles."""
    return APIResponse(
        data={
            "providers": [
                {
                    "name": "Open-Meteo",
                    "role": "Weather features + forecast inputs",
                    "description": (
                        "Open-Meteo provides free weather data including temperature, humidity, "
                        "wind speed, precipitation, and air quality indices. It is used as the "
                        "primary source for future-facing meteorological features that drive AQI predictions."
                    ),
                    "url": "https://open-meteo.com/",
                    "data_used": [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "dew_point_2m",
                        "pressure_msl",
                        "wind_speed_10m",
                        "wind_direction_10m",
                        "precipitation",
                        "cloud_cover",
                        "Air quality indices (US AQI, PM2.5, PM10)",
                    ],
                    "used_for": "training features, inference features",
                },
                {
                    "name": "AQICN",
                    "role": "Observed station AQI — ground truth labels",
                    "description": (
                        "AQICN provides real-time observed AQI and pollutant readings from monitoring "
                        "stations. This is the measured reality layer used for supervised training labels "
                        "and evaluation targets. "
                    ),
                    "url": "https://aqicn.org/",
                    "station": get("providers.aqicn.station_id", "A546205"),
                    "city": "Hyderabad, Pakistan",
                    "data_used": [
                        "Observed AQI",
                        "PM2.5",
                        "PM10",
                        "NO2",
                        "O3",
                        "SO2",
                        "CO",
                    ],
                    "used_for": "training labels, evaluation targets, validation",
                },
            ],
            "merge_strategy": {
                "description": (
                    "Data is merged on hourly timestamp in Asia/Karachi timezone. "
                    "Open-Meteo contributes weather features (predictors). "
                    "AQICN contributes observed AQI labels (targets). "
                    "The model learns the relationship between weather conditions and future air quality."
                ),
                "merge_key": "timestamp (hourly, Asia/Karachi)",
                "target_construction": "AQI at t+24h, t+48h, t+72h from AQICN observed values",
            },
            "fallback_logic": (
                "1. Try AQICN observed station data for labels. "
                "2. Use Open-Meteo weather regardless for future-facing inputs. "
                "3. If AQICN is missing, skip that row from supervised training. "
                "4. Continue inference with the latest valid Open-Meteo feature set."
            ),
        }
    )
