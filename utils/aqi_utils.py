"""AQI classification and alert utilities."""

from enum import Enum

from utils.config import get


class AQICategory(str, Enum):
    GOOD = "Good"
    MODERATE = "Moderate"
    UNHEALTHY_SENSITIVE = "Unhealthy for Sensitive Groups"
    UNHEALTHY = "Unhealthy"
    VERY_UNHEALTHY = "Very Unhealthy"
    HAZARDOUS = "Hazardous"
    UNKNOWN = "Unknown"


THRESHOLDS = get("aqi_thresholds", {})
ALERT_THRESHOLD = get("alerts.alert_threshold", 200)

_CATEGORY_COLORS = {
    AQICategory.GOOD: "#00e400",
    AQICategory.MODERATE: "#ffff00",
    AQICategory.UNHEALTHY_SENSITIVE: "#ff7e00",
    AQICategory.UNHEALTHY: "#ff0000",
    AQICategory.VERY_UNHEALTHY: "#8f3f97",
    AQICategory.HAZARDOUS: "#7e0023",
    AQICategory.UNKNOWN: "#808080",
}

_CATEGORY_HEALTH_ADVICE = {
    AQICategory.GOOD: "Air quality is satisfactory. Enjoy outdoor activities.",
    AQICategory.MODERATE: "Air quality is acceptable. Sensitive individuals should limit prolonged exertion.",
    AQICategory.UNHEALTHY_SENSITIVE: "Sensitive groups may experience health effects. Limit outdoor activity.",
    AQICategory.UNHEALTHY: "Everyone may begin to experience health effects. Reduce outdoor activity.",
    AQICategory.VERY_UNHEALTHY: "Health warnings of emergency conditions. Avoid all outdoor activity.",
    AQICategory.HAZARDOUS: "Health emergency. Everyone should stay indoors.",
}


def classify_aqi(aqi_value: float) -> AQICategory:
    if aqi_value < 0:
        return AQICategory.UNKNOWN
    if aqi_value <= 50:
        return AQICategory.GOOD
    if aqi_value <= 100:
        return AQICategory.MODERATE
    if aqi_value <= 150:
        return AQICategory.UNHEALTHY_SENSITIVE
    if aqi_value <= 200:
        return AQICategory.UNHEALTHY
    if aqi_value <= 300:
        return AQICategory.VERY_UNHEALTHY
    return AQICategory.HAZARDOUS


def is_alert_level(aqi_value: float) -> bool:
    return aqi_value >= ALERT_THRESHOLD


def category_color(category: AQICategory) -> str:
    return _CATEGORY_COLORS.get(category, "#808080")


def category_advice(category: AQICategory) -> str:
    return _CATEGORY_HEALTH_ADVICE.get(category, "")


def dominant_pollutant(pm25: float, pm10: float, no2: float = 0, o3: float = 0) -> str:
    pollutants = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "O3": o3,
    }
    return max(pollutants, key=lambda k: pollutants.get(k, 0) or 0)
