"""Alert route handlers."""

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Query

from backend.schemas import APIResponse, AlertRecord
from utils.config import get

router = APIRouter()
DATA_DIR = Path(get("storage.data_dir", "./data"))
ALERTS_FILE = DATA_DIR / "raw" / "logs" / "alerts.jsonl"


@router.get("/alerts", response_model=APIResponse)
async def get_alerts(
    limit: int = Query(default=20, ge=1, le=100, description="Number of recent alerts to return"),
):
    """Get recent AQI alerts."""
    alerts: List[dict] = []

    if ALERTS_FILE.exists():
        with open(ALERTS_FILE) as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    alerts.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        alerts.reverse()

    return APIResponse(
        data=alerts,
        meta={"total_alerts": len(alerts), "limit": limit},
    )


@router.get("/alerts/thresholds", response_model=APIResponse)
async def get_alert_thresholds():
    """Get AQI alert threshold definitions."""
    thresholds = get("aqi_thresholds", {})
    return APIResponse(
        data={
            "thresholds": thresholds,
            "alert_trigger": get("alerts.alert_threshold", 200),
            "description": "Alerts are triggered when AQI >= alert_trigger",
        }
    )
