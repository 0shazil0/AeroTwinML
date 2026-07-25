"""Pipeline status route handlers."""

import json
from pathlib import Path

from fastapi import APIRouter

from backend.schemas import APIResponse
from utils.config import get

router = APIRouter()
DATA_DIR = Path(get("storage.data_dir", "./data"))


@router.get("/pipeline/status", response_model=APIResponse)
async def get_pipeline_status():
    """Get current pipeline health and last run status."""
    # Hourly pipeline status
    hourly_path = DATA_DIR / "processed" / "pipeline_status.json"
    hourly_status = None
    if hourly_path.exists():
        with open(hourly_path) as f:
            hourly_status = json.load(f)

    # Daily pipeline status
    daily_path = DATA_DIR / "processed" / "daily_status.json"
    daily_status = None
    if daily_path.exists():
        with open(daily_path) as f:
            daily_status = json.load(f)

    # Data freshness
    merged_path = DATA_DIR / "processed" / "merged_hourly" / "merged_latest.parquet"
    data_freshness = None
    if merged_path.exists():
        import pandas as pd

        try:
            df = pd.read_parquet(merged_path)
            if "timestamp" in df.columns and len(df) > 0:
                latest_ts = pd.to_datetime(df["timestamp"]).max()
                data_freshness = latest_ts.isoformat()
        except Exception:
            pass

    return APIResponse(
        data={
            "hourly_pipeline": hourly_status,
            "daily_pipeline": daily_status,
            "data_freshness": {
                "latest_timestamp": data_freshness,
                "status": "healthy" if data_freshness else "no_data",
            },
        }
    )
