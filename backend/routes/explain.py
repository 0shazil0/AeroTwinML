"""Explainability route handlers."""

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter

from backend.schemas import APIResponse, ExplanationResponse, FeatureImportance
from utils.config import get
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

DATA_DIR = Path(get("storage.data_dir", "./data"))


def _generate_fallback_explanation() -> dict:
    """Generate explanation without SHAP — uses correlation-based importance."""
    # Try loading feature data to compute basic importance
    try:
        import pandas as pd
        import numpy as np

        merged_path = DATA_DIR / "processed" / "merged_hourly" / "merged_latest.parquet"
        if not merged_path.exists():
            return _empty_explanation()

        df = pd.read_parquet(merged_path)
        aqi_col = "aqi" if "aqi" in df.columns else "us_aqi"
        if aqi_col not in df.columns:
            return _empty_explanation()

        # Compute correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()[aqi_col].dropna()

        # Top features
        top = corr.abs().sort_values(ascending=False).head(8)
        drivers = []
        for feat, val in top.items():
            if feat == aqi_col or feat.startswith("om_forecast"):
                continue
            drivers.append(
                {
                    "feature": feat,
                    "importance": round(abs(val), 4),
                    "direction": "positive" if val > 0 else "negative",
                }
            )
            if len(drivers) >= 6:
                break

        # Natural language
        if drivers:
            top_driver = drivers[0]
            direction_word = "increase" if top_driver["direction"] == "positive" else "decrease"
            nl = (
                f"AQI is most strongly influenced by {top_driver['feature'].replace('_', ' ')} "
                f"(correlation: {top_driver['importance']:.3f}, direction: {direction_word}). "
            )
        else:
            nl = "Insufficient data for reliable explanation."

        return {
            "top_drivers": drivers,
            "natural_language": nl,
            "global_importance": drivers,
            "method": "correlation_fallback",
        }
    except Exception as e:
        logger.error("Fallback explanation failed: %s", e)
        return _empty_explanation()


def _empty_explanation() -> dict:
    return {
        "top_drivers": [],
        "natural_language": "No data available for explanation. Run the pipeline first.",
        "global_importance": [],
        "method": "none",
    }


@router.get("/explain/latest", response_model=APIResponse)
async def get_latest_explanation():
    """Get explanation for the latest prediction."""
    explanation = _generate_fallback_explanation()

    from datetime import datetime

    return APIResponse(
        data={
            "top_drivers": explanation.get("top_drivers", []),
            "natural_language": explanation.get("natural_language", ""),
            "global_importance": explanation.get("global_importance", []),
            "method": explanation.get("method", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }
    )


@router.get("/explain/feature-importance", response_model=APIResponse)
async def get_global_feature_importance():
    """Get global feature importance across all predictions."""
    explanation = _generate_fallback_explanation()
    return APIResponse(
        data={
            "features": explanation.get("global_importance", []),
            "method": explanation.get("method", "unknown"),
        }
    )
