"""Model inference — load trained model and generate predictions."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.registry import get_latest_model
from utils.aqi_utils import classify_aqi, is_alert_level
from utils.logging import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Loads model, runs predictions, returns structured forecasts."""

    def __init__(self):
        self.model = get_latest_model()
        self.feature_cols: List[str] = []

    def predict(
        self,
        features: pd.DataFrame,
        horizons: List[int] = None,
    ) -> Dict[str, Any]:
        """Run inference on feature DataFrame.

        Returns:
            {
                "current_aqi": float,
                "forecast": {
                    "24h": {"aqi": float, "category": str, "alert": bool},
                    "48h": {...},
                    "72h": {...}
                },
                "timestamp": str,
                "model_info": {...}
            }
        """
        if self.model is None:
            logger.warning("No model loaded — using fallback")
            return self._fallback_forecast(features)

        if horizons is None:
            horizons = [24, 48, 72]

        result = {"forecast": {}, "timestamp": pd.Timestamp.now().isoformat()}

        # Get current AQI if available
        aqi_col = self._find_aqi_col(features)
        if aqi_col:
            latest = features[aqi_col].dropna()
            result["current_aqi"] = float(latest.iloc[-1]) if len(latest) > 0 else 0.0
        else:
            result["current_aqi"] = 0.0

        # Prepare feature vector
        feature_cols = self.feature_cols or [c for c in features.columns if c not in (
            "timestamp", "source", "station_name", "city", "country",
            "dominant_pollutant", "merged_at", "fetched_at", "latitude", "longitude",
        ) and not c.startswith("target_")]

        if not feature_cols:
            feature_cols = [c for c in features.columns if features[c].dtype in (np.float64, np.float32, np.int64, np.int32)]

        X = features[feature_cols].fillna(0).values[-1:]

        for h in horizons:
            try:
                pred = float(self.model.predict(X)[0])
            except Exception as e:
                logger.error("Prediction error for %dh: %s", h, e)
                pred = result["current_aqi"]

            result["forecast"][f"{h}h"] = {
                "aqi": round(pred, 1),
                "category": classify_aqi(pred).value,
                "alert": is_alert_level(pred),
            }

        result["model_info"] = {
            "type": type(self.model).__name__,
            "features_used": len(feature_cols),
        }

        return result

    def _find_aqi_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["aqi", "us_aqi", "om_forecast_aqi"]:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    def _fallback_forecast(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Simple fallback when no trained model exists."""
        aqi_col = self._find_aqi_col(features)
        current = 0.0
        if aqi_col:
            vals = features[aqi_col].dropna()
            current = float(vals.iloc[-1]) if len(vals) > 0 else 0.0

        # Naive persistence
        result = {
            "current_aqi": round(current, 1),
            "forecast": {
                "24h": {
                    "aqi": round(current, 1),
                    "category": classify_aqi(current).value,
                    "alert": is_alert_level(current),
                },
                "48h": {
                    "aqi": round(current, 1),
                    "category": classify_aqi(current).value,
                    "alert": is_alert_level(current),
                },
                "72h": {
                    "aqi": round(current, 1),
                    "category": classify_aqi(current).value,
                    "alert": is_alert_level(current),
                },
            },
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_info": {"type": "fallback_persistence", "features_used": 0},
        }
        return result
