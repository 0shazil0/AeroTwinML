"""Model inference — load trained model and generate predictions."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.registry import get_latest_model
from utils.aqi_utils import classify_aqi, is_alert_level
from utils.logging import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Loads model(s), runs predictions, returns structured forecasts.

    Supports per-horizon models: separate models for 24h, 48h, 72h.
    Falls back to single model if per-horizon models aren't available.
    """

    def __init__(self):
        self.model = None
        self.models_by_horizon: dict = {}
        self.feature_cols: List[str] = []
        self._load_models()

    def _load_models(self):
        """Load per-horizon models, falling back to single model.

        Loading order:
        1. Per-horizon pkl files (24h, 48h, 72h) — best differentiated forecasts
        2. Single aqi_forecaster_latest.pkl — works for all horizons
        3. No model — uses Open-Meteo AQI fallback (shows 'fallback_om_forecast')
        """
        from models.registry import get_models_by_horizon, get_latest_model
        from pathlib import Path

        MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
        logger.info("Model artifacts directory: %s (exists=%s)", MODEL_DIR, MODEL_DIR.exists())

        # Log what files are present
        if MODEL_DIR.exists():
            pkl_files = list(MODEL_DIR.glob("*.pkl"))
            logger.info("Available pkl files: %s", [f.name for f in pkl_files])
        else:
            logger.warning("Model artifacts directory does not exist — no trained models available")

        # Try per-horizon models first
        horizon_models = get_models_by_horizon()
        if horizon_models:
            self.models_by_horizon = horizon_models
            # Use feature_cols from the first horizon model (all should be identical)
            for h_entry in horizon_models.values():
                saved_cols = h_entry.get("feature_cols", [])
                if saved_cols:
                    self.feature_cols = saved_cols
                    logger.info("Loaded feature_cols from model pkl: %d features", len(saved_cols))
                    break
            logger.info("Loaded per-horizon models: %s", list(horizon_models.keys()))
        else:
            logger.info("Per-horizon models not found — trying single model")
            # Fallback: single model for all horizons
            self.model = get_latest_model()
            if self.model:
                logger.info("Loaded single model: %s", type(self.model).__name__)
            else:
                logger.warning(
                    "No trained model found in artifacts/. "
                    "Run 'python -m pipelines.daily_pipeline' to train and save models. "
                    "Falling back to Open-Meteo forecast AQI."
                )

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
        if self.model is None and not self.models_by_horizon:
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

        # Determine global feature columns (for single-model fallback)
        # Priority: 1) saved feature_cols from pkl 2) dynamic selection
        global_feature_cols = self.feature_cols or [
            c for c in features.columns if c not in (
                "timestamp", "source", "station_name", "city", "country",
                "dominant_pollutant", "merged_at", "fetched_at", "latitude", "longitude",
            ) and not c.startswith("target_")
            and features[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]

        for h in horizons:
            horizon_key = f"{h}h"
            h_entry = self.models_by_horizon.get(horizon_key, {})
            model_for_h = h_entry.get("model") or self.model

            if model_for_h is None:
                pred = result["current_aqi"]
            else:
                try:
                    # Use the feature_cols saved WITH this specific horizon's model
                    # This is the fix for feature drift (57 vs 49 mismatch)
                    horizon_feature_cols = h_entry.get("feature_cols") or global_feature_cols

                    if horizon_feature_cols:
                        # Build feature vector using ONLY the columns the model was trained on
                        # Columns missing from live data are filled with 0 (safe default)
                        available = [c for c in horizon_feature_cols if c in features.columns]
                        missing = [c for c in horizon_feature_cols if c not in features.columns]
                        if missing:
                            logger.debug("Filling %d missing feature(s) with 0: %s", len(missing), missing[:5])

                        X_df = features.reindex(columns=horizon_feature_cols, fill_value=0)
                        X = X_df.fillna(0).values[-1:]
                    else:
                        X = features[global_feature_cols].fillna(0).values[-1:]

                    pred = float(model_for_h.predict(X)[0])
                    pred = max(0.0, min(500.0, pred))  # Clamp to valid AQI range
                except Exception as e:
                    logger.error("Prediction error for %s: %s", horizon_key, e)
                    pred = result["current_aqi"]

            result["forecast"][horizon_key] = {
                "aqi": round(pred, 1),
                "category": classify_aqi(pred).value,
                "alert": is_alert_level(pred),
            }

        model_names = {k: v.get("model_name", "?") for k, v in self.models_by_horizon.items()}
        result["model_info"] = {
            "type": "per_horizon" if self.models_by_horizon else (type(self.model).__name__ if self.model else "none"),
            "features_used": len(self.feature_cols) if self.feature_cols else len(global_feature_cols),
            "horizon_models": model_names if model_names else None,
        }

        return result

    def _find_aqi_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["aqi", "us_aqi", "om_forecast_aqi"]:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    def _fallback_forecast(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Fallback when no trained model exists.

        Uses Open-Meteo's forecast AQI for future horizons if available,
        otherwise falls back to persistence (repeat current value).
        """
        aqi_col = self._find_aqi_col(features)
        current = 0.0
        if aqi_col:
            vals = features[aqi_col].dropna()
            current = float(vals.iloc[-1]) if len(vals) > 0 else 0.0

        # Try to use Open-Meteo's forecast AQI for future horizons
        om_col = "om_forecast_aqi" if "om_forecast_aqi" in features.columns else None
        om_values = []
        if om_col:
            om_values = features[om_col].dropna().tolist()

        result = {
            "current_aqi": round(current, 1),
            "forecast": {},
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_info": {"type": "fallback_om_forecast", "features_used": 0},
        }

        # Use Open-Meteo forecast AQI for each horizon if available
        for h in [24, 48, 72]:
            if len(om_values) >= h:
                pred = float(om_values[h - 1])
            elif om_values:
                # Use the last available OM forecast value
                pred = float(om_values[-1])
            else:
                pred = current

            result["forecast"][f"{h}h"] = {
                "aqi": round(pred, 1),
                "category": classify_aqi(pred).value,
                "alert": is_alert_level(pred),
            }

        if not om_values:
            result["model_info"]["type"] = "fallback_persistence"

        return result
