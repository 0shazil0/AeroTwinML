"""Model explainability — SHAP and LIME integration for AQI predictions."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.config import get
from utils.logging import get_logger
from utils.storage import load_parquet

logger = get_logger(__name__)

DATA_DIR = Path(get("storage.data_dir", "./data"))


class ModelExplainer:
    """Wraps SHAP and provides natural-language explanations for AQI predictions."""

    def __init__(self, model: Any = None, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or []
        self.shap_explainer = None
        self.shap_values = None
        self.background_data: Optional[np.ndarray] = None

    def fit_shap(self, X_background: np.ndarray) -> bool:
        """Initialize SHAP explainer with background data."""
        if self.model is None:
            logger.warning("No model provided for SHAP")
            return False

        try:
            import shap

            # Try TreeExplainer first (for XGBoost, LightGBM, Random Forest)
            if hasattr(self.model, "get_booster") or hasattr(self.model, "estimators_"):
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info("Using SHAP TreeExplainer")
            else:
                # Fallback: KernelExplainer with subset of background
                background = X_background[: min(100, len(X_background))]
                self.background_data = background
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict, background
                )
                logger.info("Using SHAP KernelExplainer (fallback)")

            return True
        except ImportError:
            logger.warning("SHAP not installed — using correlation fallback")
            return False
        except Exception as e:
            logger.error("SHAP initialization failed: %s", e)
            return False

    def explain(self, X: np.ndarray) -> Optional[Dict[str, Any]]:
        """Generate SHAP explanation for given input."""
        if self.shap_explainer is None:
            return None

        try:
            import shap

            shap_vals = self.shap_explainer.shap_values(X)
        except Exception as e:
            logger.error("SHAP values computation failed: %s", e)
            return None

        # Handle multi-output SHAP
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]  # take first output

        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[-1]  # latest prediction

        # Create feature importance list
        feature_impacts = []
        for i, name in enumerate(self.feature_names):
            if i < len(shap_vals):
                feature_impacts.append(
                    {
                        "feature": name,
                        "shap_value": round(float(shap_vals[i]), 4),
                        "abs_importance": round(abs(float(shap_vals[i])), 4),
                        "direction": "positive" if shap_vals[i] > 0 else "negative",
                    }
                )

        # Sort by absolute importance
        feature_impacts.sort(key=lambda x: x["abs_importance"], reverse=True)

        # Generate natural language
        nl = self._generate_nl_explanation(feature_impacts[:5])

        return {
            "top_drivers": feature_impacts[:10],
            "natural_language": nl,
            "global_importance": feature_impacts,
            "method": "shap",
        }

    def _generate_nl_explanation(self, top_drivers: List[Dict]) -> str:
        """Generate human-readable explanation from SHAP values."""
        if not top_drivers:
            return "No significant feature drivers identified."

        positive = [d for d in top_drivers if d["direction"] == "positive"]
        negative = [d for d in top_drivers if d["direction"] == "negative"]

        sentences = []

        if positive:
            pos_features = [d["feature"].replace("_", " ") for d in positive[:3]]
            sentences.append(
                f"AQI is predicted to rise due to {', '.join(pos_features)}."
            )

        if negative:
            neg_features = [d["feature"].replace("_", " ") for d in negative[:3]]
            sentences.append(
                f"AQI improvement is driven by {', '.join(neg_features)}."
            )

        if not sentences:
            strongest = top_drivers[0]
            direction = "increases" if strongest["direction"] == "positive" else "decreases"
            sentences.append(
                f"The strongest factor is {strongest['feature'].replace('_', ' ')}, "
                f"which {direction} AQI."
            )

        return " ".join(sentences)


class LIMEExplainer:
    """LIME-based explainability as alternative to SHAP."""

    def __init__(self, model: Any, feature_names: List[str], training_data: np.ndarray):
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data

    def explain(self, X: np.ndarray) -> Optional[Dict[str, Any]]:
        """Generate LIME explanation."""
        try:
            import lime
            import lime.lime_tabular

            explainer = lime.lime_tabular.LimeTabularExplainer(
                self.training_data,
                feature_names=self.feature_names,
                mode="regression",
                discretize_continuous=True,
            )

            exp = explainer.explain_instance(X[-1], self.model.predict, num_features=10)

            drivers = []
            for feat, weight in exp.as_list():
                drivers.append(
                    {
                        "feature": feat,
                        "importance": round(abs(weight), 4),
                        "direction": "positive" if weight > 0 else "negative",
                        "weight": round(weight, 4),
                    }
                )

            nl = self._generate_nl_explanation(drivers)

            return {
                "top_drivers": drivers[:10],
                "natural_language": nl,
                "global_importance": drivers,
                "method": "lime",
            }

        except ImportError:
            logger.warning("LIME not installed")
            return None
        except Exception as e:
            logger.error("LIME explanation failed: %s", e)
            return None

    def _generate_nl_explanation(self, drivers: List[Dict]) -> str:
        if not drivers:
            return "No significant feature drivers identified."

        top = drivers[0]
        direction = "increases" if top["direction"] == "positive" else "decreases"
        return (
            f"The most influential factor is {top['feature']}, which {direction} "
            f"the predicted AQI (weight: {top['weight']:.3f})."
        )


def correlation_explanation(features_df: pd.DataFrame, target_col: str = "aqi") -> Dict[str, Any]:
    """Correlation-based explanation as fallback when SHAP/LIME unavailable."""
    if target_col not in features_df.columns:
        if "us_aqi" in features_df.columns:
            target_col = "us_aqi"
        else:
            return {"top_drivers": [], "natural_language": "No target column available."}

    numeric = features_df.select_dtypes(include=[np.number])
    correlations = numeric.corr()[target_col].dropna().drop(target_col, errors="ignore")

    drivers = []
    for feat, corr in correlations.abs().sort_values(ascending=False).head(10).items():
        drivers.append(
            {
                "feature": feat,
                "correlation": round(corr, 4),
                "importance": round(abs(correlations[feat]), 4),
                "direction": "positive" if correlations[feat] > 0 else "negative",
            }
        )

    nl = ""
    if drivers:
        top = drivers[0]
        direction = "increase" if top["direction"] == "positive" else "decrease"
        nl = (
            f"AQI shows the strongest correlation with {top['feature']} "
            f"(r={top['correlation']:.3f}, {direction})."
        )

    return {
        "top_drivers": drivers,
        "natural_language": nl,
        "global_importance": drivers,
        "method": "correlation",
    }
