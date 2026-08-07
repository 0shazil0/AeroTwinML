"""Model training framework — baselines, tree models, walk-forward validation, MLflow tracking."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.config import get
from utils.logging import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(get("storage.data_dir", "./data")).parent / "models" / "artifacts"


class BaseModel(ABC):
    """Abstract base for all forecasting models."""

    def __init__(self, name: str):
        self.name = name
        self.model: Any = None
        self.feature_names: List[str] = []
        self.target_horizons = get("pipeline.features.target_horizons_hours", [24, 48, 72])

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved: %s", path)

    @staticmethod
    def load(path: Path) -> "BaseModel":
        import joblib

        return joblib.load(path)


class PersistenceModel(BaseModel):
    """Predicts the last known value — simplest baseline."""

    def __init__(self):
        super().__init__("persistence")
        self.last_value: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceModel":
        self.last_value = float(y[-1]) if len(y) > 0 else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = len(X) if len(X.shape) > 0 else 1
        return np.full(n, self.last_value)


class SeasonalNaiveModel(BaseModel):
    """Predicts the value from 24h ago — captures daily seasonality."""

    def __init__(self):
        super().__init__("seasonal_naive")
        self.seasonal_value: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SeasonalNaiveModel":
        # Use the value 24 steps back as prediction
        if len(y) >= 24:
            self.seasonal_value = float(y[-24])
        elif len(y) > 0:
            self.seasonal_value = float(y[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = len(X) if len(X.shape) > 0 else 1
        return np.full(n, self.seasonal_value)


class SklearnWrapper(BaseModel):
    """Wrapper for sklearn-compatible models."""

    def __init__(self, name: str, model: Any):
        super().__init__(name)
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnWrapper":
        self.model.fit(X, y)
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def walk_forward_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_size: float = 0.7,
    step: int = 24,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create walk-forward validation splits for time series.

    Returns list of (train_df, test_df) tuples, each test_df being 'step' hours.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_size)

    splits = []
    current = train_end
    while current < n:
        train = df.iloc[:current]
        test_end = min(current + step, n)
        test = df.iloc[current:test_end]
        splits.append((train, test))
        current += step

    return splits


def evaluate_model(
    model: BaseModel,
    X: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """Calculate regression metrics."""
    y_pred = model.predict(X)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    return metrics


def train_and_evaluate_horizon(
    model_class: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    horizon_name: str,
) -> Dict[str, Any]:
    """Train a model for a single horizon and return metrics."""
    model = model_class()
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    return {
        "horizon": horizon_name,
        "model_name": model.name,
        "metrics": {f"{k}_{horizon_name}": v for k, v in metrics.items()},
        "model": model,
    }


def build_models_for_horizons(
    feature_cols: List[str],
    target_cols: Dict[str, str],  # {"24h": "target_aqi_24h", ...}
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Train multiple model types across all horizons with proper feature-target alignment."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    results = {}
    model_classes = {
        "ridge": lambda: SklearnWrapper("ridge", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        "random_forest": lambda: SklearnWrapper("random_forest", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
        "gradient_boosting": lambda: SklearnWrapper("gradient_boosting", GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
    }

    # Try XGBoost and LightGBM if available
    try:
        import xgboost as xgb
        model_classes["xgboost"] = lambda: SklearnWrapper("xgboost", xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42))
    except ImportError:
        logger.warning("XGBoost not installed")

    try:
        import lightgbm as lgb
        model_classes["lightgbm"] = lambda: SklearnWrapper("lightgbm", lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1))
    except ImportError:
        logger.warning("LightGBM not installed")

    for horizon_key, target_col in target_cols.items():
        # Ensure target_col exists
        if target_col not in train_df.columns or target_col not in test_df.columns:
            logger.warning("Target column %s missing from data — skipping %s", target_col, horizon_key)
            continue

        # Correct alignment: Drop rows where target_col is NaN directly from DataFrames
        tr_clean = train_df.dropna(subset=[target_col])
        te_clean = test_df.dropna(subset=[target_col])

        if len(tr_clean) < 10 or len(te_clean) < 5:
            logger.warning("Skipping horizon %s — insufficient clean data (%d train, %d test)",
                           horizon_key, len(tr_clean), len(te_clean))
            continue

        X_tr = tr_clean[feature_cols].fillna(0).values
        y_tr = tr_clean[target_col].values
        X_te = te_clean[feature_cols].fillna(0).values
        y_te = te_clean[target_col].values

        horizon_results = {}

        # Baselines
        baselines = {
            "persistence": PersistenceModel(),
            "seasonal_naive": SeasonalNaiveModel(),
        }
        for bname, bmodel in baselines.items():
            bmodel.fit(X_tr, y_tr)
            metrics = evaluate_model(bmodel, X_te, y_te)
            horizon_results[bname] = {
                "horizon": horizon_key,
                "model_name": bname,
                "metrics": {f"{k}_{horizon_key}": v for k, v in metrics.items()},
                "model": bmodel,
            }

        # ML models
        for mname, mfactory in model_classes.items():
            try:
                result = train_and_evaluate_horizon(
                    mfactory,
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    horizon_key,
                )
                horizon_results[mname] = result
            except Exception as e:
                logger.error("Error training %s for %s: %s", mname, horizon_key, e)

        results[horizon_key] = horizon_results

    return results


def find_best_model(results: Dict[str, Any], primary_metric: str = "rmse_24h") -> Tuple[str, str, Any, Dict]:
    """Find the best model across all horizons based on primary metric."""
    best_score = float("inf")
    best_model = None
    best_mname = ""
    best_horizon = ""

    for horizon_key, horizon_results in results.items():
        for mname, result in horizon_results.items():
            metrics = result.get("metrics", {})
            metric_key = primary_metric
            # Try horizon-specific metric first
            if primary_metric.replace("_24h", f"_{horizon_key}") in metrics:
                metric_key = primary_metric.replace("_24h", f"_{horizon_key}")

            score = metrics.get(metric_key, float("inf"))
            if score < best_score:
                best_score = score
                best_model = result["model"]
                best_mname = mname
                best_horizon = horizon_key

    all_metrics = {}
    for horizon_key, horizon_results in results.items():
        for mname, result in horizon_results.items():
            all_metrics[f"{horizon_key}_{mname}"] = result.get("metrics", {})

    return best_mname, best_horizon, best_model, all_metrics


def find_best_models_per_horizon(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Find the best model for EACH horizon independently.

    Returns:
        {"24h": {"model_name": str, "model": BaseModel, "metrics": dict},
         "48h": {...}, "72h": {...}}
    """
    best_per_horizon = {}

    for horizon_key, horizon_results in results.items():
        best_score = float("inf")
        best_entry = None

        for mname, result in horizon_results.items():
            metrics = result.get("metrics", {})
            # Find the RMSE metric for this specific horizon
            metric_key = f"rmse_{horizon_key}"
            score = metrics.get(metric_key, float("inf"))

            if score < best_score:
                best_score = score
                best_entry = {
                    "model_name": mname,
                    "model": result["model"],
                    "metrics": metrics,
                    "rmse": best_score,
                }

        if best_entry:
            best_per_horizon[horizon_key] = best_entry

    return best_per_horizon
