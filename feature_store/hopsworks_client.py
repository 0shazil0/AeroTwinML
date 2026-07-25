"""Hopsworks Feature Store + Model Registry client.

Handles all interactions with the managed Hopsworks platform:
  - Feature Store: write/read feature groups for training and inference
  - Model Registry: register trained models, load latest for inference

Connection: eu-west.cloud.hopsworks.ai (free tier)
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)

# Singleton connection
_project = None
_fs = None
_mr = None
_ms = None


def _connect():
    """Lazy-connect to Hopsworks. Called once, cached globally."""
    global _project, _fs, _mr, _ms
    if _project is not None:
        return

    api_key = os.getenv("HOPSWORKS_API_KEY", "")
    project_name = os.getenv("HOPSWORKS_PROJECT", "AeroTwinML")
    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    if not api_key:
        logger.warning("HOPSWORKS_API_KEY not set — Hopsworks features disabled")
        return

    try:
        import hopsworks

        _project = hopsworks.login(
            project=project_name,
            host=host,
            port=443,
            api_key_value=api_key,
        )
        _fs = _project.get_feature_store()
        _mr = _project.get_model_registry()
        _ms = _project.get_model_serving()
        logger.info("Connected to Hopsworks project: %s", _project.name)
    except ImportError:
        logger.warning("hopsworks package not installed — run: pip install 'hopsworks[python]'")
    except Exception as e:
        logger.error("Hopsworks connection failed: %s", e)


def is_available() -> bool:
    _connect()
    return _fs is not None


# ─── Feature Store ───────────────────────────────────────────────

def write_feature_group(
    name: str,
    df: pd.DataFrame,
    version: int = 1,
    description: str = "",
    primary_key: List[str] = None,
    event_time: str = "timestamp",
    online_enabled: bool = True,
) -> bool:
    """Write a DataFrame to a Hopsworks feature group. Creates it if missing.

    Args:
        name: Feature group name (e.g. 'merged_hourly')
        df: DataFrame to write
        version: Feature group version
        description: Human-readable description
        primary_key: Columns forming the primary key (default: ['timestamp'])
        event_time: Column used as event time for time-travel queries
        online_enabled: Whether to enable low-latency online serving

    Returns:
        True if write succeeded, False otherwise.
    """
    _connect()
    if _fs is None:
        logger.warning("Hopsworks FS unavailable — saving locally")
        _save_local_fallback(name, df)
        return False

    if primary_key is None:
        primary_key = ["timestamp", "source"] if "source" in df.columns else ["timestamp"]

    try:
        try:
            fg = _fs.get_feature_group(name=name, version=version)
        except Exception:
            fg = _fs.create_feature_group(
                name=name,
                version=version,
                description=description or f"Auto-created: {name}",
                primary_key=primary_key,
                event_time=event_time,
                online_enabled=online_enabled,
            )
            logger.info("Created feature group: %s v%d", name, version)

        fg.insert(df)
        logger.info("Wrote %d rows to Hopsworks FG: %s v%d", len(df), name, version)
        return True
    except Exception as e:
        logger.error("Feature group write failed: %s", e)
        _save_local_fallback(name, df)
        return False


def read_feature_group(
    name: str,
    version: int = 1,
    online: bool = False,
) -> Optional[pd.DataFrame]:
    """Read from a Hopsworks feature group.

    Args:
        name: Feature group name
        version: Feature group version
        online: If True, read from online store (low latency). If False, offline.

    Returns:
        DataFrame or None if unavailable.
    """
    _connect()
    if _fs is None:
        return _load_local_fallback(name)

    try:
        fg = _fs.get_feature_group(name=name, version=version)
        df = fg.read(online=online) if online else fg.read()
        logger.info("Read %d rows from Hopsworks FG: %s v%d", len(df), name, version)
        return df
    except Exception as e:
        logger.warning("Feature group read failed: %s — trying local fallback", e)
        return _load_local_fallback(name)


# ─── Model Registry ───────────────────────────────────────────────

def register_model(
    model_obj: Any,
    model_name: str = "aqi_forecaster",
    metrics: Optional[Dict[str, float]] = None,
    description: str = "",
    input_example: Optional[Any] = None,
) -> Optional[int]:
    """Register a trained model in the Hopsworks Model Registry.

    Args:
        model_obj: Trained model object (sklearn, xgboost, etc.)
        model_name: Name to register under
        metrics: Dict of metric_name → value
        description: Version description
        input_example: Example input for schema inference

    Returns:
        Model version number, or None on failure.
    """
    _connect()
    if _mr is None:
        logger.warning("Hopsworks MR unavailable — saving locally")
        _save_model_local_fallback(model_obj, model_name)
        return None

    try:
        import joblib
        import tempfile
        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema

        # Save model to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            joblib.dump(model_obj, tmp.name)
            model_path = tmp.name

        # Create schema from input example if provided
        input_schema = None
        output_schema = None
        if input_example is not None:
            try:
                import numpy as np
                input_schema = Schema(np.array(input_example))
                output_schema = Schema(np.array([0.0]))
            except Exception:
                pass

        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema) if input_schema else None

        # Create or get model
        try:
            hw_model = _mr.get_model(name=model_name)
        except Exception:
            hw_model = _mr.create_model(name=model_name, description=description)

        # Register new version
        hw_model.save(
            model_path,
            model_schema=model_schema,
            metrics=metrics or {},
        )

        # Cleanup temp file
        os.unlink(model_path)

        versions = hw_model.versions
        latest_version = max(v.version for v in versions) if versions else 1
        logger.info("Registered model in Hopsworks: %s v%d", model_name, latest_version)
        return latest_version

    except Exception as e:
        logger.error("Hopsworks model registration failed: %s", e)
        _save_model_local_fallback(model_obj, model_name)
        return None


def get_latest_model(model_name: str = "aqi_forecaster") -> Optional[Any]:
    """Load the latest model version from Hopsworks Model Registry.

    Args:
        model_name: Model name to load

    Returns:
        Trained model object, or None if unavailable.
    """
    _connect()
    if _mr is None:
        return _load_model_local_fallback(model_name)

    try:
        hw_model = _mr.get_model(name=model_name)
        versions = hw_model.versions
        if not versions:
            logger.warning("No versions found for model: %s", model_name)
            return _load_model_local_fallback(model_name)

        latest = sorted(versions, key=lambda v: v.version)[-1]
        model_dir = latest.download()

        import joblib
        model_files = list(Path(model_dir).glob("*.pkl"))
        if model_files:
            model = joblib.load(str(model_files[0]))
            logger.info("Loaded model from Hopsworks: %s v%d", model_name, latest.version)
            return model

        # Try sklearn
        try:
            from hsml import sklearn
            return None  # hsml loading handled differently
        except Exception:
            pass

        return _load_model_local_fallback(model_name)

    except Exception as e:
        logger.warning("Could not load model from Hopsworks: %s", e)
        return _load_model_local_fallback(model_name)


def log_metrics_to_store(metrics: Dict[str, float], run_name: str = "") -> bool:
    """Log training metrics to Hopsworks (stored as a feature group for tracking)."""
    _connect()
    if _fs is None:
        return False

    try:
        metrics["run_name"] = run_name
        metrics["logged_at"] = datetime.now().isoformat()
        df = pd.DataFrame([metrics])
        write_feature_group(
            name="training_metrics",
            df=df,
            version=1,
            description="Model training metrics per run",
            primary_key=["run_name", "logged_at"],
        )
        return True
    except Exception as e:
        logger.warning("Metric logging failed: %s", e)
        return False


# ─── Local Fallbacks (when Hopsworks is unreachable) ─────────────

_LOCAL_DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


def _save_local_fallback(name: str, df: pd.DataFrame):
    path = _LOCAL_DATA_DIR / "processed" / "features" / f"{name}_latest.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _load_local_fallback(name: str) -> Optional[pd.DataFrame]:
    path = _LOCAL_DATA_DIR / "processed" / "features" / f"{name}_latest.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def _save_model_local_fallback(model_obj, model_name: str):
    import joblib
    path = _LOCAL_DATA_DIR.parent / "models" / "artifacts" / f"{model_name}_latest.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_obj, path)
    logger.info("Model saved locally: %s", path)


def _load_model_local_fallback(model_name: str) -> Optional[Any]:
    import joblib
    path = _LOCAL_DATA_DIR.parent / "models" / "artifacts" / f"{model_name}_latest.pkl"
    if path.exists():
        return joblib.load(path)
    return None
