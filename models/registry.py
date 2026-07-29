"""Model registry integration — Hopsworks Model Registry with MLflow fallback.

Primary: Hopsworks Model Registry (serverless, managed)
Fallback: MLflow (local/Docker, for dev without Hopsworks credentials)
Last resort: Local pickle files
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from feature_store.hopsworks_client import (
    is_available as hopsworks_available,
    register_model as hopsworks_register,
    get_latest_model as hopsworks_get_model,
    log_metrics_to_store,
)
from models.trainer import find_best_model
from utils.config import get
from utils.logging import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(get("storage.data_dir", "./data")).parent / "models" / "artifacts"
REGISTRY_BACKEND = os.getenv("REGISTRY_BACKEND", "hopsworks")  # hopsworks | mlflow | local


def log_experiment(
    results: Dict[str, Any],
    feature_cols: list,
    target_cols: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Log training results to the active registry backend."""
    best_mname, best_horizon, best_model, all_metrics = find_best_model(results)

    run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # Build flat metrics dict
    flat_metrics: Dict[str, float] = {}
    for key, metric_dict in all_metrics.items():
        for metric_name, metric_val in metric_dict.items():
            if isinstance(metric_val, (int, float)):
                flat_metrics[f"{key}_{metric_name}"] = float(metric_val)

    # Add metadata
    flat_metrics["best_model_name"] = float(hash(best_mname) % 1000) if best_mname else 0
    flat_metrics["n_features"] = float(len(feature_cols))
    flat_metrics["n_horizons"] = float(len(target_cols))

    if hopsworks_available():
        # Log metrics to Hopsworks feature group
        log_metrics_to_store(flat_metrics, run_name)

        # Register best model to Hopsworks Model Registry
        if best_model is not None:
            underlying = best_model.model if hasattr(best_model, "model") else best_model
            hopsworks_register(
                model_obj=underlying,
                model_name="aqi_forecaster",
                metrics={"rmse_24h": flat_metrics.get("24h_persistence_rmse_24h", 0)},
                description=f"Trained on {len(feature_cols)} features, best: {best_mname} @ {best_horizon}",
            )
    else:
        # MLflow fallback
        _log_to_mlflow(results, feature_cols, target_cols, params)

    logger.info("Experiment logged: best=%s, horizon=%s", best_mname, best_horizon)


def _log_to_mlflow(
    results: Dict[str, Any],
    feature_cols: list,
    target_cols: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """MLflow fallback when Hopsworks is unavailable."""
    try:
        import mlflow
        import mlflow.sklearn

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "aqi_predictor"))

        best_mname, best_horizon, best_model, all_metrics = find_best_model(results)

        with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                "n_features": len(feature_cols),
                "target_horizons": str(list(target_cols.keys())),
                "best_model": best_mname,
                "best_horizon": best_horizon,
                **(params or {}),
            })
            for metric_name, metric_dict in all_metrics.items():
                mlflow.log_metrics(metric_dict)
            if best_model is not None:
                if hasattr(best_model, "model") and best_model.model is not None:
                    mlflow.sklearn.log_model(best_model.model, f"model_{best_mname}_{best_horizon}")
        logger.info("MLflow run logged: best=%s", best_mname)
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)


def get_latest_model(model_name: str = "aqi_forecaster") -> Optional[Any]:
    """Load latest model — tries Hopsworks first, then MLflow, then local pickle."""
    # Tier 1: Hopsworks
    if hopsworks_available():
        model = hopsworks_get_model(model_name)
        if model is not None:
            return model

    # Tier 2: MLflow
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        )
        models = client.search_registered_models()
        if models:
            latest = models[0].latest_versions[0]
            model_uri = f"models:/{models[0].name}/{latest.version}"
            return mlflow.sklearn.load_model(model_uri)
    except Exception:
        pass

    # Tier 3: Local fallback
    import joblib
    local_paths = sorted(MODEL_DIR.glob("best_model_*.pkl"), reverse=True)
    if local_paths:
        return joblib.load(local_paths[0])

    return None


def register_model(model_name: str = "aqi_forecaster") -> Optional[str]:
    """Register latest model. Redirects to Hopsworks if available."""
    if hopsworks_available():
        # Already registered during log_experiment — check if it succeeded
        result = hopsworks_register(
            model_obj=None,  # Already saved during log_experiment
            model_name=model_name,
        )
        if result is not None:
            logger.info("Model registered via Hopsworks: %s", model_name)
            return f"hopsworks:/{model_name}"
        logger.warning("Hopsworks registration returned None")
        return None

    # MLflow fallback
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        )
        runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)
        if runs:
            result = mlflow.register_model(f"runs:/{runs[0].info.run_id}/model", model_name)
            return f"models:/{model_name}/{result.version}"
    except Exception as e:
        logger.warning("MLflow registration failed: %s", e)

    return None
