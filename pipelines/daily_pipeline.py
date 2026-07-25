"""Daily pipeline — backfill, feature generation → Hopsworks FS, training, model registry.

Runs on GitHub Actions daily at 2 AM. Entirely serverless — no persistent servers.
- Backfills 2 years of historical data from Open-Meteo + OpenAQ
- Builds supervised training dataset
- Trains all models, selects best by RMSE at 24h
- Registers best model to Hopsworks Model Registry (or MLflow)
- Writes feature table to Hopsworks Feature Store
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from feature_store.feature_builder import FeatureBuilder
from feature_store.hopsworks_client import (
    write_feature_group,
    is_available as hopsworks_available,
)
from ingestion.orchestrator import IngestionOrchestrator
from models.registry import log_experiment, register_model
from models.trainer import build_models_for_horizons, find_best_model
from utils.config import get
from utils.logging import setup_logger
from utils.storage import load_parquet, save_json, save_parquet
from utils.time_utils import format_iso, now_local

logger = setup_logger("daily_pipeline")

DATA_DIR = Path(get("storage.data_dir", "./data"))


def run_daily_pipeline() -> dict:
    status = {
        "pipeline": "daily",
        "started_at": format_iso(now_local()),
        "steps": {},
        "success": False,
        "backend": "hopsworks" if hopsworks_available() else "local",
    }

    try:
        # Step 1: Backfill — 2 years via Open-Meteo + OpenAQ
        logger.info("=== Step 1: Backfill (2 years) ===")
        end_date = now_local().strftime("%Y-%m-%d")
        start_date = (now_local() - timedelta(days=730)).strftime("%Y-%m-%d")
        orchestrator = IngestionOrchestrator()
        backfill_df = orchestrator.backfill(start_date, end_date, use_openaq=True)
        observed_count = (
            backfill_df["aqi"].notna().sum() if "aqi" in backfill_df.columns else 0
        )
        status["steps"]["backfill"] = {
            "status": "ok",
            "rows": len(backfill_df),
            "range": f"{start_date} → {end_date}",
            "observed_labels": observed_count,
        }
        logger.info("Backfill: %d rows, %d with observed AQI labels", len(backfill_df), observed_count)

        # Step 2: Build training dataset
        logger.info("=== Step 2: Build Training Dataset ===")
        merged_path = DATA_DIR / "processed" / "merged_hourly" / "merged_latest.parquet"
        try:
            df = load_parquet(merged_path)
        except FileNotFoundError:
            logger.warning("No merged data found, using backfill only")
            df = backfill_df

        if df.empty:
            status["steps"]["features"] = {"status": "error", "detail": "No data"}
            status["completed_at"] = format_iso(now_local())
            _save_daily_status(status)
            return status

        builder = FeatureBuilder(df)
        featured = builder.build_all()
        train_df = builder.get_training_data()

        # Write training feature table to Hopsworks FS
        if hopsworks_available():
            write_feature_group(
                name="aqi_training_features",
                df=featured,
                version=1,
                description=f"Daily training features — {len(train_df)} rows, {len(featured.columns)} cols",
                primary_key=["timestamp"],
                online_enabled=False,
            )
            logger.info("Training features written to Hopsworks FS")

        feature_path = DATA_DIR / "processed" / "features" / f"features_daily_{now_local().strftime('%Y%m%d')}.parquet"
        save_parquet(featured, feature_path)

        status["steps"]["features"] = {
            "status": "ok",
            "total_rows": len(featured),
            "train_rows": len(train_df),
            "features": len([c for c in featured.columns if not c.startswith("target_")]),
        }

        if len(train_df) < 720:
            logger.warning("Insufficient training data: %d < 720 rows", len(train_df))
            status["steps"]["training"] = {"status": "skipped", "detail": "Insufficient data"}
            status["completed_at"] = format_iso(now_local())
            _save_daily_status(status)
            return status

        # Step 3: Train models
        logger.info("=== Step 3: Model Training ===")
        feature_cols = [
            c for c in featured.columns
            if not c.startswith("target_")
            and c not in ("timestamp", "source", "station_name", "city", "country",
                          "dominant_pollutant", "merged_at", "fetched_at", "latitude", "longitude")
            and featured[c].dtype in ("float64", "float32", "int64", "int32")
        ]
        target_cols = {"24h": "target_aqi_24h", "48h": "target_aqi_48h", "72h": "target_aqi_72h"}

        split_idx = int(len(train_df) * 0.8)
        train_split = train_df.iloc[:split_idx]
        test_split = train_df.iloc[split_idx:]

        results = build_models_for_horizons(feature_cols, target_cols, train_split, test_split)
        best_mname, best_horizon, best_model, all_metrics = find_best_model(results, "rmse_24h")

        status["steps"]["training"] = {
            "status": "ok",
            "best_model": best_mname,
            "best_horizon": best_horizon,
            "models_trained": sum(len(v) for v in results.values()),
        }

        # Step 4: Register to Hopsworks Model Registry (or MLflow fallback)
        logger.info("=== Step 4: Model Registry ===")
        try:
            log_experiment(results, feature_cols, target_cols)
            model_uri = register_model()
            status["steps"]["registry"] = {
                "status": "ok",
                "model_uri": model_uri or "hopsworks:/aqi_forecaster",
                "backend": "hopsworks" if hopsworks_available() else "mlflow/local",
            }
        except Exception as e:
            logger.warning("Registry failed: %s", e)
            status["steps"]["registry"] = {"status": "warning", "detail": str(e)}

        # Step 5: Save local fallback
        if best_model is not None:
            local_path = DATA_DIR.parent / "models" / "artifacts" / f"best_model_{now_local().strftime('%Y%m%d')}.pkl"
            best_model.save(local_path)

        status["success"] = True

    except Exception as e:
        logger.exception("Daily pipeline failed: %s", e)
        status["success"] = False
        status["error"] = str(e)

    status["completed_at"] = format_iso(now_local())
    _save_daily_status(status)
    return status


def _save_daily_status(status: dict):
    path = DATA_DIR / "processed" / "daily_status.json"
    save_json(status, path)


if __name__ == "__main__":
    result = run_daily_pipeline()
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["success"] else 1)
