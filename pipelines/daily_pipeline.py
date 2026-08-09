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

import pandas as pd

from feature_store.feature_builder import FeatureBuilder
from feature_store.hopsworks_client import (
    write_feature_group,
    is_available as hopsworks_available,
)
from ingestion.orchestrator import IngestionOrchestrator
from models.registry import log_experiment, register_model, save_models_by_horizon
from models.trainer import build_models_for_horizons, find_best_model, find_best_models_per_horizon
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
        # Step 1: Fetch training data via standalone acquisition script
        logger.info("=== Step 1: Data Acquisition (2 years) ===")
        end_date = (now_local() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (now_local() - timedelta(days=731)).strftime("%Y-%m-%d")

        # Use the standalone fetcher which handles Open-Meteo + OpenAQ correctly
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "scripts.fetch_training_data",
                "--start", start_date,
                "--end", end_date,
                "--years", "2",
            ],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout for full 2-year fetch
        )
        logger.info("Acquisition output:\n%s", result.stdout)
        if result.returncode != 0:
            logger.warning("Acquisition script warnings:\n%s", result.stderr)

        # Load the produced CSV
        csv_path = DATA_DIR / "backfill" / f"training_data_{start_date}_{end_date}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            observed_count = df["aqi"].notna().sum() if "aqi" in df.columns else 0
            logger.info("Loaded training CSV: %d rows, %d with observed labels", len(df), observed_count)
        else:
            # Fall back to orchestrator if CSV not produced
            logger.warning("CSV not found at %s — using orchestrator fallback", csv_path)
            orchestrator = IngestionOrchestrator()
            df = orchestrator.backfill(start_date, end_date, use_openaq=True)
            observed_count = df["aqi"].notna().sum() if "aqi" in df.columns else 0

        status["steps"]["backfill"] = {
            "status": "ok",
            "rows": len(df) if df is not None else 0,
            "range": f"{start_date} → {end_date}",
            "observed_labels": observed_count,
        }
        logger.info("Backfill: %d rows, %d with observed AQI labels", len(df) if df is not None else 0, observed_count)

        # Step 2: Build training dataset
        logger.info("=== Step 2: Build Training Dataset ===")
        merged_path = DATA_DIR / "processed" / "merged_hourly" / "merged_latest.parquet"
        if df is None or df.empty:
            try:
                df = load_parquet(merged_path)
            except FileNotFoundError:
                logger.warning("No data available")
                status["steps"]["features"] = {"status": "error", "detail": "No data"}
                status["completed_at"] = format_iso(now_local())
                _save_daily_status(status)
                return status

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
        # Guard: skip if no target columns exist (e.g. zero observed labels fetched)
        target_cols_map = {
            "24h": "target_aqi_24h",
            "48h": "target_aqi_48h",
            "72h": "target_aqi_72h",
        }
        available_targets = {k: v for k, v in target_cols_map.items() if v in train_df.columns}
        if not available_targets:
            logger.warning("No target columns found in training data — skipping training")
            status["steps"]["training"] = {"status": "skipped", "detail": "No observed labels in backfill. Run again after hourly pipeline accumulates data."}
            status["completed_at"] = format_iso(now_local())
            _save_daily_status(status)
            return status

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

        # Find best model per horizon (24h, 48h, 72h independently)
        best_per_horizon = find_best_models_per_horizon(results)

        status["steps"]["training"] = {
            "status": "ok",
            "best_model": best_mname,
            "best_horizon": best_horizon,
            "models_trained": sum(len(v) for v in results.values()),
            "per_horizon": {
                h: {"model": e["model_name"], "rmse": round(e.get("rmse", 0), 3)}
                for h, e in best_per_horizon.items()
            },
        }

        # Write full metrics table to disk so dashboard can display it
        metrics_summary = {
            "trained_at": format_iso(now_local()),
            "n_train_rows": split_idx,
            "n_test_rows": len(train_df) - split_idx,
            "n_features": len(feature_cols),
            "best_model": best_mname,
            "best_horizon": best_horizon,
            "per_horizon_best": {
                h: {
                    "model_name": e["model_name"],
                    "rmse": round(e.get("rmse", 0), 3),
                    "mae": round(e["metrics"].get(f"mae_{h}", 0), 3) if "metrics" in e else None,
                    "r2": round(e["metrics"].get(f"r2_{h}", 0), 3) if "metrics" in e else None,
                }
                for h, e in best_per_horizon.items()
            },
            "all_models": {
                f"{h}_{mname}": {
                    "horizon": h,
                    "model_name": mname,
                    "metrics": result.get("metrics", {}),
                }
                for h, horizon_results in results.items()
                for mname, result in horizon_results.items()
                if mname not in ("persistence", "seasonal_naive")  # Exclude trivial baselines
            },
        }
        save_json(metrics_summary, DATA_DIR / "processed" / "training_metrics.json")

        # Print a clear metrics table to GitHub Actions logs
        logger.info("\n" + "=" * 60)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        for h, entry in best_per_horizon.items():
            rmse = round(entry.get("rmse", 0), 2)
            mae = round(entry["metrics"].get(f"mae_{h}", 0), 2) if "metrics" in entry else "?"
            r2 = round(entry["metrics"].get(f"r2_{h}", 0), 3) if "metrics" in entry else "?"
            logger.info("  Horizon %s | Best: %-20s | RMSE: %6.2f | MAE: %6.2f | R2: %5.3f",
                         h, entry["model_name"], rmse, mae if isinstance(mae, float) else 0, r2 if isinstance(r2, float) else 0)
        logger.info("=" * 60 + "\n")

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

        # Step 5: Save local fallback AND run inference for dashboard
        if best_model is not None:
            local_path = DATA_DIR.parent / "models" / "artifacts" / f"best_model_{now_local().strftime('%Y%m%d')}.pkl"
            best_model.save(local_path)

            # Also save to the standard inference path
            import joblib
            fallback_path = DATA_DIR.parent / "models" / "artifacts" / "aqi_forecaster_latest.pkl"
            joblib.dump(best_model, fallback_path)

            # Save per-horizon models for differentiated forecasts
            if best_per_horizon:
                save_models_by_horizon(best_per_horizon, feature_cols=feature_cols)
                logger.info("Saved per-horizon models: %s",
                            {h: e["model_name"] for h, e in best_per_horizon.items()})

            # Run inference on the latest data to produce forecast
            try:
                from models.inference import InferenceEngine
                from pipelines.hourly_pipeline import _to_json_safe, _embed_history_from_df

                engine = InferenceEngine()
                engine.model = best_model  # Use full wrapper (baselines have .model=None)

                # Multi-city: generate per-city forecasts
                cities = train_df["city"].unique().tolist() if "city" in train_df.columns else [None]
                city_forecasts = {}

                for city_name in cities:
                    if city_name is not None:
                        city_df = train_df[train_df["city"] == city_name]
                    else:
                        city_df = train_df

                    if city_df.empty:
                        continue

                    fc = engine.predict(city_df.tail(200))

                    # Embed weather + pollutant data
                    latest_row = city_df.iloc[-1] if len(city_df) > 0 else None
                    if latest_row is not None:
                        fc["weather"] = {
                            "temperature": _to_json_safe(latest_row.get("temperature_2m")),
                            "humidity": _to_json_safe(latest_row.get("relative_humidity_2m")),
                            "pressure": _to_json_safe(latest_row.get("pressure_msl")),
                            "wind_speed": _to_json_safe(latest_row.get("wind_speed_10m")),
                            "wind_direction": _to_json_safe(latest_row.get("wind_direction_10m")),
                            "precipitation": _to_json_safe(latest_row.get("precipitation")),
                            "cloud_cover": _to_json_safe(latest_row.get("cloud_cover")),
                        }
                        fc["pollutants"] = {
                            "pm2_5": _to_json_safe(latest_row.get("pm2_5")),
                            "pm10": _to_json_safe(latest_row.get("pm10")),
                            "no2": _to_json_safe(latest_row.get("no2")),
                            "o3": _to_json_safe(latest_row.get("o3")),
                            "so2": _to_json_safe(latest_row.get("so2")),
                            "co": _to_json_safe(latest_row.get("co")),
                        }
                        fc["station"] = str(latest_row.get("station_name") or "OpenAQ")

                    fc["city"] = city_name
                    _embed_history_from_df(fc, city_df)
                    city_forecasts[city_name or "default"] = fc
                    logger.info("  %s: current_aqi=%.1f", city_name or "default", fc.get("current_aqi", 0))

                # Build combined forecast JSON
                if len(city_forecasts) == 1:
                    forecast = list(city_forecasts.values())[0]
                else:
                    primary = list(city_forecasts.values())[0]
                    forecast = {**primary, "cities": city_forecasts}

                save_json(forecast, DATA_DIR / "processed" / "predictions" / "forecast_latest.json")
                logger.info("Forecast generated and saved")
            except Exception as e:
                logger.warning("Could not generate forecast: %s", e)

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
