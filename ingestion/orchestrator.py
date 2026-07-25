"""Ingestion orchestrator — coordinates all providers and merges data.

Provider roles:
  - Open-Meteo: weather features (predictors) — always available, no auth
  - OpenAQ: primary observed AQI/pollutant labels — has 2+ years historical data
  - AQICN: secondary observed labels — live data only, kept for redundancy

Label priority for training: OpenAQ > AQICN > Open-Meteo forecast AQI
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from ingestion.providers.aqicn import AQICNProvider
from ingestion.providers.openmeteo import OpenMeteoProvider
from ingestion.providers.openaq import OpenAQProvider
from utils.config import get
from utils.logging import get_logger
from utils.storage import load_parquet, save_parquet
from utils.time_utils import floor_hour, format_iso, now_local

logger = get_logger(__name__)

DATA_DIR = Path(get("storage.data_dir", "./data"))
MERGED_DIR = DATA_DIR / "processed" / "merged_hourly"
BACKFILL_DIR = DATA_DIR / "backfill"

# Labels from observed sources override Open-Meteo forecast columns
LABEL_PRIORITY = ["openaq", "aqicn"]


class IngestionOrchestrator:
    def __init__(self):
        self.open_meteo = OpenMeteoProvider()
        self.aqicn = AQICNProvider()
        self.openaq = OpenAQProvider()

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch latest data from all providers.

        Returns dict keyed by provider name: 'open_meteo', 'aqicn', 'openaq'.
        """
        logger.info("Starting ingestion run — all providers")
        results = {}

        # Open-Meteo (always — features)
        try:
            om_df = self.open_meteo.run()
            results["open_meteo"] = om_df
            logger.info("  Open-Meteo: %d rows", len(om_df))
        except Exception as e:
            logger.warning("Open-Meteo fetch failed: %s", e)
            results["open_meteo"] = pd.DataFrame()

        # OpenAQ (primary labels)
        try:
            oa_df = self.openaq.run()
            results["openaq"] = oa_df
            logger.info("  OpenAQ: %d rows", len(oa_df))
        except Exception as e:
            logger.warning("OpenAQ fetch failed: %s", e)
            results["openaq"] = pd.DataFrame()

        # AQICN (secondary labels)
        try:
            aq_df = self.aqicn.run()
            results["aqicn"] = aq_df
            logger.info("  AQICN: %d rows", len(aq_df))
        except Exception as e:
            logger.warning("AQICN fetch failed: %s", e)
            results["aqicn"] = pd.DataFrame()

        return results

    def merge(self, om_df: pd.DataFrame, observed_dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge Open-Meteo weather with observed labels from one or more sources.

        Observed sources are applied in priority order — later sources don't
        overwrite labels already set by higher-priority sources.

        Args:
            om_df: Open-Meteo DataFrame (weather features)
            observed_dfs: List of DataFrames from observed sources (OpenAQ, AQICN),
                          in priority order (highest first).

        Returns:
            Merged DataFrame with weather features + best available observed labels.
        """
        if om_df.empty:
            logger.warning("Open-Meteo returned empty data — merge may be incomplete")

        om_cols = [
            "timestamp",
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "cloud_cover",
            "us_aqi",
            "pm2_5",
            "pm10",
        ]

        # Rename Open-Meteo pollutant columns to avoid collision with observed labels
        om_renames = {
            "us_aqi": "om_forecast_aqi",
            "pm2_5": "om_forecast_pm25",
            "pm10": "om_forecast_pm10",
        }
        om_sub = om_df[[c for c in om_cols if c in om_df.columns]].copy()
        om_sub.rename(columns=om_renames, inplace=True)

        # Start with Open-Meteo as the base
        merged = om_sub.copy()

        # Merge observed sources one by one, higher priority first
        label_cols = ["aqi", "pm2_5", "pm10", "no2", "o3", "so2", "co", "station_name", "dominant_pollutant"]

        for obs_df in observed_dfs:
            if obs_df.empty:
                continue

            available_cols = ["timestamp"] + [c for c in label_cols if c in obs_df.columns]
            obs_sub = obs_df[available_cols].copy()
            obs_sub["_has_obs"] = True

            # Merge — keep existing labels if already set by higher-priority source
            merged = pd.merge(merged, obs_sub, on="timestamp", how="left", suffixes=("", "_new"))

            # For each label column, only fill if currently null
            for col in label_cols:
                new_col = f"{col}_new"
                if new_col in merged.columns and col in merged.columns:
                    merged[col] = merged[col].fillna(merged[new_col])
                    merged.drop(columns=[new_col], inplace=True)
                elif new_col in merged.columns:
                    merged.rename(columns={new_col: col}, inplace=True)

        merged = merged.sort_values("timestamp").reset_index(drop=True)
        merged["merged_at"] = format_iso(now_local())

        logger.info("Merged table: %d rows, %d with observed AQI labels",
                     len(merged), merged["aqi"].notna().sum() if "aqi" in merged.columns else 0)
        return merged

    def save_merged(self, df: pd.DataFrame) -> Path:
        """Append to canonical merged table."""
        now = floor_hour(now_local())
        path = MERGED_DIR / f"merged_{now.strftime('%Y%m%d_%H')}.parquet"
        save_parquet(df, path)

        # Update rolling merged file
        full_path = MERGED_DIR / "merged_latest.parquet"
        try:
            existing = load_parquet(full_path)
        except FileNotFoundError:
            existing = pd.DataFrame()

        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        save_parquet(combined, full_path)

        return path

    def backfill(
        self,
        start_date: str,
        end_date: str,
        use_openaq: bool = True,
    ) -> pd.DataFrame:
        """Backfill historical data from all available sources.

        This is the critical training data pipeline:
        1. Fetch historical weather from Open-Meteo (always available, 2+ years)
        2. Fetch historical observed AQI from OpenAQ (2+ years if station has data)
        3. Merge weather features + observed labels into supervised training rows

        Args:
            start_date: YYYY-MM-DD start
            end_date: YYYY-MM-DD end
            use_openaq: Whether to fetch historical labels from OpenAQ

        Returns:
            Merged DataFrame with weather features + observed AQI labels.
        """
        logger.info("=== Backfill: %s → %s ===", start_date, end_date)
        logger.info("Fetching historical weather from Open-Meteo...")
        om_hist = self.open_meteo.fetch_historical(start_date, end_date)
        logger.info("  Open-Meteo: %d rows", len(om_hist))

        observed_dfs = []

        if use_openaq:
            logger.info("Fetching historical observed AQI from OpenAQ...")
            try:
                oa_hist = self.openaq.fetch_historical(start_date, end_date)
                logger.info("  OpenAQ: %d hourly rows with observed labels", len(oa_hist))
                if not oa_hist.empty:
                    observed_dfs.append(oa_hist)
            except Exception as e:
                logger.warning("OpenAQ backfill failed: %s", e)

        # Also try AQICN if we have recent live data accumulated
        try:
            aqicn_merged = MERGED_DIR / "merged_latest.parquet"
            if aqicn_merged.exists():
                existing = load_parquet(aqicn_merged)
                aqicn_data = existing[existing["source"] == "aqicn"] if "source" in existing.columns else pd.DataFrame()
                if not aqicn_data.empty:
                    logger.info("  Found %d AQICN rows from live ingestion", len(aqicn_data))
        except Exception:
            pass

        # Merge: Open-Meteo weather + observed labels (OpenAQ priority first)
        if not om_hist.empty:
            merged = self.merge(om_hist, observed_dfs)
            version = "train_v2" if use_openaq else "train_v1"
            path = BACKFILL_DIR / version / f"backfill_{start_date}_{end_date}.parquet"
            save_parquet(merged, path)
            logger.info("Backfill saved: %s (%d rows, %d with observed AQI)",
                         path, len(merged),
                         merged["aqi"].notna().sum() if "aqi" in merged.columns else 0)
            return merged

        logger.warning("Backfill returned empty data")
        return pd.DataFrame()

    def run_full_cycle(self) -> pd.DataFrame:
        """Run a complete live ingestion cycle: fetch all → merge → save."""
        results = self.fetch_all()

        om_df = results.get("open_meteo", pd.DataFrame())

        # Build observed_dfs list in priority order: OpenAQ first, then AQICN
        observed_dfs = []
        for source in LABEL_PRIORITY:
            df = results.get(source, pd.DataFrame())
            if not df.empty:
                observed_dfs.append(df)

        if om_df.empty:
            logger.warning("Open-Meteo data unavailable — using whatever is available")
            # Try to construct from any available source
            all_dfs = [df for df in results.values() if not df.empty]
            if all_dfs:
                om_df = all_dfs[0]

        merged = self.merge(om_df, observed_dfs)
        if not merged.empty:
            self.save_merged(merged)
        return merged
