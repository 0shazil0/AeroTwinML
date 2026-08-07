"""Feature engineering pipeline — builds all feature groups for ML models."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.config import get
from utils.logging import get_logger
from utils.storage import load_parquet, save_parquet

logger = get_logger(__name__)

DATA_DIR = Path(get("storage.data_dir", "./data"))
FEATURES_DIR = DATA_DIR / "processed" / "features"


class FeatureBuilder:
    """Builds time, lag, rolling, weather, and interaction features.

    Usage:
        builder = FeatureBuilder(df)
        featured = builder.build_all()
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

    def build_all(self) -> pd.DataFrame:
        """Run all feature groups in sequence."""
        self._add_time_features()
        self._add_city_feature()
        self._add_lag_features()
        self._add_rolling_features()
        self._add_weather_features()
        self._add_interaction_features()
        self._add_targets()
        self._add_classification_labels()
        return self.df

    def _add_time_features(self) -> None:
        """Add cyclical and categorical time features."""
        ts = self.df["timestamp"]
        self.df["hour"] = ts.dt.hour.astype(int)
        self.df["day"] = ts.dt.day.astype(int)
        self.df["day_of_week"] = ts.dt.dayofweek.astype(int)
        self.df["month"] = ts.dt.month.astype(int)
        self.df["weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        self.df["season"] = ts.dt.month.map(season_map)

        # Cyclical encodings
        self.df["hour_sin"] = np.sin(2 * np.pi * self.df["hour"] / 24)
        self.df["hour_cos"] = np.cos(2 * np.pi * self.df["hour"] / 24)
        self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)
        self.df["day_of_week_sin"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["day_of_week_cos"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)

    def _add_city_feature(self) -> None:
        """Encode city column as numeric feature for multi-city training."""
        if "city" not in self.df.columns:
            return
        cities = self.df["city"].dropna().unique()
        city_map = {name: i for i, name in enumerate(sorted(cities))}
        self.df["city_encoded"] = self.df["city"].map(city_map).fillna(0).astype(int)
        logger.info("City feature added: %s", city_map)

    def _get_aqi_col(self) -> str:
        """Return the best available AQI column to use as label."""
        if "aqi" in self.df.columns and self.df["aqi"].notna().sum() > 0:
            return "aqi"
        if "om_forecast_aqi" in self.df.columns:
            return "om_forecast_aqi"
        return "us_aqi" if "us_aqi" in self.df.columns else None

    def _get_pm25_col(self) -> Optional[str]:
        if "pm2_5" in self.df.columns and self.df["pm2_5"].notna().sum() > 0:
            return "pm2_5"
        return None

    def _get_pm10_col(self) -> Optional[str]:
        if "pm10" in self.df.columns and self.df["pm10"].notna().sum() > 0:
            return "pm10"
        return None

    def _add_lag_features(self) -> None:
        """Add lagged AQI and pollutant values (grouped by city if multi-city)."""
        aqi_col = self._get_aqi_col()
        pm25_col = self._get_pm25_col()
        pm10_col = self._get_pm10_col()

        lag_windows = get("pipeline.features.lag_hours", [1, 6, 24, 72])
        has_city = "city" in self.df.columns

        for col, col_name in [(aqi_col, "aqi"), (pm25_col, "pm25"), (pm10_col, "pm10")]:
            if col is None:
                continue
            for lag in lag_windows:
                if has_city:
                    self.df[f"{col_name}_lag_{lag}"] = self.df.groupby("city")[col].shift(lag)
                else:
                    self.df[f"{col_name}_lag_{lag}"] = self.df[col].shift(lag)

    def _add_rolling_features(self) -> None:
        """Add rolling statistics (grouped by city if multi-city)."""
        aqi_col = self._get_aqi_col()
        rolling_windows = get("pipeline.features.rolling_windows_hours", [6, 24])

        if aqi_col is None:
            return

        has_city = "city" in self.df.columns

        for window in rolling_windows:
            if has_city:
                grouped = self.df.groupby("city")[aqi_col]
                self.df[f"aqi_roll_mean_{window}"] = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
                self.df[f"aqi_roll_std_{window}"] = grouped.transform(lambda x: x.rolling(window, min_periods=1).std())
                self.df[f"aqi_roll_min_{window}"] = grouped.transform(lambda x: x.rolling(window, min_periods=1).min())
                self.df[f"aqi_roll_max_{window}"] = grouped.transform(lambda x: x.rolling(window, min_periods=1).max())
            else:
                roll = self.df[aqi_col].rolling(window=window, min_periods=1)
                self.df[f"aqi_roll_mean_{window}"] = roll.mean()
                self.df[f"aqi_roll_std_{window}"] = roll.std()
                self.df[f"aqi_roll_min_{window}"] = roll.min()
                self.df[f"aqi_roll_max_{window}"] = roll.max()

    def _add_weather_features(self) -> None:
        """Pass through and clean weather features (grouped by city if multi-city)."""
        weather_cols = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "cloud_cover",
        ]
        has_city = "city" in self.df.columns
        for col in weather_cols:
            if col in self.df.columns:
                if has_city:
                    self.df[col] = self.df.groupby("city")[col].transform(lambda x: x.ffill().bfill())
                else:
                    self.df[col] = self.df[col].ffill().bfill()

    def _add_interaction_features(self) -> None:
        """Add engineered interaction features."""
        if "temperature_2m" in self.df.columns and "relative_humidity_2m" in self.df.columns:
            self.df["humidity_x_temperature"] = (
                self.df["relative_humidity_2m"] * self.df["temperature_2m"]
            )

        pm25_col = self._get_pm25_col()
        if "wind_speed_10m" in self.df.columns and pm25_col and pm25_col in self.df.columns:
            self.df["wind_x_pm25"] = self.df["wind_speed_10m"] * self.df[pm25_col]

        pm10_col = self._get_pm10_col()
        if "precipitation" in self.df.columns and pm10_col and pm10_col in self.df.columns:
            self.df["rain_x_pm10"] = self.df["precipitation"] * self.df[pm10_col]

        aqi_col = self._get_aqi_col()
        if aqi_col and aqi_col in self.df.columns:
            has_city = "city" in self.df.columns
            # 6-hour change rate
            if has_city:
                self.df["aqi_change_rate"] = self.df.groupby("city")[aqi_col].transform(lambda x: x.diff(6) / 6)
            else:
                self.df["aqi_change_rate"] = self.df[aqi_col].diff(6) / 6

    def _add_targets(self) -> None:
        """Create future AQI targets at 24h, 48h, 72h horizons (grouped by city if multi-city)."""
        aqi_col = self._get_aqi_col()
        pm25_col = self._get_pm25_col()
        pm10_col = self._get_pm10_col()

        horizons = get("pipeline.features.target_horizons_hours", [24, 48, 72])
        has_city = "city" in self.df.columns

        if aqi_col:
            for h in horizons:
                if has_city:
                    self.df[f"target_aqi_{h}h"] = self.df.groupby("city")[aqi_col].shift(-h)
                else:
                    self.df[f"target_aqi_{h}h"] = self.df[aqi_col].shift(-h)

        if pm25_col:
            if has_city:
                self.df["target_pm25_24h"] = self.df.groupby("city")[pm25_col].shift(-24)
            else:
                self.df["target_pm25_24h"] = self.df[pm25_col].shift(-24)

        if pm10_col:
            if has_city:
                self.df["target_pm10_24h"] = self.df.groupby("city")[pm10_col].shift(-24)
            else:
                self.df["target_pm10_24h"] = self.df[pm10_col].shift(-24)

    def _add_classification_labels(self) -> None:
        """Add AQI category labels derived from target columns."""
        from utils.aqi_utils import classify_aqi

        target_cols = [c for c in self.df.columns if c.startswith("target_aqi_")]
        for col in target_cols:
            self.df[f"{col}_category"] = self.df[col].apply(
                lambda x: classify_aqi(x) if pd.notna(x) else None
            )

    def get_training_data(self) -> pd.DataFrame:
        """Return clean training data with features + targets, dropping rows with NaN targets."""
        target_cols = [c for c in self.df.columns if c.startswith("target_aqi_") and not c.endswith("_category")]
        if not target_cols:
            logger.warning("No target columns found")
            return self.df

        train_df = self.df.dropna(subset=target_cols, how="all").copy()
        train_df = train_df.dropna(subset=[target_cols[0]]).copy()  # require at least 24h target
        logger.info("Training data: %d rows after dropping NaN targets", len(train_df))
        return train_df

    def get_inference_data(self) -> pd.DataFrame:
        """Return latest row(s) with features but no target requirement."""
        return self.df.tail(72).copy()  # enough lag rows


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function: build all features from a DataFrame."""
    builder = FeatureBuilder(df)
    return builder.build_all()


def build_and_save(input_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    """Load merged data, build features, save result."""
    df = load_parquet(input_path)
    featured = build_features(df)

    if output_path:
        save_parquet(featured, output_path)
    else:
        path = FEATURES_DIR / f"features_{pd.Timestamp.now().strftime('%Y%m%d_%H')}.parquet"
        save_parquet(featured, path)

    return featured
