"""Tests for feature engineering."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from feature_store.feature_builder import FeatureBuilder


@pytest.fixture
def sample_df():
    """Create a 200-hour sample dataset for testing."""
    base = datetime(2026, 7, 20, 0)
    timestamps = [base + timedelta(hours=i) for i in range(200)]

    np.random.seed(42)
    aqi = np.clip(30 + np.random.randn(200) * 20 + np.sin(np.arange(200) * 2 * np.pi / 24) * 15, 0, 300)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "aqi": aqi,
        "pm2_5": aqi * 0.8 + np.random.randn(200) * 5,
        "pm10": aqi * 1.1 + np.random.randn(200) * 10,
        "temperature_2m": 25 + np.sin(np.arange(200) * 2 * np.pi / 24) * 8 + np.random.randn(200) * 2,
        "relative_humidity_2m": np.clip(50 + np.random.randn(200) * 15, 0, 100),
        "wind_speed_10m": np.abs(3 + np.random.randn(200) * 2),
        "wind_direction_10m": np.random.uniform(0, 360, 200),
        "precipitation": np.abs(np.random.randn(200)) * 2,
        "cloud_cover": np.clip(np.random.uniform(0, 100, 200), 0, 100),
    })
    return df


class TestFeatureBuilder:
    def test_build_all_creates_expected_columns(self, sample_df):
        builder = FeatureBuilder(sample_df)
        result = builder.build_all()

        # Time features
        assert "hour" in result.columns
        assert "day" in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

        # Lag features
        assert "aqi_lag_1" in result.columns
        assert "aqi_lag_24" in result.columns
        assert "pm25_lag_1" in result.columns

        # Rolling features
        assert "aqi_roll_mean_6" in result.columns
        assert "aqi_roll_mean_24" in result.columns

        # Interaction features
        assert "humidity_x_temperature" in result.columns
        assert "aqi_change_rate" in result.columns

        # Targets
        assert "target_aqi_24h" in result.columns
        assert "target_aqi_48h" in result.columns
        assert "target_aqi_72h" in result.columns

    def test_targets_have_correct_shift(self, sample_df):
        builder = FeatureBuilder(sample_df)
        result = builder.build_all()

        # target_aqi_24h at t=0 should match aqi at t=24
        aqi_t24 = result["aqi"].iloc[24]
        target_t0 = result["target_aqi_24h"].iloc[0]
        assert abs(target_t0 - aqi_t24) < 0.001

    def test_get_training_data_drops_nan_targets(self, sample_df):
        builder = FeatureBuilder(sample_df)
        builder.build_all()
        train = builder.get_training_data()

        # Should be less than original because last rows have NaN targets
        assert len(train) < len(sample_df)

    def test_cyclical_encodings_range(self, sample_df):
        builder = FeatureBuilder(sample_df)
        result = builder.build_all()

        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()
