"""Tests for AQI utility functions."""

import pytest
from utils.aqi_utils import classify_aqi, is_alert_level, category_color, AQICategory


class TestAQIClassification:
    def test_good(self):
        assert classify_aqi(0) == AQICategory.GOOD
        assert classify_aqi(25) == AQICategory.GOOD
        assert classify_aqi(50) == AQICategory.GOOD

    def test_moderate(self):
        assert classify_aqi(51) == AQICategory.MODERATE
        assert classify_aqi(80) == AQICategory.MODERATE
        assert classify_aqi(100) == AQICategory.MODERATE

    def test_unhealthy_sensitive(self):
        assert classify_aqi(101) == AQICategory.UNHEALTHY_SENSITIVE
        assert classify_aqi(130) == AQICategory.UNHEALTHY_SENSITIVE

    def test_unhealthy(self):
        assert classify_aqi(151) == AQICategory.UNHEALTHY
        assert classify_aqi(180) == AQICategory.UNHEALTHY

    def test_very_unhealthy(self):
        assert classify_aqi(201) == AQICategory.VERY_UNHEALTHY
        assert classify_aqi(250) == AQICategory.VERY_UNHEALTHY

    def test_hazardous(self):
        assert classify_aqi(301) == AQICategory.HAZARDOUS
        assert classify_aqi(500) == AQICategory.HAZARDOUS

    def test_negative_returns_unknown(self):
        assert classify_aqi(-1) == AQICategory.UNKNOWN


class TestAlerts:
    def test_alert_threshold(self):
        assert not is_alert_level(100)
        assert not is_alert_level(199)
        assert is_alert_level(200)
        assert is_alert_level(250)
        assert is_alert_level(500)

    def test_category_colors(self):
        assert category_color(AQICategory.GOOD) == "#00e400"
        assert category_color(AQICategory.HAZARDOUS) == "#7e0023"
        assert category_color(AQICategory.UNKNOWN) == "#808080"
