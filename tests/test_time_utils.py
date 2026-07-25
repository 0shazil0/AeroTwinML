"""Tests for time utility functions."""

import pytest
from datetime import datetime, timezone

from utils.time_utils import floor_hour, LOCAL_TZ


class TestTimeUtils:
    def test_floor_hour(self):
        dt = datetime(2026, 7, 23, 14, 35, 22)
        result = floor_hour(dt)
        assert result.minute == 0
        assert result.second == 0
        assert result.microsecond == 0
        assert result.hour == 14

    def test_floor_hour_midnight(self):
        dt = datetime(2026, 7, 23, 0, 0, 0)
        result = floor_hour(dt)
        assert result.hour == 0
        assert result.minute == 0

    def test_now_local_has_tz(self):
        from utils.time_utils import now_local
        result = now_local()
        assert result.tzinfo is not None

    def test_timezone_is_karachi(self):
        assert "Karachi" in str(LOCAL_TZ) or "Asia" in str(LOCAL_TZ)
