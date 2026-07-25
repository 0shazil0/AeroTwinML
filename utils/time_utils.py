"""Time utilities — all timestamps in Asia/Karachi."""

from datetime import datetime, timezone

import pytz

from utils.config import get

_TZ_STR = get("city.timezone", "Asia/Karachi")
LOCAL_TZ = pytz.timezone(_TZ_STR)


def now_local() -> datetime:
    return datetime.now(LOCAL_TZ)


def utc_to_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LOCAL_TZ)


def local_to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = LOCAL_TZ.localize(dt)
    return dt.astimezone(timezone.utc)


def floor_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def format_iso(dt: datetime) -> str:
    return dt.isoformat()
