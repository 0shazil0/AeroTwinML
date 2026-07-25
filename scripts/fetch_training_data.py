"""
Standalone data acquisition script for AeroTwinML.

Fetches training data from both sources and produces a clean CSV:
  1. Open-Meteo Archive API -> historical weather (temperature, humidity, wind, etc.)
  2. OpenAQ v3 API -> observed PM2.5, PM10, NO2, O3 measurements
  3. Merges on hourly timestamp -> writes training CSV

Usage:
    python scripts/fetch_training_data.py --start 2024-01-01 --end 2026-07-25
    python scripts/fetch_training_data.py --years 2
    python scripts/fetch_training_data.py --provider openaq --years 1

Output: data/backfill/training_data_YYYY-MM-DD_YYYY-MM-DD.csv
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Load .env file so this script works standalone
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import requests

# --- Config ---

LAT = float(os.getenv("LATITUDE", "25.396"))
LON = float(os.getenv("LONGITUDE", "68.357"))
TZ = os.getenv("TIMEZONE", "Asia/Karachi")
OPENAQ_KEY = os.getenv("OPENAQ_API_KEY") or os.getenv("OPEN_API_KEY", "")
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")
AQICN_STATION = os.getenv("AQICN_STATION", "A546205")
OPENAQ_LOCATION_ID = int(os.getenv("OPENAQ_LOCATION_ID", "4889110"))
OUTPUT_DIR = Path("data/backfill")

OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENAQ_BASE = "https://api.openaq.org/v3"


def fetch_openmeteo(start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch historical weather from Open-Meteo Archive API."""
    print(f"\n[Open-Meteo] Fetching weather: {start} -> {end}")
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": (
            "temperature_2m,relative_humidity_2m,dew_point_2m,"
            "pressure_msl,wind_speed_10m,wind_direction_10m,"
            "precipitation,cloud_cover"
        ),
        "timezone": TZ,
    }
    try:
        resp = requests.get(OPENMETEO_ARCHIVE, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            print("   No hourly data in response")
            return None

        df = pd.DataFrame({"timestamp": pd.to_datetime(times)})
        for key, values in hourly.items():
            if key != "time":
                df[key] = values

        df["timestamp"] = df["timestamp"].dt.tz_localize(TZ)
        print(f"   OK: {len(df)} hourly rows, {len(df.columns)-1} weather variables")
        return df
    except Exception as e:
        print(f"   FAILED: {e}")
        return None


def discover_openaq_sensors(location_id: int) -> dict:
    """Discover sensor IDs from OpenAQ location endpoint."""
    print(f"\n[OpenAQ] Discovering sensors for location {location_id}")
    headers = {"X-API-Key": OPENAQ_KEY}
    try:
        resp = requests.get(
            f"{OPENAQ_BASE}/locations/{location_id}",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            print("   No location results")
            return {}

        sensors = results[0].get("sensors", [])
        sensor_map = {}
        for s in sensors:
            param = s.get("parameter", {})
            name = param.get("name", "").lower()
            sid = s.get("id")
            if name and sid:
                sensor_map[name] = int(sid)
                print(f"   {name} -> sensor_id={sid}")

        print(f"   Found {len(sensor_map)} sensors")
        return sensor_map
    except Exception as e:
        print(f"   FAILED: {e}")
        return {}


def fetch_openaq_sensor(
    sensor_id: int,
    param_name: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Fetch historical hourly measurements for one sensor."""
    headers = {"X-API-Key": OPENAQ_KEY}
    all_rows = []

    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    page_count = 0

    while current < end_dt:
        chunk_end = min(current + timedelta(days=90), end_dt)
        page = 1
        while True:
            try:
                resp = requests.get(
                    f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                    headers=headers,
                    params={
                        "datetime_from": f"{current.strftime('%Y-%m-%d')}T00:00:00",
                        "datetime_to": f"{chunk_end.strftime('%Y-%m-%d')}T23:59:59",
                        "limit": 1000,
                        "page": page,
                        "sort": "asc",
                    },
                    timeout=60,
                )

                if resp.status_code == 429:
                    print(f"   Rate limited. Waiting 60s...")
                    time.sleep(60)
                    continue

                if resp.status_code != 200:
                    print(f"   HTTP {resp.status_code} on page {page}")
                    break

                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break

                for r in results:
                    period = r.get("period", {})
                    dt_from = period.get("datetimeFrom", {})
                    ts = dt_from.get("local") or dt_from.get("utc")
                    if ts is None:
                        continue
                    try:
                        ts = pd.Timestamp(ts).floor("h")
                    except Exception:
                        continue

                    all_rows.append({
                        "timestamp": ts,
                        "parameter": param_name,
                        "value": r.get("value"),
                    })

                found = data.get("meta", {}).get("found", 0)
                if isinstance(found, str):
                    found = int(found.lstrip(">"))
                if page * 1000 >= found:
                    break
                page += 1
                page_count += 1
                if page_count % 10 == 0:
                    print(f"   {param_name}: {len(all_rows)} records so far...")
                time.sleep(0.3)

            except Exception as e:
                print(f"   ERROR: {e}")
                time.sleep(5)
                break

        current = chunk_end + timedelta(days=1)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df = df.pivot_table(
        index="timestamp",
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()

    rename_map = {"pm25": "pm2_5"}
    df.rename(columns=rename_map, inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip("_") for col in df.columns]

    print(f"   OK {param_name}: {len(df)} hourly rows")
    return df


def fetch_openaq_all(start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch all available parameters from OpenAQ."""
    if not OPENAQ_KEY:
        print("\nWARNING: OPENAQ_API_KEY not set -- skipping OpenAQ")
        return None

    sensor_map = discover_openaq_sensors(OPENAQ_LOCATION_ID)
    if not sensor_map:
        return None

    desired_params = ["pm25", "pm10", "no2", "o3", "so2", "co"]
    param_dfs = []

    for param_name in desired_params:
        if param_name not in sensor_map:
            print(f"   WARNING: {param_name} not available at this station")
            continue

        sensor_id = sensor_map[param_name]
        print(f"\n[OpenAQ] Fetching {param_name} from sensor {sensor_id}: {start} -> {end}")
        df = fetch_openaq_sensor(sensor_id, param_name, start, end)
        if df is not None and not df.empty:
            param_dfs.append(df)
        time.sleep(1)

    if not param_dfs:
        return None

    result = param_dfs[0]
    for df in param_dfs[1:]:
        result = pd.merge(result, df, on="timestamp", how="outer")

    if "pm2_5" in result.columns:
        result["aqi"] = result["pm2_5"].apply(
            lambda x: round((x / 35.4) * 100, 1) if pd.notna(x) else None
        )

    result["source"] = "openaq_api"
    result["station_id"] = OPENAQ_LOCATION_ID
    result = result.sort_values("timestamp").reset_index(drop=True)

    print(f"\n[OpenAQ] Total: {len(result)} hourly rows with observed labels")
    return result


def fetch_aqicn_latest() -> Optional[pd.DataFrame]:
    """Pull latest AQICN reading — live data only, no historical API."""
    if not AQICN_TOKEN:
        print("\nWARNING: AQICN_TOKEN not set -- skipping AQICN")
        return None

    print(f"\n[AQICN] Fetching latest reading for station {AQICN_STATION}")
    try:
        url = f"https://api.waqi.info/feed/{AQICN_STATION}/"
        resp = requests.get(url, params={"token": AQICN_TOKEN}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            print(f"   AQICN API error: {data.get('data')}")
            return None

        d = data.get("data", {})
        iaqi = d.get("iaqi", {})
        time_info = d.get("time", {})

        def _extract(obj):
            if isinstance(obj, dict):
                return obj.get("v")
            return obj

        ts_iso = time_info.get("iso") or time_info.get("s")
        ts = pd.Timestamp(ts_iso).floor("h") if ts_iso else None

        record = {
            "timestamp": ts,
            "aqi": d.get("aqi"),
            "pm2_5": _extract(iaqi.get("pm25")),
            "pm10": _extract(iaqi.get("pm10")),
            "no2": _extract(iaqi.get("no2")),
            "o3": _extract(iaqi.get("o3")),
            "so2": _extract(iaqi.get("so2")),
            "co": _extract(iaqi.get("co")),
            "station_name": AQICN_STATION,
            "source": "aqicn_live",
        }
        df = pd.DataFrame([record])
        print(f"   OK AQICN: AQI={record['aqi']}, PM2.5={record['pm2_5']}, PM10={record['pm10']}")
        return df
    except Exception as e:
        print(f"   FAILED: {e}")
        return None


def merge_and_save(
    weather_df: pd.DataFrame,
    openaq_df: Optional[pd.DataFrame],
    aqicn_df: Optional[pd.DataFrame],
    start: str,
    end: str,
) -> Path:
    """Merge weather + OpenAQ + AQICN and save to CSV."""
    print(f"\n[Merge] Combining weather + observed labels...")
    merged = weather_df.copy()

    # Merge OpenAQ first (primary source, has history)
    if openaq_df is not None and not openaq_df.empty:
        merged = pd.merge(merged, openaq_df, on="timestamp", how="left")
        label_count = merged["aqi"].notna().sum() if "aqi" in merged.columns else 0
        print(f"   OpenAQ: {label_count} rows with observed AQI labels")
    else:
        print("   OpenAQ: no data")

    # Merge AQICN as secondary (fills gaps in OpenAQ, adds PM10/NO2/O3/SO2/CO)
    if aqicn_df is not None and not aqicn_df.empty:
        # Append AQICN row if its timestamp isn't already covered
        if "timestamp" in aqicn_df.columns:
            aqicn_df["timestamp"] = pd.to_datetime(aqicn_df["timestamp"])
            merged = pd.merge(merged, aqicn_df, on="timestamp", how="left", suffixes=("", "_aqicn"))
            # Fill missing values from AQICN columns
            for col in ["aqi", "pm2_5", "pm10", "no2", "o3", "so2", "co"]:
                aqicn_col = f"{col}_aqicn"
                if aqicn_col in merged.columns and col in merged.columns:
                    merged[col] = merged[col].fillna(merged[aqicn_col])
                    merged.drop(columns=[aqicn_col], inplace=True)
                elif aqicn_col in merged.columns:
                    merged.rename(columns={aqicn_col: col}, inplace=True)
        label_count = merged["aqi"].notna().sum() if "aqi" in merged.columns else 0
        print(f"   +AQICN: {label_count} total rows with observed AQI labels")

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"training_data_{start}_{end}.csv"
    path = OUTPUT_DIR / filename
    merged.to_csv(path, index=False)
    print(f"\nSaved: {path} ({len(merged)} rows x {len(merged.columns)} cols)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="AeroTwinML -- Training Data Acquisition",
        epilog="""
Examples:
  python scripts/fetch_training_data.py --start 2024-01-01 --end 2026-07-25
  python scripts/fetch_training_data.py --years 2
  python scripts/fetch_training_data.py --provider openaq --years 1
        """,
    )
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--years", type=float, default=2, help="Years of data to fetch")
    parser.add_argument("--provider", choices=["all", "openmeteo", "openaq"], default="all")

    args = parser.parse_args()

    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=int(args.years * 365) + 1)).strftime("%Y-%m-%d")

    print("=" * 60)
    print("  AeroTwinML -- Training Data Acquisition")
    print(f"  Range: {start_date} -> {end_date}")
    print(f"  Location: {LAT}, {LON} | TZ: {TZ}")
    print("=" * 60)

    weather_df = None
    openaq_df = None
    aqicn_df = None

    if args.provider in ("all", "openmeteo"):
        weather_df = fetch_openmeteo(start_date, end_date)

    if args.provider in ("all", "openaq"):
        openaq_df = fetch_openaq_all(start_date, end_date)

    # Always pull AQICN for secondary pollutant coverage (PM10, NO2, O3, etc.)
    aqicn_df = fetch_aqicn_latest()

    if weather_df is not None:
        merge_and_save(weather_df, openaq_df, aqicn_df, start_date, end_date)
    elif openaq_df is not None:
        merge_and_save(openaq_df, None, aqicn_df, start_date, end_date)
    elif aqicn_df is not None:
        merge_and_save(aqicn_df, None, None, start_date, end_date)
    else:
        print("\nFAILED: No data fetched -- check API keys and network")


if __name__ == "__main__":
    main()
