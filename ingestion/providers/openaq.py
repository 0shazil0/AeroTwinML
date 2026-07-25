"""OpenAQ v3 provider — fetches historical and real-time observed air quality measurements.

OpenAQ provides free access to global air quality data from monitoring stations.
This provider replaces AQICN for historical backfill and serves as the primary
ground-truth label source.

API: https://docs.openaq.org/
Rate limits: 60 req/min, 2000 req/hour

For Hyderabad station 4889110, sensors typically include:
  - pm25 (PM2.5 in µg/m³)
  - pm10 (PM10 in µg/m³)
  - no2, o3, so2, co (various units)
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from ingestion.providers.base import BaseProvider
from utils.config import get
from utils.logging import get_logger
from utils.time_utils import floor_hour, now_local, utc_to_local

logger = get_logger(__name__)

OPENAQ_BASE = "https://api.openaq.org/v3"
REQUEST_DELAY = 0.3  # seconds between API calls to respect rate limits


class OpenAQProvider(BaseProvider):
    def __init__(self):
        super().__init__("openaq")
        self.api_key = os.getenv("OPENAQ_API_KEY") or ""
        self.location_id = int(os.getenv("OPENAQ_LOCATION_ID") or "4889110")
        self.city = get("city.name", "Hyderabad")
        self.timeout = get("providers.openaq.timeout_seconds", 30)
        self._sensor_map: Dict[str, int] = {}  # param_name → sensor_id, populated on first use

    def _headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key, "Accept": "application/json"}

    def _ensure_sensor_map(self) -> None:
        """Fetch location info to discover sensor IDs for each parameter."""
        if self._sensor_map:
            return

        self.logger.info("Discovering sensors for location %d", self.location_id)
        try:
            resp = requests.get(
                f"{OPENAQ_BASE}/locations/{self.location_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self.logger.error("Failed to fetch location %d: %s", self.location_id, e)
            return

        # The response structure: {"results": [{..., "sensors": [...]}]}
        results = data.get("results", [])
        sensors = []
        if isinstance(results, list) and len(results) > 0:
            loc = results[0]
            sensors = loc.get("sensors", [])
            self.logger.info("Location %d has %d sensors registered", self.location_id, len(sensors))

        # If not found, try top-level sensors key (some API versions)
        if not sensors:
            sensors = data.get("sensors", [])
            self.logger.info("Tried top-level sensors: %d found", len(sensors))

        for s in sensors:
            param = s.get("parameter", {})
            # Parameter can be a dict or a string depending on API version
            if isinstance(param, dict):
                param_name = param.get("name", param.get("id", "")).strip().lower()
            else:
                param_name = str(param).strip().lower()
            sensor_id = s.get("id")
            if param_name and sensor_id:
                self._sensor_map[param_name] = int(sensor_id)
                self.logger.info("  Found sensor: %s → id=%d", param_name, int(sensor_id))

        if not self._sensor_map:
            self.logger.warning("Auto-discovery failed — using hardcoded fallback sensors for Hyderabad")
            # Fallback: common sensor IDs for typical parameters at Hyderabad stations
            # These will be discovered automatically on first successful API call,
            # but if the location endpoint doesn't return sensors, we try direct lookup.
            self._sensor_map = self._discover_via_measurements()

    def _discover_via_measurements(self) -> Dict[str, int]:
        """Fallback: discover sensors by querying the /latest endpoint for this location."""
        fallback: Dict[str, int] = {}
        try:
            resp = requests.get(
                f"{OPENAQ_BASE}/locations/{self.location_id}/latest",
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            self.logger.info("Latest measurements returned %d results", len(results))
            for r in results:
                param = r.get("parameter", {})
                param_name = param.get("name", param.get("id", "")).strip().lower() if isinstance(param, dict) else str(param).strip().lower()
                sensor_id = r.get("sensorsId") or r.get("sensorId") or r.get("sensor_id")
                if param_name and sensor_id:
                    fallback[param_name] = int(sensor_id)
                    self.logger.info("  Fallback sensor: %s → id=%d", param_name, int(sensor_id))
        except Exception as e:
            self.logger.error("Fallback sensor discovery also failed: %s", e)
        return fallback

    def fetch_raw(self) -> Dict[str, Any]:
        """Fetch latest measurement for each parameter from each sensor."""
        self._ensure_sensor_map()

        results: Dict[str, Any] = {"measurements": {}, "fetched_at": datetime.now().isoformat()}

        for param_name, sensor_id in self._sensor_map.items():
            try:
                resp = requests.get(
                    f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                    headers=self._headers(),
                    params={
                        "limit": 1,
                        "sort": "desc",
                        "order_by": "datetimeFrom.utc",
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                recs = data.get("results", [])
                if recs:
                    results["measurements"][param_name] = recs[0]
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                self.logger.warning("Failed to fetch %s (sensor %d): %s", param_name, sensor_id, e)

        return results

    def normalize(self, raw: Dict[str, Any]) -> pd.DataFrame:
        """Convert OpenAQ measurements to a single-row DataFrame matching AQICN column names."""
        measurements = raw.get("measurements", {})

        record: Dict[str, Any] = {
            "timestamp": floor_hour(now_local()),
            "station_name": f"openaq_{self.location_id}",
            "city": self.city,
            "country": "PK",
            "source": "openaq",
        }

        # Parameter name → column name mapping
        param_col_map = {
            "pm25": "pm2_5",
            "pm10": "pm10",
            "no2": "no2",
            "o3": "o3",
            "so2": "so2",
            "co": "co",
        }

        for param_name, measurement in measurements.items():
            col = param_col_map.get(param_name, param_name)
            value = measurement.get("value")
            if value is not None:
                record[col] = float(value)

            # Extract timestamp from the measurement if available
            period = measurement.get("period", {})
            dt_to = period.get("datetimeTo", {})
            local_time = dt_to.get("local") or dt_to.get("utc")
            if local_time:
                try:
                    record["timestamp"] = floor_hour(
                        utc_to_local(pd.Timestamp(local_time).to_pydatetime())
                    )
                except Exception:
                    pass

        # Compute AQI from PM2.5 and PM10 (simplified — uses max of individual pollutant AQIs)
        if not any(k in record for k in ("aqi",)):
            record["aqi"] = self._estimate_aqi_from_pollutants(record)

        df = pd.DataFrame([record])
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate value ranges — same as AQICN provider."""
        if df.empty:
            return df

        checks = {
            "aqi": (0, 999),
            "pm2_5": (0, 1000),
            "pm10": (0, 1000),
            "no2": (0, 500),
            "o3": (0, 500),
            "so2": (0, 500),
            "co": (0, 200),
        }

        for col, (lo, hi) in checks.items():
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask & (df[col] < lo), col] = None
                df.loc[mask & (df[col] > hi), col] = None

        return df

    def fetch_historical(
        self,
        start_date: str,
        end_date: str,
        parameters: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch historical hourly measurements from OpenAQ for the given date range.

        This pulls data from the /hours endpoint for each sensor, which returns
        hourly averages — perfect for merging with Open-Meteo's hourly weather.

        Parameters:
            start_date: YYYY-MM-DD start (inclusive)
            end_date: YYYY-MM-DD end (inclusive)
            parameters: List of parameter names to fetch. If None, fetches all available.

        Returns:
            DataFrame with columns: timestamp, pm2_5, pm10, no2, o3, so2, co, aqi, source
        """
        self._ensure_sensor_map()

        if parameters is None:
            parameters = list(self._sensor_map.keys())

        all_records: List[Dict[str, Any]] = []
        total_calls = 0
        max_calls_per_hour = 1900  # Leave headroom under the 2000 limit

        for param_name in parameters:
            sensor_id = self._sensor_map.get(param_name)
            if not sensor_id:
                self.logger.warning("No sensor found for parameter: %s", param_name)
                continue

            # Fetch in 90-day chunks to manage response size and pagination
            current_start = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            self.logger.info(
                "Fetching historical %s from OpenAQ: %s → %s",
                param_name, start_date, end_date,
            )

            while current_start < end_dt and total_calls < max_calls_per_hour:
                chunk_end = min(current_start + timedelta(days=90), end_dt)
                chunk_start_str = current_start.strftime("%Y-%m-%d")
                chunk_end_str = chunk_end.strftime("%Y-%m-%d")

                page = 1
                while True and total_calls < max_calls_per_hour:
                    try:
                        resp = requests.get(
                            f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                            headers=self._headers(),
                            params={
                                "datetime_from": f"{chunk_start_str}T00:00:00+05:00",
                                "datetime_to": f"{chunk_end_str}T23:59:59+05:00",
                                "limit": 1000,
                                "page": page,
                                "sort": "asc",
                            },
                            timeout=self.timeout,
                        )
                        total_calls += 1
                        time.sleep(REQUEST_DELAY)

                        if resp.status_code == 429:
                            self.logger.warning("Rate limited. Waiting 60s...")
                            time.sleep(60)
                            continue

                        resp.raise_for_status()
                        data = resp.json()
                        results = data.get("results", [])

                        if not results:
                            break

                        for r in results:
                            period = r.get("period", {})
                            dt_from = period.get("datetimeFrom", {})
                            local_ts = dt_from.get("local") or dt_from.get("utc")
                            if not local_ts:
                                continue

                            try:
                                ts = floor_hour(
                                    utc_to_local(pd.Timestamp(local_ts).to_pydatetime())
                                )
                            except Exception:
                                continue

                            all_records.append({
                                "timestamp": ts,
                                "parameter": param_name,
                                "value": r.get("value"),
                            })

                        # Check if more pages
                        found = data.get("meta", {}).get("found", 0)
                        # OpenAQ returns ">1000" as a string for large results
                        if isinstance(found, str):
                            found = int(found.lstrip(">"))
                        if page * 1000 >= found:
                            break
                        page += 1

                    except requests.RequestException as e:
                        self.logger.error("OpenAQ fetch error: %s", e)
                        time.sleep(5)
                        break

                current_start = chunk_end + timedelta(days=1)

                # Progress report
                self.logger.info(
                    "  %s: %d records so far (%d API calls)",
                    param_name, len(all_records), total_calls,
                )

        if not all_records:
            self.logger.warning("No historical data fetched from OpenAQ")
            return pd.DataFrame()

        # Pivot: one row per timestamp, one column per parameter
        df = pd.DataFrame(all_records)
        pivoted = df.pivot_table(
            index="timestamp",
            columns="parameter",
            values="value",
            aggfunc="mean",
        ).reset_index()

        # Rename columns to match the canonical schema
        param_col_map = {
            "pm25": "pm2_5",
            "pm10": "pm10",
            "no2": "no2",
            "o3": "o3",
            "so2": "so2",
            "co": "co",
        }
        pivoted.rename(columns=param_col_map, inplace=True)

        # Estimate AQI from available pollutants
        pivoted["aqi"] = pivoted.apply(self._estimate_aqi_from_pollutants, axis=1)
        pivoted["source"] = "openaq"
        pivoted["station_name"] = f"openaq_{self.location_id}"

        pivoted = pivoted.sort_values("timestamp").reset_index(drop=True)

        self.logger.info(
            "OpenAQ historical fetch complete: %d hourly rows, %d columns",
            len(pivoted), len(pivoted.columns),
        )
        return pivoted

    def _estimate_aqi_from_pollutants(self, row) -> Optional[float]:
        """Estimate AQI from individual pollutants using EPA breakpoints.

        This is a simplified AQI computation. The actual AQI would require
        piecewise linear interpolation across breakpoint tables for each pollutant.
        Here we take the maximum of individual pollutant concentrations as a proxy.

        For a proper implementation, use the EPA AQI breakpoint tables.
        """
        # Simplified: return the max normalized pollutant value
        # PM2.5 breakpoint: 0-12=Good, 12.1-35.4=Moderate, 35.5-55.4=Unhealthy Sensitive, ...
        # This is a rough proxy — replace with full EPA calculation for production
        pm25 = row.get("pm2_5") if isinstance(row, dict) else getattr(row, "pm2_5", None)
        pm10 = row.get("pm10") if isinstance(row, dict) else getattr(row, "pm10", None)

        if pd.isna(pm25) and pd.isna(pm10):
            return None

        # Simple linear scaling as a fallback aqi estimate
        aqi_from_pm25 = (pm25 / 35.4) * 100 if pd.notna(pm25) else 0
        aqi_from_pm10 = (pm10 / 150) * 100 if pd.notna(pm10) else 0

        return round(max(aqi_from_pm25, aqi_from_pm10), 1)
