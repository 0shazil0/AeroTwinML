"""Central configuration loader — reads YAML + env overrides."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "settings.yaml"


def _load_yaml() -> Dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _override_from_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    env_map = {
        "AQICN_TOKEN": ("providers", "aqicn", "token"),
        "AQICN_STATION": ("providers", "aqicn", "station_id"),
        "LATITUDE": ("city", "latitude"),
        "LONGITUDE": ("city", "longitude"),
        "CITY_NAME": ("city", "name"),
        "TIMEZONE": ("city", "timezone"),
        "API_PORT": ("api", "port"),
        "DASHBOARD_PORT": ("dashboard", "port"),
        "MLFLOW_TRACKING_URI": ("model_registry", "mlflow_uri"),
        "REGISTRY_BACKEND": ("model_registry", "backend"),
        "HOPSWORKS_API_KEY": ("hopsworks", "api_key"),
        "HOPSWORKS_PROJECT": ("hopsworks", "project"),
        "HOPSWORKS_HOST": ("hopsworks", "host"),
    }
    for env_key, (section, *keys) in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            target = cfg.setdefault(section, {})
            for k in keys[:-1]:
                target = target.setdefault(k, {})
            target[keys[-1]] = val
    return cfg


_cfg = _override_from_env(_load_yaml())


def get(key_path: str, default: Any = None) -> Any:
    """Get config value by dot-separated path, e.g. 'city.latitude'."""
    keys = key_path.split(".")
    val = _cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


def all_config() -> Dict[str, Any]:
    return _cfg
