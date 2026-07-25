# Pearls AQI Predictor — Full Documentation & Guide

> **End-to-end MLOps system forecasting Air Quality Index in Hyderabad, Pakistan for the next 72 hours.**  
> Three-source architecture (Open-Meteo + OpenAQ + AQICN) · Serverless on GitHub Actions + Hopsworks · 29 tests passing · 7-page dark dashboard · Canvas Digital Twin

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Quick Start — 60-Second Setup](#2-quick-start--60-second-setup)
3. [Project Structure — Every File Explained](#3-project-structure--every-file-explained)
4. [Data Strategy — Deep Dive](#4-data-strategy--deep-dive)
5. [Ingestion Layer — Provider Architecture](#5-ingestion-layer--provider-architecture)
6. [Feature Engineering — Complete Reference](#6-feature-engineering--complete-reference)
7. [Model Training & Evaluation](#7-model-training--evaluation)
8. [MLflow Registry & Experiment Tracking](#8-mlflow-registry--experiment-tracking)
9. [SHAP Explainability — How It Works](#9-shap-explainability--how-it-works)
10. [Pipeline Automation — Hourly & Daily](#10-pipeline-automation--hourly--daily)
11. [FastAPI Backend — Full API Reference](#11-fastapi-backend--full-api-reference)
12. [Flask Dashboard — Every Page Documented](#12-flask-dashboard--every-page-documented)
13. [Digital Twin — Technical Deep Dive](#13-digital-twin--technical-deep-dive)
14. [Alerting System](#14-alerting-system)
15. [Docker Deployment Guide](#15-docker-deployment-guide)
16. [CI/CD with GitHub Actions](#16-cicd-with-github-actions)
17. [Testing Suite](#17-testing-suite)
18. [Configuration Reference](#18-configuration-reference)
19. [Developer Guide — Extending the System](#19-developer-guide--extending-the-system)
20. [Troubleshooting & FAQ](#20-troubleshooting--faq)

---

## 1. Overview & Architecture

### 1.1 What It Does

The Pearls AQI Predictor forecasts air quality 24, 48, and 72 hours ahead using weather data as input and observed station measurements as supervision. It's a complete MLOps pipeline — not just a notebook.

### 1.2 Core Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Three data sources** (Open-Meteo + OpenAQ + AQICN) | Open-Meteo provides future weather (genuinely unknown at prediction time). OpenAQ provides 2+ years of observed station AQI labels. AQICN provides live data as a secondary source. The model learns `weather → future AQI` instead of copying one forecast from another. |
| **Serverless architecture** | All pipelines run on GitHub Actions. Hopsworks provides managed Feature Store + Model Registry. The Flask dashboard deploys to Streamlit Cloud. Zero persistent servers to manage. |
| **Hopsworks Feature Store** | Features written to Hopsworks FS every hour — available for training, inference, and time-travel queries. Online store enables low-latency serving. |
| **Hopsworks Model Registry** | Trained models registered with full lineage: feature version, training metrics, artifacts. Inference loads latest version automatically. |
| **Regression + Classification** | Regression predicts numeric AQI. A lightweight classification layer maps it to health categories (Good → Hazardous). |
| **Walk-forward validation** | Time-series aware splits prevent data leakage. Random splits would cheat by training on future data. |
| **SHAP + correlation fallback** | TreeExplainer for tree models, correlation-based importance when SHAP isn't available. Explainability always works. |
| **Server-rendered Flask + Plotly.js** | Fast template rendering with interactive charts. No heavy SPA framework needed. |

### 1.3 Architecture Diagram (Serverless)

```
┌─────────────────────────────────────────────────────────────────────┐
│                       GitHub Actions CI/CD                           │
│         ┌───────────────────┐     ┌──────────────────┐              │
│         │  Hourly Pipeline  │     │  Daily Pipeline   │              │
│         │  (Every 60 min)   │     │  (2 AM daily)     │              │
│         │  · Ingest + Feat. │     │  · Backfill 2yr   │              │
│         │  · Infer + Alerts │     │  · Train + Register│              │
│         └────────┬──────────┘     └────────┬─────────┘              │
└──────────────────┼────────────────────────┼─────────────────────────┘
                   │                        │
                   ▼                        ▼
         ┌─────────────────────────────────────────────────┐
         │              Data Ingestion Layer                 │
         │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
         │  │Open-Meteo│  │  OpenAQ  │  │  AQICN   │       │
         │  │(weather) │  │(2yr hist)│  │(live)    │       │
         │  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
         │       └────────┬────┴──────────┬──┘              │
         │                ▼               ▼                  │
         │         ┌─────────────────────────┐              │
         │         │  Merge by Hour (label   │              │
         │         │  priority: OA > AQICN)  │              │
         │         └────────────┬────────────┘              │
         └──────────────────────┼───────────────────────────┘
                                │
                                ▼
         ┌─────────────────────────────────────────────────┐
         │            Feature Engineering Layer              │
         │  · Time (cyclical)       · Lags (1-72h)          │
         │  · Rolling stats         · Interactions          │
         │  · Weather passthrough   · Targets (t+24,48,72)  │
         └──────────────┬──────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
┌─────────────────────┐    ┌─────────────────────┐
│  Hopsworks           │    │  Hopsworks           │
│  Feature Store       │    │  Model Registry      │
│  · aqi_features      │    │  · aqi_forecaster    │
│  · training_metrics  │    │  · Version history   │
│  · Online serving    │    │  · Metrics tracking  │
└──────────┬───────────┘    └──────────┬──────────┘
           │                           │
           ▼                           ▼
  ┌─────────────────┐        ┌─────────────────┐
  │  Model Training  │        │   Inference      │
  │  · Persistence   │        │   Engine         │
  │  · Ridge         │        │   · Load from    │
  │  · Random Forest │        │     Hopsworks MR │
  │  · XGBoost       │        │   · Predict 24h, │
  │  · LightGBM      │        │     48h, 72h     │
  └─────────────────┘        └────────┬────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
               ┌──────────────────┐     ┌──────────────────┐
               │  FastAPI Backend │     │  Flask Dashboard  │
               │  (Port 8000)     │◄────│  (Streamlit Cloud)│
               │  10 REST routes  │     │  7 pages + DT     │
               └──────────────────┘     └──────────────────┘
```

### 1.4 What Runs Where

| Component | Location | How |
|-----------|----------|-----|
| **Hourly pipeline** | GitHub Actions | `python -m pipelines.hourly_pipeline` every 60 min |
| **Daily pipeline** | GitHub Actions | `python -m pipelines.daily_pipeline` at 2 AM |
| **Feature Store** | Hopsworks Cloud | `feature_store/hopsworks_client.py` |
| **Model Registry** | Hopsworks Cloud | `models/registry.py` → Hopsworks MR |
| **Inference** | GitHub Actions (hourly) | `models/inference.py` loads from Hopsworks |
| **Dashboard** | Streamlit Cloud | Flask app with Hopsworks API reads |
| **API** | GitHub Actions (optional) | FastAPI for local dev; replaced by Hopsworks FS reads at runtime |
     └──────────────┴──────────────┴───────────────┘
                        │
                  aqi-network
                        │
                 ┌──────┴──────┐
                 │    data/    │  (shared volume)
                 │  models/    │
                 │  mlruns/    │
                 └─────────────┘
```

---

## 2. Quick Start — 60-Second Setup

### Prerequisites

- **Python 3.10+** (for local dev) or **Docker + Docker Compose** (recommended)
- **AQICN API token** — free at https://aqicn.org/data-platform/token/

### Option A: Docker (Recommended — One Command)

```bash
git clone <repo-url> && cd aqi-predictor
cp .env.example .env
# Open .env in your editor and paste your AQICN_TOKEN
docker compose up --build
```

After startup (≈2 minutes for first build):

| Service | URL | Purpose |
|---------|-----|---------|
| **Dashboard** | http://localhost:5000 | All 7 UI pages |
| **API** | http://localhost:8000 | REST endpoints |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger |
| **API ReDoc** | http://localhost:8000/redoc | Alternative docs |
| **MLflow** | http://localhost:5001 | Experiment tracking |

### Option B: Local Development

```bash
# Windows
cd aqi-predictor
scripts\setup.bat

# Linux/macOS
bash scripts/setup.sh

# Edit .env with your AQICN_TOKEN

# Terminal 1 — Start API
python -m backend.main

# Terminal 2 — Start Dashboard
python -m dashboards.app

# Terminal 3 — Run one ingestion cycle
python -m pipelines.hourly_pipeline

# Terminal 4 — Run training (after some data exists)
python -m pipelines.daily_pipeline
```

### Verify Everything Works

```bash
# Check API health
curl http://localhost:8000/health

# Check forecast
curl http://localhost:8000/api/v1/aqi/forecast

# Run tests
python -m pytest tests/ -v
```

---

## 3. Project Structure — Every File Explained

```
aqi-predictor/
│
├── .env.example              # Template for secrets. Copy to .env and fill tokens.
├── .gitignore                # Excludes .env, data/, model artifacts, __pycache__
├── requirements.txt          # All Python dependencies (pinned ranges)
├── Dockerfile                # Multi-service base image (Python 3.11-slim)
├── docker-compose.yml        # 5 services: api, dashboard, trainer, mlflow, ingestion
├── requirements.txt          # Full deps for local dev (includes jupyter, xgboost, testing)
├── requirements-docker.txt   # Lean deps for Docker build (core-only, builds faster)
├── README.md                 # Quick-start and overview
│
├── .github/
│   └── workflows/
│       ├── hourly.yml        # GitHub Actions: runs every hour
│       └── daily.yml         # GitHub Actions: runs at 2 AM daily
│
├── configs/
│   └── settings.yaml         # Central configuration. All values overrideable via ENV.
│
├── scripts/
│   ├── setup.sh              # Bash setup: create .env, install deps, make dirs
│   └── setup.bat             # Windows setup script
│
├── utils/                    # Shared utilities (no business logic)
│   ├── config.py             # YAML config loader + env override system
│   ├── logging.py            # Structured logging setup
│   ├── time_utils.py         # Timezone handling (all → Asia/Karachi)
│   ├── storage.py            # Read/write JSON, Parquet, CSV helpers
│   └── aqi_utils.py          # AQI classification, colors, alerts
│
├── ingestion/                # Data ingestion layer
│   ├── providers/
│   │   ├── base.py           # Abstract BaseProvider: fetch → normalize → validate
│   │   ├── openmeteo.py      # OpenMeteoProvider: air quality + weather API client
│   │   └── aqicn.py          # AQICNProvider: station feed API client
│   ├── orchestrator.py       # IngestionOrchestrator: runs all providers, merges data
│   └── fetch.py              # CLI: python -m ingestion.fetch [--backfill]
│
├── feature_store/
│   └── feature_builder.py    # FeatureBuilder: 50+ features across 7 groups
│
├── models/                   # ML lifecycle
│   ├── trainer.py            # Model training: baselines, sklearn, XGBoost, LightGBM
│   ├── registry.py           # MLflow integration: log runs, register models
│   ├── inference.py          # InferenceEngine: load model, predict, classify
│   └── explainer.py          # SHAP TreeExplainer + LIME + correlation fallback
│
├── pipelines/                # Orchestration
│   ├── hourly_pipeline.py    # 5-step: ingest → features → infer → alerts → quality
│   └── daily_pipeline.py     # 5-step: backfill → features → train → register → save
│
├── backend/                  # FastAPI REST API
│   ├── main.py               # FastAPI app: lifespan, CORS, route registration
│   ├── schemas.py            # Pydantic models for all request/response types
│   └── routes/
│       ├── aqi.py            # GET /aqi/current, /forecast, /history, /pollutants
│       ├── alerts.py         # GET /alerts, /alerts/thresholds
│       ├── explain.py        # GET /explain/latest, /explain/feature-importance
│       ├── pipeline.py       # GET /pipeline/status
│       └── sources.py        # GET /data-sources
│
├── dashboards/               # Flask server-rendered UI
│   ├── app.py                # Flask app: 8 routes, proxy to FastAPI backend
│   ├── static/
│   │   └── css/
│   │       └── style.css     # Full dark theme design system (500+ lines)
│   └── templates/
│       ├── base.html         # Base layout: nav, sidebar, alert banner
│       ├── index.html        # Dashboard: hero AQI, weather, stats, alerts
│       ├── forecast.html     # Forecast: 72h timeline, charts, pollutant bars
│       ├── analytics.html    # Analytics: trends, diurnal, scatter, correlations
│       ├── explainability.html # SHAP bars, waterfall, natural language
│       ├── pipeline.html     # Pipeline: live health, step-by-step, freshness
│       ├── data_sources.html # Data sources: provider cards, merge viz
│       ├── digital_twin.html # Digital Twin: Canvas 2.5D + time slider + particles
│       └── settings.html     # Settings: thresholds, endpoints, config
│
├── notebooks/
│   └── 01_eda.ipynb          # EDA: trends, decomposition, correlations, missingness
│
├── tests/                    # 29 tests, all passing
│   ├── test_aqi_utils.py     # Classification, alerts, colors (9 tests)
│   ├── test_time_utils.py    # Timezone, floor_hour (4 tests)
│   ├── test_features.py      # FeatureBuilder output (4 tests)
│   ├── test_models.py        # Baselines, metrics, walk-forward (5 tests)
│   └── test_api.py           # API endpoints (7 tests)
│
├── docs/
│   ├── report.md             # Technical report (standalone)
│   └── full-guide.md         # This file — complete documentation
│
└── data/                     # Gitignored — generated at runtime
    ├── raw/open_meteo/       # Raw JSON responses
    ├── raw/aqicn/            # Raw JSON responses
    ├── raw/logs/             # alerts.jsonl
    ├── processed/             # Merged, cleaned, features, predictions
    └── backfill/              # Training datasets
```

---

## 4. Data Strategy — Deep Dive

### 4.1 The Two-Source Design

```
┌──────────────────────┐          ┌──────────────────────┐
│     Open-Meteo        │          │       AQICN           │
│  (free, no API key)   │          │  (free token needed)  │
├──────────────────────┤          ├──────────────────────┤
│ ROLE: Features        │          │ ROLE: Labels          │
│                       │          │                       │
│ What it provides:     │          │ What it provides:     │
│ • temperature_2m      │          │ • aqi (observed)      │
│ • relative_humidity   │          │ • pm2_5 (observed)    │
│ • dew_point           │          │ • pm10 (observed)     │
│ • pressure_msl        │          │ • no2, o3, so2, co    │
│ • wind_speed_10m      │          │                       │
│ • wind_direction_10m  │          │ Station: A546205      │
│ • precipitation       │          │ City: Hyderabad, PK   │
│ • cloud_cover         │          │                       │
│ • forecast AQI (4d)   │          │                       │
├──────────────────────┤          ├──────────────────────┤
│ Used for:             │          │ Used for:             │
│ Training features ✓   │          │ Training labels ✓     │
│ Inference features ✓  │          │ Evaluation targets ✓  │
└──────────────────────┘          └──────────────────────┘
```

### 4.2 Why This Separation Matters

**Wrong approach:** Train on AQI forecast from one API, evaluate against another API's forecast. You'd just be learning the relationship between two forecasts — not predicting reality.

**Correct approach:** Open-Meteo provides weather (what will happen in the atmosphere). AQICN provides actual measured AQI (what actually happened). The model learns:

```
weather_conditions(t) → actual_air_quality(t + N)
```

### 4.3 Merge Logic (Implementation Detail)

```python
# From ingestion/orchestrator.py
def merge(self, om_df, aq_df):
    # Merge on hourly timestamp in Asia/Karachi
    # Open-Meteo columns → weather features
    # AQICN columns → observed labels
    # Join: outer (keep all timestamps)
    # Deduplicate columns
    merged = pd.merge(om_sub, aq_sub, on="timestamp", how="outer")
```

Key rule: **Merge on datetime only, never on provider IDs.**

### 4.4 Fallback Strategy

```
1. Try AQICN observed station data for labels
2. Use Open-Meteo weather regardless (always available)
3. If AQICN is missing for a row → skip from supervised training
4. Continue inference with latest valid Open-Meteo features
```

### 4.5 Training Row Construction

At time `t`:

| Column | Source | Value |
|--------|--------|-------|
| `temperature_2m` | Open-Meteo | Weather at time t |
| `relative_humidity_2m` | Open-Meteo | Weather at time t |
| ... (all weather features) | Open-Meteo | |
| `aqi_lag_1` | Computed | AQI at t-1 |
| `aqi_lag_24` | Computed | AQI at t-24 |
| ... (all lag features) | Computed | |
| **`target_aqi_24h`** | AQICN shifted | AQI at t+24 |
| **`target_aqi_48h`** | AQICN shifted | AQI at t+48 |
| **`target_aqi_72h`** | AQICN shifted | AQI at t+72 |

---

## 5. Ingestion Layer — Provider Architecture

### 5.1 Abstract Base Class

```python
# ingestion/providers/base.py
class BaseProvider(ABC):
    def __init__(self, name: str):
        self.name = name
        self.raw_dir = DATA_DIR / "raw" / name

    @abstractmethod
    def fetch_raw(self) -> Dict[str, Any]: ...

    @abstractmethod
    def normalize(self, raw: Dict) -> pd.DataFrame: ...

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def run(self) -> pd.DataFrame:
        """Full pipeline: fetch → normalize → validate → save raw."""
        raw = self.fetch_raw()
        self._save_raw(raw)
        df = self.normalize(raw)
        df = self.validate(df)
        return df
```

Every provider must implement three methods. Adding a new provider means creating one new file that extends `BaseProvider`.

### 5.2 Open-Meteo Provider

**Endpoints hit:**
- `https://air-quality-api.open-meteo.com/v1/air-quality` → current + hourly AQI, PM2.5, PM10, O₃, NO₂, SO₂, CO
- `https://api.open-meteo.com/v1/forecast` → current + hourly temperature, humidity, pressure, wind, precipitation, cloud

**Rate limiting:** 300ms delay between calls (free tier allows ~600 req/min).

**Validation rules:**
| Column | Min | Max |
|--------|-----|-----|
| relative_humidity_2m | 0 | 100 |
| wind_speed_10m | 0 | — |
| precipitation | 0 | — |
| cloud_cover | 0 | 100 |
| pm2_5 | 0 | — |
| pm10 | 0 | — |
| dew_point_2m | -50 | 60 |
| pressure_msl | 800 | 1100 |

**Historical backfill:**
```python
provider.fetch_historical("2024-01-01", "2026-07-23")
```
Fetches both air quality and weather history in one call per endpoint.

### 5.3 AQICN Provider

**Endpoint:** `https://api.waqi.info/feed/A546205/?token=YOUR_TOKEN`

**Response structure parsed:**
```json
{
  "status": "ok",
  "data": {
    "aqi": 85,
    "dominentpol": "pm25",
    "iaqi": {
      "pm25": {"v": 42.5},
      "pm10": {"v": 65.2},
      "no2": {"v": 12.3}
    },
    "time": {"iso": "2026-07-23T14:00:00+05:00"},
    "city": {"name": "Hyderabad", "country": "PK"}
  }
}
```

**The `iaqi` field** can be either a dict `{"v": value}` or a direct float. The provider handles both:
```python
pm2_5 = iaqi.get("pm25", {}).get("v") if isinstance(iaqi.get("pm25"), dict) else iaqi.get("pm25")
```

**Time parsing:** Tries ISO format first, then string format, falls back to current local time.

### 5.4 Running Ingestion Manually

```bash
# Single fetch cycle
python -m ingestion.fetch --fetch

# Historical backfill
python -m ingestion.fetch --backfill --start 2024-01-01 --end 2026-07-23
```

---

## 6. Feature Engineering — Complete Reference

### 6.1 FeatureBuilder Class

```python
from feature_store.feature_builder import FeatureBuilder

builder = FeatureBuilder(df)       # df must have 'timestamp' column
featured = builder.build_all()     # returns DataFrame with all features
train = builder.get_training_data() # drops rows with NaN targets
```

### 6.2 All Feature Groups

#### A. Time Features (10 columns)
| Column | Type | Description |
|--------|------|-------------|
| `hour` | int 0-23 | Hour of day |
| `day` | int 1-31 | Day of month |
| `day_of_week` | int 0-6 | Monday=0, Sunday=6 |
| `month` | int 1-12 | Month number |
| `weekend` | int 0/1 | 1 if Saturday or Sunday |
| `season` | int 0-3 | Dec-Feb=0, Mar-May=1, Jun-Aug=2, Sep-Nov=3 |
| `hour_sin` | float [-1,1] | sin(2π × hour / 24) |
| `hour_cos` | float [-1,1] | cos(2π × hour / 24) |
| `month_sin` | float [-1,1] | sin(2π × month / 12) |
| `month_cos` | float [-1,1] | cos(2π × month / 12) |

Cyclical encodings preserve the circular nature of time — 23:00 and 00:00 are adjacent, and December (12) wraps to January (1).

#### B. Lag Features (12 columns)
| Column | Shift |
|--------|-------|
| `aqi_lag_1` | t-1 hour |
| `aqi_lag_6` | t-6 hours |
| `aqi_lag_24` | t-24 hours (yesterday same hour) |
| `aqi_lag_72` | t-72 hours (3 days ago) |
| `pm25_lag_1` | t-1 hour |
| `pm25_lag_24` | t-24 hours |
| `pm10_lag_1` | t-1 hour |
| `pm10_lag_24` | t-24 hours |

**Priority:** Uses AQICN `aqi` column if available, falls back to Open-Meteo's `us_aqi`.

#### C. Rolling Statistics (8 columns)
| Column | Window | Statistic |
|--------|--------|-----------|
| `aqi_roll_mean_6` | 6 hours | Mean |
| `aqi_roll_mean_24` | 24 hours | Mean |
| `aqi_roll_std_24` | 24 hours | Standard deviation |
| `aqi_roll_min_24` | 24 hours | Minimum |
| `aqi_roll_max_24` | 24 hours | Maximum |

#### D. Weather Features (8 columns, passthrough)
`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `pressure_msl`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`, `cloud_cover`

Missing values filled with forward-fill then backward-fill.

#### E. Interaction Features (4 columns)
| Column | Formula |
|--------|---------|
| `humidity_x_temperature` | humidity × temperature |
| `wind_x_pm25` | wind_speed × PM2.5 |
| `rain_x_pm10` | precipitation × PM10 |
| `aqi_change_rate` | ΔAQI / 6h (slope over last 6 hours) |

#### F. Target Columns (3-5 columns)
| Column | Meaning |
|--------|---------|
| `target_aqi_24h` | AQI at t+24 (shifted -24) |
| `target_aqi_48h` | AQI at t+48 |
| `target_aqi_72h` | AQI at t+72 |
| `target_pm25_24h` | PM2.5 at t+24 |
| `target_pm10_24h` | PM10 at t+24 |

#### G. Classification Labels
For each `target_aqi_Xh`, a corresponding `target_aqi_Xh_category` column with values:
`Good`, `Moderate`, `Unhealthy for Sensitive Groups`, `Unhealthy`, `Very Unhealthy`, `Hazardous`

---

## 7. Model Training & Evaluation

### 7.1 Model Catalog

#### Baseline Models (must-beat bar)

| Model | Class | What It Does |
|-------|-------|-------------|
| **Persistence** | `PersistenceModel` | Predicts last known AQI for all horizons. If AQI was 85, predicts 85 forever. |
| **Seasonal Naive** | `SeasonalNaiveModel` | Predicts the value from exactly 24 hours ago. Captures daily rhythm. |

```python
# Usage
model = PersistenceModel()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# Every prediction = last value in y_train
```

#### ML Models (scikit-learn wrapped)

| Model | Class | Hyperparameters |
|-------|-------|----------------|
| **Ridge Regression** | `SklearnWrapper("ridge", Ridge(alpha=1.0))` | L2 penalty |
| **Random Forest** | `SklearnWrapper("rf", RandomForestRegressor(n=100, depth=10))` | Ensemble of 100 trees |
| **Gradient Boosting** | `SklearnWrapper("gb", GradientBoostingRegressor(n=100, depth=5))` | Sequential boosting |
| **XGBoost** | `SklearnWrapper("xgb", XGBRegressor(n=100, depth=6, lr=0.1))` | If installed |
| **LightGBM** | `SklearnWrapper("lgb", LGBMRegressor(n=100, depth=6, lr=0.1))` | If installed |

XGBoost and LightGBM are optional — the system gracefully skips them if not installed.

### 7.2 Walk-Forward Validation

```python
def walk_forward_split(df, feature_cols, target_col, train_size=0.7, step=24):
    """
    Split such that training always precedes testing.
    
    Example with 100 rows, train_size=0.7, step=24:
      Split 1: train=[0:70],  test=[70:94]
      Split 2: train=[0:94],  test=[94:100]  (truncated to available data)
    
    Returns list of (train_df, test_df) tuples.
    """
```

Why not random split? In time series, `t+1` is correlated with `t`. Random splitting puts future data in training and past data in testing — the model learns patterns it shouldn't know. Walk-forward simulates real forecasting: train on the past, test on the future.

### 7.3 Metrics

For each model × horizon combination:

```python
metrics = {
    "rmse_24h": 12.5,   # Root Mean Squared Error — primary metric
    "mae_24h": 9.8,     # Mean Absolute Error
    "r2_24h": 0.78,     # R² (1.0 = perfect, 0.0 = mean baseline)
}
```

The best model is selected by **lowest RMSE at 24h horizon**.

### 7.4 Training Entry Point

```python
# From models/trainer.py
from models.trainer import build_models_for_horizons, find_best_model

feature_cols = ['hour_sin', 'hour_cos', 'aqi_lag_1', ...]
target_cols = {
    "24h": "target_aqi_24h",
    "48h": "target_aqi_48h",
    "72h": "target_aqi_72h",
}

results = build_models_for_horizons(feature_cols, target_cols, train_df, test_df)
best_name, best_horizon, best_model, all_metrics = find_best_model(results, "rmse_24h")
```

---

## 8. Hopsworks Model Registry & Feature Store

### 8.1 What Runs on Hopsworks

| Service | Hopsworks Entity | What's Stored | Purpose |
|---------|-----------------|---------------|---------|
| **Feature Store** | `aqi_features` FG | Hourly features (time, lags, rolling, weather, interactions) | Training + online inference |
| **Feature Store** | `aqi_training_features` FG | Daily training dataset with targets | Model training |
| **Feature Store** | `training_metrics` FG | RMSE, MAE, R² per run | Experiment tracking |
| **Model Registry** | `aqi_forecaster` | Trained sklearn/xgboost models with version history | Inference loading |

### 8.2 Feature Store Flow

```python
# feature_store/hopsworks_client.py — used by both pipelines

# 1. Hourly: write live features
write_feature_group("aqi_features", featured_df, version=1, online_enabled=True)

# 2. Daily: write training features
write_feature_group("aqi_training_features", featured_df, version=1)

# 3. Read for training or inference
df = read_feature_group("aqi_features", version=1, online=False)
```

### 8.3 Model Registry Flow

```python
# models/registry.py

# 1. Train → Register to Hopsworks MR
log_experiment(results, feature_cols, target_cols)   # Auto-registers best model
# → Hopsworks MR: aqi_forecaster v1, v2, v3...

# 2. Inference → Load latest from Hopsworks
model = get_latest_model("aqi_forecaster")
# Tries: Hopsworks MR → MLflow (fallback) → local pickle (last resort)
```

### 8.4 Metrics Tracking

Training run metrics are stored as a feature group:
```python
log_metrics_to_store({"rmse_24h": 12.5, "mae_24h": 9.3}, run_name="train_20260723_0200")
# → Feature group: training_metrics v1
```

### 8.5 Fallback Behavior (Graceful Degradation)

If Hopsworks API key is missing or unreachable:
1. Features saved locally to `data/processed/features/`
2. Models saved locally to `models/artifacts/`
3. MLflow becomes the secondary fallback (if running)
4. System continues operating fully — just without managed FS/MR

This means local dev works identically with or without Hopsworks credentials.

---

## 9. SHAP Explainability — How It Works

### 9.1 Three-Tier System

```
Tier 1: SHAP TreeExplainer     ← XGBoost, LightGBM, Random Forest
    ↓ (if SHAP unavailable)
Tier 2: LIME                   ← Tabular explainer on training data
    ↓ (if LIME unavailable)
Tier 3: Correlation fallback   ← Pearson correlation from historical data
```

### 9.2 SHAP Implementation

```python
# models/explainer.py
class ModelExplainer:
    def fit_shap(self, X_background):
        if hasattr(self.model, 'estimators_'):
            self.shap_explainer = shap.TreeExplainer(self.model)  # Fast!
        else:
            self.shap_explainer = shap.KernelExplainer(...)       # Slower fallback
    
    def explain(self, X):
        shap_vals = self.shap_explainer.shap_values(X)
        # Returns per-feature contribution values
```

### 9.3 What the API Returns

```json
{
  "top_drivers": [
    {"feature": "aqi_lag_1", "shap_value": 12.5, "direction": "positive"},
    {"feature": "wind_speed_10m", "shap_value": -8.3, "direction": "negative"},
    {"feature": "relative_humidity_2m", "shap_value": 5.1, "direction": "positive"}
  ],
  "natural_language": "AQI is predicted to rise due to aqi lag 1, relative humidity 2m. AQI improvement is driven by wind speed 10m.",
  "method": "shap"
}
```

### 9.4 Natural Language Generation

Template-based approach:
- Positive SHAP drivers → "AQI is predicted to rise due to [features]."
- Negative SHAP drivers → "AQI improvement is driven by [features]."
- If balanced → Describes the strongest single factor.

---

## 10. Pipeline Automation — Hourly & Daily

### 10.1 Hourly Pipeline (5 Steps)

```
python -m pipelines.hourly_pipeline
```

```
START
  │
  ▼
[Step 1: Ingestion]
  Fetch Open-Meteo (air quality + weather)
  Fetch AQICN (station data)
  Validate ranges, normalize timestamps
  Merge on hourly timestamp → merged DataFrame
  │
  ▼
[Step 2: Feature Engineering]
  Build 50+ features: time, lag, rolling, weather, interactions
  Save feature table as Parquet
  │
  ▼
[Step 3: Inference]
  Load best model from MLflow (or local fallback)
  Predict AQI at 24h, 48h, 72h
  Classify predictions (Good...Hazardous)
  Save forecast to forecast_latest.json
  │
  ▼
[Step 4: Alerts Check]
  If current AQI ≥ 200 → create alert
  If any forecast ≥ 200 → create alert
  Append alerts to alerts.jsonl
  │
  ▼
[Step 5: Quality Check]
  Check for empty dataset
  Check missing critical columns (aqi, pm2_5, pm10)
  Save quality report
  │
  ▼
END (write pipeline_status.json)
```

**Runtime:** ~5-10 seconds per run (API calls dominate).

### 10.2 Daily Pipeline (5 Steps)

```
python -m pipelines.daily_pipeline
```

```
START
  │
  ▼
[Step 1: Backfill]
  Fetch last 30 days of historical Open-Meteo data
  (AQICN historical data used where available)
  Save to data/backfill/train_v1/
  │
  ▼
[Step 2: Build Training Dataset]
  Load merged data
  Build all features
  Generate supervised targets
  Filter to rows with valid labels
  │
  ▼
[Step 3: Train Models]
  Train: Persistence, Seasonal Naive, Ridge, RF, GB, XGBoost, LightGBM
  For each: train on 80%, evaluate on 20% walk-forward
  Compare RMSE, MAE, R² across all horizons
  │
  ▼
[Step 4: MLflow Registration]
  Log experiment: params, metrics, artifacts
  Register best model (lowest RMSE at 24h)
  │
  ▼
[Step 5: Save Artifacts]
  Save best model locally (fallback for when MLflow is down)
  Write daily_status.json
  │
  ▼
END
```

**Runtime:** ~1-3 minutes (depends on data volume and models installed).

### 10.3 Status Output Format

```json
{
  "pipeline": "hourly",
  "started_at": "2026-07-23T14:00:00+05:00",
  "completed_at": "2026-07-23T14:00:12+05:00",
  "success": true,
  "steps": {
    "ingestion": {"status": "ok", "rows_fetched": 96},
    "features": {"status": "ok", "features_generated": 52, "rows": 96},
    "inference": {"status": "ok", "current_aqi": 85.3, "forecast_24h": 92.1},
    "alerts": {"status": "ok", "alerts": []},
    "quality": {"status": "ok", "issues": 0}
  }
}
```

---

## 11. FastAPI Backend — Full API Reference

### 11.1 Base URL

```
http://localhost:8000
```

### 11.2 Endpoints

#### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check. Returns `{"status": "healthy", "version": "1.0.0"}` |
| `GET` | `/api/v1/status` | Full system status including current AQI and pipeline health |

#### AQI Endpoints

| Method | Path | Params | Description |
|--------|------|--------|-------------|
| `GET` | `/api/v1/aqi/current` | — | Current AQI, category, health advice, weather snapshot, PM2.5/PM10 |
| `GET` | `/api/v1/aqi/forecast` | — | 72h forecast: 24h, 48h, 72h predictions with categories and alerts |
| `GET` | `/api/v1/aqi/history` | `hours` (1-720, default 168) | Historical AQI, PM, temperature, humidity, wind |
| `GET` | `/api/v1/aqi/pollutants` | — | Current PM2.5, PM10, NO₂, O₃, SO₂, CO, dominant pollutant |

#### Alerts

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/alerts` | Recent alerts. Query: `?limit=20` |
| `GET` | `/api/v1/alerts/thresholds` | AQI category definitions and alert trigger |

#### Explainability

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/explain/latest` | Top SHAP drivers + natural language for latest prediction |
| `GET` | `/api/v1/explain/feature-importance` | Global feature importance across all predictions |

#### Pipeline & Data

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/pipeline/status` | Hourly + daily pipeline status, data freshness |
| `GET` | `/api/v1/data-sources` | Provider roles, merge strategy, fallback logic |

### 11.3 Response Format

All endpoints return:
```json
{
  "status": "ok",
  "data": { ... },
  "meta": { "city": "Hyderabad", ... },
  "timestamp": "2026-07-23T14:00:00"
}
```

When data is unavailable:
```json
{
  "status": "no_data",
  "data": {},
  "meta": {},
  "timestamp": "..."
}
```

### 11.4 Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs — Try every endpoint directly
- **ReDoc:** http://localhost:8000/redoc — Clean reference docs

---

## 12. Flask Dashboard — Every Page Documented

### 12.1 Architecture

The dashboard is a Flask app that proxies all data from the FastAPI backend:

```
Browser ──► Flask (:5000) ──► FastAPI (:8000) ──► Data files
```

Flask routes call `api_get("/api/v1/...")` which makes HTTP requests to the backend, then renders Jinja2 templates with the data.

### 12.2 Page-by-Page Reference

#### 1. Home Dashboard (`/`)

**Layout:**
```
┌──────────────────────────────────────────────────┐
│  ┌─────────────────┐   ┌──────────────────────┐  │
│  │   HERO AQI       │   │  Current Weather     │  │
│  │      85          │   │  Temp  | Humidity    │  │
│  │   Moderate       │   │  Wind  | Pressure    │  │
│  │   "Air quality.."│   │                      │  │
│  └─────────────────┘   └──────────────────────┘  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌────────────────┐  │
│  │PM2.5 │ │PM10  │ │ Dom. │ │ Last Updated   │  │
│  │ 42.5 │ │ 65.2 │ │ PM25 │ │ 14:00 PKT      │  │
│  └──────┘ └──────┘ └──────┘ └────────────────┘  │
│  ┌─────────────────┐   ┌──────────────────────┐  │
│  │ Forecast Preview │   │  Recent Alerts       │  │
│  │ +24h: 92   Mod. │   │  (none or list)      │  │
│  │ +48h: 105  Sens.│   │                      │  │
│  │ +72h: 78   Mod. │   │                      │  │
│  └─────────────────┘   └──────────────────────┘  │
└──────────────────────────────────────────────────┘
```

**Special behavior:** AQI number animates counting up from 0 on page load. The hero section glows green (safe) or red (danger) based on AQI.

#### 2. Forecast (`/forecast`)

- **3 forecast cards** — Large numbers for +24h, +48h, +72h with category badges
- **Forecast trend chart** — Plotly line chart with 3-hour intervals, alert threshold line at 200
- **Pollutant breakdown** — Horizontal bar chart for PM2.5, PM10, NO₂, O₃, SO₂, CO
- **Historical context chart** — 7-day AQI trend for comparison

#### 3. Analytics (`/analytics`)

- **Timeline** — Full historical AQI line chart
- **Diurnal pattern** — Bar chart: average AQI by hour of day
- **Weather vs AQI scatter** — Temperature vs AQI scatter plot
- **Pollutant comparison** — Multi-line chart for 7-day pollutant trends

#### 4. Explainability (`/explainability`)

- **Natural language summary** — Bold paragraph explaining the prediction
- **Feature importance bar chart** — Horizontal bars colored green (positive) / red (negative)
- **Top drivers list** — Ranked features with ↑/↓ arrows and SHAP values
- **Waterfall chart** — Cumulative contribution of features to final prediction

#### 5. Pipeline (`/pipeline`)

- **Health indicators** — Green/yellow/red dots for each pipeline step
- **Step-by-step status** — Each step shows its name, status, and details (rows fetched, etc.)
- **Data freshness** — Latest observation timestamp, green if recent
- **Auto-refresh** — Updates every 30 seconds via JavaScript `fetch()`

#### 6. Data Sources (`/data-sources`)

- **Provider cards** — Open-Meteo and AQICN with roles, descriptions, data fields
- **Merge visualization** — Visual flow: Open-Meteo + AQICN → ML Model
- **Fallback strategy** — Numbered list with green check indicators

#### 7. Settings (`/settings`)

- City config (read-only display)
- System endpoints (API, Dashboard, MLflow URLs)
- AQI threshold table with color swatches

---

## 13. Digital Twin — Technical Deep Dive

### 13.1 What It Is

The Digital Twin is a Canvas-based 2.5D visualization of Hyderabad that renders in real-time in the browser. It transforms the AQI forecast into a visual simulation of the city's atmosphere.

### 13.2 Rendering Pipeline (Each Frame)

```
1. Clear canvas
2. Draw sky gradient (color from AQI forecast)
3. If AQI layer active: draw radial AQI heat gradient
4. Draw terrain (simplified ground shape)
5. Draw road grid (semi-transparent lines)
6. Draw building silhouettes (9 structures with window lights)
7. Draw landmarks (7 points: station, industrial, residential, green, commercial)
8. Update particle positions (wind/pollution simulation)
9. Draw particles
10. Update HUD panel (AQI number, category, wind speed)
11. requestAnimationFrame(loop)
```

### 13.3 AQI Color Mapping

```javascript
function aqiToColor(aqi) {
    if (aqi <= 50)  → { r: 0-128,  g: 100-228,  b: 50-100 }   // Green tones
    if (aqi <= 100) → { r: 0-180,  g: 200,       b: 50-0 }     // Yellow-green
    if (aqi <= 150) → { r: 255,    g: 140-0,     b: 0 }        // Orange
    if (aqi <= 200) → { r: 255,    g: 70-0,      b: 0-30 }     // Red
    else            → { r: 140,    g: 20,        b: 30 }        // Maroon
}
```

The sky gradient, building glow, and particle colors all use this AQI → color mapping, making the entire scene shift as AQI changes.

### 13.4 Interaction Controls

| Control | How |
|---------|-----|
| **Time slider** | Drag 0-72h. Updates sky, heat, particles, HUD. |
| **Layer toggle** | AQI Heat / PM2.5 / Wind / Particles buttons |
| **Play/Pause** | Auto-advances time slider at 300ms intervals |
| **Pan** | Click + drag anywhere on canvas |
| **Zoom** | Scroll wheel (0.5x – 3x range) |
| **Hover tooltip** | Shows landmark name on hover |

### 13.5 Particle System

- **80 particles** with random positions, velocities, sizes, and lifetimes
- **Wind strength** proportional to AQI (higher AQI → faster particles)
- **Color** based on current AQI forecast
- **Lifetime** random 50-150 frames, particles respawn at edges

### 13.6 Landmarks

| Name | Type | Visual |
|------|------|--------|
| City Center | city | (label only) |
| Station A546205 | station | Pulsing cyan circle with glow |
| Industrial Zone | industrial | Orange rectangle |
| Residential North | residential | (label only) |
| Residential South | residential | (label only) |
| Green Belt | green | Green semi-transparent circle |
| Commercial Hub | commercial | (label only) |

---

## 14. Alerting System

### 14.1 Thresholds

| Category | AQI | Color | Alert Triggered? |
|----------|-----|-------|-----------------|
| Good | 0–50 | `#00e400` | No |
| Moderate | 51–100 | `#ffff00` | No |
| Unhealthy (Sensitive) | 101–150 | `#ff7e00` | No |
| Unhealthy | 151–200 | `#ff0000` | No |
| **Very Unhealthy** | **201–300** | `#8f3f97` | **Yes** |
| **Hazardous** | **301+** | `#7e0023` | **Yes** |

Alert threshold: **AQI ≥ 200**.

### 14.2 Alert Flow

```
Inference Engine produces forecast
         │
         ▼
_check_alerts(forecast)
         │
    ┌────┴────┐
    │         │
  AQI≥200?  AQI<200?
    │         │
    ▼         ▼
Create     Skip
alert       │
    │         │
    └────┬────┘
         ▼
_save_alerts() → alerts.jsonl
         │
         ▼
Dashboard: showAlert() banner
```

### 14.3 Alert Storage Format (JSONL)

```json
{"type": "current_aqi", "aqi": 245, "level": "very_unhealthy", "timestamp": "2026-07-23T14:00:00"}
{"type": "forecast_48h", "aqi": 210, "category": "Very Unhealthy", "horizon": "48h", "timestamp": "2026-07-23T14:00:00"}
```

### 14.4 Dashboard Alert Banner

When alerts exist, a red gradient banner slides down from the top of every page:
```
⚠ Hazardous Air Quality Alert: AQI is 245 — Very Unhealthy
```

The banner is dismissible and has a slide-down animation.

---

## 15. Docker Deployment Guide

### 15.1 docker-compose.yml Services

```yaml
services:
  api:          # FastAPI on :8000
    build: .
    depends_on: []
    volumes: [./data, ./models/artifacts]
    healthcheck: curl localhost:8000/health every 30s

  dashboard:    # Flask on :5000
    build: .
    depends_on: [api (healthy)]
    volumes: [./data, ./dashboards]

  trainer:      # Runs daily_pipeline once, then exits
    build: .
    volumes: [./data, ./models/artifacts, ./mlruns]
    restart: "no"

  mlflow:       # MLflow UI on :5001
    build: .
    command: mlflow server --host 0.0.0.0 --port 5001 ...
    volumes: [./mlruns, ./models/artifacts]

  ingestion:    # Runs hourly_pipeline
    build: .
    volumes: [./data]
    restart: unless-stopped
```

**Note:** Docker builds use `requirements-docker.txt` (lean set: pandas, scikit-learn, fastapi, flask, shap, mlflow, plotly) instead of the full `requirements.txt` (which includes jupyter, xgboost, lightgbm, matplotlib, seaborn, pytest, black, flake8 — for local dev only). This keeps the Docker image smaller and builds faster.

### 15.2 Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/app/data` | All raw/processed data persistence |
| `./models/artifacts` | `/app/models/artifacts` | Trained model files |
| `./mlruns` | `/app/mlruns` and `/mlflow/mlruns` | MLflow tracking data |
| `./dashboards` | `/app/dashboards` | Template hot-reload for development |

### 15.3 Build & Run

```bash
# First time (build images)
docker compose up --build

# Subsequent runs
docker compose up

# Background mode
docker compose up -d

# View logs
docker compose logs -f api
docker compose logs -f dashboard

# Stop
docker compose down

# Full reset (remove volumes)
docker compose down -v
```

### 15.4 Environment Variables

All configurable via `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `AQICN_TOKEN` | (required) | Your AQICN API token |
| `API_PORT` | 8000 | FastAPI port |
| `DASHBOARD_PORT` | 5000 | Flask port |
| `MLFLOW_TRACKING_URI` | http://localhost:5001 | MLflow server URL |
| `DATA_DIR` | ./data | Data storage path |
| `LOG_LEVEL` | INFO | Python log level |

---

## 16. CI/CD with GitHub Actions

### 16.1 Hourly Workflow (`.github/workflows/hourly.yml`)

```yaml
on:
  schedule:
    - cron: "0 * * * *"   # Every hour at minute 0
  workflow_dispatch:        # Manual trigger from GitHub UI
```

Runs: Checkout → Install Python 3.11 → `pip install` → `python -m pipelines.hourly_pipeline`

Secrets needed: `AQICN_TOKEN`, `MLFLOW_TRACKING_URI`

### 16.2 Daily Workflow (`.github/workflows/daily.yml`)

```yaml
on:
  schedule:
    - cron: "0 2 * * *"   # 2 AM UTC daily
  workflow_dispatch:
```

Runs: Same as hourly + `python -m pytest tests/ -v` after training.

### 16.3 Manual Trigger

1. Go to GitHub repo → **Actions** tab
2. Select "Hourly AQI Pipeline" or "Daily Retraining Pipeline"
3. Click **Run workflow** → **Run workflow**

---

## 17. Testing Suite

### 17.1 Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific file
python -m pytest tests/test_features.py -v

# With coverage
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=html
```

### 17.2 Test Catalog

| File | Tests | What's Covered |
|------|-------|---------------|
| `test_aqi_utils.py` | 9 | All 7 AQI categories (Good → Hazardous), negative→Unknown, alert threshold at 200, color codes |
| `test_time_utils.py` | 4 | `floor_hour()` strips minutes/seconds, midnight handled, `now_local()` has timezone, tz is Asia/Karachi |
| `test_features.py` | 4 | FeatureBuilder creates all expected columns, target shift is correct (t vs t+24), training data drops NaN targets, cyclical encodings are in [-1,1] |
| `test_models.py` | 5 | Persistence predicts last value, Seasonal Naive uses t-24, perfect model gets RMSE=0 R²=1, imperfect model gets positive error, walk-forward creates valid splits |
| `test_api.py` | 7 | Health endpoint returns 200, all AQI routes return 200 (with no data), pipeline status works, data sources returns providers |

**Total: 29 tests, 29 passing, 0 skipped, 0 warnings.**

### 17.3 Test Fixture Pattern

```python
@pytest.fixture
def sample_df():
    """200-hour synthetic dataset with realistic patterns."""
    base = datetime(2026, 7, 20, 0)
    timestamps = [base + timedelta(hours=i) for i in range(200)]
    
    np.random.seed(42)
    aqi = np.clip(30 + randn(200) * 20 + sin(2π/24 * range(200)) * 15, 0, 300)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "aqi": aqi,
        "pm2_5": aqi * 0.8 + noise,
        "temperature_2m": 25 + daily_sine + noise,
        ...
    })
```

---

## 18. Configuration Reference

### 18.1 Full `configs/settings.yaml`

```yaml
city:
  name: Hyderabad
  country: Pakistan
  latitude: 25.396          # Hyderabad city center
  longitude: 68.357
  timezone: Asia/Karachi    # PKT = UTC+5

providers:
  open_meteo:
    base_url: https://air-quality-api.open-meteo.com/v1/air-quality
    weather_url: https://api.open-meteo.com/v1/forecast
    rate_limit_per_min: 60
    timeout_seconds: 30

  aqicn:
    base_url: https://api.waqi.info
    station_id: A546205     # Hyderabad station
    rate_limit_per_min: 10
    timeout_seconds: 30

pipeline:
  ingestion:
    fetch_interval_minutes: 60
    max_gap_fill_hours: 72

  features:
    lag_hours: [1, 6, 24, 72]           # Lags for AQI/PM features
    rolling_windows_hours: [6, 24]       # Rolling stats windows
    target_horizons_hours: [24, 48, 72]  # Prediction horizons

  training:
    validation_strategy: walk_forward
    test_split_ratio: 0.2
    retrain_cadence: daily
    min_training_samples: 168            # One week of hourly data

aqi_thresholds:
  good: [0, 50]
  moderate: [51, 100]
  unhealthy_sensitive: [101, 150]
  unhealthy: [151, 200]
  very_unhealthy: [201, 300]
  hazardous: [301, 999]
  alert_threshold: 200

alerts:
  enabled: true
  cooldown_hours: 3

storage:
  raw_format: json
  processed_format: parquet
  db_type: sqlite
  data_dir: ./data

model_registry:
  backend: mlflow
  metric_primary: rmse_24h
  metric_secondary: [rmse_48h, rmse_72h, mae_24h, r2_24h]

api:
  host: 0.0.0.0
  port: 8000
  cors_origins: ["http://localhost:5000", "http://localhost:8501"]

dashboard:
  host: 0.0.0.0
  port: 5000
  theme: dark
  refresh_interval_seconds: 60

logging:
  level: INFO
  format: "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
```

### 18.2 Override via Environment Variables

```bash
# Any config value can be overridden:
export AQICN_TOKEN="your_token"
export LATITUDE="24.8607"       # Karachi instead of Hyderabad
export LONGITUDE="67.0011"
export CITY_NAME="Karachi"
export API_PORT="9000"
```

The `utils/config.py` module loads YAML first, then checks for matching environment variables and overrides them. The priority: **ENV > YAML**.

---

## 19. Developer Guide — Extending the System

### 19.1 Adding a New City

1. **Update config:**
   ```yaml
   # configs/settings.yaml
   city:
     name: Karachi
     latitude: 24.8607
     longitude: 67.0011
     timezone: Asia/Karachi
   ```

2. **Find the AQICN station ID:** Search https://aqicn.org for the city and use the station code.

3. **Update `.env`:**
   ```
   AQICN_STATION=XXXXXX
   ```

4. **Run backfill for the new city:**
   ```bash
   python -m ingestion.fetch --backfill --start 2024-01-01 --end 2026-07-23
   ```

5. **Retrain:**
   ```bash
   python -m pipelines.daily_pipeline
   ```

### 19.2 Adding a New Data Provider

1. Create `ingestion/providers/newprovider.py`:
   ```python
   from ingestion.providers.base import BaseProvider

   class NewProvider(BaseProvider):
       def __init__(self):
           super().__init__("new_provider")

       def fetch_raw(self):
           # Your API call here
           return raw_data

       def normalize(self, raw):
           # Convert to DataFrame with timestamp column
           return df

       def validate(self, df):
           # Apply range checks
           return df
   ```

2. Register in `ingestion/orchestrator.py`:
   ```python
   class IngestionOrchestrator:
       def __init__(self):
           self.open_meteo = OpenMeteoProvider()
           self.aqicn = AQICNProvider()
           self.new_provider = NewProvider()  # Add here

       def fetch_all(self):
           om_df = self.open_meteo.run()
           aq_df = self.aqicn.run()
           np_df = self.new_provider.run()    # And use
           ...
   ```

### 19.3 Adding a New ML Model

```python
# In models/trainer.py, within build_models_for_horizons()

model_classes["my_new_model"] = lambda: SklearnWrapper(
    "my_new_model",
    MyNewRegressor(param1=..., param2=...)
)
```

That's it — the training loop automatically picks it up.

### 19.4 Adding a New Dashboard Page

1. Create `dashboards/templates/newpage.html`:
   ```html
   {% extends "base.html" %}
   {% block title %}New Page{% endblock %}
   {% block content %}
   <h2>My New Page</h2>
   <div class="card">Content here</div>
   {% endblock %}
   ```

2. Add route in `dashboards/app.py`:
   ```python
   @app.route("/newpage")
   def newpage():
       data = api_get("/api/v1/some/endpoint")
       return render_template("newpage.html", data=data.get("data", {}))
   ```

3. Add nav item in `base.html` sidebar.

### 19.5 Adding a New API Endpoint

1. Create `backend/routes/newroute.py`:
   ```python
   from fastapi import APIRouter
   from backend.schemas import APIResponse

   router = APIRouter()

   @router.get("/new-endpoint", response_model=APIResponse)
   async def my_new_endpoint():
       return APIResponse(data={"message": "Hello"})
   ```

2. Register in `backend/main.py`:
   ```python
   from backend.routes import newroute
   app.include_router(newroute.router, prefix="/api/v1", tags=["New"])
   ```

---

## 20. Troubleshooting & FAQ

### Common Issues

#### "No module named 'utils'"
```
PYTHONPATH not set. Run from aqi-predictor/ directory.
Fix: Set PYTHONPATH=. or run scripts via python -m
```

#### AQICN returns empty data
```
Station A546205 might be temporarily offline.
The system handles this gracefully — it skips supervised training
for those rows but continues inference with Open-Meteo data.
```

#### "MLflow tracking URI not reachable"
```
MLflow container might not be running.
The system falls back to local model storage automatically.
Start MLflow: docker compose up mlflow
```

#### Dashboard shows "No data available"
```
The pipeline hasn't run yet.
Run: docker compose up trainer  (runs daily pipeline once)
Or: python -m pipelines.hourly_pipeline
```

#### Docker build takes too long
```
First build installs all Python dependencies (~2 minutes).
Subsequent builds use Docker cache.
Use: docker compose up  (without --build if unchanged)
```

#### Port 8000 or 5000 already in use
```
Change ports in .env:
  API_PORT=8001
  DASHBOARD_PORT=5001
Or kill the existing process:
  Windows: netstat -ano | findstr :8000  →  taskkill /PID XXXX /F
  Linux:   lsof -i :8000  →  kill -9 XXXX
```

#### Tests fail with "No module named 'backend'"
```
Run from aqi-predictor/ directory, not from tests/ subdirectory.
Correct: cd aqi-predictor && python -m pytest tests/ -v
```

### FAQ

**Q: Why not use only one data source?**
A: If you use only Open-Meteo's forecast AQI as both input and target, you're just learning to copy a forecast — not predicting actual air quality. The dual-source design ensures the model learns the real weather→AQI relationship.

**Q: Can this work without AQICN data?**
A: Partially. Without AQICN labels, you can't train supervised models. The system will run inference using the fallback (persistence) model and Open-Meteo weather data. For full capability, you need an AQICN token (free).

**Q: How much historical data is needed?**
A: Minimum 168 hours (1 week) for training. More is better — 30+ days gives meaningful patterns.

**Q: Can I deploy this to production?**
A: The architecture supports it. Replace SQLite with PostgreSQL, add authentication to the API, use a reverse proxy (nginx), and switch GitHub Actions to a proper orchestrator (Airflow, Prefect). The modular structure makes these changes straightforward.

**Q: Does the Digital Twin use real map data?**
A: No — it's a simplified artistic representation using Canvas 2D. Landmarks are manually positioned. For a production version, you could integrate Mapbox or Leaflet for real geographic rendering.

**Q: How do I change the prediction horizons?**
A: Edit `configs/settings.yaml`:
```yaml
pipeline:
  features:
    target_horizons_hours: [12, 24, 36, 48]  # Custom horizons
```

**Q: What Python version?**
A: 3.10+. Tested on 3.11. Uses modern features like `str | Path` type hints in some modules.

---

## Appendix A: Quick Command Reference

```bash
# Setup
cp .env.example .env          # Create env file
bash scripts/setup.sh          # Install deps, create dirs

# Data
python -m ingestion.fetch --fetch                          # Single fetch
python -m ingestion.fetch --backfill --start 2024-01-01    # Historical

# Pipelines
python -m pipelines.hourly_pipeline    # Full hourly cycle
python -m pipelines.daily_pipeline     # Full daily cycle

# Servers
python -m backend.main                 # Start API (:8000)
python -m dashboards.app               # Start dashboard (:5000)

# MLflow
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db

# Testing
python -m pytest tests/ -v             # All tests
python -m pytest tests/test_api.py -v  # Specific file

# Docker
docker compose up --build              # Build + start all
docker compose up -d                   # Detached mode
docker compose logs -f api             # Follow API logs
docker compose down                    # Stop all
docker compose down -v                 # Stop + remove volumes

# Jupyter (EDA)
jupyter notebook notebooks/01_eda.ipynb
```

---

## Appendix B: File Count Summary

| Category | Count |
|----------|-------|
| Python modules | 39 |
| HTML templates | 9 (base + 7 pages + settings) |
| CSS files | 1 (500+ lines) |
| YAML configs | 1 |
| Docker files | 2 (Dockerfile + compose) |
| CI/CD workflows | 2 |
| Documentation | 3 (README + report + this guide) |
| Test files | 5 (29 tests) |
| Shell scripts | 2 |
| **Total project files** | **59+** |

---

*Documentation version: 1.0.0 — Generated 2026-07-23*
