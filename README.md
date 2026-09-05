# 🌫️ AeroTwinML — Serverless AQI Forecasting & 3D Atmospheric Digital Twin

[![Live Demo](https://img.shields.io/badge/Live%20Demo-aerotwinml.onrender.com-00C49F?style=for-the-badge&logo=render&logoColor=white)](https://aerotwinml.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Serverless_MLOps-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/0shazil0/AeroTwinML/actions)
[![Feature Store](https://img.shields.io/badge/Hopsworks-Feature_Store-FF6F00?style=for-the-badge)](https://www.hopsworks.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **Autonomous, Serverless Machine Learning System for 72-Hour Air Quality Index (AQI) Forecasting with an Interactive 3D Atmospheric Digital Twin for Hyderabad and Karachi, Pakistan.**

---

## 🌐 Live Web Application

The live production application is deployed and publicly accessible on Render:

### 👉 [https://aerotwinml.onrender.com/](https://aerotwinml.onrender.com/)

- **Hosting Platform**: [Render](https://render.com/) (Web Service)
- **Zero-Cost Serverless MLOps**: Background data ingestion, feature engineering, ML inference, and retraining pipelines run autonomously on **GitHub Actions**, automatically committing fresh predictions and pipeline telemetry back to the repository without requiring 24/7 cloud server compute.
- **Continuous Deployment**: Any push to `main` or hourly forecast snapshot updates automatically reflect on the live dashboard.

---

## 📑 Table of Contents

1. [Key Highlights & Features](#-key-highlights--features)
2. [Live Application Pages](#-live-application-pages)
3. [System Architecture](#-system-architecture)
4. [Data Strategy & Multi-Source Ingestion](#-data-strategy--multi-source-ingestion)
5. [Feature Store & MLOps Pipeline](#-feature-store--mlops-pipeline)
6. [Machine Learning Modeling & Evaluation](#-machine-learning-modeling--evaluation)
7. [Explainable AI (XAI)](#-explainable-ai-xai)
8. [Interactive 3D Digital Twin](#-interactive-3d-digital-twin)
9. [REST API Specification](#-rest-api-specification)
10. [Project Structure](#-project-structure)
11. [Local Development & Setup](#-local-development--setup)
12. [CI/CD Automation](#-cicd-automation)
13. [Testing Suite](#-testing-suite)
14. [License](#-license)

---

## 🚀 Key Highlights & Features

- **Multi-Horizon 72-Hour Forecasting**: Hourly updated predictions for **+24h, +48h, and +72h** horizons, capturing air quality trends, diurnal variations, and meteorological impacts.
- **Multi-City Support**: Native calibration for **Hyderabad** (Primary: Lat 25.396, Lon 68.357) and **Karachi** (Lat 24.868, Lon 67.082), Pakistan.
- **Interactive 2.5D / 3D Digital Twin**: High-performance HTML5 Canvas simulation rendering atmospheric particle physics, dynamic wind vector fields, AQI pollution heatmaps, and a time-travel scrubber.
- **Explainable AI (SHAP)**: Integrated SHapley Additive exPlanations to explain exact feature contributions for every forecast step with human-readable rationales.
- **Hopsworks Feature Store Integration**: Production feature groups (`aqi_features`, `aqi_training_features`, `training_metrics`) for time-series feature engineering and zero-leakage training.
- **Dual Framework Architecture**:
  - **Flask UI**: Standalone, lightweight server-rendered dark-mode dashboard tailored for free-tier cloud deployment (Render).
  - **FastAPI REST API**: High-performance OpenAPI backend for programmatic data access and integration.
- **Automated Health & Safety Alerts**: Instant threshold detection (Moderate, Unhealthy, Hazardous) following US EPA standards with specific medical and outdoor safety guidelines.
- **Resilient Fallback Mechanics**: Automatic data imputation, backward-search historical fill, and synthetic pollutant calibration ensuring 100% dashboard uptime even during third-party station API outages.

---

## 🖥️ Live Application Pages

The web application features an 8-view dark-mode responsive interface:

| View | Route | Description |
|---|---|---|
| **📊 Dashboard** | `/` | Hero AQI dial, US EPA category status, real-time meteorological metrics, pollutant breakdown (PM2.5, PM10, NO₂, O₃, SO₂, CO), and health advice. |
| **📈 Forecast** | `/forecast` | 72-hour forecast timeline with 24h/48h/72h milestone cards, trajectory charts, and confidence intervals. |
| **🔍 Analytics** | `/analytics` | Historical trends (7d / 30d / 90d), diurnal cycles (hour-of-day patterns), pollutant correlation heatmaps, and distribution plots. |
| **🧠 Explainability** | `/explainability` | SHAP waterfall and bar charts, top positive/negative feature drivers, and natural-language prediction explanations. |
| **⚙ Pipeline** | `/pipeline` | Live pipeline health monitor displaying execution status across Ingestion, Validation, Feature Store, Inference, and Alerts. |
| **🗄 Data Sources** | `/data-sources` | Telemetry for upstream data providers (Open-Meteo, AQICN, OpenAQ), station latency, sync timestamps, and fallback logic. |
| **🌐 Digital Twin** | `/digital-twin` | Interactive 2.5D / 3D particle physics simulation of atmospheric dispersion across urban zones with wind vectors and time controls. |
| **🔧 Settings** | `/settings` | Alert threshold customization, auto-refresh configuration, telemetry options, and location toggles. |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Data Ingestion Providers                               │
│       Open-Meteo (Weather Features)   │   AQICN / OpenAQ (Station Ground Truth) │
└───────────────────────┬─────────────────────────────────┬───────────────────────┘
                        │                                 │
                        ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Serverless GitHub Actions Orchestration                      │
│                                                                                 │
│   Hourly Pipeline (`hourly.yml`):          Daily Pipeline (`daily.yml`):        │
│   • Data Ingestion & Fallback Checks       • 2-Year Historical Backfill         │
│   • Cyclical & Lag Feature Generation      • Model Retraining & Walk-Forward CV │
│   • 24h / 48h / 72h Model Inference        • Hopsworks Feature Store Sync       │
│   • Real-Time Alert Evaluation             • Model Registry Artifact Updates    │
│   • Automated Git Snapshot Commit          • Pytest Test Suite Validation       │
└───────────────────────┬─────────────────────────────────┬───────────────────────┘
                        │                                 │
                        ▼                                 ▼
┌─────────────────────────────────────┐   ┌───────────────────────────────────────┐
│       Hopsworks Feature Store       │   │           ML Model Registry           │
│   • aqi_features                    │   │   • LightGBM / XGBoost Regressors     │
│   • aqi_training_features           │   │   • Random Forest & Ridge Baselines   │
│   • training_metrics                │   │   • SHAP Explainer Artifacts          │
└──────────────────┬──────────────────┘   └───────────────────┬───────────────────┘
                   │                                          │
                   └────────────────────┬─────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Deployment & Serving Layer                             │
│                                                                                 │
│      Render Web Service (Live):              FastAPI REST Backend (Port 8000):  │
│      👉 https://aerotwinml.onrender.com/     • /api/v1/aqi/current              │
│      • Flask UI + Jinja2 + Plotly.js         • /api/v1/aqi/forecast             │
│      • 3D Canvas Digital Twin Engine         • /api/v1/explain/latest           │
│      • Continuous Deployment on Git Push     • /api/v1/pipeline/status          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📡 Data Strategy & Multi-Source Ingestion

AeroTwinML operates on a **dual-source strategy** separating predictors from ground truth labels:

| Provider | Role | Parameters Captured |
|---|---|---|
| **Open-Meteo** | Predictor Features | Temperature, relative humidity, dew point, surface pressure, wind speed, wind direction, precipitation, cloud cover. |
| **AQICN / WAQI** | Ground Truth Station Labels | Real-time station AQI, PM2.5, PM10, NO₂, O₃, SO₂, CO (Station `A546205` - Hyderabad). |
| **OpenAQ** | Historical Backfill & Multi-City | Multi-year historical measurements (Location `4889110` - Hyderabad, `4791924` - Karachi). |

### Merge & Fallback Strategy
- **Timestamp Alignment**: Ingested hourly in local `Asia/Karachi` time (UTC+5).
- **Graceful Degradation**: If AQICN station labels experience downtime, Open-Meteo weather inputs continue driving inference with cached baseline ratios. Missing values never interrupt live serving.

---

## 🧠 Feature Store & MLOps Pipeline

Feature engineering transforms raw weather and air quality observations into high-signal time-series predictors:

- **Temporal Lags**: $t-1\text{h}$, $t-6\text{h}$, $t-24\text{h}$, $t-72\text{h}$ of AQI and key weather parameters.
- **Rolling Aggregations**: 6-hour and 24-hour rolling means, standard deviations, and min/max ranges.
- **Cyclical Encodings**: $\sin/\cos$ transformations of hour-of-day ($0-23$) and month-of-year ($1-12$) to model diurnal and seasonal seasonality.
- **Multi-Horizon Targets**: Supervised shift targets for $+24\text{h}$, $+48\text{h}$, and $+72\text{h}$.

### Hopsworks Integration
Feature groups maintained in Hopsworks:
1. `aqi_features`: Real-time streaming feature group updated hourly.
2. `aqi_training_features`: Cleaned, validated training datasets with train/test split tags.
3. `training_metrics`: Evaluation logs across model versions.

---

## 🔬 Machine Learning Modeling & Evaluation

The training engine evaluates multiple model families using **time-series walk-forward validation** (strictly avoiding future lookahead bias):

### Evaluated Model Families
- **Baselines**: Persistence Model, Seasonal Naive (24-hour lag), Ridge Regression.
- **Ensemble & Boosting**: Random Forest Regressor, XGBoost, LightGBM Regressor.

### Validation Metrics

Walk-forward validation across 72-hour forecast horizons:

| Horizon | Model | Target Metric | Purpose |
|---|---|---|---|
| **+24 Hours** | LightGBM / RF | RMSE, MAE, $R^2$ | Immediate next-day planning and high-risk alerts |
| **+48 Hours** | LightGBM / XGBoost | RMSE, MAE, $R^2$ | Medium-range pollution trajectory |
| **+72 Hours** | LightGBM / Ridge | RMSE, MAE, $R^2$ | Extended weekend and multi-day trend forecasting |

---

## 💡 Explainable AI (XAI)

AeroTwinML demystifies black-box machine learning predictions using **SHAP (SHapley Additive exPlanations)**:
- **Global Importance**: Highlights which meteorological factors (e.g., wind stagnation, high relative humidity, temperature inversions) historically contribute most to pollution build-up.
- **Local Attribution**: Quantifies exactly how many AQI points each feature added or subtracted for the latest prediction.
- **Natural Language Insights**: Converts numeric SHAP outputs into clear, actionable bullet points displayed directly on the UI (e.g., *"Low wind speed (1.8 m/s) trapped particulate matter, increasing AQI by +18.4 points"*).

---

## 🌐 Interactive 3D Digital Twin

The Digital Twin view (`/digital-twin`) provides a 2.5D / 3D atmospheric simulation rendered on HTML5 Canvas:
- **Atmospheric Particle Physics**: Thousands of simulated air particles responding in real time to actual wind direction and velocity.
- **Zonal AQI Heatmap**: Interpolated color gradients representing pollution density across urban zones in Hyderabad and Karachi.
- **Interactive Controls**:
  - Time-Travel Scrubber: Slide through past 24 hours to future 72-hour forecast states.
  - Altitude Layer Slicing: Inspect ground level vs. elevated inversion layers.
  - Wind Vector Overlays: Toggle velocity arrows and flow fields.

---

## 🔌 REST API Specification

AeroTwinML provides a FastAPI REST API with automatic Swagger documentation at `/docs`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and timestamp |
| `GET` | `/api/v1/aqi/current` | Current observed AQI, category, and live weather conditions |
| `GET` | `/api/v1/aqi/forecast` | 72-hour forecasted AQI and pollutant trajectories |
| `GET` | `/api/v1/aqi/history?hours=168` | Historical AQI and meteorological readings |
| `GET` | `/api/v1/aqi/pollutants` | Current breakdown of PM2.5, PM10, NO₂, O₃, SO₂, and CO |
| `GET` | `/api/v1/alerts` | Active and historical air quality alerts |
| `GET` | `/api/v1/explain/latest` | Local SHAP explanation for the latest inference cycle |
| `GET` | `/api/v1/explain/feature-importance` | Global feature importance rankings |
| `GET` | `/api/v1/pipeline/status` | Execution status and latency of MLOps pipelines |
| `GET` | `/api/v1/data-sources` | Metadata, sync status, and latency of ingestion APIs |

---

## 📁 Project Structure

```
aqi-predictor/
├── backend/                # FastAPI REST API
│   ├── routes/             # API endpoint handlers
│   ├── main.py             # FastAPI entry point
│   └── schemas.py          # Pydantic response models
├── configs/                # Central configuration
│   └── settings.yaml       # City coordinates, provider URLs, thresholds
├── dashboards/             # Flask Web Application (Deployed on Render)
│   ├── app.py              # Flask application routes
│   ├── templates/          # Jinja2 HTML templates (8 pages)
│   │   ├── index.html          # Dashboard (/)
│   │   ├── forecast.html       # 72-hour forecast (/forecast)
│   │   ├── analytics.html      # Historical trends (/analytics)
│   │   ├── explainability.html # SHAP explanations (/explainability)
│   │   ├── pipeline.html       # Pipeline health (/pipeline)
│   │   ├── data_sources.html   # Upstream sources (/data-sources)
│   │   ├── digital_twin.html   # 3D Digital Twin simulation (/digital-twin)
│   │   ├── settings.html       # User preferences (/settings)
│   │   └── base.html           # Layout shell & navigation
│   └── static/             # CSS stylesheets, JS scripts, visual assets
├── data/                   # Data directory
│   ├── raw/                # Ingested provider payloads & alert logs
│   ├── processed/          # Merged parquets, predictions & pipeline status
│   │   └── predictions/    # forecast_latest.json
│   └── backfill/           # Multi-year training datasets
├── docs/                   # Technical documentation & reports
│   ├── report.md           # Engineering & architecture report
│   └── full-guide.md       # Comprehensive system guide
├── feature_store/          # Feature engineering & Hopsworks client
│   ├── builder.py          # Lag & rolling window feature generation
│   └── hopsworks_client.py # Hopsworks API integration
├── ingestion/              # Ingestion layer
│   └── providers/          # Open-Meteo, AQICN, and OpenAQ connectors
├── models/                 # Machine Learning pipeline
│   ├── trainer.py          # Walk-forward model training
│   ├── inference.py        # Multi-horizon prediction engine
│   ├── explainer.py        # SHAP calculation engine
│   ├── registry.py         # Hopsworks / MLflow model tracking
│   └── artifacts/          # Serialized models (.pkl)
├── pipelines/              # Automation scripts
│   ├── hourly_pipeline.py  # Hourly inference & forecast generation
│   └── daily_pipeline.py   # Daily backfill & retraining pipeline
├── scripts/                # Helper scripts
│   ├── setup.bat           # Windows environment setup
│   └── setup.sh            # Unix environment setup
├── tests/                  # Pytest test suite (31 tests)
├── .github/workflows/      # GitHub Actions CI/CD workflows
│   ├── hourly.yml          # Hourly pipeline runner (cron: "0 * * * *")
│   └── daily.yml           # Daily retraining runner (cron: "0 2 * * *")
├── docker-compose.yml      # Multi-container local orchestration
├── Dockerfile              # Container definition for API / Dashboard
├── requirements.txt        # Full Python dependencies
├── requirements-docker.txt # Lightweight container dependencies
└── README.md               # Project documentation
```

---

## 💻 Local Development & Setup

### Prerequisites
- Python 3.10 or 3.11
- Git
- Docker & Docker Compose (optional, for containerized run)

### Option 1: Local Virtual Environment

```bash
# 1. Clone the repository
git clone https://github.com/0shazil0/AeroTwinML.git
cd AeroTwinML

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and insert your AQICN_TOKEN, OPENAQ_API_KEY, and HOPSWORKS_API_KEY

# 5. Run the Flask Dashboard (Render replica)
python -m dashboards.app
# Access at http://localhost:5000

# 6. Run the FastAPI REST Backend (in another terminal)
python -m backend.main
# Access at http://localhost:8000 and docs at http://localhost:8000/docs

# 7. Run an on-demand hourly prediction cycle
python -m pipelines.hourly_pipeline
```

### Option 2: Docker Compose

```bash
# Build and run API, Dashboard, Trainer, and MLflow
docker compose up --build
```

Services will be mapped to:
- **Dashboard**: [http://localhost:5000](http://localhost:5000)
- **API**: [http://localhost:8000](http://localhost:8000)
- **API Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking**: [http://localhost:5001](http://localhost:5001)

---

## ⚙️ CI/CD Automation

AeroTwinML runs 100% serverlessly via GitHub Actions:

- **Hourly Pipeline (`.github/workflows/hourly.yml`)**:
  - Scheduled via cron: `0 * * * *` (at the top of every hour).
  - Fetches fresh meteorological data from Open-Meteo and observed AQI from AQICN.
  - Builds feature vectors and executes multi-horizon inference.
  - Evaluates threshold alerts.
  - Automatically commits updated predictions to `data/processed/predictions/forecast_latest.json` with `[skip ci]`.
  - Render auto-detects changes and updates the live site immediately.
- **Daily Pipeline (`.github/workflows/daily.yml`)**:
  - Scheduled via cron: `0 2 * * *` (2:00 AM UTC daily).
  - Backfills historical datasets from OpenAQ & Open-Meteo.
  - Retrains Random Forest, LightGBM, XGBoost, and Ridge models.
  - Evaluates walk-forward cross-validation metrics.
  - Registers the best performing model to Hopsworks Model Registry.
  - Executes the automated Pytest test suite.

---

## 🧪 Testing Suite

The repository includes a comprehensive Pytest test suite covering AQI calculations, time zones, feature engineering, baselines, model inference, and REST endpoints:

```bash
python -m pytest tests/ -v
```

```
tests/test_api.py::TestAPIHealth::test_health_check PASSED
tests/test_api.py::TestAqiRoutes::test_current_aqi_no_data PASSED
tests/test_api.py::TestAqiRoutes::test_forecast_no_data PASSED
tests/test_aqi_utils.py::TestAQIClassification::test_hazardous PASSED
tests/test_features.py::TestFeatureBuilder::test_build_all_creates_expected_columns PASSED
tests/test_models.py::TestWalkForwardSplit::test_walk_forward_split PASSED
tests/test_models.py::TestInferenceEngine::test_per_horizon_inference_when_single_model_is_none PASSED
...
======================= 31 passed in 11.47s =======================
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.
