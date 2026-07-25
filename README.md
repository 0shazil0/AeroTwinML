# 🌫 Pearls AQI Predictor → AeroTwinML

**Serverless AQI forecasting for Hyderabad, Pakistan — 72-hour predictions using Open-Meteo + OpenAQ + Hopsworks.**

```
# Everything runs on GitHub Actions + Hopsworks — zero servers to manage
python -m pipelines.hourly_pipeline  # or just push to main — CI runs it
```

## Architecture Overview

The system predicts Air Quality Index (AQI) for the next 24, 48, and 72 hours using weather data from Open-Meteo as features and observed AQI from AQICN as labels. A trained ML model learns the mapping from weather conditions to future air quality.

### Data Flow

```
Hourly Cycle:            Daily Cycle:
┌─────────────┐           ┌─────────────┐
│ Fetch Open-  │           │ Backfill     │
│ Meteo + AQICN│           │ History      │
└──────┬───────┘           └──────┬───────┘
       │                          │
       ▼                          ▼
┌─────────────┐           ┌─────────────┐
│ Validate &   │           │ Build        │
│ Normalize    │           │ Features     │
└──────┬───────┘           └──────┬───────┘
       │                          │
       ▼                          ▼
┌─────────────┐           ┌─────────────┐
│ Merge on     │           │ Train Models │
│ Timestamp    │           │ (RF, XGBoost,│
└──────┬───────┘           │ LightGBM)    │
       │                   └──────┬───────┘
       ▼                          │
┌─────────────┐                   ▼
│ Feature      │           ┌─────────────┐
│ Engineering  │           │ MLflow       │
└──────┬───────┘           │ Registry     │
       │                   └──────┬───────┘
       ▼                          │
┌─────────────┐                   ▼
│ Run          │           ┌─────────────┐
│ Inference    │           │ Register     │
└──────┬───────┘           │ Best Model   │
       │                   └─────────────┘
       ▼
┌─────────────┐
│ Check Alerts │
│ + SHAP Explan│
└─────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- AQICN API token ([get one here](https://aqicn.org/data-platform/token/))

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repo
git clone <repo-url> && cd aqi-predictor

# 2. Set up environment
cp .env.example .env
# Edit .env with your AQICN_TOKEN

# 3. Start everything
docker compose up --build
```

Services will be available at:
| Service | URL |
|---------|-----|
| Dashboard | http://localhost:5000 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |

### Option 2: Local Development

```bash
# Windows
scripts\setup.bat
# Linux/macOS
bash scripts/setup.sh

# Start API
python -m backend.main

# Start Dashboard (in another terminal)
python -m dashboards.app

# Run hourly pipeline
python -m pipelines.hourly_pipeline
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/aqi/current` | GET | Current AQI + weather |
| `/api/v1/aqi/forecast` | GET | 72-hour forecast |
| `/api/v1/aqi/history?hours=168` | GET | Historical data |
| `/api/v1/aqi/pollutants` | GET | Pollutant breakdown |
| `/api/v1/alerts` | GET | Recent alerts |
| `/api/v1/explain/latest` | GET | SHAP explanation |
| `/api/v1/explain/feature-importance` | GET | Global importance |
| `/api/v1/pipeline/status` | GET | Pipeline health |
| `/api/v1/data-sources` | GET | Data source info |

---

## Dashboard Pages

1. **Dashboard** — Current AQI hero, weather snapshot, quick stats, alerts
2. **Forecast** — 72-hour forecast timeline, pollutant breakdown, charts
3. **Analytics** — Historical trends, diurnal patterns, weather correlations
4. **Explainability** — SHAP feature importance, natural language explanations
5. **Pipeline** — Real-time pipeline health, step-by-step status
6. **Data Sources** — Provider roles, merge logic, fallback strategy
7. **Digital Twin** — 2.5D interactive view of Hyderabad with AQI heat map, time slider, particle animations

---

## Project Structure

```
aqi-predictor/
├── app/                    # Application layer
├── backend/                # FastAPI backend
│   ├── routes/             # API route handlers
│   ├── main.py             # App entry point
│   └── schemas.py          # Pydantic models
├── configs/                # Configuration files
├── dashboards/             # Flask UI
│   ├── app.py              # Dashboard server
│   ├── templates/          # Jinja2 templates (7 pages)
│   └── static/             # CSS, JS, assets
├── data/                   # Data storage
│   ├── raw/                # Raw JSON from providers
│   ├── processed/          # Merged, cleaned, features
│   └── backfill/           # Training datasets
├── feature_store/          # Feature engineering
├── ingestion/              # Data ingestion
│   └── providers/          # Open-Meteo & AQICN clients
├── models/                 # ML models
│   ├── trainer.py          # Training framework
│   ├── registry.py         # MLflow integration
│   ├── inference.py        # Prediction engine
│   └── explainer.py        # SHAP explainability
├── notebooks/              # EDA & experimentation
├── pipelines/              # Automation scripts
│   ├── hourly_pipeline.py  # Hourly run
│   └── daily_pipeline.py   # Daily retraining
├── tests/                  # Test suite (29 tests)
├── utils/                  # Shared utilities
├── .github/workflows/      # CI/CD (GitHub Actions)
├── docker-compose.yml      # Multi-service Docker
├── Dockerfile              # Container definition
└── README.md               # This file
```

---

## Tech Stack

### Data
- **Open-Meteo** — Free weather and air quality data (features)
- **AQICN** — Station-level observed AQI (labels)

### ML
- **Scikit-learn** — Ridge, Random Forest, Gradient Boosting
- **XGBoost / LightGBM** — Gradient boosted trees
- **SHAP** — Model explainability
- **MLflow** — Experiment tracking & model registry

### Backend
- **FastAPI** — High-performance REST API
- **Pydantic** — Data validation
- **Uvicorn** — ASGI server

### Frontend
- **Flask** — Server-rendered dashboard
- **Jinja2** — HTML templating
- **Plotly.js / Chart.js** — Interactive charts
- **Canvas API** — Digital Twin visualization

### DevOps
- **Docker Compose** — Container orchestration
- **GitHub Actions** — CI/CD scheduling

---

## Data Strategy

Two sources with distinct roles:

| Source | Role | Used For |
|--------|------|----------|
| **Open-Meteo** | Weather features, forecast inputs | Temperature, humidity, wind, precipitation, cloud cover |
| **AQICN** | Observed ground truth | AQI, PM2.5, PM10, NO₂, O₃, SO₂, CO |

**Merge strategy:** Data is merged on hourly timestamp in `Asia/Karachi`. Open-Meteo contributes weather predictors. AQICN provides target labels shifted by 24h, 48h, 72h for supervised training.

---

## Model Training

### Baseline Models
- **Persistence** — Predicts last known value
- **Seasonal Naive** — Uses value from 24h ago
- **Ridge Regression** — Linear baseline

### Advanced Models
- **Random Forest** — Ensemble of decision trees
- **XGBoost** — Gradient boosting with regularization
- **LightGBM** — Fast gradient boosting

### Validation
- **Walk-forward splits** — Time-series aware, no data leakage
- **Metrics** — RMSE, MAE, R² per horizon (24h, 48h, 72h)

---

## Deployment

The system runs as four Docker services:

```yaml
services:
  api        # FastAPI backend (port 8000)
  dashboard  # Flask UI (port 5000)
  trainer    # Daily retraining (runs once)
  mlflow     # Experiment tracking (port 5001)
```

Automated via GitHub Actions:
- **Hourly** — Ingestion + Features + Inference
- **Daily** — Backfill + Retraining + Registry update

---

## Testing

```bash
python -m pytest tests/ -v
```

29 tests covering: AQI classification, time utilities, feature engineering, model training, baselines, walk-forward splits, and API endpoints.

---

## License

MIT
