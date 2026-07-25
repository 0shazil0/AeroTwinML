# Pearls AQI Predictor — Technical Report

## 1. Executive Summary

The Pearls AQI Predictor is an end-to-end MLOps system that forecasts Air Quality Index (AQI) in Hyderabad, Pakistan for the next 72 hours. It uses a dual-source architecture — Open-Meteo for weather features and AQICN for observed labels — to learn the mapping from meteorological conditions to future air quality. The system includes automated hourly ingestion, feature engineering, model training with walk-forward validation, SHAP explainability, a modern dark-themed dashboard with a Digital Twin page, and Dockerized deployment.

---

## 2. Architecture

### 2.1 System Design

The system follows a modular MLOps architecture with clear separation of concerns:

```
┌────────────────────────────────────────────────────┐
│                  GitHub Actions CI/CD               │
│             (Hourly + Daily Scheduling)              │
└──────────────┬─────────────────┬───────────────────┘
               │                 │
               ▼                 ▼
     ┌─────────────────┐  ┌──────────────┐
     │ Hourly Pipeline │  │Daily Pipeline │
     │ • Ingestion     │  │ • Backfill    │
     │ • Features      │  │ • Training    │
     │ • Inference     │  │ • Registry    │
     │ • Alerts        │  │ • Evaluation  │
     └────────┬────────┘  └──────┬───────┘
              │                  │
              ▼                  ▼
     ┌─────────────────────────────────────┐
     │           FastAPI Backend            │
     │   (REST API + All Endpoints)         │
     └──────────────┬──────────────────────┘
                    │
                    ▼
     ┌─────────────────────────────────────┐
     │         Flask Dashboard UI           │
     │  7 Pages: Dashboard, Forecast,       │
     │  Analytics, Explainability,          │
     │  Pipeline, Data Sources,             │
     │  Digital Twin                         │
     └─────────────────────────────────────┘
```

### 2.2 Docker Services

- **API** (`port 8000`) — FastAPI serving all REST endpoints
- **Dashboard** (`port 5000`) — Flask rendering 7 interactive pages
- **Trainer** — Runs daily retraining pipeline
- **MLflow** (`port 5001`) — Experiment tracking and model registry

---

## 3. Data Strategy

### 3.1 Dual-Source Design

| Provider | Role | Data Provided |
|----------|------|---------------|
| Open-Meteo | Features (predictors) | Temperature, humidity, dew point, pressure, wind speed, wind direction, precipitation, cloud cover |
| AQICN | Labels (targets) | Observed AQI, PM2.5, PM10, NO₂, O₃, SO₂, CO (Station A546205) |

**Key principle:** Open-Meteo provides weather inputs used to forecast future AQI. AQICN provides measured reality used as supervision. The model learns the relationship: `weather(t) → AQI(t+N)`.

### 3.2 Merge Logic

Data is merged on hourly timestamps in `Asia/Karachi`. No provider-specific identifiers are used — only datetime alignment. AQICN may have gaps; these rows are excluded from supervised training but weather data is preserved for inference.

### 3.3 Fallback Strategy

1. Try AQICN observed station data for labels
2. Use Open-Meteo weather regardless (always available)
3. Skip rows with missing AQICN from supervised training
4. Continue inference with latest Open-Meteo features

---

## 4. Feature Engineering

### 4.1 Feature Groups

**Time Features:**
- hour, day, day_of_week, month, weekend, season
- Cyclical encodings: hour_sin, hour_cos, month_sin, month_cos, day_of_week_sin, day_of_week_cos

**Lag Features:**
- AQI: t-1, t-6, t-24, t-72
- PM2.5: t-1, t-24
- PM10: t-1, t-24

**Rolling Statistics:**
- AQI rolling mean (6h, 24h), std (24h), min (24h), max (24h)

**Weather Features:**
- temperature_2m, relative_humidity_2m, dew_point_2m, pressure_msl, wind_speed_10m, wind_direction_10m, precipitation, cloud_cover

**Interaction Features:**
- humidity × temperature
- wind × PM2.5
- rain × PM10
- AQI change rate (6h slope)

### 4.2 Target Construction

Supervised targets at time `t`:
- `target_aqi_24h` = AQI at t+24
- `target_aqi_48h` = AQI at t+48
- `target_aqi_72h` = AQI at t+72

Classification labels derived from regression output:
- Good (0-50), Moderate (51-100), Unhealthy for Sensitive (101-150), Unhealthy (151-200), Very Unhealthy (201-300), Hazardous (301+)

---

## 5. Modeling Approach

### 5.1 Baseline Models

| Model | Description | Purpose |
|-------|-------------|---------|
| Persistence | Predicts last known AQI | Simplest baseline — any model must beat this |
| Seasonal Naive | Uses value from 24h ago | Captures daily seasonality |
| Ridge Regression | Linear model with L2 regularization | Linear baseline |

### 5.2 Tree-Based Models

| Model | Parameters | Notes |
|-------|-----------|-------|
| Random Forest | n_estimators=100, max_depth=10 | Ensemble, handles non-linearity |
| Gradient Boosting | n_estimators=100, max_depth=5 | Sequential boosting |
| XGBoost | n_estimators=100, max_depth=6, lr=0.1 | Regularized boosting |
| LightGBM | n_estimators=100, max_depth=6, lr=0.1 | Histogram-based, fast |

### 5.3 Validation Strategy

**Walk-forward validation** (time-series split) is used instead of random splits to prevent data leakage. Training proceeds on earlier periods, testing on subsequent windows. This properly simulates real-world forecasting where future data is unseen.

### 5.4 Metrics

- **RMSE** (Root Mean Squared Error) — Primary metric, penalizes large errors
- **MAE** (Mean Absolute Error) — Interpretable
- **R²** (Coefficient of Determination) — Variance explained

Evaluated separately for each horizon (24h, 48h, 72h).

---

## 6. Model Registry (MLflow)

MLflow tracks:
- Data version and feature version
- Model parameters
- Training date range
- Evaluation metrics per horizon
- Model artifacts

The best model (lowest RMSE at 24h) is registered to the MLflow Model Registry. The inference engine loads the latest registered model automatically.

---

## 7. Explainability

### 7.1 SHAP Integration

- **TreeExplainer** for XGBoost, LightGBM, Random Forest
- **KernelExplainer** as fallback
- Per-prediction feature contribution values
- Global feature importance via mean |SHAP|

### 7.2 Correlation Fallback

When SHAP is unavailable (no trained tree model), a correlation-based importance rank is computed from historical data. This ensures the explainability page always shows useful information.

### 7.3 Natural Language Generation

Template-based NL from top SHAP drivers:
- "AQI is predicted to rise due to high humidity and low wind speed"
- "Air quality expected to improve due to increasing precipitation"

---

## 8. Dashboard

### 8.1 Design System

- **Theme:** Dark (#0a0a14 primary, #1a1a35 cards)
- **Typography:** System sans-serif, 96px hero numbers
- **Components:** Rounded cards, glow effects, animated counters, status badges
- **Charts:** Plotly.js (interactive, dark-themed)

### 8.2 Page Descriptions

1. **Dashboard** — Hero AQI number with animated counter, weather snapshot, PM stats, recent alerts
2. **Forecast** — 72-hour timeline, forecast chart, pollutant breakdown bars, historical context
3. **Analytics** — Timeline, diurnal pattern, temperature vs AQI scatter, pollutant comparison
4. **Explainability** — Natural language summary, horizontal SHAP bars, waterfall chart
5. **Pipeline** — Live health indicators, step-by-step status with color coding, data freshness
6. **Data Sources** — Provider cards, merge logic visualization, fallback strategy
7. **Digital Twin** — Canvas-based 2.5D Hyderabad view with AQI heat map, landmark markers, particle animation, time slider (0-72h), play/pause

### 8.3 Digital Twin Details

The Digital Twin renders Hyderabad as a simplified 2.5D scene:
- **Sky** — Gradient colored by current AQI level
- **Buildings** — Silhouette blocks with window lights colored by AQI
- **AQI Heat** — Radial gradient overlay based on forecast AQI
- **Landmarks** — Markers for station A546205, industrial zones, residential areas
- **Particles** — Animated wind/pollution particles with intensity tied to AQI
- **Controls** — Layer toggle (AQI Heat / PM2.5 / Wind / Particles), time slider, play button
- **Interaction** — Pan, zoom, hover tooltips on landmarks

---

## 9. Pipeline Automation

### 9.1 Hourly Pipeline

1. Fetch latest Open-Meteo + AQICN data
2. Validate and normalize
3. Merge on hourly timestamp
4. Generate 50+ features
5. Run inference using best registered model
6. Check alert thresholds (AQI ≥ 200)
7. Persist predictions and quality report

### 9.2 Daily Pipeline

1. Backfill recent 30 days of historical data
2. Build training dataset with features + targets
3. Train all models (Ridge, RF, GB, XGBoost, LightGBM)
4. Evaluate on walk-forward validation splits
5. Log to MLflow
6. Register best model
7. Save metrics for dashboard

### 9.3 Scheduling

GitHub Actions provides serverless scheduling:
- **Hourly** — Runs every hour (`0 * * * *`)
- **Daily** — Runs at 2 AM daily (`0 2 * * *`)

Both can be triggered manually via workflow_dispatch.

---

## 10. Alerting System

### 10.1 Thresholds

| Category | AQI Range | Alert? |
|----------|-----------|--------|
| Good | 0-50 | No |
| Moderate | 51-100 | No |
| Unhealthy (Sensitive) | 101-150 | No |
| Unhealthy | 151-200 | No |
| Very Unhealthy | 201-300 | **Yes** |
| Hazardous | 301+ | **Yes** |

### 10.2 Alert Behavior

- Alert banner appears at top of all pages when AQI ≥ 200
- Banner shows health advisory text
- Alerts logged to JSONL file with timestamp
- 3-hour cooldown prevents repeated identical alerts

---

## 11. Testing

### 11.1 Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| AQI Utils | 9 | Classification, alerts, colors |
| Time Utils | 4 | Floor hour, timezone, localization |
| Feature Builder | 4 | Column creation, target alignment, cyclical encodings, training data |
| Models | 5 | Baselines, metrics, walk-forward split |
| API | 7 | Health, AQI routes, pipeline, data sources |

**Total: 29 tests, all passing.**

---

## 12. Challenges & Solutions

### 12.1 AQICN Data Sparsity
AQICN station coverage for Hyderabad may have gaps. The system handles this by marking missing labels and excluding those rows from supervised training while continuing inference with available weather data.

### 12.2 Time-Series Validation
Random train/test splits cause data leakage in forecasting. Walk-forward validation was implemented to properly simulate real-world conditions.

### 12.3 Multi-Horizon Forecasting
Instead of one model for all horizons, separate evaluations for 24h, 48h, and 72h ensure each horizon gets appropriate attention.

---

## 13. Future Improvements

1. **Deep Learning** — LSTM/GRU models for sequence learning once classical baselines are stable
2. **Multi-Station** — Expand beyond Hyderabad to other Pakistani cities
3. **Real-Time Streaming** — Kafka/Pulsar for true streaming ingestion
4. **Probability Forecasts** — Uncertainty quantification with prediction intervals
5. **Airflow Migration** — Replace GitHub Actions with Airflow for complex DAGs
6. **PostgreSQL** — Upgrade from SQLite for production-grade storage

---

## 14. Conclusion

The Pearls AQI Predictor delivers a complete, production-style AQI forecasting system. It demonstrates proper MLOps practices — dual-source data ingestion, robust feature engineering, time-series-aware model validation, MLflow experiment tracking, SHAP explainability, and Dockerized deployment. The 7-page dashboard with a standout Digital Twin page makes the project visually impressive while maintaining technical rigor.
