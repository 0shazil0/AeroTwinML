@echo off
REM AQI Predictor — Windows setup script
echo === AQI Predictor Setup ===

if not exist .env (
    echo Creating .env from .env.example...
    copy .env.example .env
    echo Please edit .env with your actual API tokens before running.
)

echo Installing Python dependencies...
pip install -r requirements.txt

echo Creating data directories...
python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) or (Path(d)/'.gitkeep').touch() for d in ['data/raw/open_meteo','data/raw/aqicn','data/raw/logs','data/processed/merged_hourly','data/processed/cleaned','data/processed/features','data/processed/predictions','data/backfill/train_v1','data/quality','models/artifacts','mlruns']]; print('Done.')"

echo === Setup complete ===
echo Run: docker compose up --build
