#!/bin/bash
# AQI Predictor — One-time setup script
set -e

echo "=== AQI Predictor Setup ==="

# Check for .env
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please edit .env with your actual API tokens before running."
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create data directories
echo "Creating data directories..."
python -c "
from pathlib import Path
dirs = [
    'data/raw/open_meteo', 'data/raw/aqicn', 'data/raw/logs',
    'data/processed/merged_hourly', 'data/processed/cleaned',
    'data/processed/features', 'data/processed/predictions',
    'data/backfill/train_v1', 'data/quality',
    'models/artifacts', 'mlruns',
]
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / '.gitkeep').touch()
print('Done.')
"

echo "=== Setup complete ==="
echo "Run: docker compose up --build"
