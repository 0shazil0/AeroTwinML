"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

# Add parent to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Create the test app without the actual imports that might fail
# We'll test routes directly with mocked data


class TestAPIHealth:
    def test_health_endpoint_imports(self):
        """Verify that the app module can be imported."""
        # Just verify the structure exists
        from backend import main
        assert main.app is not None
        assert main.app.title == "Pearls AQI Predictor"

    def test_health_check(self):
        """Test the health endpoint."""
        from backend.main import app
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestAqiRoutes:
    def test_current_aqi_no_data(self):
        from backend.main import app
        client = TestClient(app)

        response = client.get("/api/v1/aqi/current")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_forecast_no_data(self):
        from backend.main import app
        client = TestClient(app)

        response = client.get("/api/v1/aqi/forecast")
        assert response.status_code == 200

    def test_history_no_data(self):
        from backend.main import app
        client = TestClient(app)

        response = client.get("/api/v1/aqi/history?hours=24")
        assert response.status_code == 200


class TestPipelineRoutes:
    def test_pipeline_status(self):
        from backend.main import app
        client = TestClient(app)

        response = client.get("/api/v1/pipeline/status")
        assert response.status_code == 200


class TestDataSourcesRoutes:
    def test_data_sources(self):
        from backend.main import app
        client = TestClient(app)

        response = client.get("/api/v1/data-sources")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "providers" in data["data"]
