"""Tests for model training and baselines."""

import pytest
import numpy as np
from models.trainer import PersistenceModel, SeasonalNaiveModel, evaluate_model


class TestBaselines:
    def test_persistence_model(self):
        model = PersistenceModel()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])

        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == 3
        assert preds[0] == 30  # last value

    def test_seasonal_naive_with_enough_data(self):
        model = SeasonalNaiveModel()
        y = np.arange(1, 50, dtype=float)  # values 1..49, index 25 = value 26

        model.fit(np.zeros((49, 3)), y)
        preds = model.predict(np.zeros((3, 3)))

        assert len(preds) == 3
        assert preds[0] == 26.0  # y[-24] = value at index 49-24=25, which is 26

    def test_evaluate_model_perfect_prediction(self):
        class PerfectModel:
            def predict(self, X):
                return np.array([10, 20, 30])

        model = PerfectModel()
        X = np.array([[1], [2], [3]])
        y = np.array([10, 20, 30])

        metrics = evaluate_model(model, X, y)
        assert metrics["rmse"] == 0
        assert metrics["mae"] == 0
        assert metrics["r2"] == 1.0

    def test_evaluate_model_imperfect(self):
        class BadModel:
            def predict(self, X):
                return np.array([0, 0, 0])

        model = BadModel()
        X = np.array([[1], [2], [3]])
        y = np.array([10, 20, 30])

        metrics = evaluate_model(model, X, y)
        assert metrics["rmse"] > 0
        assert metrics["r2"] < 1


class TestWalkForwardSplit:
    def test_walk_forward_split(self):
        from models.trainer import walk_forward_split
        import pandas as pd
        from datetime import datetime, timedelta

        base = datetime(2026, 1, 1)
        df = pd.DataFrame({
            "timestamp": [base + timedelta(hours=i) for i in range(100)],
            "feature": np.random.randn(100),
            "target": np.random.randn(100),
        })

        splits = walk_forward_split(df, ["feature"], "target", train_size=0.7, step=10)
        assert len(splits) >= 1
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0
            assert len(train) + len(test) <= 100


class TestInferenceEngine:
    def test_per_horizon_inference_when_single_model_is_none(self):
        import pandas as pd
        from models.inference import InferenceEngine

        class DummyHorizonModel:
            def __init__(self, pred_val):
                self.pred_val = pred_val

            def predict(self, X):
                return np.array([self.pred_val])

        engine = InferenceEngine()
        engine.model = None
        engine.models_by_horizon = {
            "24h": {"model": DummyHorizonModel(42.0), "model_name": "dummy_24"},
            "48h": {"model": DummyHorizonModel(55.0), "model_name": "dummy_48"},
            "72h": {"model": DummyHorizonModel(68.0), "model_name": "dummy_72"},
        }

        df = pd.DataFrame({"aqi": [50.0], "temperature_2m": [25.0]})
        fc = engine.predict(df)

        assert fc["model_info"]["type"] == "per_horizon"
        assert fc["forecast"]["24h"]["aqi"] == 42.0
        assert fc["forecast"]["48h"]["aqi"] == 55.0
        assert fc["forecast"]["72h"]["aqi"] == 68.0
