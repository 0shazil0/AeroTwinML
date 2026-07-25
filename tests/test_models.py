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
