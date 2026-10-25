from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import make_features
from src.data.splitting import build_windows
from src.evaluation.backtest import backtest, prediction_table
from src.evaluation.metrics import forecast_metrics
from src.models import create_model
from src.training.trainer import train_model


@pytest.fixture
def batch():
    index = pd.date_range("2024-01-01", periods=240, freq="15min", tz="UTC")
    frame = pd.DataFrame({"load": 10 + np.arange(240) % 8}, index=index)
    history, known = make_features(frame, "load", [], "UTC")
    return build_windows(history, known, frame.load, (0, 240), 16, 12, 12)


def test_metrics_percentages_and_zero_policy():
    result = forecast_metrics(np.array([0, 10, 20]), np.array([1, 12, 18]))
    assert result["WAPE"] == pytest.approx(100*5/30)
    assert result["MAPE"] == pytest.approx(15)
    assert result["MAE"] == pytest.approx(5/3)
    assert result["RMSE"] == pytest.approx(np.sqrt(3))
    assert result["n"] == 3 and result["mape_n"] == 2
    assert forecast_metrics(np.zeros(2), np.ones(2))["WAPE"] is None


def test_metrics_reject_invalid_shapes_and_nan():
    with pytest.raises(ValueError):
        forecast_metrics(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        forecast_metrics(np.array([np.nan]), np.array([1]))


def test_seasonal_naive_repeats_only_observed_cycle(batch):
    model = train_model("seasonal_naive", batch, {"period": 8}).model
    result = backtest(model, batch, batch_size=3)
    np.testing.assert_array_equal(result.predictions, batch.targets)
    assert result.metrics["WAPE"] == 0
    assert len(result.per_lead) == 12
    table = prediction_table(batch, result.predictions)
    assert len(table) == result.metrics["n"]
    assert table.lead.max() == 12


@pytest.mark.parametrize("name,params", [("naive", {}), ("ridge", {"alpha": 1}), ("extra_trees", {"n_estimators": 4, "max_depth": 3, "n_jobs": 1})])
def test_default_models_fit_forecast_and_save(batch, tmp_path, name, params):
    import joblib
    model = train_model(name, batch, params).model
    result = backtest(model, batch)
    assert result.predictions.shape == batch.targets.shape
    restored = joblib.load(model.save(tmp_path / name))
    np.testing.assert_allclose(restored.predict(batch.inputs), result.predictions)


def test_invalid_model_does_not_fall_back_to_baseline():
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("missing")


def test_archived_models_are_not_active_fallbacks(batch):
    with pytest.raises(ValueError, match="archived"):
        train_model("hdsf", batch)
