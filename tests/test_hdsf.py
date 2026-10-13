"""Independent numerical and causal checks for the transferred method."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from src.data.features import make_features
from src.data.splitting import build_windows, complete_days, adaptive_day_splits
from src.evaluation.backtest import score_predictions
from src.models.proposed.local import Local24Forecaster, LocalCurveForecaster
from src.models.proposed.simplex import fit_simplex, segment_ids, purged_folds


def test_simplex_agrees_with_independent_linear_program():
    rng = np.random.default_rng(17)
    P = rng.normal(size=(57, 2))
    y = .23*P[:, 0]+.77*P[:, 1]+rng.normal(0, .1, 57)
    weights = fit_simplex(P, y)
    n = len(y)
    objective = np.r_[np.zeros(2), np.ones(n)]
    constraints = np.r_[np.c_[P, -np.eye(n)], np.c_[-P, -np.eye(n)]]
    result = linprog(objective, A_ub=constraints, b_ub=np.r_[y, -y],
                     A_eq=np.r_[np.ones(2), np.zeros(n)][None], b_eq=[1.], bounds=[(0., None)]*(n+2), method="highs")
    assert result.success
    assert (weights >= 0).all() and weights.sum() == pytest.approx(1.)
    assert np.abs(P@weights-y).sum() == pytest.approx(result.fun)


def test_half_day_segments_and_adaptive_boundaries():
    np.testing.assert_array_equal(segment_ids(96), np.repeat([0, 1], 48))
    np.testing.assert_array_equal(segment_ids(672), np.repeat(np.arange(14), 48))
    assert adaptive_day_splits(31) == {"train": (0, 1344), "validation": (1344, 2112), "test": (2112, 2976)}
    assert adaptive_day_splits(120) == {"train": (0, 6912), "validation": (6912, 9216), "test": (9216, 11520)}


def test_partial_days_trim_without_compressing_missing_interior():
    index = pd.date_range("2024-01-01 06:00", "2024-02-03 04:30", freq="15min", tz="UTC")
    frame = pd.DataFrame({"load": 1.}, index=index)
    frame.iloc[200, 0] = np.nan
    result, report = complete_days(frame, "UTC")
    assert result.index[0].hour == 0 and result.index[-1].hour == 23
    assert len(result) == 32*96
    assert result.load.isna().sum() == 1
    assert report["dropped_edge_rows"] == len(frame)-len(result)


def test_seven_day_cv_purges_every_overlapping_target():
    origins = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    times = origins.asi8
    for mode in ("loo", "blocked"):
        for fit, held in purged_folds(origins, 672, folds=5, mode=mode):
            for i in fit:
                for j in held:
                    assert abs(times[i]-times[j]) >= 7*86400*1_000_000_000


def load_batch(days=45, horizon=96):
    index = pd.date_range("2024-01-01", periods=days*96, freq="15min", tz="UTC")
    frame = pd.DataFrame({"load": 100+np.sin(np.arange(len(index))*2*np.pi/96)}, index=index)
    history, known = make_features(frame, "load", [], "UTC")
    return frame, history, known, build_windows(history, known, frame.load, (0, len(frame)), 1344, horizon, 96)


def test_target_unavailable_edges_trim_without_dropping_interior_missing_days():
    index = pd.date_range("2024-01-01", periods=35*96, freq="15min", tz="UTC")
    frame = pd.DataFrame({"load": 100., "other": 1.}, index=index)
    frame.iloc[:3*96, 0] = np.nan
    frame.iloc[-96:, 0] = np.nan
    frame.iloc[8*96, 0] = np.nan
    result, info = complete_days(frame, "UTC", "load")
    assert len(result) == 31*96
    assert result.index[0] == pd.Timestamp("2024-01-04", tz="UTC")
    assert result.load.isna().sum() == 1
    assert info["target_edge_rows_dropped"] == 4*96


def test_local_curve_uses_only_observed_input_and_handles_zero_days():
    frame, history, known, batch = load_batch(horizon=672)
    model = LocalCurveForecaster().fit(batch)
    original = model.predict(batch.inputs.take(slice(0, 1)))
    origin = batch.inputs.origins[0]
    frame.loc[origin:, "load"] = 1e9
    changed_history, changed_known = make_features(frame, "load", [], "UTC")
    changed = build_windows(changed_history, changed_known, frame.load, (1344, 2016), 1344, 672, 96)
    np.testing.assert_array_equal(model.predict(changed.inputs), original)
    zeros = type(batch.inputs)(np.zeros_like(batch.inputs.history), batch.inputs.future_known,
                              batch.inputs.origins, batch.inputs.target_times)
    assert np.isfinite(model.predict(zeros)).all()
    assert np.count_nonzero(model.predict(zeros)) == 0


def test_local24_history_only_and_serialization(tmp_path):
    pytest.importorskip("lightgbm")
    import joblib
    _, _, _, batch = load_batch()
    model = Local24Forecaster(rounds=3, seeds=(101,), jobs=1).fit(batch)
    original = model.predict(batch.inputs)
    restored = joblib.load(model.save(tmp_path/"local24"))
    np.testing.assert_allclose(restored.predict(batch.inputs), original)
    assert not hasattr(batch.inputs, "targets")


def test_daily_metrics_aggregate_before_scoring_and_nonoverlap():
    _, _, _, batch = load_batch(horizon=672)
    noise = np.tile(np.r_[np.ones(48)*10, np.ones(48)*-10], 7)
    result = score_predictions(batch, batch.targets+noise, anchor=batch.inputs.origins[0])
    assert result.metrics["MAPE"] > 9
    assert result.daily_metrics["MAPE"] < 1e-4
    assert result.daily_metrics["n"] == len(batch)*7
    origins = pd.to_datetime(result.nonoverlap_metrics["origins"])
    assert (np.diff(origins.asi8) >= 7*86400*1_000_000_000).all()


def test_fusion_configuration_unchanged_by_test_label_pollution(monkeypatch):
    pytest.importorskip("lightgbm")
    from types import SimpleNamespace
    from src.models.proposed import hdsf
    from src.data.splitting import WindowBatch

    class FakeFoundation:
        def __init__(self, name, **kwargs):
            self.name = name
            self.cache = SimpleNamespace(fingerprint=b'{}')

        def fit(self, batch):
            return self

        def predict(self, inputs):
            cycle = inputs.history[:, -96:, 0]
            bias = (sum(map(ord, self.name)) % 7 - 3)*.1
            return np.tile(cycle, (1, inputs.future_known.shape[1]//96))+bias

    monkeypatch.setattr(hdsf, "FoundationForecaster", FakeFoundation)
    frame, history, known, _ = load_batch(days=120)
    train = build_windows(history, known, frame.load, (0, 72*96), 1344, 96, 96)
    validation = build_windows(history, known, frame.load, (72*96, 96*96), 1344, 96, 96)
    first = hdsf.HDSFForecaster(weights_cache="unused", local_rounds=2).fit(train).calibrate(validation)
    frame.iloc[96*96:, 0] = 1e12
    changed_history, changed_known = make_features(frame, "load", [], "UTC")
    changed_train = build_windows(changed_history, changed_known, frame.load, (0, 72*96), 1344, 96, 96)
    changed_validation = build_windows(changed_history, changed_known, frame.load, (72*96, 96*96), 1344, 96, 96)
    second = hdsf.HDSFForecaster(weights_cache="unused", local_rounds=2).fit(changed_train).calibrate(changed_validation)
    assert first.configuration == second.configuration
    np.testing.assert_allclose(first.weights_b.sum(1), 1.)
    assert (first.weights_b >= 0.).all()
    with pytest.raises(RuntimeError, match="frozen"):
        first.fit(train)
