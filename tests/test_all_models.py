import numpy as np
import pandas as pd
import pytest

from src.data.splitting import ForecastInputs
from src.models import COMPARISON_MODELS, create_model
from src.utils import load_config


def test_all_requested_families_are_registered_and_in_default_configs():
    assert len(COMPARISON_MODELS) == 18
    for path in ("configs/models.yaml", "configs/hdsf_models.yaml"):
        config = load_config(path)
        assert set(COMPARISON_MODELS) <= set(config["default_models"])
        assert set(COMPARISON_MODELS) <= set(config["models"])


def test_batched_sarima_matches_stock_kalman_forecasts_and_isolates_samples():
    from threadpoolctl import threadpool_limits

    model = create_model("sarima", {"seasonal_order": (1, 0, 0, 8)})
    model.params = np.array([.8, .1, .1, .001])
    model._plans = {}
    x = .7 + .04*np.sin(np.arange(48)*2*np.pi/8)
    histories = np.stack([x, .8*x+.01, x[::-1]])[:, :, None]
    inputs = ForecastInputs(histories, np.zeros((3, 16, 1)),
        pd.date_range("2024-01-01", periods=3, tz="UTC"), np.zeros((3, 16)))
    predictions = model.predict(inputs)
    with threadpool_limits(limits=1):
        reference = np.stack([model._spec(h[:, 0]).filter(model.params).forecast(16) for h in histories])
    np.testing.assert_allclose(predictions, reference, rtol=0, atol=1e-12)
    changed = histories.copy()
    changed[1:] = 1000
    poisoned = ForecastInputs(changed, inputs.future_known, inputs.origins, inputs.target_times)
    np.testing.assert_array_equal(model.predict(poisoned)[0], predictions[0])
    # Checkpoints saved before the batched filter have no cached-plan attribute.
    del model._plans
    np.testing.assert_allclose(model.predict(inputs), reference, rtol=0, atol=1e-12)


def test_google_training_windows_obey_cutoff_and_domain_boundaries():
    from types import SimpleNamespace
    from src.data.google_panel import training_panel

    values = np.arange(31*96).reshape(1, 31, 96).repeat(2, axis=0).astype(float)
    values[1] += 10000
    data = SimpleNamespace(target=values, n_series=2)
    batch, ids = training_panel(data, 14, 7, 672)
    assert len(batch) == 2
    np.testing.assert_array_equal(ids, [[0, 7], [1, 7]])
    np.testing.assert_array_equal(batch.targets[0], values[0, 7:14].reshape(-1))
    np.testing.assert_array_equal(batch.inputs.history[1, :, 0], values[1, :7].reshape(-1))
    before = batch.targets.copy()
    data.target[:, 14:] = 999999
    changed, _ = training_panel(data, 14, 7, 672)
    np.testing.assert_array_equal(before, changed.targets)


@pytest.mark.parametrize("name", ["lstm", "tcn", "dlinear"])
def test_deep_model_checkpoint_restores_identical_predictions(tmp_path, name):
    from src.data.splitting import WindowBatch
    from src.models.deep.forecasters import TorchForecaster

    rng = np.random.default_rng(5)
    inputs = ForecastInputs(rng.normal(size=(4, 16, 2)).astype(np.float32), np.zeros((4, 12, 1), np.float32),
        pd.date_range("2024-01-01", periods=4, tz="UTC"), np.zeros((4, 12)))
    batch = WindowBatch(inputs, rng.normal(size=(4, 12)).astype(np.float32))
    model = TorchForecaster(name, epochs=1, hidden_size=4, kernel_size=3).fit(batch)
    restored = TorchForecaster.load(model.save(tmp_path/name))
    np.testing.assert_array_equal(model.predict(inputs), restored.predict(inputs))


def test_compact_tabular_features_preserve_all_observed_lags_and_parallel_outputs():
    from src.data.splitting import WindowBatch

    rng = np.random.default_rng(17)
    inputs = ForecastInputs(rng.normal(size=(64, 16, 2)).astype(np.float32),
        np.zeros((64, 4, 1), np.float32), pd.date_range("2024-01-01", periods=64, tz="UTC"), np.zeros((64, 4)))
    targets = inputs.history[:, -1, 0:1]*np.arange(1, 5)[None, :]
    batch = WindowBatch(inputs, targets.astype(np.float32))
    params = dict(n_estimators=4, min_child_samples=4, num_leaves=4, n_jobs=1, verbosity=-1,
                  feature_mode="observed_history")
    sequential = create_model("lightgbm", {**params, "output_jobs": 1}).fit(batch)
    parallel = create_model("lightgbm", {**params, "output_jobs": 2}).fit(batch)
    np.testing.assert_array_equal(parallel._features(inputs)[:, :16], inputs.history[:, :, 0])
    np.testing.assert_allclose(sequential.predict(inputs), parallel.predict(inputs), rtol=0, atol=1e-12)


def test_centered_innovations_sarima_matches_stock_forecast():
    from threadpoolctl import threadpool_limits
    values = .7 + .03*np.sin(np.arange(96)*2*np.pi/8) + np.random.default_rng(8).normal(0, .01, 96)
    model = create_model("sarima", {"seasonal_order": (1, 0, 0, 8), "estimation": "innovations_mle",
                                    "center": True, "maxiter": 20}).fit_series(values)
    history = np.stack([values[-48:], .9*values[-48:]])[:, :, None]
    inputs = ForecastInputs(history, np.zeros((2, 16, 1)), pd.date_range("2024-01-01", periods=2, tz="UTC"), np.zeros((2, 16)))
    predicted = model.predict(inputs)
    with threadpool_limits(limits=1):
        expected = np.stack([model._spec(row[:, 0]-model.level).filter(model.params).forecast(16)+model.level
                             for row in history])
    # Stationary covariance initialization can amplify floating-point differences
    # in the independently reconstructed gain; this is 1e-8 utilization units.
    np.testing.assert_allclose(predicted, expected, rtol=0, atol=1e-8)
