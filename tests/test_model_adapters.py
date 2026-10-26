"""Real-weight integration tests for the frozen foundation adapters (A1/A2) and A3.

These are integration-style tests, not mocks: they load the **real** frozen
Chronos-2 / TiRex-2 checkpoints from the local offline cache and assert the frozen
contract of the Stage-2 main path.  They are skipped (not faked) when the offline
weights cache is absent, and they are kept small in wall-clock cost by sharing
module-scoped adapter fixtures.

Covered contract
----------------
* offline checkpoint resolution: snapshot path, revision, file SHA256, no download;
* lazy loading (checkpoint hashed at construction, tensors loaded on first predict);
* ``[B, H]`` shape / float64 dtype / finiteness / plausible kW magnitude for
  ``H = 16, 32, 96, 288`` with ``L = 1344``;
* determinism of repeated calls (<= 1e-6) and batch invariance (a series predicted
  alone equals the same series inside a two-row batch);
* TiRex-2 long-horizon recursion at ``H = 672``: documented block size / hops /
  predicted-medians-only context update, first native block identical to a direct
  ``H = 320`` forecast, and no API path for true future values;
* P50 policy (Chronos-2 quantile 0.5; TiRex-2 median quantile index 4);
* insufficient / non-finite / ragged context and invalid horizons raise clearly;
* a missing or partial weights cache raises ``FileNotFoundError`` without any
  download attempt.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.models.foundation import (
    Chronos2Forecaster,
    TiRex2Forecaster,
    resolve_checkpoint,
)
from src.models.power_main import build_power_forecasters
from src.models.power_main.ensemble import build_power_ensemble
from src.utils import load_config

#: Local frozen HF_HUB_CACHE-layout weights directory (prepared outside the run).
WEIGHTS_CACHE = Path(
    os.environ.get(
        "POWERFORECAST_WEIGHTS_CACHE",
        "D:/AAA_Project/PowerData_Prediction/.model-cache",
    )
)
FOUNDATION_CONFIG = "configs/models/foundation.yaml"
EXPERIMENT_CONFIG = "configs/experiments/power_main_v1.yaml"
CONTEXT_POINTS = 1344
HORIZONS = (16, 32, 96, 288)

pytestmark = pytest.mark.skipif(
    not WEIGHTS_CACHE.is_dir(),
    reason=(
        f"frozen foundation weights cache not available at {WEIGHTS_CACHE}; "
        "set POWERFORECAST_WEIGHTS_CACHE to run the real-weight adapter tests"
    ),
)


def make_context(batch: int = 2, points: int = CONTEXT_POINTS) -> np.ndarray:
    """Deterministic, noise-free ``[B, L]`` kW context (two distinct series)."""
    slots = np.arange(points, dtype=np.float64)
    day = np.sin(2.0 * np.pi * slots / 96.0)
    week = np.sin(2.0 * np.pi * slots / (7.0 * 96.0))
    levels = np.array([8000.0, 12000.0][:batch], dtype=np.float64)
    return np.stack([level + 400.0 * day + 150.0 * week for level in levels])


@pytest.fixture(scope="module")
def chronos2() -> Chronos2Forecaster:
    return Chronos2Forecaster(weights_cache=WEIGHTS_CACHE)


@pytest.fixture(scope="module")
def tirex2() -> TiRex2Forecaster:
    return TiRex2Forecaster(weights_cache=WEIGHTS_CACHE)


# ---------------------------------------------------------------------------
# offline checkpoint evidence + lazy loading
# ---------------------------------------------------------------------------


def test_offline_checkpoint_evidence_and_lazy_loading(
    chronos2: Chronos2Forecaster, tirex2: TiRex2Forecaster
) -> None:
    context = make_context()
    for adapter, model_id in ((chronos2, "amazon/chronos-2"), (tirex2, "NX-AI/TiRex-2")):
        # Weights are NOT loaded before the first predict: the builder/hash step only
        # resolved and hashed the frozen checkpoint files.
        assert adapter.load_seconds is None
        assert adapter.checkpoint.model_id == model_id
        assert adapter.checkpoint.source == "local_cache"
        assert adapter.checkpoint.snapshot_dir.is_dir()
        assert adapter.checkpoint.revision == adapter.checkpoint.snapshot_dir.name
        assert set(adapter.checkpoint.files)
        weight_files = {"model.safetensors", "model.ckpt"}
        assert sum(adapter.checkpoint.file_bytes.values()) > 100_000_000
        for name in adapter.checkpoint.files:
            assert len(adapter.checkpoint.file_sha256[name]) == 64
            if name in weight_files:
                assert adapter.checkpoint.file_bytes[name] > 100_000_000
        assert weight_files & set(adapter.checkpoint.files)
        description = adapter.describe()
        assert description["name"] == adapter.name
        assert description["requires_fit"] is False
        assert description["device"] == "cpu"
        assert description["dtype"] == "float32"
        assert description["checkpoint"]["snapshot_dir"] == str(adapter.checkpoint.snapshot_dir)
        assert "lazy" in description["loading"]

    # The first real call loads the weights and records the load time.
    first = np.asarray(chronos2.predict(context, 16))
    assert first.shape == (2, 16)
    assert chronos2.load_seconds is not None and chronos2.load_seconds > 0.0
    assert chronos2.last_inference_seconds is not None
    assert tirex2.predict(context, 16).shape == (2, 16)
    assert tirex2.load_seconds is not None and tirex2.load_seconds > 0.0


# ---------------------------------------------------------------------------
# shapes / finiteness / plausibility / determinism
# ---------------------------------------------------------------------------


def test_predict_shapes_dtype_finite_plausible_kw(
    chronos2: Chronos2Forecaster, tirex2: TiRex2Forecaster
) -> None:
    context = make_context()
    for adapter in (chronos2, tirex2):
        for horizon in HORIZONS:
            forecast = np.asarray(adapter.predict(context, horizon))
            assert forecast.shape == (2, horizon), (adapter.name, horizon, forecast.shape)
            assert forecast.dtype == np.float64
            assert np.isfinite(forecast).all()
            assert not np.isnan(forecast).any()
            # Synthetic context is ~8-12 MW, so predictions must be in a similar kW range.
            assert 100.0 < float(forecast.mean()) < 100_000.0
            assert float(forecast.min()) > -1_000_000.0


def test_repeated_calls_are_deterministic(
    chronos2: Chronos2Forecaster, tirex2: TiRex2Forecaster
) -> None:
    context = make_context()
    for adapter in (chronos2, tirex2):
        first = np.asarray(adapter.predict(context, 96))
        second = np.asarray(adapter.predict(context, 96))
        assert np.array_equal(first, second), adapter.name
        assert float(np.max(np.abs(first - second))) <= 1e-6


def test_batch_invariance(chronos2: Chronos2Forecaster, tirex2: TiRex2Forecaster) -> None:
    """A series predicted alone == the same series inside a two-row batch."""
    context = make_context()
    alone_context = context[[0]]
    for adapter in (chronos2, tirex2):
        alone = np.asarray(adapter.predict(alone_context, 96))
        batched = np.asarray(adapter.predict(context, 96))
        assert alone.shape == (1, 96) and batched.shape == (2, 96)
        # Default batch_size=1 processes rows individually -> bit-identical here.
        assert np.array_equal(alone[0], batched[0]), adapter.name
        np.testing.assert_allclose(alone[0], batched[0], rtol=1e-6, atol=1e-6)
        # The second series must be unaffected by the first one as well.
        assert np.isfinite(batched[1]).all()


# ---------------------------------------------------------------------------
# TiRex-2 long-horizon recursion
# ---------------------------------------------------------------------------


def test_tirex2_long_horizon_recursion_uses_predicted_medians_only(
    tirex2: TiRex2Forecaster,
) -> None:
    context = make_context()
    native = int(tirex2.native_horizon)
    assert native == 320

    long_forecast = np.asarray(tirex2.predict(context, 672))
    assert long_forecast.shape == (2, 672)
    assert np.isfinite(long_forecast).all()

    recursion = tirex2.describe()["recursion"]
    assert recursion["used"] is True
    assert recursion["block_size"] == 320
    assert recursion["hops"] == 2  # 320 + 320 + 32 -> two continuation hops
    assert recursion["context_update"] == "predicted_median_only"
    assert recursion["blocks"] == 3
    policy = tirex2.describe()["recursion_policy"]
    assert "never consumed" in policy["true_future_values"]

    # The first native block is exactly a plain 320-step forecast from the same
    # context: the rollout only continues after that first block.
    direct_native = np.asarray(tirex2.predict(context, native))
    assert np.array_equal(long_forecast[:, :native], direct_native)

    # There is structurally no way to feed true future values into predict().
    future = np.full((2, 672), 1234.5)
    with pytest.raises(TypeError):
        tirex2.predict(context, 672, future_kw=future)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# P50 policy
# ---------------------------------------------------------------------------


def test_p50_policy_is_reported_and_verified(
    chronos2: Chronos2Forecaster, tirex2: TiRex2Forecaster
) -> None:
    context = make_context()
    chronos2.predict(context, 16)
    policy = chronos2.describe()["point_policy"]
    assert policy["quantile"] == 0.5
    assert "quantile_levels=[0.5]" in policy["selection"]
    assert policy["cross_learning"] is False

    tirex2.predict(context, 16)
    policy = tirex2.describe()["point_policy"]
    assert policy["median_index"] == 4
    assert policy["quantile_levels"] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]
    assert abs(float(policy["quantile_levels"][policy["median_index"]]) - 0.5) <= 1e-6
    assert abs(float(policy["verified_median_quantile_level"]) - 0.5) <= 1e-6
    assert tirex2.native_horizon == 320


# ---------------------------------------------------------------------------
# invalid inputs
# ---------------------------------------------------------------------------


def test_insufficient_context_is_rejected(
    chronos2: Chronos2Forecaster, tirex2: TiRex2Forecaster
) -> None:
    short = make_context(points=10)
    for adapter in (chronos2, tirex2):
        with pytest.raises(ValueError, match="at least 96 context points"):
            adapter.predict(short, 16)


def test_non_finite_context_is_rejected(
    chronos2: Chronos2Forecaster, tirex2: TiRex2Forecaster
) -> None:
    context = make_context()
    for adapter in (chronos2, tirex2):
        for bad_value in (np.nan, np.inf, -np.inf):
            damaged = context.copy()
            damaged[1, 500] = bad_value
            with pytest.raises(ValueError, match="non-finite context"):
                adapter.predict(damaged, 16)


def test_malformed_context_and_horizon_are_rejected(
    chronos2: Chronos2Forecaster, tirex2: TiRex2Forecaster
) -> None:
    context = make_context()
    for adapter in (chronos2, tirex2):
        with pytest.raises(ValueError, match=r"2-D \[B, L\]"):
            adapter.predict(context[0], 16)  # 1-D
        with pytest.raises(ValueError, match="ragged|dense numeric"):
            adapter.predict([[1.0, 2.0], [3.0]], 16)  # ragged rows
        with pytest.raises(ValueError, match="integer horizon"):
            adapter.predict(context, 16.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="horizon >= 1"):
            adapter.predict(context, 0)
        with pytest.raises(ValueError, match="horizon >= 1"):
            adapter.predict(context, -5)


# ---------------------------------------------------------------------------
# missing weights: explicit failure, never a download or random state
# ---------------------------------------------------------------------------


def test_missing_weights_cache_fails_explicitly(tmp_path: Path) -> None:
    empty_cache = tmp_path / "empty-hf-cache"
    empty_cache.mkdir()
    for model_id, filenames in (
        ("amazon/chronos-2", ("config.json", "model.safetensors")),
        ("NX-AI/TiRex-2", ("model-config.yaml", "model.ckpt")),
    ):
        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_checkpoint(model_id, empty_cache, filenames, allow_download=False)
        message = str(excinfo.value)
        assert model_id in message
        assert str(empty_cache) in message
        assert "Downloads are disabled" in message
        assert "No random, partial or substituted weights" in message

    with pytest.raises(FileNotFoundError):
        Chronos2Forecaster(weights_cache=empty_cache)
    with pytest.raises(FileNotFoundError):
        TiRex2Forecaster(weights_cache=empty_cache)


def test_partial_weights_cache_fails_explicitly(tmp_path: Path) -> None:
    """A snapshot with only one of the two required files must be rejected."""
    cache = tmp_path / "partial-hf-cache"
    snapshot = cache / "models--amazon--chronos-2" / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    refs = cache / "models--amazon--chronos-2" / "refs"
    refs.mkdir()
    (refs / "main").write_text("deadbeef", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_checkpoint(
            "amazon/chronos-2",
            cache,
            ("config.json", "model.safetensors"),
            allow_download=False,
        )
    assert "model.safetensors" in str(excinfo.value)


# ---------------------------------------------------------------------------
# builder contract
# ---------------------------------------------------------------------------


def test_builder_exposes_the_five_stage2_models_without_loading_weights() -> None:
    config = load_config(FOUNDATION_CONFIG)
    forecasters = build_power_forecasters(config)
    assert list(forecasters) == [
        "seasonal_naive",
        "three_day_mean",
        "chronos2",
        "tirex2",
        "power_ensemble",
    ]
    for name, model in forecasters.items():
        assert model.name == name
        assert model.requires_fit is False
        assert callable(model.predict)
        assert callable(model.describe)
    # Foundation weights are not loaded by the builder: only resolved and hashed.
    assert forecasters["chronos2"].load_seconds is None
    assert forecasters["tirex2"].load_seconds is None


def test_builder_from_experiment_config_does_not_load_weights() -> None:
    config = load_config(EXPERIMENT_CONFIG)
    forecasters = build_power_forecasters(
        config, models=["seasonal_naive", "power_ensemble"]
    )
    assert list(forecasters) == ["seasonal_naive", "chronos2", "tirex2", "power_ensemble"]
    assert forecasters["chronos2"].load_seconds is None
    assert forecasters["tirex2"].load_seconds is None
    ensemble = forecasters["power_ensemble"]
    assert ensemble.requires_fit is False
    assert list(ensemble.weights) == [0.5, 0.5]
    description = ensemble.describe()
    assert description["weights"] == [0.5, 0.5]
    assert description["fixed_weight_policy"]["search_allowed"] is False
    assert "NOT a main/auxiliary" in description["weight_scope"]


def test_real_ensemble_is_the_exact_mean_of_the_two_members(
    chronos2: Chronos2Forecaster, tirex2: TiRex2Forecaster
) -> None:
    """A3 built from the already-loaded adapters equals the exact 0.5/0.5 mean."""
    ensemble = build_power_ensemble({"chronos2": chronos2, "tirex2": tirex2})
    context = make_context(batch=1, points=CONTEXT_POINTS)
    ensemble_forecast = np.asarray(ensemble.predict(context, 16))
    expected = 0.5 * np.asarray(chronos2.predict(context, 16)) + 0.5 * np.asarray(
        tirex2.predict(context, 16)
    )
    assert np.array_equal(ensemble_forecast, expected)
    np.testing.assert_allclose(ensemble_forecast, expected, rtol=0.0, atol=0.0)
