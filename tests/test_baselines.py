"""Tests for the two real baselines, registry status, metrics and scoring.

Covers the acceptance items owned by this slice:

* T16 -- exact numerical baseline behaviour (previous-day same-slot profile;
  three-day **slot-wise** mean, not a constant mean);
* T09 -- every supported horizon ``H`` produces ``[B, H]``, including pure
  power-only inputs (``F=0``, ``K=0``);
* planned interfaces are ``NotImplementedError`` only and planned names can
  never be built/supported;
* T17 -- metrics recomputation with the documented near-zero MAPE policy;
* T18 -- multi-level scoring: 7d per-day rows and overlap flag, 4h/8h never
  labelled as a daily mean.

All data is built in memory from :mod:`src.data.contracts`; these tests do not
depend on synthetic-data or window-builder modules owned by other slices.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.contracts import (
    ContractError,
    ForecastBatch,
    ForecastInputs,
    InsufficientDataError,
    LabelOrigin,
    TargetKind,
    TargetScope,
    TargetSpec,
)
from src.evaluation.backtest import score_batch
from src.evaluation.metrics import forecast_metrics
from src.models import (
    MODEL_REGISTRY,
    ModelStatus,
    SeasonalNaiveForecaster,
    ThreeDayMeanForecaster,
    build_forecaster,
    describe_models,
    implemented_models,
    load_model_config,
    planned_models,
)
from src.models.fusion import PlannedFusion
from src.models.power_backbone import PlannedPowerBackbone
from src.models.resource_encoder import PlannedResourceEncoder
from src.models.target_mapping import PlannedTargetMapper
from src.training.runner import predict_batch

POINTS_PER_DAY = 96
ANCHOR = pd.Timestamp("2026-01-01T00:00:00")
SLOT = np.arange(POINTS_PER_DAY, dtype=float)
#: A non-constant, exactly known 96-point profile (no noise anywhere).
BASE_PROFILE = 120.0 + 40.0 * np.sin(2.0 * np.pi * SLOT / POINTS_PER_DAY) + 0.5 * SLOT

TARGET_SPEC = TargetSpec(
    kind=TargetKind.ACTIVE_POWER,
    scope=TargetScope.FACILITY,
    unit="kW",
    label_origin=LabelOrigin.SYNTHETIC_SIMULATED,
    measurement_boundary="synthetic facility main meter",
    sampling_semantics="interval_mean_15min",
    synthetic=True,
)


# ---------------------------------------------------------------------------
# in-memory batch helpers (contracts only, no synthetic/windows dependency)
# ---------------------------------------------------------------------------


def _profiles(days: int) -> list[np.ndarray]:
    """One distinct noiseless 96-point profile per trace day."""
    return [BASE_PROFILE + 3.0 * day for day in range(days)]


def _series(days: int) -> np.ndarray:
    return np.concatenate(_profiles(days))


def _grid(points: int) -> np.ndarray:
    return pd.date_range(ANCHOR, periods=points, freq="15min").to_numpy(dtype="datetime64[ns]")


def _make_inputs(
    series: np.ndarray,
    horizon: int,
    *,
    samples: int = 1,
    valid: np.ndarray | None = None,
) -> ForecastInputs:
    series = np.asarray(series, dtype=float)
    length = series.shape[0]
    past = np.tile(series, (samples, 1))
    if valid is None:
        validity = np.ones((samples, length), dtype=bool)
    else:
        validity = np.asarray(valid, dtype=bool)
        assert validity.shape == (length,), "valid mask must cover the whole series"
        validity = np.tile(validity, (samples, 1))
    grid = _grid(length + horizon)
    return ForecastInputs(
        past_power_kw=past,
        past_power_valid=validity,
        past_timestamps=np.tile(grid[:length], (samples, 1)),
        origin_timestamps=np.repeat(grid[length], samples),
        target_timestamps=np.tile(grid[length:], (samples, 1)),
        target_spec=TARGET_SPEC,
        domain_ids=tuple(f"synthetic_domain_{index}" for index in range(samples)),
        dataset_id="synthetic_facility",
    )


def _make_inputs_at(
    series: np.ndarray,
    horizon: int,
    origin_indices: list[int],
    context_points: int,
) -> ForecastInputs:
    series = np.asarray(series, dtype=float)
    grid = _grid(series.shape[0])
    past = np.stack([series[index - context_points : index] for index in origin_indices])
    validity = np.ones_like(past, dtype=bool)
    return ForecastInputs(
        past_power_kw=past,
        past_power_valid=validity,
        past_timestamps=np.stack([grid[index - context_points : index] for index in origin_indices]),
        origin_timestamps=np.array(
            [grid[index] for index in origin_indices], dtype="datetime64[ns]"
        ),
        target_timestamps=np.stack([grid[index : index + horizon] for index in origin_indices]),
        target_spec=TARGET_SPEC,
        domain_ids=tuple(f"synthetic_domain_{index}" for index in range(len(origin_indices))),
        dataset_id="synthetic_facility",
    )


def _tile_profile(profile: np.ndarray, horizon: int) -> np.ndarray:
    repeats = int(np.ceil(horizon / profile.size))
    return np.tile(profile, repeats)[:horizon]


def _future_values(horizon: int) -> np.ndarray:
    return _tile_profile(BASE_PROFILE, horizon)


def _make_batch(series: np.ndarray, horizon: int, *, samples: int = 1) -> ForecastBatch:
    inputs = _make_inputs(series, horizon, samples=samples)
    targets = np.tile(_future_values(horizon), (samples, 1))
    return ForecastBatch(
        inputs=inputs,
        targets_kw=targets,
        target_valid_mask=np.ones_like(targets, dtype=bool),
    )


# ---------------------------------------------------------------------------
# T16: exact baseline numerics
# ---------------------------------------------------------------------------


def test_seasonal_naive_reproduces_previous_day_same_slot_exactly() -> None:
    profiles = _profiles(5)
    series = np.concatenate(profiles)
    model = SeasonalNaiveForecaster()

    for horizon in (16, 96, 288, 672):
        inputs = _make_inputs(series, horizon)
        predicted = model.predict(inputs)
        expected = _tile_profile(profiles[-1], horizon)
        assert predicted.shape == (1, horizon)
        assert predicted.dtype == np.float64
        np.testing.assert_array_equal(predicted, expected[None, :])
        # Sanity: this profile is not a constant, so the equality above is meaningful.
        assert float(np.ptp(profiles[-1])) > 1.0
        assert not np.allclose(predicted[0], float(profiles[-1].mean()))


def test_three_day_mean_is_slotwise_mean_not_constant_mean() -> None:
    profiles = _profiles(5)
    series = np.concatenate(profiles)
    model = ThreeDayMeanForecaster()
    expected_profile = np.mean(np.stack(profiles[-3:]), axis=0)
    constant_mean = float(series[-288:].mean())

    for horizon in (16, 96, 288, 672):
        inputs = _make_inputs(series, horizon)
        predicted = model.predict(inputs)
        expected = _tile_profile(expected_profile, horizon)
        assert predicted.shape == (1, horizon)
        np.testing.assert_allclose(predicted, expected[None, :], rtol=0.0, atol=1e-12)
        # Same 96 slots averaged across the previous three days -- not one constant.
        assert not np.allclose(predicted[0], constant_mean)
        assert float(np.ptp(predicted[0])) > 1.0
        # Slot 0 must be the mean of the same slot on the last three days.
        assert predicted[0, 0] == pytest.approx(
            float(np.mean([profiles[-3][0], profiles[-2][0], profiles[-1][0]]))
        )
        # Slot 1 differs from slot 0 because the profile varies by slot.
        assert predicted[0, 0] != pytest.approx(predicted[0, 1])


def test_baselines_raise_insufficient_data_for_short_or_incomplete_context() -> None:
    series = _series(5)

    with pytest.raises(InsufficientDataError, match="context_length"):
        SeasonalNaiveForecaster().predict(_make_inputs(series[-95:], 16))
    with pytest.raises(InsufficientDataError, match="context_length"):
        ThreeDayMeanForecaster().predict(_make_inputs(series[-287:], 16))

    # Total length is enough, but the trailing window contains a missing point.
    naive_mask = np.ones(series.size, dtype=bool)
    naive_mask[-5] = False
    with pytest.raises(InsufficientDataError, match="invalid"):
        SeasonalNaiveForecaster().predict(_make_inputs(series, 16, valid=naive_mask))

    three_day_mask = np.ones(series.size, dtype=bool)
    three_day_mask[-250] = False  # inside the required trailing 288 points
    with pytest.raises(InsufficientDataError, match="invalid"):
        ThreeDayMeanForecaster().predict(_make_inputs(series, 16, valid=three_day_mask))

    # A gap outside the required trailing window is legal (missing power is never
    # silently treated as 0 kW, but it also does not invalidate the window).
    old_gap = np.ones(series.size, dtype=bool)
    old_gap[:20] = False
    assert SeasonalNaiveForecaster().predict(_make_inputs(series, 16, valid=old_gap)).shape == (1, 16)
    assert ThreeDayMeanForecaster().predict(_make_inputs(series, 16, valid=old_gap)).shape == (1, 16)


def test_fit_is_a_documented_no_op_for_parameter_free_baselines() -> None:
    series = _series(5)
    batch = _make_batch(series, 96)
    for model in (SeasonalNaiveForecaster(), ThreeDayMeanForecaster()):
        assert model.requires_fit is False
        assert model.fit(batch) is model
        with pytest.raises(ContractError):
            model.fit(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T09: all horizons, pure-power (F=0, K=0) inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [16, 32, 96, 288, 672])
def test_all_horizons_return_correct_shape_without_covariates(horizon: int) -> None:
    profiles = _profiles(5)
    series = np.concatenate(profiles)
    inputs = _make_inputs(series, horizon, samples=3)
    assert inputs.n_covariates == 0
    assert inputs.n_future_known == 0

    expected_seasonal = _tile_profile(profiles[-1], horizon)
    expected_three_day = _tile_profile(np.mean(np.stack(profiles[-3:]), axis=0), horizon)

    for model, expected in (
        (SeasonalNaiveForecaster(), expected_seasonal),
        (ThreeDayMeanForecaster(), expected_three_day),
    ):
        predicted = model.predict(inputs)
        assert predicted.shape == (3, horizon)
        assert predicted.dtype == np.float64
        assert bool(np.isfinite(predicted).all())
        for sample in range(3):
            np.testing.assert_allclose(predicted[sample], expected, rtol=0.0, atol=1e-12)


def test_baseline_save_and_load_roundtrip(tmp_path) -> None:
    series = _series(5)
    inputs = _make_inputs(series, 96)
    for model in (SeasonalNaiveForecaster(), ThreeDayMeanForecaster()):
        destination = model.save(tmp_path / f"{model.name}_state.json")
        assert destination.is_file()
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["model_type"] == model.name
        assert payload["version"] == model.version
        assert payload["requires_fit"] is False
        loaded = type(model).load(destination)
        assert loaded.name == model.name
        np.testing.assert_array_equal(model.predict(inputs), loaded.predict(inputs))

    with pytest.raises(ContractError):
        SeasonalNaiveForecaster.load(tmp_path / "missing_state.json")
    wrong_type = ThreeDayMeanForecaster.save(
        ThreeDayMeanForecaster(), tmp_path / "three_day.json"
    )
    with pytest.raises(ContractError):
        SeasonalNaiveForecaster.load(wrong_type)


# ---------------------------------------------------------------------------
# registry: exactly two implemented models, planned is never supported
# ---------------------------------------------------------------------------


def test_registry_declares_exactly_two_implemented_baselines() -> None:
    assert implemented_models() == ("seasonal_naive", "three_day_mean")
    planned = set(planned_models())
    assert {
        "lightgbm",
        "power_backbone",
        "resource_pretrain",
        "target_mapping",
        "fusion",
    } <= planned
    assert not set(implemented_models()) & planned

    for name in implemented_models():
        assert MODEL_REGISTRY[name].status is ModelStatus.IMPLEMENTED
        assert MODEL_REGISTRY[name].requires_fit is False
    for name in planned_models():
        assert MODEL_REGISTRY[name].status is ModelStatus.PLANNED
        assert name not in implemented_models()

    model = build_forecaster("seasonal_naive")
    assert model.name == "seasonal_naive"
    assert model.requires_fit is False
    three_day = build_forecaster("three_day_mean", season_length=96)
    assert three_day.describe()["status"] == "implemented"

    with pytest.raises(ContractError, match="planned"):
        build_forecaster("lightgbm")
    for planned_name in planned_models():
        with pytest.raises(ContractError, match="planned"):
            build_forecaster(planned_name)
    with pytest.raises(ContractError, match="Unknown model"):
        build_forecaster("not_a_real_model")
    with pytest.raises(ContractError):
        build_forecaster("")

    described = describe_models()
    assert described["seasonal_naive"]["status"] == "implemented"
    assert described["lightgbm"]["status"] == "planned"
    assert described["fusion"]["status"] == "planned"


def test_baselines_yaml_matches_registry_and_marks_planned_models() -> None:
    config = load_model_config()
    assert config["default_model"] == "seasonal_naive"
    assert set(config["models"]) == {"seasonal_naive", "three_day_mean", "lightgbm"}
    assert config["models"]["seasonal_naive"]["status"] == "implemented"
    assert config["models"]["three_day_mean"]["status"] == "implemented"
    assert config["models"]["lightgbm"]["status"] == "planned"
    # A planned model in the config is never promoted to supported by loading it.
    assert "lightgbm" not in implemented_models()


# ---------------------------------------------------------------------------
# planned research interfaces: explicit NotImplementedError only
# ---------------------------------------------------------------------------


def test_planned_interfaces_raise_not_implemented() -> None:
    encoder = PlannedResourceEncoder()
    assert encoder.status is ModelStatus.PLANNED
    with pytest.raises(NotImplementedError, match="TODO_RESEARCH"):
        encoder.pretrain([])
    with pytest.raises(NotImplementedError, match="TODO_RESEARCH"):
        encoder.encode(None)  # type: ignore[arg-type]

    mapper = PlannedTargetMapper()
    assert mapper.status is ModelStatus.PLANNED
    with pytest.raises(NotImplementedError, match="TODO_RESEARCH"):
        mapper.fit(None, np.zeros(4, dtype=float))  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="TODO_RESEARCH"):
        mapper.predict_aux(None, np.zeros(4, dtype=float))  # type: ignore[arg-type]

    fusion = PlannedFusion()
    assert fusion.status is ModelStatus.PLANNED
    assert fusion.describe()["global_fusion_weight"] is None
    with pytest.raises(NotImplementedError, match="TODO_RESEARCH"):
        fusion.predict(np.zeros((1, 96)), None, None)  # type: ignore[arg-type]

    backbone = PlannedPowerBackbone()
    assert backbone.status is ModelStatus.PLANNED
    with pytest.raises(NotImplementedError, match="TODO_RESEARCH"):
        backbone.predict(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="TODO_RESEARCH"):
        backbone.fit(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T17: metrics, hand-computed case and the documented near-zero policy
# ---------------------------------------------------------------------------


def test_metrics_t17_hand_computed_case() -> None:
    metrics = forecast_metrics(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
    assert tuple(metrics) == (
        "MAPE",
        "WAPE",
        "MAE",
        "RMSE",
        "n",
        "mape_threshold",
        "mape_n",
        "mape_excluded",
        "wape_defined",
        "mape_defined",
    )
    assert metrics["MAPE"] == pytest.approx(7.5)
    assert metrics["WAPE"] == pytest.approx(100.0 * 20.0 / 300.0)
    assert metrics["WAPE"] == pytest.approx(6.666666666666667)
    assert metrics["MAE"] == pytest.approx(10.0)
    assert metrics["RMSE"] == pytest.approx(10.0)
    assert metrics["n"] == 2
    assert metrics["mape_threshold"] == 0.0
    assert metrics["mape_n"] == 2
    assert metrics["mape_excluded"] == 0
    assert metrics["wape_defined"] is True
    assert metrics["mape_defined"] is True


def test_metrics_near_zero_policy_and_undefined_wape() -> None:
    # Default policy: only exactly-zero truth is excluded, nothing is replaced by 1.
    metrics = forecast_metrics(np.array([0.0, 100.0]), np.array([10.0, 110.0]))
    assert metrics["n"] == 2
    assert metrics["mape_excluded"] == 1
    assert metrics["mape_n"] == 1
    assert metrics["mape_defined"] is True
    assert metrics["MAPE"] == pytest.approx(10.0)
    assert metrics["WAPE"] == pytest.approx(100.0 * 20.0 / 100.0)

    # Scale-free relative threshold: |y| <= threshold * mean(|y|) is excluded.
    relative = forecast_metrics(
        np.array([0.4, 100.0]),
        np.array([1.4, 110.0]),
        mape_relative_threshold=0.01,
    )
    assert relative["mape_threshold"] == pytest.approx(0.01 * (100.4 / 2.0))
    assert relative["mape_threshold"] == pytest.approx(0.502)
    assert relative["mape_excluded"] == 1
    assert relative["mape_n"] == 1
    assert relative["mape_defined"] is True
    assert relative["MAPE"] == pytest.approx(10.0)

    # All truth ~zero: MAPE and WAPE are explicitly undefined, not silently 0/NaN.
    undefined = forecast_metrics(np.array([0.0, 0.0]), np.array([1.0, 2.0]))
    assert undefined["MAPE"] is None
    assert undefined["mape_defined"] is False
    assert undefined["mape_n"] == 0
    assert undefined["mape_excluded"] == 2
    assert undefined["WAPE"] is None
    assert undefined["wape_defined"] is False
    assert undefined["MAE"] == pytest.approx(1.5)
    assert undefined["RMSE"] == pytest.approx(np.sqrt(2.5))


def test_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        forecast_metrics(np.array([100.0, 200.0]), np.array([100.0]))
    with pytest.raises(ValueError):
        forecast_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        forecast_metrics(np.array([np.nan, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        forecast_metrics(np.array([1.0, 1.0]), np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        forecast_metrics(np.array([1.0]), np.array([1.0]), mape_relative_threshold=-0.1)
    with pytest.raises(ValueError):
        forecast_metrics(np.array([1.0]), np.array([1.0]), mape_relative_threshold=float("nan"))


# ---------------------------------------------------------------------------
# runner validation: a broken model output can never pass silently
# ---------------------------------------------------------------------------


class _WrongShapeModel:
    name = "wrong_shape"
    requires_fit = False

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        return np.zeros((inputs.batch_size, inputs.horizon + 1), dtype=float)


class _NonFiniteModel:
    name = "non_finite"
    requires_fit = False

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        return np.full((inputs.batch_size, inputs.horizon), np.nan)


class _NegativeModel:
    name = "negative"
    requires_fit = False

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        return np.full((inputs.batch_size, inputs.horizon), -5.0)


def test_predict_batch_rejects_wrong_shape_and_non_finite_output() -> None:
    inputs = _make_inputs(_series(5), 96, samples=2)
    with pytest.raises(ContractError, match="shape"):
        predict_batch(_WrongShapeModel(), inputs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-finite"):
        predict_batch(_NonFiniteModel(), inputs)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        predict_batch(SeasonalNaiveForecaster(), inputs, batch_size=0)


def test_predict_batch_matches_direct_predict_and_records_clipping() -> None:
    series = _series(5)
    inputs = _make_inputs(series, 96, samples=3)
    model = SeasonalNaiveForecaster()
    direct = model.predict(inputs)
    chunked, diagnostics = predict_batch(model, inputs, batch_size=1)
    np.testing.assert_array_equal(chunked, direct)
    assert diagnostics["clipped"] is False
    assert diagnostics["clip_min"] is None
    assert diagnostics["n_clipped"] == 0
    assert diagnostics["model"] == "seasonal_naive"
    assert diagnostics["requires_fit"] is False
    assert diagnostics["negative_predictions"] == 0

    clipped, clip_diagnostics = predict_batch(_NegativeModel(), inputs, clip_min=0.0)  # type: ignore[arg-type]
    assert clip_diagnostics["clipped"] is True
    assert clip_diagnostics["clip_min"] == 0.0
    assert clip_diagnostics["n_clipped"] == clipped.size
    assert clip_diagnostics["negative_predictions"] == clipped.size
    assert bool((clipped == 0.0).all())


# ---------------------------------------------------------------------------
# T18: multi-level scoring, daily-mean naming, 7d overlap
# ---------------------------------------------------------------------------


def test_score_batch_levels_and_daily_mean_naming() -> None:
    series = _series(5)

    # 24h: two distinct origins, 96 points each -> one forecast day per origin,
    # labelled daily_mean_power; distinct label windows are not overlapping.
    origin_indices = [2 * POINTS_PER_DAY, 3 * POINTS_PER_DAY]
    inputs_24h = _make_inputs_at(
        series, 96, origin_indices, context_points=2 * POINTS_PER_DAY
    )
    targets_24h = np.stack([series[index : index + 96] for index in origin_indices])
    batch_24h = ForecastBatch(
        inputs=inputs_24h,
        targets_kw=targets_24h,
        target_valid_mask=np.ones_like(targets_24h, dtype=bool),
    )
    predictions_24h = targets_24h + 5.0
    report = score_batch(batch_24h, predictions_24h)
    assert report.daily_mean_kind == "daily_mean_power"
    assert report.daily_mean is not None
    assert len(report.per_forecast_day) == 1
    assert len(report.per_lead) == 96
    assert len(report.per_origin) == 2
    assert report.overlap is None
    assert report.labels["unit"] == "kW"
    assert report.labels["scope"] == "facility"
    assert report.labels["label_origin"] == "synthetic_simulated"
    assert report.labels["synthetic"] is True
    assert report.labels["mape_threshold"] == report.aggregate["mape_threshold"]
    assert report.per_origin["origin_timestamp"].notna().all()
    assert report.aggregate["MAE"] == pytest.approx(5.0)

    # 4h: the mean of the forecast window is window_mean_power, never a daily mean.
    batch_4h = _make_batch(series, 16, samples=2)
    report_4h = score_batch(batch_4h, np.tile(_future_values(16), (2, 1)) + 2.0)
    assert report_4h.daily_mean_kind == "window_mean_power"
    assert report_4h.daily_mean is not None
    assert report_4h.daily_mean_kind != "daily_mean_power"
    assert report_4h.per_forecast_day.empty
    assert len(report_4h.per_lead) == 16

    # The JSON payload for metrics.json is JSON-safe and keeps every level.
    payload = report.to_dict()
    assert json.dumps(payload, allow_nan=False)  # raises on NaN/inf
    assert payload["horizon"] == 96
    assert payload["n_origins"] == 2
    assert payload["aggregate"]["MAPE"] == pytest.approx(report.aggregate["MAPE"])
    assert len(payload["per_lead"]) == 96
    assert payload["daily_mean_kind"] == "daily_mean_power"


def test_score_batch_7d_reports_forecast_days_and_overlap() -> None:
    full_series = _series(16)
    horizon = 672
    origin_indices = [6 * POINTS_PER_DAY, 7 * POINTS_PER_DAY]
    inputs = _make_inputs_at(full_series, horizon, origin_indices, context_points=4 * POINTS_PER_DAY)
    targets = np.stack([full_series[index : index + horizon] for index in origin_indices])
    batch = ForecastBatch(
        inputs=inputs,
        targets_kw=targets,
        target_valid_mask=np.ones_like(targets, dtype=bool),
    )
    predictions = targets + 1.0
    report = score_batch(batch, predictions)

    assert report.daily_mean_kind == "daily_mean_power"
    assert len(report.per_forecast_day) == 7
    assert list(report.per_forecast_day["forecast_day"]) == list(range(1, 8))
    assert len(report.per_origin) == 2
    assert len(report.per_lead) == horizon
    assert report.aggregate["MAE"] == pytest.approx(1.0)
    assert report.aggregate["MAPE"] == pytest.approx(float(np.mean(1.0 / targets)) * 100.0)
    assert report.aggregate["WAPE"] == pytest.approx(100.0 / float(targets.mean()))

    # Consecutive 7d origins share ground truth -> non-independent errors.
    assert report.overlap is not None
    assert report.overlap["overlapping"] is True
    assert report.overlap["unique_target_timestamps"] == horizon + POINTS_PER_DAY
    assert "NOT independent" in report.overlap["note"]
    assert json.dumps(report.to_dict(), allow_nan=False)
