"""Tests for model A3: the frozen 0.5/0.5 equal-weight main-path ensemble.

No foundation weights are loaded here.  The tests cover:

* exact arithmetic of the fixed weighted mean;
* the strict constructor policy (only ``(0.5, 0.5)``; negative / non-summing /
  searched weights are rejected) and the absence of any weight-search API;
* member output validation and error propagation with the failing member named;
* ``build_power_ensemble`` input validation (method, members, substitutions);
* the frozen configuration (``configs/models/foundation.yaml``) and the builder's
  refusal of non-0.5/0.5 ensemble configurations;
* the B0/B1 context wrappers match the existing baseline models exactly on a shared
  context (regression guard for the additive ``predict_from_context`` refactor).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.contracts import (
    ContractError,
    ForecastInputs,
    InsufficientDataError,
    LabelOrigin,
    TargetKind,
    TargetScope,
    TargetSpec,
)
from src.models.baselines import SeasonalNaiveForecaster, ThreeDayMeanForecaster
from src.models.power_main import build_power_forecasters
from src.models.power_main.ensemble import (
    ENSEMBLE_METHOD,
    FIXED_WEIGHTS,
    EnsembleMemberError,
    FixedWeightedEnsemble,
    build_power_ensemble,
)
from src.utils import load_config

POINTS_PER_DAY = 96
ANCHOR = pd.Timestamp("2026-01-01T00:00:00")
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
# deterministic member stubs
# ---------------------------------------------------------------------------


class ConstantMember:
    """Member returning a fixed ``[B, H]`` array (or a deliberate failure)."""

    requires_fit = False

    def __init__(
        self,
        name: str,
        *,
        value: float | None = None,
        output: np.ndarray | None = None,
        nan: bool = False,
        raises: Exception | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.output = output
        self.nan = nan
        self.raises = raises
        self.shape = shape

    def predict(self, context_kw: np.ndarray, horizon: int) -> np.ndarray:
        if self.raises is not None:
            raise self.raises
        context = np.asarray(context_kw, dtype=float)
        batch = context.shape[0]
        if self.shape is not None:
            return np.zeros(self.shape, dtype=np.float64)
        if self.output is not None:
            return np.asarray(self.output, dtype=np.float64).copy()
        result = np.full((batch, horizon), float(self.value), dtype=np.float64)
        if self.nan:
            result[0, 0] = np.nan
        return result

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": "constant_stub", "requires_fit": False}


def make_members(chronos_value: float, tirex_value: float) -> tuple[ConstantMember, ConstantMember]:
    return (
        ConstantMember("chronos2", value=chronos_value),
        ConstantMember("tirex2", value=tirex_value),
    )


# ---------------------------------------------------------------------------
# fixed arithmetic
# ---------------------------------------------------------------------------


def test_equal_weight_mean_is_exact() -> None:
    chronos = ConstantMember("chronos2", output=np.array([[100.0, 110.0]]))
    tirex = ConstantMember("tirex2", output=np.array([[120.0, 130.0]]))
    ensemble = FixedWeightedEnsemble(members=(chronos, tirex), weights=(0.5, 0.5))
    forecast = np.asarray(ensemble.predict(np.zeros((1, 1344)), 2))
    np.testing.assert_array_equal(forecast, np.array([[110.0, 120.0]]))
    assert forecast.shape == (1, 2)
    assert forecast.dtype == np.float64


def test_fixed_weights_are_declared_in_describe() -> None:
    ensemble = FixedWeightedEnsemble(members=make_members(1.0, 3.0), weights=(0.5, 0.5))
    description = ensemble.describe()
    assert description["name"] == "power_ensemble"
    assert description["requires_fit"] is False
    assert description["method"] == ENSEMBLE_METHOD
    assert description["weights"] == [0.5, 0.5]
    assert description["fixed_weight_policy"]["weights"] == [0.5, 0.5]
    assert description["fixed_weight_policy"]["search_allowed"] is False
    assert "NOT a main/auxiliary" in description["weight_scope"]
    assert FIXED_WEIGHTS == (0.5, 0.5)


# ---------------------------------------------------------------------------
# strict constructor policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weights",
    [
        (0.6, 0.4),
        (0.4, 0.6),
        (0.5, 0.6),
        (0.4, 0.4),  # does not sum to 1
        (0.2, 0.3),
        (-0.5, 1.5),  # negative weight
        (1.0, 0.0),  # sums to 1 but is not the frozen equal pair
        (0.0, 1.0),
        (0.5,),
        (0.5, 0.5, 0.0),
    ],
)
def test_only_fixed_equal_weights_are_accepted(weights: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        FixedWeightedEnsemble(members=make_members(1.0, 3.0), weights=weights)


def test_ensemble_requires_exactly_two_distinct_members() -> None:
    chronos, tirex = make_members(1.0, 3.0)
    with pytest.raises(ValueError, match="exactly 2 member"):
        FixedWeightedEnsemble(members=(chronos,), weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="exactly 2 member"):
        FixedWeightedEnsemble(members=(chronos, tirex, ConstantMember("x")), weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="distinct"):
        FixedWeightedEnsemble(members=(chronos, chronos), weights=(0.5, 0.5))


def test_ensemble_rejects_members_without_predict_or_requiring_fit() -> None:
    class NoPredict:
        requires_fit = False
        name = "nope"

    class RequiresFit(ConstantMember):
        requires_fit = True

    with pytest.raises(ValueError, match="callable predict"):
        FixedWeightedEnsemble(members=(NoPredict(), ConstantMember("tirex2")), weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="requires fit"):
        FixedWeightedEnsemble(
            members=(RequiresFit("chronos2", value=1.0), ConstantMember("tirex2")),
            weights=(0.5, 0.5),
        )


def test_no_weight_search_api_exists() -> None:
    ensemble = FixedWeightedEnsemble(members=make_members(1.0, 3.0), weights=(0.5, 0.5))
    for attribute in (
        "fit",
        "tune",
        "optimize_weights",
        "search_weights",
        "grid_search",
        "calibrate",
        "set_weights",
    ):
        assert not hasattr(ensemble, attribute), attribute
    # The weights are frozen after construction.
    with pytest.raises(Exception):
        ensemble.weights = (0.6, 0.4)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# member validation and error propagation
# ---------------------------------------------------------------------------


def test_member_shape_mismatch_is_named() -> None:
    chronos = ConstantMember("chronos2", shape=(2, 3))
    tirex = ConstantMember("tirex2", value=5.0)
    ensemble = FixedWeightedEnsemble(members=(chronos, tirex), weights=(0.5, 0.5))
    with pytest.raises(EnsembleMemberError) as excinfo:
        ensemble.predict(np.zeros((2, 1344)), 2)
    assert "chronos2" in str(excinfo.value)
    assert "shape" in str(excinfo.value)


def test_member_nan_output_is_named() -> None:
    chronos = ConstantMember("chronos2", value=5.0)
    tirex = ConstantMember("tirex2", value=5.0, nan=True)
    ensemble = FixedWeightedEnsemble(members=(chronos, tirex), weights=(0.5, 0.5))
    with pytest.raises(EnsembleMemberError) as excinfo:
        ensemble.predict(np.zeros((2, 1344)), 2)
    assert "tirex2" in str(excinfo.value)


def test_member_error_is_propagated_with_member_identity() -> None:
    chronos = ConstantMember("chronos2", raises=ValueError("no weights for this series"))
    tirex = ConstantMember("tirex2", value=5.0)
    ensemble = FixedWeightedEnsemble(members=(chronos, tirex), weights=(0.5, 0.5))
    with pytest.raises(EnsembleMemberError) as excinfo:
        ensemble.predict(np.zeros((2, 1344)), 2)
    message = str(excinfo.value)
    assert "chronos2" in message
    assert "no weights for this series" in message
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_ensemble_rejects_invalid_context_and_horizon() -> None:
    ensemble = FixedWeightedEnsemble(members=make_members(1.0, 3.0), weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="at least 96 context points"):
        ensemble.predict(np.zeros((1, 10)), 16)
    with pytest.raises(ValueError, match="horizon >= 1"):
        ensemble.predict(np.zeros((1, 1344)), 0)
    bad = np.zeros((1, 1344))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite context"):
        ensemble.predict(bad, 16)


# ---------------------------------------------------------------------------
# build_power_ensemble validation
# ---------------------------------------------------------------------------


def test_build_power_ensemble_accepts_the_two_named_members() -> None:
    chronos, tirex = make_members(1.0, 3.0)
    ensemble = build_power_ensemble({"chronos2": chronos, "tirex2": tirex})
    assert isinstance(ensemble, FixedWeightedEnsemble)
    assert ensemble.weights == (0.5, 0.5)
    assert [member.name for member in ensemble.members] == ["chronos2", "tirex2"]


def test_build_power_ensemble_rejects_bad_method_or_members() -> None:
    chronos, tirex = make_members(1.0, 3.0)
    with pytest.raises(ValueError, match="only supports"):
        build_power_ensemble({"chronos2": chronos, "tirex2": tirex}, method="searched_weights")
    with pytest.raises(ValueError, match="missing"):
        build_power_ensemble({"chronos2": chronos})
    with pytest.raises(ValueError, match="unexpected"):
        build_power_ensemble({"chronos2": chronos, "tirex2": tirex, "extra": chronos})
    with pytest.raises(ValueError, match="impossible to substitute"):
        build_power_ensemble({"chronos2": ConstantMember("tirex2"), "tirex2": tirex})
    with pytest.raises(ValueError, match="mapping"):
        build_power_ensemble([chronos, tirex])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# frozen configuration + builder refusal of non-0.5/0.5
# ---------------------------------------------------------------------------


def test_foundation_config_declares_fixed_equal_weights() -> None:
    config = load_config("configs/models/foundation.yaml")
    ensemble_config = config["ensemble"]
    assert ensemble_config["method"] == "fixed_weighted_mean"
    assert float(ensemble_config["chronos2_weight"]) == 0.5
    assert float(ensemble_config["tirex2_weight"]) == 0.5
    assert float(ensemble_config["chronos2_weight"]) + float(
        ensemble_config["tirex2_weight"]
    ) == 1.0


def test_builder_refuses_searched_ensemble_weights() -> None:
    config = copy.deepcopy(load_config("configs/models/foundation.yaml"))
    config["ensemble"] = {
        "method": "fixed_weighted_mean",
        "chronos2_weight": 0.6,
        "tirex2_weight": 0.4,
    }
    with pytest.raises(ValueError, match="frozen 0.5/0.5"):
        build_power_forecasters(config, models=["power_ensemble"])

    config["ensemble"] = {
        "method": "optimised_weighted_mean",
        "chronos2_weight": 0.5,
        "tirex2_weight": 0.5,
    }
    with pytest.raises(ValueError, match="fixed_weighted_mean"):
        build_power_forecasters(config, models=["power_ensemble"])


# ---------------------------------------------------------------------------
# B0/B1 context wrappers == unchanged baseline models (refactor regression guard)
# ---------------------------------------------------------------------------


def _make_inputs(series: np.ndarray, horizon: int) -> ForecastInputs:
    series = np.asarray(series, dtype=float)
    length = series.shape[0]
    grid = pd.date_range(ANCHOR, periods=length + horizon, freq="15min").to_numpy(
        dtype="datetime64[ns]"
    )
    return ForecastInputs(
        past_power_kw=series[None, :],
        past_power_valid=np.ones((1, length), dtype=bool),
        past_timestamps=grid[:length][None, :],
        origin_timestamps=grid[length : length + 1],
        target_timestamps=grid[length:][None, :],
        target_spec=TARGET_SPEC,
        domain_ids=("synthetic_domain_0",),
        dataset_id="synthetic_facility",
    )


def _series(points: int = 3 * POINTS_PER_DAY) -> np.ndarray:
    slots = np.arange(points, dtype=float)
    return (
        820.0
        + 60.0 * np.sin(2.0 * np.pi * slots / POINTS_PER_DAY)
        + 25.0 * np.sin(2.0 * np.pi * slots / (7.0 * POINTS_PER_DAY))
        + 0.3 * slots
    )


def test_baseline_wrappers_match_existing_models_exactly() -> None:
    config = load_config("configs/models/foundation.yaml")
    forecasters = build_power_forecasters(
        config, models=["seasonal_naive", "three_day_mean"]
    )
    series = _series()
    horizon = 96
    inputs = _make_inputs(series, horizon)
    for name, baseline_class in (
        ("seasonal_naive", SeasonalNaiveForecaster),
        ("three_day_mean", ThreeDayMeanForecaster),
    ):
        reference = baseline_class()
        expected = reference.predict(inputs)
        wrapper = forecasters[name]
        actual = wrapper.predict(inputs.past_power_kw, horizon)
        assert np.array_equal(actual, expected), name
        # The additive refactor exposes the same numbers through the raw-context path.
        direct = reference.predict_from_context(inputs.past_power_kw, horizon)
        assert np.array_equal(direct, expected), name
        assert wrapper.describe()["stage2_entry_point"] == "predict_from_context"


def test_baseline_context_path_rejects_short_or_non_finite_context() -> None:
    model = SeasonalNaiveForecaster()
    with pytest.raises(InsufficientDataError):
        model.predict_from_context(np.zeros((1, 10)), 16)
    bad = np.zeros((1, 96))
    bad[0, 0] = np.nan
    with pytest.raises(ContractError):
        model.predict_from_context(bad, 16)
    with pytest.raises(ContractError):
        model.predict_from_context([[1.0, 2.0], [3.0]], 16)  # ragged
