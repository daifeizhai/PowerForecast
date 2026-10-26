"""Task-book §9.1/§9.2 leakage and time-protocol tests at the Stage-2 pipeline level.

These tests build the very same window objects the experiment runner uses, then
prove that future power can never influence a forecast and that trace-day
numbering survives missing days.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.run_power_main import build_parser, run_experiment
from src.data.contracts import (
    TARGET_COLUMN,
    InsufficientDataError,
    DatasetRole,
    DatasetSpec,
    FeatureSpec,
    LabelOrigin,
    TargetKind,
    TargetScope,
    TargetSpec,
)
from src.data.splitting import Protocol, grid_for_frame
from src.data.windows import build_forecast_batch, plan_origins
from src.models.baselines import SeasonalNaiveForecaster, ThreeDayMeanForecaster
from src.utils import load_config
from support_stage2 import make_experiment, make_target_frame

REPO_ROOT = Path(__file__).resolve().parents[1]
POINTS_PER_DAY = 96


def target_spec() -> TargetSpec:
    return TargetSpec(
        kind=TargetKind.ACTIVE_POWER,
        scope=TargetScope.FACILITY,
        unit="kW",
        label_origin=LabelOrigin.COMPONENT_SUM,
        measurement_boundary="sum of ESIF facility sub-meter components",
        sampling_semantics="15-minute interval mean",
        synthetic=False,
    )


def dataset_spec(frame: pd.DataFrame) -> DatasetSpec:
    return DatasetSpec(
        dataset_id="nlr_esif",
        site_id="nlr_esif",
        domain_id="nlr_esif",
        role=DatasetRole.TARGET_SUPERVISED,
        native_frequency="15min",
        trace_anchor_timestamp=str(frame.index.min()),
        timezone="UTC",
        timezone_verified=False,
        interval_semantics="15-minute interval starting at timestamp",
        availability_semantics="available after interval end",
        features={},
        target=target_spec(),
        provenance={"label_type": "facility_derived_kw"},
    )


def target_frame(n_days: int = 31) -> pd.DataFrame:
    return make_target_frame(n_days).rename(columns={"power_kw": TARGET_COLUMN})


def protocol_24h() -> Protocol:
    payload = dict(load_config("configs/protocols/power_main_fixed.yaml"))
    payload["development_origins"] = [14, 16, 18, 20]
    payload["test_origins"] = list(range(22, 31))
    return Protocol.from_config(payload, horizon="24h", context_points=1344)


def build_batch(frame: pd.DataFrame):
    spec = dataset_spec(frame)
    grid = grid_for_frame(spec, frame)
    plan = plan_origins(spec, grid, protocol_24h(), phase="final-test", horizon="24h", frame=frame)
    return grid, plan, build_forecast_batch(spec, frame, plan, include_calendar=False)


def profile_predict(context: np.ndarray, horizon: int) -> np.ndarray:
    profile = np.asarray(context, dtype=float)[:, -POINTS_PER_DAY:]
    repeats = int(np.ceil(horizon / POINTS_PER_DAY))
    return np.tile(profile, (1, repeats))[:, :horizon]


# ---------------------------------------------------------------------------
# §9.1 future power can never change a prediction
# ---------------------------------------------------------------------------


def test_perturbing_future_power_does_not_change_inputs_or_predictions() -> None:
    frame = target_frame()
    _, plan, batch = build_batch(frame)
    context_before = batch.inputs.past_power_kw.copy()
    targets_before = batch.targets_kw.copy()
    predictions_before = profile_predict(context_before, batch.horizon)

    tampered = frame.copy()
    # Perturb only values strictly after the LAST origin: no window's context may
    # legitimately contain them, so every forecast must be unchanged.
    last_origin = plan.origins[-1]
    future_mask = tampered.index > last_origin
    assert future_mask.any()
    tampered.loc[future_mask, TARGET_COLUMN] = tampered.loc[future_mask, TARGET_COLUMN] * 1.5 + 100.0

    _, _, tampered_batch = build_batch(tampered)
    np.testing.assert_array_equal(context_before, tampered_batch.inputs.past_power_kw)
    predictions_after = profile_predict(tampered_batch.inputs.past_power_kw, tampered_batch.horizon)
    np.testing.assert_array_equal(predictions_before, predictions_after)
    # the test is meaningful only because some labels did change
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(targets_before, tampered_batch.targets_kw)


def test_context_timestamps_strictly_precede_each_origin() -> None:
    _, plan, batch = build_batch(target_frame())
    origins = np.asarray(batch.inputs.origin_timestamps)
    assert bool((batch.inputs.past_timestamps < origins[:, None]).all())
    # the last context slot is exactly one 15-minute step before the origin
    assert bool(
        (batch.inputs.past_timestamps[:, -1] == origins - np.timedelta64(15, "m")).all()
    )
    assert bool((batch.inputs.target_timestamps[:, 0] == origins).all())


def test_main_path_receives_no_auxiliary_or_known_future_features() -> None:
    _, _, batch = build_batch(target_frame())
    assert batch.inputs.n_covariates == 0
    assert batch.inputs.n_future_known == 0
    assert batch.inputs.covariate_names == ()
    assert batch.inputs.future_known_names == ()
    assert batch.inputs.future_known is None


def test_forecasters_are_only_handed_context_and_horizon(tmp_path: Path) -> None:
    """Registry-level proof that predict() never sees labels or future features."""
    received: list[tuple[np.ndarray, int]] = []

    class Recorder:
        name = "seasonal_naive"
        requires_fit = False

        def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
            received.append((np.asarray(context, dtype=float).copy(), int(horizon)))
            return profile_predict(context, horizon)

        def describe(self) -> dict:
            return {"name": self.name, "requires_fit": False}

    def factory(config, args, model_names):  # noqa: ANN001
        return {name: Recorder() for name in model_names}

    config_path, _ = make_experiment(tmp_path)
    args = build_parser().parse_args(
        ["--config", str(config_path), "--output", str(tmp_path / "runs"), "--horizons", "24h", "--splits", "test", "--models", "seasonal_naive", "--no-figures"]
    )
    assert run_experiment(args, forecaster_factory=factory) == 0
    assert len(received) == 1
    context, horizon = received[0]
    assert horizon == 96
    assert context.shape == (9, 1344)
    assert np.isfinite(context).all()
    # the recorded context is exactly the batch context (history only)
    _, _, batch = build_batch(target_frame())
    np.testing.assert_array_equal(context, batch.inputs.past_power_kw)


# ---------------------------------------------------------------------------
# §9.2 trace-day numbering with missing days
# ---------------------------------------------------------------------------


def test_missing_interior_day_does_not_shift_trace_days_or_origins() -> None:
    frame = target_frame()
    complete_grid, complete_plan, _ = build_batch(frame)

    gapped = frame.drop(frame.index[10 * POINTS_PER_DAY : 11 * POINTS_PER_DAY])
    gapped_grid, gapped_plan, gapped_batch = build_batch(gapped)

    # anchors, day numbers and origins are unchanged
    assert gapped_grid.anchor == complete_grid.anchor
    assert gapped_plan.days == complete_plan.days
    assert [str(origin) for origin in gapped_plan.origins] == [
        str(origin) for origin in complete_plan.origins
    ]
    for origin in gapped_plan.origins:
        assert gapped_grid.day_number(origin) == complete_grid.day_number(origin)
    # Missing slots remain missing; they are never re-compressed.  For an origin
    # Dk the 1344-slot context covers trace days (k-14)..(k-1), so a gap on trace
    # day 10 appears at offset (24-k)*96 inside later windows and not at all once
    # the window starts after day 10.
    for row, day in enumerate(gapped_plan.days):
        offset = (24 - day) * POINTS_PER_DAY
        if 0 <= offset <= 1344 - POINTS_PER_DAY:
            assert not bool(
                gapped_batch.inputs.past_power_valid[row, offset : offset + POINTS_PER_DAY].any()
            ), f"gap must be invalid inside the D{day} window"
        else:
            assert bool(gapped_batch.inputs.past_power_valid[row].all())


def test_incomplete_context_is_flagged_and_baselines_reject_the_window() -> None:
    frame = target_frame()
    _, plan, _ = build_batch(frame)
    first_origin = plan.origins[0]
    damaged = frame.copy()
    # break the last context day of the first origin -> that window is unusable
    last_context_day = (first_origin - pd.Timedelta(days=1))
    damaged.loc[last_context_day : first_origin - pd.Timedelta(minutes=15), TARGET_COLUMN] = np.nan
    _, _, batch = build_batch(damaged)
    assert not bool(batch.inputs.past_power_valid[0, -POINTS_PER_DAY:].all())
    with pytest.raises(InsufficientDataError):
        SeasonalNaiveForecaster().predict(batch.inputs.slice(slice(0, 1)))
    with pytest.raises(InsufficientDataError):
        ThreeDayMeanForecaster().predict(batch.inputs.slice(slice(0, 1)))


def test_labels_are_masked_not_shifted_when_target_missing() -> None:
    frame = target_frame()
    damaged = frame.copy()
    # remove the first test label day (D23) for all origins that predict it
    label_day_start = damaged.index[22 * POINTS_PER_DAY]
    damaged.loc[label_day_start : label_day_start + pd.Timedelta(hours=24) - pd.Timedelta(minutes=15), TARGET_COLUMN] = np.nan
    _, plan, batch = build_batch(damaged)
    d23_mask = batch.inputs.target_timestamps == label_day_start.to_datetime64()
    assert d23_mask.any()
    assert not bool(batch.target_valid_mask[d23_mask].any())
    # the D23 origin still exists and its labels (D24) are intact
    assert 23 in plan.days
    d24_mask = batch.inputs.target_timestamps == (label_day_start + pd.Timedelta(hours=24)).to_datetime64()
    assert bool(batch.target_valid_mask[d24_mask].all())
