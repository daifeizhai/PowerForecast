"""Leakage-blocking tests (T13, T14, T21).

The contracts raise ``LeakageError`` on their own when a *usable* input claims to be available
after its origin; these tests prove that (a) the window builder never needs that error because
it masks unavailable values out, (b) explicitly bypassing the mask is rejected, and (c) training
cutoffs and unverified timezones are handled honestly instead of being assumed away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.contracts import (
    TARGET_COLUMN,
    AuditStatus,
    DatasetRole,
    DatasetSpec,
    FeatureSpec,
    ForecastInputs,
    InsufficientDataError,
    IntervalStatistic,
    LabelOrigin,
    LeakageError,
    TargetEligibilityError,
    TargetKind,
    TargetScope,
    TargetSpec,
)
from src.data.features import CALENDAR_FEATURE_NAMES
from src.data.splitting import (
    Protocol,
    grid_for_frame,
    require_training_windows,
    training_window_audit,
)
from src.data.windows import (
    build_forecast_inputs,
    build_pretrain_batch,
    plan_origins,
)
from src.utils import load_config

ANCHOR = "2026-01-01T00:00:00"
STEP = pd.Timedelta(15, "min")
PPD = 96
DEFAULT = "configs/protocols/default.yaml"


# ---------------------------------------------------------------------------
# in-memory fixtures (this file must not depend on src.data.synthetic)
# ---------------------------------------------------------------------------


def make_target_spec() -> TargetSpec:
    return TargetSpec(
        kind=TargetKind.ACTIVE_POWER,
        scope=TargetScope.FACILITY,
        unit="kw",
        label_origin=LabelOrigin.SYNTHETIC_SIMULATED,
        measurement_boundary="synthetic facility meter",
        sampling_semantics="interval mean over the labelled 15-minute slot",
        synthetic=True,
    )


def make_spec(
    *,
    features: dict[str, FeatureSpec] | None = None,
    domain_id: str = "domain_a",
    anchor: str | None = ANCHOR,
    timezone: str = "UTC",
    timezone_verified: bool = True,
    role: DatasetRole = DatasetRole.TARGET_SUPERVISED,
    labelled: bool = True,
) -> DatasetSpec:
    if features is None:
        features = {
            "cpu_util": FeatureSpec(
                raw_name="cpu_util",
                canonical_name="cpu_util",
                unit="ratio",
                statistic=IntervalStatistic.INTERVAL_MEAN,
                audit_status=AuditStatus.VERIFIED,
            ),
            "gpu_util": FeatureSpec(
                raw_name="gpu_util", canonical_name="gpu_util", unit="ratio"
            ),
        }
    return DatasetSpec(
        dataset_id="synthetic_demo",
        site_id="test_site",
        domain_id=domain_id,
        role=role,
        native_frequency="15min",
        trace_anchor_timestamp=anchor,
        timezone=timezone,
        timezone_verified=timezone_verified,
        interval_semantics="left-labelled 15-minute interval mean (synthetic)",
        availability_semantics="available_at = interval end (synthetic assumption)",
        features=dict(features),
        target=make_target_spec() if labelled else None,
        source_version="test_v1",
        provenance={"synthetic": True},
        audit_status=AuditStatus.PARTIAL,
    )


def make_frame(
    *,
    days: int = 31,
    start_day: int = 0,
    anchor: str = ANCHOR,
    covariates: tuple[str, ...] = ("cpu_util", "gpu_util"),
    seed: int = 0,
) -> pd.DataFrame:
    anchor_ts = pd.Timestamp(anchor)
    index = pd.date_range(
        anchor_ts + pd.Timedelta(start_day, "D"),
        periods=days * PPD,
        freq="15min",
        name="timestamp",
    )
    rng = np.random.default_rng(seed)
    slot = (index.asi8 - anchor_ts.value) // STEP.value
    day = slot // PPD
    power = (
        900.0
        + 180.0 * np.sin(2.0 * np.pi * slot / PPD)
        + 4.0 * day
        + rng.normal(0.0, 1.5, index.size)
    )
    data: dict[str, np.ndarray] = {TARGET_COLUMN: power}
    if "cpu_util" in covariates:
        data["cpu_util"] = np.clip(
            0.5 + 0.2 * np.sin(2.0 * np.pi * slot / PPD) + rng.normal(0.0, 0.02, index.size),
            0.0,
            1.0,
        )
    if "gpu_util" in covariates:
        data["gpu_util"] = np.clip(
            0.4 + 0.15 * np.cos(2.0 * np.pi * slot / PPD) + rng.normal(0.0, 0.02, index.size),
            0.0,
            1.0,
        )
    return pd.DataFrame(data, index=index)


def build_default_inputs(
    *,
    frame: pd.DataFrame | None = None,
    spec: DatasetSpec | None = None,
    origins: list[int] | None = None,
    context_points: int = 96,
) -> tuple[ForecastInputs, DatasetSpec, pd.DataFrame]:
    frame = make_frame(days=31) if frame is None else frame
    spec = make_spec() if spec is None else spec
    grid = grid_for_frame(spec, frame)
    protocol = Protocol.from_config(
        load_config(DEFAULT),
        horizon="24h",
        origins=origins if origins is not None else [20],
        context_points=context_points,
    )
    plan = plan_origins(spec, grid, protocol, phase="development", frame=frame)
    return build_forecast_inputs(spec, frame, plan), spec, frame


# ---------------------------------------------------------------------------
# T13: unavailable past covariates are masked, never delivered
# ---------------------------------------------------------------------------


def test_large_availability_lag_masks_recent_covariates_and_is_not_delivered() -> None:
    lag = pd.Timedelta(6, "h")
    features = {
        "cpu_util": FeatureSpec(
            raw_name="cpu_util",
            canonical_name="cpu_util",
            unit="ratio",
            statistic=IntervalStatistic.INTERVAL_MEAN,
            availability_lag=lag,
            notes="assumed 6h reporting latency for this test",
        )
    }
    inputs, spec, _ = build_default_inputs(
        frame=make_frame(days=31, covariates=("cpu_util",)),
        spec=make_spec(features=features),
        context_points=96,
    )

    mask = inputs.past_covariate_mask[0, :, 0]
    available = inputs.past_covariate_available_at[0, :, 0]
    origin = inputs.origin_timestamps[0]

    # availability = interval end + declared lag, exactly as audited.
    expected_available = inputs.past_timestamps[0] + np.timedelta64(15, "m") + lag.to_timedelta64()
    assert np.all(available == expected_available)

    expected_usable = available <= origin
    assert np.array_equal(mask, expected_usable)
    # 6h = 24 slots of the newest history cannot be delivered at the origin.
    assert int(mask.sum()) == 96 - 24
    assert not mask[-24:].any()
    assert mask[: 96 - 24].all()
    # Masked entries never carry a value that could be mistaken for an observation.
    assert np.all(np.isnan(inputs.past_covariates[0, -24:, 0]))
    assert np.all(np.isfinite(inputs.past_covariates[0, : 96 - 24, 0]))


def test_usable_covariate_after_origin_is_rejected_by_the_contract() -> None:
    inputs, spec, _ = build_default_inputs(context_points=96)
    batch, length = inputs.past_timestamps.shape
    width = inputs.n_covariates
    assert (batch, length, width) == (1, 96, 2)

    # Force every entry to be "usable" while claiming availability one slot after the origin.
    leaky_available = np.broadcast_to(
        inputs.origin_timestamps[:, None, None] + np.timedelta64(15, "m"),
        (batch, length, width),
    ).astype("datetime64[ns]")
    with pytest.raises(LeakageError):
        ForecastInputs(
            past_power_kw=inputs.past_power_kw,
            past_power_valid=inputs.past_power_valid,
            past_timestamps=inputs.past_timestamps,
            origin_timestamps=inputs.origin_timestamps,
            target_timestamps=inputs.target_timestamps,
            target_spec=spec.target,
            domain_ids=inputs.domain_ids,
            dataset_id=inputs.dataset_id,
            past_covariates=inputs.past_covariates,
            past_covariate_mask=np.ones((batch, length, width), dtype=bool),
            past_covariate_available_at=leaky_available,
            covariate_names=inputs.covariate_names,
        )


# ---------------------------------------------------------------------------
# T13: real future values may not be smuggled in as known-future features
# ---------------------------------------------------------------------------


def test_real_future_values_are_rejected_as_known_future_but_calendar_is_accepted() -> None:
    frame = make_frame(days=31)
    inputs, spec, frame = build_default_inputs(frame=frame, context_points=1344)

    # The delivered known-future block is exactly the audited calendar, available at origin.
    assert inputs.future_known_names == CALENDAR_FEATURE_NAMES
    assert inputs.future_known_mask.all()
    assert np.all(inputs.future_known_available_at == inputs.origin_timestamps[:, None, None])
    # Calendar features alone are accepted: constructing those inputs already succeeded.
    calendar_only = ForecastInputs(
        past_power_kw=inputs.past_power_kw,
        past_power_valid=inputs.past_power_valid,
        past_timestamps=inputs.past_timestamps,
        origin_timestamps=inputs.origin_timestamps,
        target_timestamps=inputs.target_timestamps,
        target_spec=spec.target,
        domain_ids=inputs.domain_ids,
        dataset_id=inputs.dataset_id,
        future_known=inputs.future_known,
        future_known_mask=inputs.future_known_mask,
        future_known_available_at=inputs.future_known_available_at,
        future_known_names=inputs.future_known_names,
    )
    assert calendar_only.n_future_known == len(CALENDAR_FEATURE_NAMES)

    batch, horizon = inputs.target_timestamps.shape
    future_cpu = frame["cpu_util"].reindex(pd.DatetimeIndex(inputs.target_timestamps[0]))
    future_cpu_values = future_cpu.to_numpy(dtype=float).reshape(batch, horizon, 1)
    # The actual future measurements are only available when their own slot happens.
    real_available = inputs.target_timestamps[:, :, None].astype("datetime64[ns]")
    with pytest.raises(LeakageError):
        ForecastInputs(
            past_power_kw=inputs.past_power_kw,
            past_power_valid=inputs.past_power_valid,
            past_timestamps=inputs.past_timestamps,
            origin_timestamps=inputs.origin_timestamps,
            target_timestamps=inputs.target_timestamps,
            target_spec=spec.target,
            domain_ids=inputs.domain_ids,
            dataset_id=inputs.dataset_id,
            future_known=future_cpu_values,
            future_known_mask=np.ones((batch, horizon, 1), dtype=bool),
            future_known_available_at=real_available,
            future_known_names=("cpu_util",),
        )


def test_declared_future_known_feature_is_not_delivered_as_known_future() -> None:
    features = {
        "cpu_util": FeatureSpec(
            raw_name="cpu_util",
            canonical_name="cpu_util",
            unit="ratio",
            future_known=True,
            notes="declared known-future upstream, but no published plan/provenance exists here",
        )
    }
    inputs, _, _ = build_default_inputs(
        frame=make_frame(days=31, covariates=("cpu_util",)),
        spec=make_spec(features=features),
        context_points=96,
    )
    # The only delivered known-future block remains the calendar.
    assert inputs.future_known_names == CALENDAR_FEATURE_NAMES
    assert "cpu_util" not in inputs.future_known_names
    # The declared feature stays available as an ordinary (audited) past covariate.
    assert inputs.covariate_names == ("cpu_util",)
    assert inputs.metadata["declared_but_not_delivered_future_known_features"] == ["cpu_util"]


# ---------------------------------------------------------------------------
# T14: training-cutoff feasibility
# ---------------------------------------------------------------------------


def test_training_cutoff_blocks_long_context_and_long_horizon_samples() -> None:
    audit = training_window_audit(
        train_cutoff_day=14, context_points=1344, horizon_points=672
    )
    assert audit["feasible_windows"] == 0
    assert audit["insufficient_training_windows"] is True
    assert isinstance(audit["reason"], str) and audit["reason"]
    assert audit["train_cutoff_day"] == 14
    assert audit["context_points"] == 1344
    assert audit["horizon_points"] == 672

    short_horizon = training_window_audit(
        train_cutoff_day=14, context_points=1344, horizon_points=96
    )
    assert short_horizon["feasible_windows"] == 0
    assert short_horizon["insufficient_training_windows"] is True

    with pytest.raises(InsufficientDataError) as excinfo:
        require_training_windows(
            train_cutoff_day=14, context_points=1344, horizon_points=96
        )
    assert "Insufficient supervised training windows" in str(excinfo.value)


def test_training_window_audit_boundaries_are_exact() -> None:
    # 1344 history + 672 labels = exactly 2016 slots = 21 trace days, so a cutoff at the end
    # of trace day 21 leaves exactly one legal origin (trace day 14, labels on days 15..21).
    enough = training_window_audit(
        train_cutoff_day=21, context_points=1344, horizon_points=672
    )
    assert enough["feasible_windows"] == 1
    assert enough["insufficient_training_windows"] is False
    assert enough["reason"] is None
    assert enough["first_feasible_origin_slot"] == 1344
    assert enough["train_cutoff_last_slot"] == 21 * PPD - 1
    assert require_training_windows(
        train_cutoff_day=21, context_points=1344, horizon_points=672
    )["feasible_windows"] == 1

    # One day earlier there is no legality window left.
    assert (
        training_window_audit(train_cutoff_day=20, context_points=1344, horizon_points=672)[
            "feasible_windows"
        ]
        == 0
    )

    # A one-day context with a one-day horizon: origins on days 2..14 fit a day-14 cutoff.
    daily = training_window_audit(
        train_cutoff_day=14, context_points=96, horizon_points=96
    )
    assert daily["feasible_windows"] == 13
    assert daily["first_feasible_origin_slot"] == 96
    assert daily["last_feasible_origin_slot"] == 13 * PPD

    # An explicit stride spaces candidate origins from the earliest legal origin.
    strided = training_window_audit(
        train_cutoff_day=21, context_points=1344, horizon_points=96, stride_points=48
    )
    assert strided["feasible_windows"] > 1
    assert strided["stride_points"] == 48


def test_missing_trace_anchor_is_rejected_not_renumbered() -> None:
    spec = make_spec(
        anchor=None,
        role=DatasetRole.UNLABELED_PRETRAIN,
        labelled=False,
        features={
            "cpu_util": FeatureSpec(raw_name="cpu_util", canonical_name="cpu_util", unit="ratio")
        },
    )
    frame = make_frame(days=5, covariates=("cpu_util",))
    with pytest.raises(ValueError) as excinfo:
        grid_for_frame(spec, frame)
    assert "trace_anchor_timestamp" in str(excinfo.value)


# ---------------------------------------------------------------------------
# T21: unverified timezone keeps the uncertainty instead of assuming local/UTC days
# ---------------------------------------------------------------------------


def test_unverified_timezone_blocks_local_time_operations_but_utc_numbering_works() -> None:
    frame = make_frame(days=31)
    spec = make_spec(timezone="America/New_York", timezone_verified=False)

    with pytest.raises(TargetEligibilityError) as excinfo:
        spec.require_verified_timezone("calendar-day alignment")
    assert "timezone_verified=False" in str(excinfo.value)

    # Trace-day numbering only needs the explicit UTC anchor; it does not depend on a verified
    # local timezone and never substitutes local clock days.
    grid = grid_for_frame(spec, frame)
    assert grid.anchor == pd.Timestamp(ANCHOR)
    assert grid.day_number(grid.anchor) == 1
    assert grid.day_number(grid.anchor + 5 * pd.Timedelta(1, "D")) == 6
    assert grid.origin_of_day(5) == grid.anchor + 5 * pd.Timedelta(1, "D")
    assert grid.day_number(grid.anchor + 96 * STEP + 3 * STEP) == 2


def test_final_holdout_cannot_be_used_as_pretraining_corpus() -> None:
    frame = make_frame(days=5, covariates=("cpu_util",))
    spec = make_spec(
        role=DatasetRole.FINAL_HOLDOUT,
        labelled=False,
        features={
            "cpu_util": FeatureSpec(raw_name="cpu_util", canonical_name="cpu_util", unit="ratio")
        },
    )
    with pytest.raises(TargetEligibilityError):
        build_pretrain_batch(spec, frame, context_points=16)
