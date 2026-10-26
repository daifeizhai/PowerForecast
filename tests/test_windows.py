"""Window construction: horizons, context lengths, F=0/K=0, trace-day numbering, audits.

All frames are built in memory (UTC ``DatetimeIndex`` named ``timestamp``, 96 slots per trace
day, columns from ``DatasetSpec.features`` plus ``TARGET_COLUMN``); the tests never depend on
``src.data.synthetic`` and never write to ``data/`` or ``results/``.
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
    ForecastBatch,
    InsufficientDataError,
    IntervalStatistic,
    LabelOrigin,
    PretrainBatch,
    TargetKind,
    TargetScope,
    TargetSpec,
)
from src.data.features import CALENDAR_FEATURE_NAMES
from src.data.splitting import (
    Protocol,
    TraceGrid,
    build_trace_grid,
    grid_for_frame,
    load_protocol,
)
from src.data.windows import (
    OriginPlan,
    audit_split,
    build_forecast_batch,
    build_forecast_inputs,
    build_pretrain_batch,
    plan_origins,
)
from src.utils import load_config

ANCHOR = "2026-01-01T00:00:00"
STEP = pd.Timedelta(15, "min")
PPD = 96
DEFAULT = "configs/protocols/default.yaml"
FIXED_24H = "configs/protocols/fixed_24h.yaml"
FIXED_7D = "configs/protocols/fixed_7d.yaml"


# ---------------------------------------------------------------------------
# in-memory fixtures
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


def default_features() -> dict[str, FeatureSpec]:
    return {
        "cpu_util": FeatureSpec(
            raw_name="cpu_util",
            canonical_name="cpu_util",
            unit="ratio",
            statistic=IntervalStatistic.INTERVAL_MEAN,
            audit_status=AuditStatus.VERIFIED,
        ),
        "gpu_util": FeatureSpec(raw_name="gpu_util", canonical_name="gpu_util", unit="ratio"),
        "task_count": FeatureSpec(raw_name="task_count", canonical_name="task_count", unit="count"),
    }


def make_spec(
    *,
    dataset_id: str = "synthetic_demo",
    domain_id: str = "domain_a",
    features: dict[str, FeatureSpec] | None = None,
    anchor: str | None = ANCHOR,
    timezone: str = "UTC",
    timezone_verified: bool = True,
    native_frequency: str = "15min",
    role: DatasetRole = DatasetRole.TARGET_SUPERVISED,
    labelled: bool = True,
) -> DatasetSpec:
    feature_map = default_features() if features is None else dict(features)
    return DatasetSpec(
        dataset_id=dataset_id,
        site_id="test_site",
        domain_id=domain_id,
        role=role,
        native_frequency=native_frequency,
        trace_anchor_timestamp=anchor,
        timezone=timezone,
        timezone_verified=timezone_verified,
        interval_semantics="left-labelled 15-minute interval mean (synthetic)",
        availability_semantics="available_at = interval end (synthetic assumption)",
        features=feature_map,
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
    covariates: tuple[str, ...] = ("cpu_util", "gpu_util", "task_count"),
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
    power = 900.0 + 180.0 * np.sin(2.0 * np.pi * slot / PPD) + 4.0 * day
    power = power + rng.normal(0.0, 1.5, index.size)
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
    if "task_count" in covariates:
        data["task_count"] = (
            20.0 + 5.0 * np.sin(2.0 * np.pi * slot / PPD) + rng.integers(0, 4, index.size)
        )
    return pd.DataFrame(data, index=index)


def default_protocol(**kwargs) -> Protocol:
    return Protocol.from_config(load_config(DEFAULT), **kwargs)


# ---------------------------------------------------------------------------
# T09: all horizons, configurable history length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("horizon", "points"),
    (("4h", 16), ("8h", 32), ("24h", 96), ("3d", 288), ("7d", 672)),
)
def test_every_horizon_produces_matching_shapes(horizon: str, points: int) -> None:
    frame = make_frame(days=31)
    spec = make_spec()
    grid = grid_for_frame(spec, frame)
    protocol = default_protocol(horizon=horizon, origins=[16, 18, 20], context_points=1344)

    plan = plan_origins(spec, grid, protocol, phase="development")
    batch = build_forecast_batch(spec, frame, plan)

    assert isinstance(batch, ForecastBatch)
    assert plan.horizon == horizon
    assert plan.horizon_points == points
    assert batch.targets_kw.shape == (3, points)
    assert batch.inputs.target_timestamps.shape == (3, points)
    assert batch.inputs.past_power_kw.shape == (3, 1344)
    # The origin is the first label timestamp and one slot after the last context slot.
    assert np.array_equal(batch.inputs.origin_timestamps, batch.inputs.target_timestamps[:, 0])
    assert np.array_equal(batch.inputs.origin_timestamps, plan.origins.to_numpy())
    assert np.all(
        batch.inputs.origin_timestamps - batch.inputs.past_timestamps[:, -1]
        == np.timedelta64(15, "m")
    )
    # Labels are the consecutive slots after the origin: no gap, no shortening.
    expected_lead = np.arange(points, dtype="int64") * STEP.value
    actual_lead = (
        batch.inputs.target_timestamps[0] - batch.inputs.origin_timestamps[0]
    ).astype("int64")
    assert np.array_equal(actual_lead, expected_lead)
    assert batch.target_valid_mask.shape == (3, points)
    assert batch.target_available_at is not None
    assert np.all(batch.inputs.target_timestamps <= batch.target_available_at)


@pytest.mark.parametrize("context_points", (96, 672, 1344))
def test_configurable_context_length(context_points: int) -> None:
    frame = make_frame(days=31)
    spec = make_spec()
    grid = grid_for_frame(spec, frame)
    protocol = default_protocol(horizon="24h", origins=[20], context_points=context_points)

    plan = plan_origins(spec, grid, protocol, phase="development")
    inputs = build_forecast_inputs(spec, frame, plan)

    assert inputs.context_length == context_points
    assert inputs.past_power_kw.shape == (1, context_points)
    assert inputs.past_timestamps.shape == (1, context_points)
    assert pd.Timestamp(inputs.past_timestamps[0, 0]) == plan.origins[0] - context_points * STEP
    assert pd.Timestamp(inputs.past_timestamps[0, -1]) == plan.origins[0] - STEP


def test_origin_14_reaches_exactly_the_trace_anchor() -> None:
    frame = make_frame(days=31)
    spec = make_spec()
    grid = grid_for_frame(spec, frame)
    protocol = default_protocol(horizon="24h", origins=[14], context_points=1344)

    plan = plan_origins(spec, grid, protocol, phase="development")
    inputs = build_forecast_inputs(spec, frame, plan)

    assert plan.days == (14,)
    assert plan.dropped == ()
    assert pd.Timestamp(inputs.past_timestamps[0, 0]) == grid.anchor
    assert pd.Timestamp(inputs.past_timestamps[0, -1]) == plan.origins[0] - STEP
    assert plan.origins[0] == grid.origin_of_day(14)


# ---------------------------------------------------------------------------
# T08: F=0, K=0 power-only path is legal
# ---------------------------------------------------------------------------


def test_power_only_frame_builds_batch_without_covariates_or_known_future() -> None:
    frame = make_frame(days=31, covariates=())
    spec = make_spec(features={})
    grid = grid_for_frame(spec, frame)
    protocol = default_protocol(horizon="24h", origins=[20], context_points=1344)

    plan = plan_origins(spec, grid, protocol, phase="development")
    batch = build_forecast_batch(spec, frame, plan, include_calendar=False)

    assert batch.inputs.past_covariates is None
    assert batch.inputs.n_covariates == 0
    assert batch.inputs.covariate_names == ()
    assert batch.inputs.past_covariate_mask is None
    assert batch.inputs.future_known is None
    assert batch.inputs.n_future_known == 0
    assert batch.inputs.future_known_names == ()
    assert batch.inputs.past_power_kw.shape == (1, 1344)
    assert batch.targets_kw.shape == (1, 96)
    assert batch.targets_kw.dtype == float
    assert np.isfinite(batch.targets_kw).all()


# ---------------------------------------------------------------------------
# covariate masks, calendar known-future, availability metadata
# ---------------------------------------------------------------------------


def test_covariate_masks_power_validity_and_calendar_known_future() -> None:
    frame = make_frame(days=31)
    context_start_slot = 20 * PPD - 1344  # 576
    covariate_nan_slots = [1000, 1001, 1002]
    power_nan_slots = [1500, 1501]
    frame.loc[frame.index[covariate_nan_slots], "cpu_util"] = np.nan
    frame.loc[frame.index[power_nan_slots], TARGET_COLUMN] = np.nan

    spec = make_spec()
    grid = grid_for_frame(spec, frame)
    protocol = default_protocol(horizon="24h", origins=[20], context_points=1344)
    plan = plan_origins(spec, grid, protocol, phase="development")
    inputs = build_forecast_inputs(spec, frame, plan)

    assert inputs.covariate_names == ("cpu_util", "gpu_util", "task_count")
    cpu = inputs.covariate_names.index("cpu_util")
    positions = [slot - context_start_slot for slot in covariate_nan_slots]
    assert inputs.past_covariate_mask[0, positions, cpu].tolist() == [False, False, False]
    assert np.all(np.isnan(inputs.past_covariates[0, positions, cpu]))
    assert bool(inputs.past_covariate_mask[0, positions[-1] + 1, cpu])

    # History power stays NaN and invalid; it is never replaced by 0 kW.
    power_positions = [slot - context_start_slot for slot in power_nan_slots]
    assert inputs.past_power_valid[0, power_positions].tolist() == [False, False]
    assert np.all(np.isnan(inputs.past_power_kw[0, power_positions]))

    # Exactly one declared covariate column, converted to kW semantics unchanged.
    assert inputs.past_covariates.shape == (1, 1344, 3)
    # Observation interval end = slot timestamp + one frequency step; lag is zero here.
    expected_available = inputs.past_timestamps[:, :, None] + np.timedelta64(15, "m")
    assert np.all(inputs.past_covariate_available_at == expected_available)
    assert np.all(inputs.past_covariate_available_at <= inputs.origin_timestamps[:, None, None])

    # The only known-future block is the audited calendar, available at the origin.
    assert inputs.future_known is not None
    assert inputs.future_known.shape == (1, 96, len(CALENDAR_FEATURE_NAMES))
    assert inputs.future_known_names == CALENDAR_FEATURE_NAMES
    assert inputs.future_known_mask.all()
    assert np.all(
        inputs.future_known_available_at == inputs.origin_timestamps[:, None, None]
    )
    assert inputs.metadata["future_known_source"] == "calendar"


# ---------------------------------------------------------------------------
# T12: one anchor per dataset, no per-domain renumbering
# ---------------------------------------------------------------------------


def test_two_domains_share_anchor_and_day_numbers_despite_leading_gap() -> None:
    domain_a = make_frame(days=31, start_day=0)
    domain_b = make_frame(days=28, start_day=3)  # first three trace days missing entirely
    spec_a = make_spec(domain_id="domain_a")
    spec_b = make_spec(domain_id="domain_b")

    grid_a = grid_for_frame(spec_a, domain_a)
    grid_b = grid_for_frame(spec_b, domain_b)

    assert grid_a.anchor == grid_b.anchor == pd.Timestamp(ANCHOR)
    assert grid_a.timestamps[0] == grid_b.timestamps[0] == pd.Timestamp(ANCHOR)

    probes = [grid_a.anchor + pd.Timedelta(offset, "D") for offset in (0, 3, 5, 10, 20, 27)]
    for probe in probes:
        assert grid_b.day_number(probe) == grid_a.day_number(probe)
        assert grid_a.day_number(probe) == int((probe - grid_a.anchor) / pd.Timedelta(1, "D")) + 1

    # Domain B's first record is trace day 4, NOT a renumbered day 1.
    assert grid_b.day_number(domain_b.index[0]) == 4
    assert grid_b.origin_of_day(3) == domain_b.index[0]
    assert grid_b.day_start(4) == domain_b.index[0]


def test_interior_missing_rows_do_not_renumber_later_days() -> None:
    frame = make_frame(days=31)
    spec = make_spec()
    full = grid_for_frame(spec, frame)
    probe = full.day_start(20)

    removed = frame.index[3 * PPD + 10 : 3 * PPD + 14]
    gapped_frame = frame.drop(removed)
    gapped = grid_for_frame(spec, gapped_frame)

    assert gapped.anchor == full.anchor
    assert gapped.n_slots == full.n_slots  # expected grid, not the observed rows
    assert gapped.day_number(probe) == full.day_number(probe) == 20
    assert gapped.day_start(20) == pd.Timestamp(ANCHOR) + pd.Timedelta(19, "D")
    # Removed slots are still expected grid slots; they are not deleted from the protocol.
    assert gapped.contains(removed).all()


def test_first_96_slots_are_trace_day_one_and_grid_starts_at_anchor() -> None:
    grid = build_trace_grid(anchor=ANCHOR, n_days=3)
    assert isinstance(grid, TraceGrid)
    assert grid.day_start(1) == grid.anchor
    assert grid.day_number(grid.anchor) == 1
    assert grid.day_number(grid.anchor + 95 * STEP) == 1
    assert grid.day_number(grid.anchor + 96 * STEP) == 2
    assert grid.origin_of_day(1) == grid.anchor + 96 * STEP
    assert grid.origin_index(1) == 96
    assert grid.day_number(grid.anchor + 2 * PPD * STEP) == 3
    assert grid.origin_of_day(2) == grid.anchor + 192 * STEP
    assert grid.timestamps[0] == grid.anchor

    # A sparse observation set never renumbers the grid to start at the first observation.
    sparse = build_trace_grid(
        anchor=ANCHOR, timestamps=[pd.Timestamp(ANCHOR) + pd.Timedelta(5, "D")]
    )
    assert sparse.timestamps[0] == pd.Timestamp(ANCHOR)
    assert sparse.day_number(sparse.timestamps[-1]) == 6

    assert grid.contains([grid.anchor, grid.anchor + PPD * STEP]).all()
    assert not grid.contains([grid.anchor + pd.Timedelta(7, "min")]).any()
    assert grid.to_dict()["anchor"] == pd.Timestamp(ANCHOR).isoformat()


# ---------------------------------------------------------------------------
# dropped origins and structured insufficiency
# ---------------------------------------------------------------------------


def test_plan_reports_dropped_origins_and_never_shortens_the_horizon() -> None:
    frame = make_frame(days=20)
    spec = make_spec()
    grid = grid_for_frame(spec, frame)
    protocol = default_protocol(horizon="24h", origins=[14, 18, 20], context_points=1344)

    plan = plan_origins(spec, grid, protocol, phase="development", frame=frame)

    assert plan.days == (14, 18)
    assert len(plan.origins) == 2
    dropped = {entry["day"]: entry for entry in plan.dropped}
    assert set(dropped) == {20}
    assert "beyond the last covered slot" in dropped[20]["reason"]
    assert dropped[20]["detail"]["origin_slot"] == 20 * PPD
    assert dropped[20]["detail"]["horizon_points"] == 96

    audit = plan.audit()
    assert [entry["day"] for entry in audit["dropped"]] == [20]
    assert audit["n_origins"] == 2
    assert [entry["day"] for entry in audit["origins"]] == [14, 18]

    # The kept windows still have the full configured horizon, not a shortened one.
    batch = build_forecast_batch(spec, frame, plan)
    assert batch.targets_kw.shape == (2, 96)
    assert batch.inputs.horizon == 96


def test_leading_gap_is_dropped_with_reason_when_frame_coverage_is_supplied() -> None:
    frame = make_frame(days=20, start_day=3)  # rows begin at trace day 4
    spec = make_spec()
    grid = grid_for_frame(spec, frame)
    protocol = default_protocol(horizon="24h", origins=[16, 18], context_points=1344)

    plan = plan_origins(spec, grid, protocol, phase="development", frame=frame)
    assert plan.days == (18,)
    assert len(plan.dropped) == 1
    assert plan.dropped[0]["day"] == 16
    assert "frame starts at" in plan.dropped[0]["reason"]

    # Without the frame the grid is geometrically long enough; the missing leading rows are
    # then reported by audit_split / masked in the batch instead of being invented.
    geometric = plan_origins(spec, grid, protocol, phase="development")
    assert geometric.days == (16, 18)
    audit = audit_split(spec, frame, geometric)
    by_day = {entry["day"]: entry for entry in audit["origins"]}
    assert by_day[16]["context_complete"] is False
    assert by_day[16]["usable_supervised_window"] is False
    assert by_day[18]["usable_supervised_window"] is True


def test_insufficient_data_raises_instead_of_shrinking_or_shifting() -> None:
    frame = make_frame(days=12)
    spec = make_spec()
    grid = grid_for_frame(spec, frame)
    protocol = default_protocol(horizon="7d", origins=[8], context_points=1344)

    with pytest.raises(InsufficientDataError) as excinfo:
        plan_origins(spec, grid, protocol, phase="development", frame=frame)
    message = str(excinfo.value)
    assert "never silently shifted" in message
    assert "horizon is never shortened" in message

    # A protocol with no configured origins never invents them from the data.
    empty = default_protocol()
    with pytest.raises(ValueError):
        plan_origins(spec, grid, empty, phase="development")


# ---------------------------------------------------------------------------
# T03/T18: unlabelled pretraining batches
# ---------------------------------------------------------------------------


def test_pretrain_batch_has_no_labels_and_native_frequency_metadata() -> None:
    anchor_ts = pd.Timestamp(ANCHOR)
    index = pd.date_range(anchor_ts, periods=300, freq="30min", name="timestamp")
    frame = pd.DataFrame(
        {
            "cpu_util": np.linspace(0.1, 0.9, 300),
            "gpu_util": np.linspace(0.9, 0.1, 300),
        },
        index=index,
    )
    spec = make_spec(
        dataset_id="unlabeled_source",
        features={
            "cpu_util": FeatureSpec(raw_name="cpu_util", canonical_name="cpu_util", unit="ratio"),
            "gpu_util": FeatureSpec(raw_name="gpu_util", canonical_name="gpu_util", unit="ratio"),
        },
        native_frequency="30min",
        role=DatasetRole.UNLABELED_PRETRAIN,
        labelled=False,
    )

    batch = build_pretrain_batch(
        spec, frame, context_points=48, stride_points=24, max_windows=5
    )

    assert isinstance(batch, PretrainBatch)
    for forbidden in ("targets_kw", "future_kw", "targets", "power_labels"):
        assert not hasattr(batch, forbidden)
    assert batch.features.shape == (5, 48, 2)
    assert batch.observed_mask.shape == (5, 48, 2)
    assert batch.timestamps.shape == (5, 48)
    assert batch.available_at.shape == (5, 48, 2)
    assert batch.feature_names == ("cpu_util", "gpu_util")
    assert batch.domain_ids == ("domain_a",) * 5
    assert batch.dataset_id == "unlabeled_source"
    assert batch.sampling_metadata["native_frequency"] == "30min"
    assert batch.sampling_metadata["time_delta_seconds"] == 1800.0
    assert batch.sampling_metadata["regular_frequency"] is True
    assert batch.sampling_metadata["context_points"] == 48
    assert batch.sampling_metadata["stride_points"] == 24
    assert batch.metadata()["contains_power_labels"] is False
    # Native 30-minute sampling is preserved, not forced onto a 96-point/day grid.
    delta = batch.timestamps[0, 1] - batch.timestamps[0, 0]
    assert delta == np.timedelta64(30, "m")
    assert np.all(batch.available_at >= batch.timestamps[:, :, None])


def test_pretrain_missing_values_keep_mask_false() -> None:
    index = pd.date_range(pd.Timestamp(ANCHOR), periods=64, freq="15min", name="timestamp")
    frame = pd.DataFrame(
        {"cpu_util": np.linspace(0.1, 0.9, 64), "gpu_util": np.linspace(0.9, 0.1, 64)},
        index=index,
    )
    frame.loc[frame.index[10], "cpu_util"] = np.nan
    spec = make_spec(
        features={
            "cpu_util": FeatureSpec(raw_name="cpu_util", canonical_name="cpu_util", unit="ratio"),
            "gpu_util": FeatureSpec(raw_name="gpu_util", canonical_name="gpu_util", unit="ratio"),
        },
        role=DatasetRole.UNLABELED_PRETRAIN,
        labelled=False,
    )

    batch = build_pretrain_batch(spec, frame, context_points=16, stride_points=16, max_windows=2)
    assert not bool(batch.observed_mask[0, 10, 0])
    assert np.isnan(batch.features[0, 10, 0])
    assert bool(batch.observed_mask[0, 9, 0])
    assert bool(batch.observed_mask[0, 11, 0])


# ---------------------------------------------------------------------------
# T10 / T11: optional presets produce the declared label coverage
# ---------------------------------------------------------------------------


def test_fixed_24h_preset_produces_nine_independent_96_point_windows() -> None:
    frame = make_frame(days=31)
    spec = make_spec()
    grid = grid_for_frame(spec, frame)
    protocol = load_protocol(FIXED_24H)

    plan = plan_origins(spec, grid, protocol, phase="final-test", frame=frame)
    assert plan.days == tuple(range(22, 31))
    assert plan.horizon_points == 96
    assert plan.overlap is False

    audit = plan.audit()
    assert [entry["label_days"] for entry in audit["origins"]] == [
        [day] for day in range(23, 32)
    ]
    assert audit["covered_label_days"] == list(range(23, 32))
    assert audit["n_label_points"] == 9 * 96
    assert audit["n_unique_label_timestamps"] == 9 * 96
    assert audit["statistically_independent_samples"] is True
    assert audit["overlap_pairs"] == []
    assert audit["overlap_note"] is None

    batch = build_forecast_batch(spec, frame, plan)
    assert batch.targets_kw.shape == (9, 96)
    assert np.all(batch.target_valid_mask)

    development = plan_origins(spec, grid, protocol, phase="development", frame=frame)
    assert development.days == (14, 16, 18, 20)


def test_fixed_7d_preset_labels_overlap_and_are_not_independent() -> None:
    frame = make_frame(days=31)
    spec = make_spec()
    grid = grid_for_frame(spec, frame)
    protocol = load_protocol(FIXED_7D)

    plan = plan_origins(spec, grid, protocol, phase="final-test", frame=frame)
    assert plan.days == (22, 23, 24)
    assert plan.horizon_points == 672
    assert plan.overlap is True

    audit = plan.audit()
    assert audit["statistically_independent_samples"] is False
    assert len(audit["overlap_pairs"]) == 2
    assert audit["overlap_pairs"][0]["overlapping_points"] == 576
    assert audit["overlap_pairs"][1]["day_a"] == 23
    assert "not independent" in (audit["overlap_note"] or "")
    assert audit["covered_label_days"] == list(range(23, 32))
    assert audit["n_label_points"] == 3 * 672
    assert audit["n_unique_label_timestamps"] < audit["n_label_points"]
    # Coverage reaches the end of the last trace day of the 31-day evaluation grid.
    assert plan.label_end == grid.timestamps[-1]
    assert grid.day_number(plan.label_end) == 31

    batch = build_forecast_batch(spec, frame, plan)
    assert batch.targets_kw.shape == (3, 672)

    development = plan_origins(spec, grid, protocol, phase="development", frame=frame)
    assert development.days == (14, 15)
    assert development.overlap is True


def test_audit_split_reports_per_window_slot_coverage() -> None:
    frame = make_frame(days=31)
    # Slots 2213-2214 belong to trace day 24, which is the label block of the window whose
    # origin is trace day 23 (origins are shifted one day from their label days).
    removed = frame.index[[23 * PPD + 5, 23 * PPD + 6]]
    broken = frame.drop(removed)
    spec = make_spec()
    grid = grid_for_frame(spec, broken)
    protocol = load_protocol(FIXED_24H)

    plan = plan_origins(spec, grid, protocol, phase="final-test", frame=broken)
    payload = audit_split(spec, broken, plan)

    assert payload["dataset"]["dataset_id"] == "synthetic_demo"
    assert payload["dataset"]["domain_id"] == "domain_a"
    assert payload["dataset"]["timezone_verified"] is True
    assert payload["target"]["unit"] == "kw"
    assert payload["anchor_policy"] == "dataset_explicit"
    assert payload["frame"]["rows"] == 31 * PPD - 2
    assert payload["frame"]["on_trace_grid"] is True
    assert payload["frame"]["off_grid_rows"] == 0

    by_day = {entry["day"]: entry for entry in payload["origins"]}
    # Window 23's labels cover trace day 24, so its two removed slots invalidate the target.
    assert by_day[23]["missing_label_points"] == 2
    assert by_day[23]["label_complete"] is False
    assert by_day[23]["target_missing_points"] == 2
    assert by_day[23]["usable_supervised_window"] is False
    # Every later window's *context* contains the removed slots: incomplete history is
    # reported, not repaired by interpolation or by shifting the window.
    assert by_day[24]["missing_context_points"] == 2
    assert by_day[24]["context_complete"] is False
    assert by_day[24]["usable_supervised_window"] is False
    assert by_day[30]["usable_supervised_window"] is False
    assert by_day[22]["usable_supervised_window"] is True
    assert by_day[22]["context_complete"] is True
    assert by_day[22]["label_complete"] is True
    assert payload["n_usable_windows"] == 1
    assert payload["all_windows_usable"] is False

    # Every origin record exposes its concrete times and trace-day numbers.
    first = payload["origins"][0]
    assert first["day"] == 22
    assert first["label_days"] == [23]
    assert first["origin"] == plan.origins[0].isoformat()
    assert first["last_context"] == (plan.origins[0] - STEP).isoformat()
    assert first["required_history_points"] == 1344
