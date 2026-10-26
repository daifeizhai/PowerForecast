"""Data-contract acceptance tests (task book T03, T04, T05, T08, T13).

The contracts are the boundary that makes "unlabelled data cannot leak power
labels", "utilisation is not kW" and "node power is not facility power" true by
construction rather than by convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.contracts import (
    AuditStatus,
    ContractError,
    DatasetRole,
    DatasetSpec,
    FeatureSpec,
    ForecastBatch,
    ForecastInputs,
    InsufficientDataError,
    IntervalStatistic,
    LabelOrigin,
    LeakageError,
    PhysicalUnitError,
    PretrainBatch,
    TargetEligibilityError,
    TargetKind,
    TargetScope,
    TargetSpec,
    from_kw,
    require_facility_active_power_kw,
    require_pretrain_batch,
    to_kw,
)

POINTS_PER_DAY = 96


def facility_synthetic_target(**overrides) -> TargetSpec:
    payload = dict(
        kind=TargetKind.ACTIVE_POWER,
        scope=TargetScope.FACILITY,
        unit="kW",
        label_origin=LabelOrigin.SYNTHETIC_SIMULATED,
        measurement_boundary="simulated facility bus",
        sampling_semantics="15-minute interval mean",
        synthetic=True,
    )
    payload.update(overrides)
    return TargetSpec(**payload)


def facility_measured_target(**overrides) -> TargetSpec:
    payload = dict(
        kind=TargetKind.ACTIVE_POWER,
        scope=TargetScope.FACILITY,
        unit="kW",
        label_origin=LabelOrigin.MEASURED,
        measurement_boundary="facility total meter",
        sampling_semantics="15-minute interval mean",
        synthetic=False,
    )
    payload.update(overrides)
    return TargetSpec(**payload)


def utilisation_target() -> TargetSpec:
    return TargetSpec(
        kind=TargetKind.POWER_UTILIZATION,
        scope=TargetScope.FACILITY,
        unit="percent",
        label_origin=LabelOrigin.POWER_UTILIZATION,
        measurement_boundary="relative utilisation panel",
        sampling_semantics="reported utilisation interval",
    )


def node_power_target() -> TargetSpec:
    return TargetSpec(
        kind=TargetKind.ACTIVE_POWER,
        scope=TargetScope.NODE,
        unit="W",
        label_origin=LabelOrigin.MEASURED,
        measurement_boundary="node PSU input",
        sampling_semantics="5-minute instantaneous sample",
    )


def timestamps(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="15min", tz="UTC")


def make_inputs(
    *,
    batch: int = 1,
    length: int = POINTS_PER_DAY,
    horizon: int = 16,
    target: TargetSpec | None = None,
    with_covariates: bool = False,
    with_future_known: bool = False,
    covariate_available_at: np.ndarray | None = None,
    future_known_available_at: np.ndarray | None = None,
) -> ForecastInputs:
    target = target or facility_synthetic_target()
    total = length + horizon
    stamps = timestamps(total)
    origins = np.tile(stamps[length].to_datetime64(), batch)
    past = np.tile(stamps[:length].to_numpy(), (batch, 1))
    future = np.tile(stamps[length:total].to_numpy(), (batch, 1))
    kwargs = {}
    if with_covariates:
        kwargs.update(
            past_covariates=np.ones((batch, length, 2)),
            past_covariate_mask=np.ones((batch, length, 2), dtype=bool),
            covariate_names=("cpu_util", "mem_util"),
        )
        if covariate_available_at is not None:
            kwargs["past_covariate_available_at"] = covariate_available_at
    if with_future_known:
        kwargs.update(
            future_known=np.ones((batch, horizon, 2)),
            future_known_mask=np.ones((batch, horizon, 2), dtype=bool),
            future_known_names=("slot_sin", "slot_cos"),
        )
        if future_known_available_at is not None:
            kwargs["future_known_available_at"] = future_known_available_at
    return ForecastInputs(
        past_power_kw=np.full((batch, length), 500.0),
        past_power_valid=np.ones((batch, length), dtype=bool),
        past_timestamps=past,
        origin_timestamps=origins,
        target_timestamps=future,
        target_spec=target,
        domain_ids=tuple(f"domain-{i}" for i in range(batch)),
        dataset_id="contract_test",
        **kwargs,
    )


def make_batch(target: TargetSpec | None = None, *, batch: int = 1, horizon: int = 16) -> ForecastBatch:
    inputs = make_inputs(batch=batch, horizon=horizon, target=target)
    return ForecastBatch(
        inputs=inputs,
        targets_kw=np.full((batch, horizon), 500.0),
        target_valid_mask=np.ones((batch, horizon), dtype=bool),
    )


# ---------------------------------------------------------------------------
# T03 - unlabelled data is structurally label-free
# ---------------------------------------------------------------------------


def test_unlabelled_dataset_spec_is_legal_without_target() -> None:
    spec = DatasetSpec(
        dataset_id="alibaba_gpu",
        site_id="alibaba",
        domain_id="alibaba_gpu",
        role=DatasetRole.UNLABELED_PRETRAIN,
        native_frequency="1min",
        features={"cpu_util": FeatureSpec("cpu_util", "cpu_util", "ratio")},
        target=None,
        audit_status=AuditStatus.PARTIAL,
    )
    assert spec.target is None
    assert spec.synthetic is False


def test_unlabelled_dataset_spec_rejects_a_power_label() -> None:
    with pytest.raises(ContractError, match="target=None"):
        DatasetSpec(
            dataset_id="bad",
            site_id="s",
            domain_id="d",
            role=DatasetRole.UNLABELED_PRETRAIN,
            features={},
            target=facility_measured_target(),
        )


def test_target_supervised_requires_a_declared_target() -> None:
    with pytest.raises(ContractError, match="TargetSpec"):
        DatasetSpec(
            dataset_id="bad",
            site_id="s",
            domain_id="d",
            role=DatasetRole.TARGET_SUPERVISED,
            timezone="UTC",
            timezone_verified=True,
            trace_anchor_timestamp="2024-01-01T00:00:00Z",
            interval_semantics="15min interval mean",
            availability_semantics="reported at interval end",
            features={},
            target=None,
        )


def test_pretrain_batch_has_no_power_label_attribute() -> None:
    stamps = timestamps(4, start="2024-01-01")
    batch = PretrainBatch(
        features=np.ones((1, 4, 2)),
        observed_mask=np.ones((1, 4, 2), dtype=bool),
        timestamps=stamps.to_numpy()[None, :],
        available_at=np.broadcast_to(stamps.to_numpy()[None, :, None], (1, 4, 2)).copy(),
        feature_names=("cpu_util", "mem_util"),
        domain_ids=("source_a",),
        sampling_metadata={"native_frequency": "1min"},
    )
    for forbidden in ("targets_kw", "targets", "future_kw", "power_labels"):
        assert not hasattr(batch, forbidden)
    with pytest.raises(ContractError, match="PretrainBatch only"):
        require_pretrain_batch(make_batch())


def test_pretrain_batch_keeps_missing_values_masked() -> None:
    stamps = timestamps(3, start="2024-01-01")
    features = np.array([[[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]]])
    mask = np.array([[[True, False], [True, True], [False, True]]])
    batch = PretrainBatch(
        features=features,
        observed_mask=mask,
        timestamps=stamps.to_numpy()[None, :],
        available_at=np.broadcast_to(stamps.to_numpy()[None, :, None], (1, 3, 2)).copy(),
        feature_names=("cpu_util", "mem_util"),
        domain_ids=("source_a",),
    )
    assert bool(batch.observed_mask[0, 0, 1]) is False
    assert np.isnan(batch.features[0, 0, 1])
    # A "missing" entry must not be reported as a real zero observation.
    assert batch.features[0, 0, 1] != 0.0
    # Claiming a value exists while it is non-finite must fail.
    with pytest.raises(ContractError, match="non-finite"):
        PretrainBatch(
            features=np.array([[[np.nan, 1.0]]]),
            observed_mask=np.array([[[True, True]]]),
            timestamps=stamps.to_numpy()[None, :1],
            available_at=np.broadcast_to(stamps.to_numpy()[None, :1, None], (1, 1, 2)).copy(),
            feature_names=("cpu_util", "mem_util"),
            domain_ids=("source_a",),
        )


# ---------------------------------------------------------------------------
# T04 - units and physical boundaries
# ---------------------------------------------------------------------------


def test_watt_to_kilowatt_conversion() -> None:
    assert to_kw([1000.0, 2500.0], "W").tolist() == [1.0, 2.5]
    assert to_kw([1.0], "MW").tolist() == [1000.0]
    assert from_kw([1.0], "W").tolist() == [1000.0]
    with pytest.raises(PhysicalUnitError):
        to_kw([50.0], "percent")
    with pytest.raises(PhysicalUnitError):
        to_kw([0.5], "ratio")


def test_utilisation_cannot_become_power() -> None:
    with pytest.raises(TargetEligibilityError):
        require_facility_active_power_kw(utilisation_target())
    with pytest.raises(PhysicalUnitError, match="power unit"):
        # A utilisation target cannot declare a kW unit.
        TargetSpec(
            kind=TargetKind.POWER_UTILIZATION,
            scope=TargetScope.FACILITY,
            unit="kW",
            label_origin=LabelOrigin.POWER_UTILIZATION,
            measurement_boundary="x",
            sampling_semantics="y",
        )


def test_utilisation_label_cannot_be_stored_as_kw_targets() -> None:
    # ForecastInputs already demands an active_power/power-unit target.
    with pytest.raises(PhysicalUnitError):
        make_inputs(target=utilisation_target())


def test_forecast_batch_requires_declared_kw_unit() -> None:
    with pytest.raises(PhysicalUnitError, match="kW"):
        make_batch(target=facility_measured_target(unit="W"))


def test_node_power_is_not_facility_power() -> None:
    node = node_power_target()
    with pytest.raises(TargetEligibilityError, match="facility"):
        require_facility_active_power_kw(node)
    # Even as an auxiliary task it must never be silently promoted.
    with pytest.raises(TargetEligibilityError, match="facility"):
        require_facility_active_power_kw(node, allow_synthetic=True)


def test_facility_measured_target_is_eligible_for_formal_task() -> None:
    require_facility_active_power_kw(facility_measured_target())
    with pytest.raises(TargetEligibilityError, match="synthetic"):
        require_facility_active_power_kw(facility_synthetic_target())
    # Synthetic labels are allowed only in explicitly synthetic/demo contexts.
    require_facility_active_power_kw(facility_synthetic_target(), allow_synthetic=True)


# ---------------------------------------------------------------------------
# T05 - label honesty
# ---------------------------------------------------------------------------


def test_synthetic_label_must_declare_synthetic_origin() -> None:
    with pytest.raises(TargetEligibilityError, match="synthetic_simulated"):
        TargetSpec(
            kind=TargetKind.ACTIVE_POWER,
            scope=TargetScope.FACILITY,
            unit="kW",
            label_origin=LabelOrigin.MEASURED,
            measurement_boundary="fake meter",
            sampling_semantics="x",
            synthetic=True,
        )
    with pytest.raises(TargetEligibilityError, match="synthetic=True"):
        TargetSpec(
            kind=TargetKind.ACTIVE_POWER,
            scope=TargetScope.FACILITY,
            unit="kW",
            label_origin=LabelOrigin.SYNTHETIC_SIMULATED,
            measurement_boundary="fake meter",
            sampling_semantics="x",
            synthetic=False,
        )


def test_component_sum_keeps_its_origin() -> None:
    component = facility_measured_target(label_origin=LabelOrigin.COMPONENT_SUM)
    assert component.to_dict()["label_origin"] == "component_sum"
    assert component.to_dict()["label_origin"] != "measured"
    require_facility_active_power_kw(component)


# ---------------------------------------------------------------------------
# T08 - F=0 / K=0 remain legal, and leakage is rejected at construction
# ---------------------------------------------------------------------------


def test_power_only_inputs_are_legal() -> None:
    inputs = make_inputs(with_covariates=False, with_future_known=False)
    assert inputs.n_covariates == 0
    assert inputs.n_future_known == 0
    assert inputs.past_covariates is None
    assert inputs.future_known is None
    batch = ForecastBatch(
        inputs=inputs,
        targets_kw=np.ones((1, inputs.horizon)),
        target_valid_mask=np.ones((1, inputs.horizon), dtype=bool),
    )
    assert batch.targets_kw.shape == (1, 16)


def test_origin_must_equal_first_target_timestamp() -> None:
    inputs = make_inputs()
    shifted = timestamps(200)[96 + 1]
    with pytest.raises(ContractError, match="origin_timestamps"):
        ForecastInputs(
            past_power_kw=inputs.past_power_kw,
            past_power_valid=inputs.past_power_valid,
            past_timestamps=inputs.past_timestamps,
            origin_timestamps=np.array([shifted.to_datetime64()]),
            target_timestamps=inputs.target_timestamps,
            target_spec=inputs.target_spec,
            domain_ids=inputs.domain_ids,
        )


def test_context_observation_at_or_after_origin_is_leakage() -> None:
    stamps = timestamps(200)
    past = stamps[1:97].to_numpy()[None, :]  # last context slot == origin
    with pytest.raises(LeakageError):
        ForecastInputs(
            past_power_kw=np.ones((1, 96)),
            past_power_valid=np.ones((1, 96), dtype=bool),
            past_timestamps=past,
            origin_timestamps=np.array([stamps[96].to_datetime64()]),
            target_timestamps=stamps[96:112].to_numpy()[None, :],
            target_spec=facility_synthetic_target(),
            domain_ids=("d",),
        )


def test_future_covariate_available_after_origin_is_leakage() -> None:
    stamps = timestamps(200)
    available = np.broadcast_to(
        stamps[96:192].to_numpy()[None, :, None], (1, 96, 2)
    ).copy()  # context slots, but availability says "at origin+something"
    available[:, -1, :] = stamps[150].to_datetime64()
    with pytest.raises(LeakageError):
        make_inputs(with_covariates=True, covariate_available_at=available)


def test_late_future_known_feature_is_leakage() -> None:
    available = np.full((1, 16, 2), timestamps(300)[100].to_datetime64())
    with pytest.raises(LeakageError):
        make_inputs(with_future_known=True, future_known_available_at=available)


def test_calendar_future_known_at_origin_is_legal() -> None:
    inputs = make_inputs(with_future_known=True)
    assert inputs.future_known is not None
    assert (inputs.future_known_available_at <= inputs.origin_timestamps[:, None, None]).all()


def test_masked_covariate_may_carry_unavailable_metadata() -> None:
    """Only usable (mask=True) entries are judged for availability."""
    inputs = make_inputs(with_covariates=True)
    available = np.broadcast_to(
        inputs.past_timestamps[:, :, None], (1, 96, 2)
    ).copy()
    available[:, -1, :] = timestamps(200)[150].to_datetime64()
    mask = np.ones((1, 96, 2), dtype=bool)
    mask[:, -1, :] = False
    rebuilt = ForecastInputs(
        past_power_kw=inputs.past_power_kw,
        past_power_valid=inputs.past_power_valid,
        past_timestamps=inputs.past_timestamps,
        origin_timestamps=inputs.origin_timestamps,
        target_timestamps=inputs.target_timestamps,
        target_spec=inputs.target_spec,
        domain_ids=inputs.domain_ids,
        past_covariates=inputs.past_covariates,
        past_covariate_mask=mask,
        past_covariate_available_at=available,
        covariate_names=inputs.covariate_names,
    )
    assert rebuilt.past_covariate_mask is not None
    assert rebuilt.past_covariate_mask[:, -1, :].sum() == 0


# ---------------------------------------------------------------------------
# batch shape / target mask validation
# ---------------------------------------------------------------------------


def test_batch_shape_mismatch_is_rejected() -> None:
    inputs = make_inputs(horizon=16)
    with pytest.raises(ContractError, match="targets_kw"):
        ForecastBatch(
            inputs=inputs,
            targets_kw=np.ones((1, 15)),
            target_valid_mask=np.ones((1, 16), dtype=bool),
        )


def test_non_finite_valid_target_is_rejected() -> None:
    inputs = make_inputs(horizon=16)
    targets = np.ones((1, 16))
    targets[0, 3] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        ForecastBatch(
            inputs=inputs,
            targets_kw=targets,
            target_valid_mask=np.ones((1, 16), dtype=bool),
        )


def test_invalid_labels_are_masked_not_zeroed() -> None:
    inputs = make_inputs(horizon=16)
    targets = np.full((1, 16), np.nan)
    mask = np.zeros((1, 16), dtype=bool)
    targets[0, :4] = 400.0
    mask[0, :4] = True
    batch = ForecastBatch(inputs=inputs, targets_kw=targets, target_valid_mask=mask)
    assert batch.target_valid_mask.sum() == 4
    assert np.isnan(batch.targets_kw[0, 5])


def test_insufficient_data_error_is_a_contract_error() -> None:
    assert issubclass(InsufficientDataError, ContractError)
