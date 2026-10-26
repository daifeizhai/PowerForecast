"""Tests for the existing preprocessing layer (quality report, causal filling, writing).

These exercise ``src/data/preprocessing.py`` as reused Stage 1 infrastructure: the
quality report, the causal forward-only fill (target untouched), the versioned
processed output with JSON sidecar, and the contract-level rejection of invalid
frames (naive time index, duplicates, missing target, non-finite values, and a
utilisation / node target disguised as facility kW).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.contracts import (
    AuditStatus,
    ContractError,
    DatasetRole,
    DatasetSpec,
    FeatureSpec,
    IntervalStatistic,
    LabelOrigin,
    PhysicalUnitError,
    TargetEligibilityError,
    TargetKind,
    TargetScope,
    TargetSpec,
    require_facility_active_power_kw,
)
from src.data.preprocessing import (
    TARGET_COLUMN,
    fill_causal,
    quality_report,
    validate_frame,
    write_processed,
)
from src.utils import file_metadata, read_json

COVARIATE = "outdoor_air_temp_c"
POINTS_PER_DAY = 96


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _measured_target(**overrides) -> TargetSpec:
    payload = dict(
        kind=TargetKind.ACTIVE_POWER,
        scope=TargetScope.FACILITY,
        unit="kW",
        label_origin=LabelOrigin.MEASURED,
        measurement_boundary="test facility total meter",
        sampling_semantics="15-minute interval mean",
    )
    payload.update(overrides)
    return TargetSpec(**payload)


def _feature(**overrides) -> FeatureSpec:
    payload = dict(
        raw_name=COVARIATE,
        canonical_name=COVARIATE,
        unit="degC",
        statistic=IntervalStatistic.INTERVAL_MEAN,
        physical_range=(-60.0, 60.0),
        native_frequency="15min",
        future_known=False,
        missing_policy="mask",
        source="test",
        audit_status=AuditStatus.VERIFIED,
    )
    payload.update(overrides)
    return FeatureSpec(**payload)


def _spec(
    *,
    target: TargetSpec | None = None,
    features: list[FeatureSpec] | None = None,
    role: DatasetRole = DatasetRole.TARGET_SUPERVISED,
    timezone_verified: bool = True,
) -> DatasetSpec:
    feature_list = features if features is not None else [_feature()]
    return DatasetSpec(
        dataset_id="test_dataset",
        site_id="test_site",
        domain_id="test_domain",
        role=role,
        native_frequency="15min",
        trace_anchor_timestamp="2024-01-01T00:00:00Z",
        timezone="UTC",
        timezone_verified=timezone_verified,
        interval_semantics="15-minute interval mean, label at interval start",
        availability_semantics="labels available at the end of their own interval",
        features={spec.canonical_name: spec for spec in feature_list},
        target=_measured_target() if target is None and role is DatasetRole.TARGET_SUPERVISED else target,
        source_version="test-v1",
        audit_status=AuditStatus.VERIFIED,
    )


def _frame(days: int = 2) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01", periods=days * POINTS_PER_DAY, freq="15min", tz="UTC", name="timestamp"
    )
    positions = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            TARGET_COLUMN: 500.0 + 10.0 * np.sin(2.0 * np.pi * positions / POINTS_PER_DAY),
            COVARIATE: 12.0 + 3.0 * np.cos(2.0 * np.pi * positions / POINTS_PER_DAY),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# quality report
# ---------------------------------------------------------------------------


def test_quality_report_counts_grid_and_target_coverage() -> None:
    spec = _spec()
    report = quality_report(_frame(days=2), spec)
    assert report["rows"] == 2 * POINTS_PER_DAY
    assert report["duplicate_timestamps"] == 0
    assert report["expected_interval_count"] == 2 * POINTS_PER_DAY
    assert report["missing_intervals"] == 0
    assert report["off_grid_rows"] == 0
    assert report["rows_before_anchor"] == 0
    assert report["trace_day_count"] == 2
    assert report["complete_trace_days"] == 2
    assert report["complete_trace_days_with_finite_target"] == 2
    assert report["anchor_matches_first_timestamp"] is True
    assert report["timezone_verified"] is True
    assert report["target_present"] is True
    assert report["target_coverage_fraction"] == pytest.approx(1.0)
    assert report["target_negative_count"] == 0
    assert report["nan_counts"][COVARIATE] == 0
    json.dumps(report)

    frame = _frame(days=2)
    frame = frame.drop([frame.index[10], frame.index[100], frame.index[150]])
    frame.loc[frame.index[20], TARGET_COLUMN] = np.nan
    report = quality_report(frame, spec)
    assert report["rows"] == 2 * POINTS_PER_DAY - 3
    assert report["missing_intervals"] == 3
    assert report["target_present"] is True
    assert report["target_coverage_count"] == 2 * POINTS_PER_DAY - 4
    # Dropping interior slots breaks whole-day completeness; numbering must not compress.
    assert report["complete_trace_days"] == 0
    assert report["complete_trace_days_with_finite_target"] == 0


def test_quality_report_requires_timezone_aware_datetime_index() -> None:
    frame = _frame(days=1)
    spec = _spec()
    naive = frame.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ContractError):
        quality_report(naive, spec)
    with pytest.raises(ContractError):
        quality_report(frame.reset_index(drop=True), spec)


def test_quality_report_handles_unlabeled_dataset() -> None:
    frame = _frame(days=1).drop(columns=[TARGET_COLUMN])
    spec = _spec(role=DatasetRole.UNLABELED_PRETRAIN, target=None)
    report = quality_report(frame, spec)
    assert report["target_present"] is False
    assert report["target_coverage_count"] is None
    assert report["complete_trace_days_with_finite_target"] is None
    assert report["declared_features"] == [COVARIATE]


def test_quality_report_counts_duplicate_timestamps() -> None:
    frame = _frame(days=1)
    duplicated = pd.concat([frame, frame.iloc[[3]]])
    report = quality_report(duplicated, _spec())
    assert report["duplicate_timestamps"] == 1
    # validate_frame is the structural gate that rejects duplicates.
    with pytest.raises(ContractError):
        validate_frame(duplicated, _spec())


# ---------------------------------------------------------------------------
# structural validation
# ---------------------------------------------------------------------------


def test_validate_frame_accepts_good_frame_and_reports_diagnostics() -> None:
    frame = _frame(days=1)
    frame.loc[frame.index[0], TARGET_COLUMN] = -5.0
    frame.loc[frame.index[1], COVARIATE] = 100.0  # outside the declared (-60, 60) range
    result = validate_frame(frame, _spec())
    assert result["valid"] is True
    assert result["target_negative_count"] == 1
    assert result["physical_range_violations"] == {COVARIATE: 1}
    assert result["physical_range_violations_are_diagnostics_only"] is True
    assert result["negative_target_is_diagnostic_not_clipped"] is True
    # Values were reported, not clipped.
    assert frame.loc[frame.index[0], TARGET_COLUMN] == -5.0
    assert frame.loc[frame.index[1], COVARIATE] == 100.0


def _mutators() -> dict:
    frame = _frame(days=1)

    def naive_index() -> pd.DataFrame:
        copy = frame.copy()
        copy.index = copy.index.tz_localize(None)
        return copy

    def duplicate() -> pd.DataFrame:
        return pd.concat([frame, frame.iloc[[5]]])

    def unsorted() -> pd.DataFrame:
        return frame.iloc[::-1]

    def wrong_index_name() -> pd.DataFrame:
        copy = frame.copy()
        copy.index.name = "time"
        return copy

    def missing_target() -> pd.DataFrame:
        return frame.drop(columns=[TARGET_COLUMN])

    def missing_feature() -> pd.DataFrame:
        return frame.drop(columns=[COVARIATE])

    def infinite_value() -> pd.DataFrame:
        copy = frame.copy()
        copy.loc[copy.index[7], COVARIATE] = np.inf
        return copy

    def non_numeric() -> pd.DataFrame:
        copy = frame.copy()
        copy[COVARIATE] = "warm"
        return copy

    return {
        "naive_index": naive_index,
        "duplicate": duplicate,
        "unsorted": unsorted,
        "wrong_index_name": wrong_index_name,
        "missing_target": missing_target,
        "missing_feature": missing_feature,
        "infinite_value": infinite_value,
        "non_numeric": non_numeric,
    }


@pytest.mark.parametrize("case", sorted(_mutators()))
def test_validate_frame_rejects_structural_violations(case: str) -> None:
    with pytest.raises(ContractError):
        validate_frame(_mutators()[case](), _spec())


def test_validate_frame_rejects_utilization_target() -> None:
    utilization = TargetSpec(
        kind=TargetKind.POWER_UTILIZATION,
        scope=TargetScope.FACILITY,
        unit="percent",
        label_origin=LabelOrigin.POWER_UTILIZATION,
        measurement_boundary="relative utilisation panel",
        sampling_semantics="reported interval",
    )
    with pytest.raises(PhysicalUnitError):
        validate_frame(_frame(days=1), _spec(target=utilization))


def test_validate_frame_rejects_target_on_unlabeled_dataset() -> None:
    spec = _spec(role=DatasetRole.UNLABELED_PRETRAIN, target=None)
    with pytest.raises(PhysicalUnitError):
        validate_frame(_frame(days=1), spec)


def test_validate_frame_rejects_node_scope_for_main_task() -> None:
    node = _measured_target(scope=TargetScope.NODE, unit="W", measurement_boundary="node PSU")
    # The preprocessing validator only enforces active_power; the facility gate rejects scope.
    with pytest.raises(TargetEligibilityError):
        require_facility_active_power_kw(node)


# ---------------------------------------------------------------------------
# causal filling
# ---------------------------------------------------------------------------


def test_fill_causal_is_forward_only_covariate_only() -> None:
    frame = _frame(days=1)
    frame.iloc[0:2, frame.columns.get_loc(COVARIATE)] = np.nan  # leading gap: never back-filled
    frame.iloc[8:12, frame.columns.get_loc(COVARIATE)] = np.nan  # 4-step gap, limit 2
    frame.iloc[5, frame.columns.get_loc(TARGET_COLUMN)] = np.nan

    filled, audit = fill_causal(frame, _spec(), fill_limit=2)
    covariate = filled[COVARIATE]
    assert covariate.iloc[0:2].isna().all()
    assert covariate.iloc[8:10].notna().all()
    assert pd.isna(covariate.iloc[10])
    assert pd.isna(covariate.iloc[11])
    # Filled values come from the known past, never from the future.
    assert covariate.iloc[8] == frame[COVARIATE].iloc[7]
    # The target is byte-identically untouched (its NaN stays NaN).
    assert pd.isna(filled[TARGET_COLUMN].iloc[5])
    np.testing.assert_array_equal(
        frame[TARGET_COLUMN].to_numpy(dtype=float, na_value=np.nan),
        filled[TARGET_COLUMN].to_numpy(dtype=float, na_value=np.nan),
    )
    assert audit["policy"] == "causal_ffill_covariates_only"
    assert audit["fill_direction"] == "forward_only"
    assert audit["backward_fill_used"] is False
    assert audit["target_column"] == TARGET_COLUMN
    assert audit["target_filled"] == 0
    assert audit["target_values_modified"] is False
    assert audit["filled_counts"][COVARIATE] == 2
    assert audit["total_filled_cells"] == 2
    json.dumps(audit)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_fill_causal_rejects_invalid_limit(limit) -> None:
    with pytest.raises(ValueError):
        fill_causal(_frame(days=1), _spec(), fill_limit=limit)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# versioned processed output
# ---------------------------------------------------------------------------


def test_write_processed_writes_versioned_parquet_and_sidecar(tmp_path: Path) -> None:
    frame = _frame(days=1)
    payload = write_processed(frame, _spec(), out_dir=tmp_path, tag="unit_test")
    assert Path(payload["file"]).is_file()
    assert Path(payload["sidecar"]).is_file()
    assert payload["sha256"] == file_metadata(payload["file"])["sha256"]
    assert payload["quality"]["rows"] == POINTS_PER_DAY
    assert payload["sampling"]["frequency"] == "15min"
    assert payload["sampling"]["trace_anchor_timestamp"] == "2024-01-01T00:00:00Z"
    assert payload["timezone"]["verified"] is True
    assert payload["synthetic"] is False
    assert payload["field_mapping"]
    assert payload["unit_conversion"]["target"]["declared_unit"] == "kW"
    assert payload["unit_conversion"]["target"]["stored_unit"] == "kW"
    assert payload["unit_conversion"]["target"]["multiply_by"] == pytest.approx(1.0)

    sidecar = read_json(payload["sidecar"])
    assert sidecar["dataset_id"] == "test_dataset"
    assert sidecar["sha256"] == payload["sha256"]
    assert sidecar["quality"]["rows"] == POINTS_PER_DAY
    assert sidecar["validation"]["valid"] is True
    assert sidecar["provenance"]["generator"] == "src.data.preprocessing.write_processed"

    # A second write must create a NEW versioned file, never overwrite the first.
    second = write_processed(frame, _spec(), out_dir=tmp_path, tag="unit_test")
    assert second["file"] != payload["file"]
    assert Path(second["file"]).is_file()
    assert Path(payload["file"]).is_file()


def test_write_processed_refuses_data_raw_destination() -> None:
    forbidden = "data/raw/preprocessing_must_not_be_written"
    with pytest.raises(ContractError):
        write_processed(_frame(days=1), _spec(), out_dir=forbidden, tag="unit_test")
    assert not Path(forbidden).exists()


def test_write_processed_rejects_invalid_frame(tmp_path: Path) -> None:
    naive = _frame(days=1)
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ContractError):
        write_processed(naive, _spec(), out_dir=tmp_path, tag="unit_test")
    duplicated = pd.concat([_frame(days=1), _frame(days=1).iloc[[0]]])
    with pytest.raises(ContractError):
        write_processed(duplicated, _spec(), out_dir=tmp_path, tag="unit_test")


def test_write_processed_unlabeled_dataset_has_no_target(tmp_path: Path) -> None:
    frame = _frame(days=1).drop(columns=[TARGET_COLUMN])
    spec = _spec(role=DatasetRole.UNLABELED_PRETRAIN, target=None)
    payload = write_processed(frame, spec, out_dir=tmp_path, tag="unit_unlabeled")
    assert payload["quality"]["target_present"] is False
    assert payload["unit_conversion"]["target"] is None
    written = pd.read_parquet(payload["file"])
    assert TARGET_COLUMN not in written.columns
