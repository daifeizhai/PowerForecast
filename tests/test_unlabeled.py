"""Acceptance tests for the unlabelled/labelled data split and synthetic sources.

Covers task-book items T03 (unlabelled source usable without labels), T06
(heterogeneous fields / masks), and the data-side of T04/T05 (semantic unit
conversion, label honesty, no fabricated facility-kW results).

Every test is fast, CPU-only, uses ``tmp_path`` for outputs, and never writes to
the project ``data/processed/`` or ``results/`` trees.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.adapters import adapt_frame, mapping_table
from src.data.contracts import (
    TARGET_COLUMN,
    AuditStatus,
    ContractError,
    DatasetRole,
    DatasetSpec,
    FeatureSpec,
    ForecastBatch,
    ForecastInputs,
    IntervalStatistic,
    LabelOrigin,
    PhysicalUnitError,
    PretrainBatch,
    TargetKind,
    TargetScope,
    TargetSpec,
    as_datetime64,
    require_pretrain_batch,
)
from src.data.loaders import (
    assert_not_raw_path,
    inspect_table,
    iter_table_batches,
    read_table,
    write_table,
)
from src.data.preprocessing import fill_causal, quality_report, validate_frame, write_processed
from src.data.synthetic import (
    SYNTHETIC_DATASET_ID,
    SyntheticSource,
    build_synthetic_dataset,
    unlabeled_frames,
)

HORIZON = 16
CONTEXT = 96
MISSING_DAY = 3
MISSING_START = MISSING_DAY * 96 + 40
MISSING_STOP = MISSING_START + 8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic():
    """One 31-day, seeded synthetic dataset shared by read-only tests."""
    return build_synthetic_dataset(days=31, seed=7)


def _pretrain_parts(source: SyntheticSource, *, length: int = 96) -> dict:
    """Keyword arguments for a PretrainBatch built from an unlabelled source."""
    frame = source.frame.iloc[:length]
    feature_names = tuple(frame.columns)
    values = frame.to_numpy(dtype=float)
    observed = np.isfinite(values)
    stamps = as_datetime64(frame.index)
    available_at = np.empty((1, length, len(feature_names)), dtype="datetime64[ns]")
    for index, name in enumerate(feature_names):
        lag_ns = int(source.spec.features[name].availability_lag.value)
        available_at[0, :, index] = stamps + np.timedelta64(lag_ns, "ns")
    return {
        "features": values[None, :, :].copy(),
        "observed_mask": observed[None, :, :].copy(),
        "timestamps": stamps[None, :].copy(),
        "available_at": available_at,
        "feature_names": feature_names,
        "domain_ids": (source.dataset_id,),
        "sampling_metadata": {
            "native_frequency": source.spec.native_frequency,
            "availability_semantics": source.spec.availability_semantics,
        },
        "dataset_id": source.dataset_id,
    }


def _pretrain_batch(source: SyntheticSource, *, length: int = 96) -> PretrainBatch:
    return PretrainBatch(**_pretrain_parts(source, length=length))


def _forecast_batch(dataset) -> ForecastBatch:
    frame = dataset.frame
    origin_position = CONTEXT
    horizon = HORIZON
    past = frame[TARGET_COLUMN].to_numpy(dtype=float)[:origin_position]
    inputs = ForecastInputs(
        past_power_kw=past[None, :],
        past_power_valid=np.isfinite(past)[None, :],
        past_timestamps=as_datetime64(frame.index[:origin_position])[None, :],
        origin_timestamps=as_datetime64(frame.index[origin_position : origin_position + 1]),
        target_timestamps=as_datetime64(
            frame.index[origin_position : origin_position + horizon]
        )[None, :],
        target_spec=dataset.spec.target,
        domain_ids=(dataset.spec.domain_id,),
        dataset_id=dataset.spec.dataset_id,
    )
    targets = frame[TARGET_COLUMN].to_numpy(dtype=float)[origin_position : origin_position + horizon]
    return ForecastBatch(
        inputs=inputs,
        targets_kw=targets[None, :],
        target_valid_mask=np.isfinite(targets)[None, :],
    )


def _adapter_spec(*, target_unit: str = "W") -> DatasetSpec:
    target = TargetSpec(
        kind=TargetKind.ACTIVE_POWER,
        scope=TargetScope.FACILITY,
        unit=target_unit,
        label_origin=LabelOrigin.MEASURED,
        measurement_boundary="unit-test facility meter",
        sampling_semantics="unit-test 15-minute interval mean",
        synthetic=False,
    )
    features = {
        "cpu_util": FeatureSpec(
            raw_name="cpu_pct",
            canonical_name="cpu_util",
            unit="percent",
            statistic=IntervalStatistic.INTERVAL_MEAN,
            physical_range=(0.0, 100.0),
            source="unit-test",
        ),
        "it_power_kw": FeatureSpec(
            raw_name="it_power_w",
            canonical_name="it_power_kw",
            unit="W",
            physical_range=(0.0, 1_000_000.0),
            source="unit-test",
        ),
    }
    return DatasetSpec(
        dataset_id="adapter_test",
        site_id="adapter_test_site",
        domain_id="adapter_test",
        role=DatasetRole.TARGET_SUPERVISED,
        native_frequency="15min",
        trace_anchor_timestamp="2024-01-01T00:00:00Z",
        timezone="UTC",
        timezone_verified=True,
        interval_semantics="unit-test interval starting at the row timestamp",
        availability_semantics="unit-test values available at interval start",
        features=features,
        target=target,
        source_version="unit-test",
        provenance={"target_raw_column": "power_w"},
        audit_status=AuditStatus.PARTIAL,
    )


def _utilisation_spec() -> DatasetSpec:
    target = TargetSpec(
        kind=TargetKind.POWER_UTILIZATION,
        scope=TargetScope.FACILITY,
        unit="percent",
        label_origin=LabelOrigin.POWER_UTILIZATION,
        measurement_boundary="relative utilisation panel, no power meter",
        sampling_semantics="reported utilisation intervals",
        synthetic=False,
    )
    return DatasetSpec(
        dataset_id="utilisation_test",
        site_id="utilisation_test_site",
        domain_id="utilisation_test",
        role=DatasetRole.TARGET_SUPERVISED,
        native_frequency="15min",
        trace_anchor_timestamp="2024-01-01T00:00:00Z",
        timezone="UTC",
        timezone_verified=True,
        interval_semantics="unit-test interval starting at the row timestamp",
        availability_semantics="unit-test values available at interval start",
        features={},
        target=target,
        source_version="unit-test",
        provenance={"target_raw_column": "power_utilization_pct"},
        audit_status=AuditStatus.UNVERIFIED,
    )


# ---------------------------------------------------------------------------
# T03: unlabelled sources are legal and carry no power label
# ---------------------------------------------------------------------------


def test_unlabelled_spec_is_legal_and_batches_have_no_label(synthetic):
    for source in synthetic.unlabeled_sources:
        assert source.spec.role is DatasetRole.UNLABELED_PRETRAIN
        assert source.spec.target is None
        assert source.spec.synthetic is True
        batch = _pretrain_batch(source, length=96)
        assert not hasattr(batch, "targets_kw")
        field_names = {field.name for field in dataclasses.fields(PretrainBatch)}
        assert "targets_kw" not in field_names
        assert "targets" not in field_names
        assert "future_kw" not in field_names
        assert require_pretrain_batch(batch) is batch
        assert batch.metadata()["contains_power_labels"] is False

    # A supervised batch must be rejected at the unlabelled entry point.
    supervised = _forecast_batch(synthetic)
    assert hasattr(supervised, "targets_kw")
    with pytest.raises(ContractError, match="PretrainBatch"):
        require_pretrain_batch(supervised)


# ---------------------------------------------------------------------------
# T06: heterogeneous unlabelled sources are read independently
# ---------------------------------------------------------------------------


def test_two_unlabelled_sources_have_different_and_independent_fields(synthetic):
    sources = synthetic.unlabeled_sources
    assert len(sources) == 2
    pairs = unlabeled_frames(synthetic)
    assert tuple(source_id for source_id, _ in pairs) == tuple(
        source.dataset_id for source in sources
    )
    source_a, source_b = sources
    (id_a, frame_a), (id_b, frame_b) = pairs

    assert id_a != id_b
    assert tuple(frame_a.columns) == source_a.spec.feature_names
    assert tuple(frame_b.columns) == source_b.spec.feature_names
    assert source_a.spec.feature_names != source_b.spec.feature_names

    # Different widths: no assumption that F is shared between sources.
    assert len(source_a.spec.feature_names) == 2
    assert len(source_b.spec.feature_names) == 3
    assert set(source_a.spec.feature_names).isdisjoint(source_b.spec.feature_names)

    # Both are indexed on the same shared anchor but are not positionally merged.
    assert frame_a.index.equals(frame_b.index)
    assert frame_a.index.name == "timestamp"
    assert str(frame_a.index.tz) == "UTC"
    assert (
        source_a.spec.trace_anchor_timestamp
        == source_b.spec.trace_anchor_timestamp
        == synthetic.spec.trace_anchor_timestamp
        == "2024-01-01T00:00:00Z"
    )
    assert frame_a.index[0] == frame_b.index[0] == synthetic.frame.index[0]
    assert not frame_a.columns.equals(frame_b.columns)

    # No source contains a power column at all.
    for frame in (frame_a, frame_b):
        assert TARGET_COLUMN not in frame.columns
        assert not [name for name in frame.columns if name.endswith("_kw")]

    # A batch keeps each source's own feature names and width, never a positional union.
    batch_a = _pretrain_batch(source_a, length=96)
    batch_b = _pretrain_batch(source_b, length=96)
    assert batch_a.feature_names == source_a.spec.feature_names
    assert batch_b.feature_names == source_b.spec.feature_names
    assert batch_a.features.shape == (1, 96, 2)
    assert batch_b.features.shape == (1, 96, 3)
    np.testing.assert_allclose(
        batch_a.features[0, :, 0], frame_a["cpu_util"].to_numpy(dtype=float)[:96], equal_nan=True
    )
    np.testing.assert_allclose(
        batch_b.features[0, :, 0], frame_b["gpu_util"].to_numpy(dtype=float)[:96], equal_nan=True
    )
    # Column 0 means different semantics in the two sources.
    assert batch_a.feature_names[0] == "cpu_util"
    assert batch_b.feature_names[0] == "gpu_util"

    # Their availability semantics differ and are carried in each spec/batch.
    assert source_a.spec.features["cpu_util"].availability_lag == pd.Timedelta(0)
    assert source_b.spec.features["gpu_util"].availability_lag == pd.Timedelta(15, "min")
    assert batch_a.sampling_metadata["native_frequency"] == "15min"
    assert "available at" in batch_a.sampling_metadata["availability_semantics"].lower()


# ---------------------------------------------------------------------------
# T06 (continued): missing values stay missing; masks are honest
# ---------------------------------------------------------------------------


def test_missing_values_stay_missing_in_pretrain_batches():
    dataset = build_synthetic_dataset(days=31, seed=7, missing_days=(MISSING_DAY,))
    source = dataset.unlabeled_sources[0]
    parts = _pretrain_parts(source, length=len(source.frame))
    batch = PretrainBatch(**parts)

    values = batch.features[0]
    mask = batch.observed_mask[0]
    nan_mask = np.isnan(values)

    # observed_mask is False exactly where the value is NaN: no coincidence, no hiding.
    np.testing.assert_array_equal(mask, ~nan_mask)
    assert int(np.count_nonzero(nan_mask)) == 8 * values.shape[1]

    expected_block = np.zeros(values.shape[0], dtype=bool)
    expected_block[MISSING_START:MISSING_STOP] = True
    for column in range(values.shape[1]):
        np.testing.assert_array_equal(nan_mask[:, column], expected_block)

    # Missing is never encoded as "measured 0".
    assert bool(np.all(np.isnan(values[nan_mask])))
    assert not bool(np.any(values[nan_mask] == 0.0))
    assert not bool(np.any(mask[nan_mask]))

    # Observed values are the real frame values (no zero filling).
    np.testing.assert_allclose(
        values[mask], source.frame.to_numpy(dtype=float)[mask], equal_nan=True
    )

    # A mask that claims True while the value is NaN is structurally invalid.
    bad = _pretrain_parts(source)
    bad["observed_mask"][0, 0, 0] = True
    bad["features"][0, 0, 0] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        PretrainBatch(**bad)


# ---------------------------------------------------------------------------
# T04/T05 data side: 31-day grid, synthetic label honesty, F=0 legal
# ---------------------------------------------------------------------------


def test_synthetic_dataset_grid_labels_and_f0():
    dataset = build_synthetic_dataset(days=31)
    frame = dataset.frame

    # 31 complete consecutive trace days, exactly 96 points each.
    assert len(frame) == 31 * 96
    per_day = np.bincount(np.arange(len(frame)) // 96, minlength=31)
    np.testing.assert_array_equal(per_day, np.full(31, 96))
    assert len(np.unique(np.arange(len(frame)) // 96)) == 31

    # Exact 15-minute UTC spacing from the anchor, no DST/local-time trickery.
    index = frame.index
    assert isinstance(index, pd.DatetimeIndex)
    assert index.name == "timestamp"
    assert str(index.tz) == "UTC"
    assert index[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    np.testing.assert_array_equal(np.diff(index.asi8), np.full(len(frame) - 1, 15 * 60 * 1_000_000_000))

    # Target column and physically plausible kW labels.
    assert TARGET_COLUMN in frame.columns
    target = frame[TARGET_COLUMN]
    assert np.isfinite(target.to_numpy(dtype=float)).all()
    assert 200.0 <= float(target.min()) and float(target.max()) <= 1500.0

    # Label honesty: synthetic_simulated, synthetic=True, never measured.
    assert dataset.spec.role is DatasetRole.TARGET_SUPERVISED
    assert dataset.spec.synthetic is True
    assert dataset.spec.provenance["synthetic"] is True
    assert dataset.spec.target is not None
    assert dataset.spec.target.label_origin is LabelOrigin.SYNTHETIC_SIMULATED
    assert dataset.spec.target.synthetic is True
    assert dataset.spec.timezone_verified is True
    assert dataset.spec.audit_status is AuditStatus.VERIFIED

    # Weekday vs weekend profile differs, and noise=0 is exactly reproducible.
    quiet = build_synthetic_dataset(days=31, noise=0.0)
    weekday = quiet.frame.loc[quiet.frame.index.dayofweek < 5, TARGET_COLUMN].mean()
    weekend = quiet.frame.loc[quiet.frame.index.dayofweek >= 5, TARGET_COLUMN].mean()
    assert float(weekend) < float(weekday)
    again = build_synthetic_dataset(days=31, noise=0.0)
    np.testing.assert_array_equal(
        quiet.frame.to_numpy(dtype=float), again.frame.to_numpy(dtype=float)
    )

    # Same seed -> same data.
    twin = build_synthetic_dataset(days=31, seed=7)
    np.testing.assert_array_equal(
        dataset.frame.to_numpy(dtype=float), twin.frame.to_numpy(dtype=float)
    )

    # covariates=False -> only the target column, so callers can build F=0 batches.
    bare = build_synthetic_dataset(days=31, covariates=False)
    assert tuple(bare.frame.columns) == (TARGET_COLUMN,)
    assert bare.spec.feature_names == ()
    assert bare.frame[TARGET_COLUMN].notna().all()

    # The default dataset id is exported for reuse.
    assert dataset.spec.dataset_id == SYNTHETIC_DATASET_ID


# ---------------------------------------------------------------------------
# T04: explicit W -> kW conversion, refusal to turn utilisation into kW
# ---------------------------------------------------------------------------


def test_adapt_frame_converts_watts_to_kw_and_refuses_utilisation():
    spec = _adapter_spec()
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:00",
                "2024-01-01 00:15",
                "2024-01-01 00:15",  # exact duplicate timestamp -> deterministic mean
                "2024-01-01 00:30",
            ],
            "power_w": [1000.0, 2000.0, 4000.0, np.inf],
            "cpu_pct": [40.0, 60.0, 80.0, np.nan],
            "it_power_w": [500.0, 1500.0, 2500.0, 3500.0],
            "undeclared_column": [1.0, 2.0, 3.0, 4.0],
        }
    )
    adapted = adapt_frame(raw, spec, timestamp_column="timestamp")

    assert list(adapted.columns) == ["cpu_util", "it_power_kw", TARGET_COLUMN]
    assert "undeclared_column" not in adapted.columns
    assert adapted.index.name == "timestamp"
    assert str(adapted.index.tz) == "UTC"
    assert adapted.index.is_monotonic_increasing

    # 1000 W -> 1.0 kW, duplicate rows averaged (2 kW & 4 kW -> 3 kW), inf -> NaN.
    assert adapted[TARGET_COLUMN].iloc[0] == pytest.approx(1.0)
    assert adapted[TARGET_COLUMN].iloc[1] == pytest.approx(3.0)
    assert np.isnan(adapted[TARGET_COLUMN].iloc[2])

    # Power feature converted as well; utilisation (%) is left untouched.
    assert adapted["it_power_kw"].iloc[0] == pytest.approx(0.5)
    assert adapted["it_power_kw"].iloc[1] == pytest.approx(2.0)
    assert adapted["cpu_util"].iloc[0] == pytest.approx(40.0)
    assert adapted["cpu_util"].iloc[1] == pytest.approx(70.0)
    assert np.isnan(adapted["cpu_util"].iloc[2])

    adaptation = adapted.attrs["adaptation"]
    assert adaptation["duplicate_timestamps_collapsed"] == 1
    assert adaptation["power_columns_converted_to_kw"][TARGET_COLUMN]["stored_unit"] == "kW"
    assert adaptation["power_columns_converted_to_kw"][TARGET_COLUMN]["factor"] == pytest.approx(1e-3)
    assert TARGET_COLUMN in adaptation["power_columns_converted_to_kw"]
    assert "cpu_util" not in adaptation["power_columns_converted_to_kw"]

    mapping = mapping_table(spec)
    assert mapping["power_w"]["canonical_name"] == TARGET_COLUMN
    assert mapping["power_w"]["is_target"] is True
    assert mapping["cpu_pct"]["canonical_name"] == "cpu_util"
    assert mapping["cpu_pct"]["unit"] == "percent"
    assert mapping["it_power_w"]["unit"] == "W"

    # A utilisation target must never be converted to kW.
    utilisation_raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00", "2024-01-01 00:15"],
            "power_utilization_pct": [50.0, 60.0],
        }
    )
    with pytest.raises(PhysicalUnitError, match="utilisation"):
        adapt_frame(utilisation_raw, _utilisation_spec(), timestamp_column="timestamp")

    # A declared raw column that is absent is a contract error, not a silent drop.
    with pytest.raises(ContractError, match="missing"):
        adapt_frame(raw.drop(columns=["power_w"]), spec, timestamp_column="timestamp")


# ---------------------------------------------------------------------------
# raw-data protection and versioned processed output
# ---------------------------------------------------------------------------


def test_writers_refuse_raw_paths(synthetic, tmp_path):
    frame = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC", name="timestamp"),
    )

    with pytest.raises(ContractError, match="raw"):
        assert_not_raw_path(tmp_path / "data" / "raw" / "x.parquet")
    with pytest.raises(ContractError, match="raw"):
        write_table(frame, tmp_path / "data" / "raw" / "x.parquet")
    with pytest.raises(ContractError, match="raw"):
        write_table(frame, "data/raw/must_not_be_written.parquet")
    assert not (Path("data/raw") / "must_not_be_written.parquet").exists()

    # The explicit escape hatch works but is never used by the pipeline.
    escaped = write_table(frame, tmp_path / "data" / "raw" / "x.parquet", allow_raw=True)
    assert escaped.is_file()

    # Normal write / read / inspect / stream round-trip.
    good = write_table(frame, tmp_path / "processed" / "table.parquet")
    loaded = read_table(good)
    assert loaded["value"].tolist() == [1.0, 2.0]
    report = inspect_table(good)
    assert report["rows"] == 2
    assert "value" in report["schema"]
    assert report["bytes"] > 0
    assert report["sample"]
    chunks = list(iter_table_batches(good, batch_size=1))
    assert len(chunks) == 2
    assert all(len(chunk) == 1 for chunk in chunks)

    # write_processed refuses data/raw/ and writes a new versioned file elsewhere.
    with pytest.raises(ContractError, match="raw"):
        write_processed(synthetic.frame, synthetic.spec, out_dir=str(tmp_path / "data" / "raw"))

    payload = write_processed(
        synthetic.frame,
        synthetic.spec,
        out_dir=str(tmp_path / "processed"),
        tag="pytest",
        extra={"test": "test_unlabeled"},
    )
    output_file = Path(payload["file"])
    sidecar = Path(payload["sidecar"])
    assert output_file.is_file()
    assert sidecar.is_file()
    assert output_file.parent == (tmp_path / "processed").resolve()
    assert "data" not in output_file.parts or output_file.parent.name == "processed"

    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["synthetic"] is True
    assert metadata["source_files"] is None
    assert metadata["provenance"]["generated"] is True
    assert metadata["spec"]["dataset_id"] == SYNTHETIC_DATASET_ID
    assert metadata["field_mapping"]
    assert metadata["unit_conversion"]["target"]["stored_unit"] == "kW"
    assert metadata["quality"]["rows"] == len(synthetic.frame)
    assert metadata["quality_mask_statistics"]["target_missing_count"] == 0
    assert metadata["sampling"]["frequency"] == "15min"
    assert metadata["timezone"]["verified"] is True
    assert metadata["sha256"] == payload["sha256"]

    # A second write never overwrites the first versioned file.
    payload_two = write_processed(
        synthetic.frame, synthetic.spec, out_dir=str(tmp_path / "processed"), tag="pytest"
    )
    assert payload_two["file"] != payload["file"]
    assert Path(payload_two["file"]).is_file()
    assert output_file.is_file()


# ---------------------------------------------------------------------------
# missing block, quality report, causal filling, structural validation
# ---------------------------------------------------------------------------


def test_missing_days_block_and_quality_report():
    dataset = build_synthetic_dataset(days=31, seed=7, missing_days=(MISSING_DAY,))
    frame = dataset.frame

    # The contiguous NaN block is in every column of the selected trace day.
    expected_positions = np.arange(MISSING_START, MISSING_STOP)
    for column in frame.columns:
        nan_positions = np.flatnonzero(~np.isfinite(frame[column].to_numpy(dtype=float)))
        np.testing.assert_array_equal(nan_positions, expected_positions)

    report = quality_report(frame, dataset.spec)
    assert report["rows"] == 31 * 96
    assert report["duplicate_timestamps"] == 0
    assert report["missing_intervals"] == 0
    assert report["complete_trace_days"] == 31
    assert report["complete_trace_days_with_finite_target"] == 30
    assert report["target_nan_count"] == 8
    assert report["target_coverage_count"] == 31 * 96 - 8
    assert report["target_coverage_fraction"] == pytest.approx(1.0 - 8 / (31 * 96))
    assert report["timezone"] == "UTC"
    assert report["timezone_verified"] is True
    for column in frame.columns:
        assert report["nan_counts"][column] == 8
        assert report["nan_fraction"][column] == pytest.approx(8 / (31 * 96))
    assert all(count == 0 for count in report["out_of_range_counts"].values())

    # Physical-range excursions are counted, never silently clipped.
    outlier = frame.copy()
    cpu_position = outlier.columns.get_loc("cpu_util")
    outlier.iloc[0, cpu_position] = 5.0
    outlier_report = quality_report(outlier, dataset.spec)
    assert outlier_report["out_of_range_counts"]["cpu_util"] == 1

    # NaN labels are legal (mask path), structural violations are not.
    diagnostics = validate_frame(frame, dataset.spec)
    assert diagnostics["valid"] is True
    assert diagnostics["target_nan_count"] == 8
    with pytest.raises(ContractError, match="sorted"):
        validate_frame(frame.iloc[::-1], dataset.spec)

    # An unlabelled frame must never carry the kW target column.
    source = dataset.unlabeled_sources[0]
    power_smuggled = source.frame.copy()
    power_smuggled[TARGET_COLUMN] = 1.0
    with pytest.raises(PhysicalUnitError, match="must not carry"):
        validate_frame(power_smuggled, source.spec)

    # fill_causal forward-fills covariates only, at most fill_limit steps, never the target.
    filled, audit = fill_causal(frame, dataset.spec, fill_limit=4)
    assert audit["policy"] == "causal_ffill_covariates_only"
    assert audit["fill_direction"] == "forward_only"
    assert audit["backward_fill_used"] is False
    assert audit["target_filled"] == 0
    assert audit["target_values_modified"] is False
    assert audit["filled_counts"]["cpu_util"] == 4
    assert audit["total_filled_cells"] == 4 * len(dataset.spec.feature_names)

    target_nan_before = int(frame[TARGET_COLUMN].isna().sum())
    target_nan_after = int(filled[TARGET_COLUMN].isna().sum())
    assert target_nan_after == target_nan_before == 8
    np.testing.assert_array_equal(
        np.isfinite(filled[TARGET_COLUMN].to_numpy(dtype=float)),
        np.isfinite(frame[TARGET_COLUMN].to_numpy(dtype=float)),
    )
    # Only the first fill_limit slots of the block use past values; nothing is back-filled.
    assert filled["cpu_util"].iloc[MISSING_START] == frame["cpu_util"].iloc[MISSING_START - 1]
    assert filled["cpu_util"].iloc[MISSING_START + 3] == frame["cpu_util"].iloc[MISSING_START - 1]
    assert np.isnan(filled["cpu_util"].iloc[MISSING_START + 4])
