"""Stage 1 target-domain pipeline acceptance tests (task book §9.2, §9.5, Milestone 2).

The tests cover both synthetic unit cases (fast, deterministic) and a bounded
real-raw integration slice verifying the verified data facts: the first 31 trace
days from the first valid 15-minute slot are 96 points/day and 100% valid.

They also verify the safety boundaries: raw files are never modified, writes below
``data/raw/`` are refused, missing intervals are never gap-compressed, non-finite
values never become zero labels, and the label type is honestly
``facility_derived_kw``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import target_power as tp
from src.data.contracts import (
    ContractError,
    LabelOrigin,
    TargetEligibilityError,
    TargetKind,
    TargetScope,
    TargetSpec,
)
from src.data.labels import LabelType, classify_target, require_main_model_label
from src.utils import load_config

CONFIG_PATH = "configs/datasets/nlr_target.yaml"
RAW_POWER = "data/raw/NLR_Dataset/esif.influx.buildingData.PUE.combined.parquet"
RAW_WEATHER = "data/raw/NLR_Dataset/esif.influx.buildingData.outside.combined.parquet"
COMPONENTS = ["it_power_kw", "cooling_kw", "hvac_kw", "plug_and_light_kw", "pump_kw"]

ANCHOR = pd.Timestamp("2016-06-12 06:00:00")
FIRST_31_MEAN_KW = 820.1847130489925
FIRST_31_MIN_KW = 606.5459999999999
FIRST_31_MAX_KW = 950.6259999999999


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _config() -> dict:
    return copy.deepcopy(load_config(CONFIG_PATH))


def _set(config: dict, dotted: str, value) -> dict:
    node: dict = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return config


def _windowless_config(**overrides) -> dict:
    """Config for synthetic frames: no declared anchor (detected from the frame)."""
    config = _config()
    _set(config, "analysis.window_start", None)
    for dotted, value in overrides.items():
        _set(config, dotted, value)
    return config


def _raw_minutes(start: str, minutes: int, *, seed: int = 7) -> pd.DataFrame:
    """1-minute raw frame with five well-behaved component channels (~kW)."""
    stamps = pd.date_range(start, periods=minutes, freq="1min")
    rng = np.random.default_rng(seed)
    base = 100.0 + rng.random(minutes) * 20.0
    payload = {"ts": stamps}
    for offset, component in enumerate(COMPONENTS):
        payload[component] = base + offset * 5.0
    return pd.DataFrame(payload)


def _write_roundtrip_inputs(config: dict | None = None):
    """Minimal consistent set of artefacts for writer tests."""
    config = config if config is not None else _windowless_config()
    raw = _raw_minutes("2024-01-01 00:00:00", 60)
    frame, components, audit = tp.build_target_power_frame_with_components(raw, config)
    weather = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01 00:00:00", periods=60, freq="1min"),
            "outdoor_air_temp": 68.0,
            "outdoor_air_humidity": 50.0,
            "unit": "outside",
        }
    )
    features, feature_audit = tp.build_target_feature_frame(weather, frame.index, config)
    metadata = tp.build_target_metadata(
        frame,
        audit,
        config,
        source_metadata=[{"role": "power_label_source", "path": "synthetic-test-frame"}],
    )
    quality = {"stage": "stage1_target_power_quality", "target": audit, "features": feature_audit}
    return config, frame, components, features, metadata, quality


# ---------------------------------------------------------------------------
# label taxonomy / gate
# ---------------------------------------------------------------------------


def test_target_spec_is_facility_derived_kw() -> None:
    spec = tp.target_spec_from_config(_config())
    assert spec.kind is TargetKind.ACTIVE_POWER
    assert spec.scope is TargetScope.FACILITY
    assert spec.unit == "kW"
    assert spec.label_origin is LabelOrigin.COMPONENT_SUM
    assert spec.synthetic is False
    assert classify_target(spec) is LabelType.FACILITY_DERIVED_KW
    require_main_model_label(classify_target(spec))  # must not raise


def test_main_model_gate_rejects_node_and_relative_labels() -> None:
    node = TargetSpec(
        kind=TargetKind.ACTIVE_POWER,
        scope=TargetScope.NODE,
        unit="W",
        label_origin=LabelOrigin.MEASURED,
        measurement_boundary="node PSU",
        sampling_semantics="per-node sample",
    )
    assert classify_target(node) is LabelType.NODE_MEASURED_KW
    with pytest.raises(TargetEligibilityError):
        require_main_model_label(classify_target(node))

    relative = TargetSpec(
        kind=TargetKind.POWER_UTILIZATION,
        scope=TargetScope.FACILITY,
        unit="percent",
        label_origin=LabelOrigin.POWER_UTILIZATION,
        measurement_boundary="utilization panel",
        sampling_semantics="reported interval",
    )
    assert classify_target(relative) is LabelType.RELATIVE_POWER
    with pytest.raises(TargetEligibilityError):
        require_main_model_label(classify_target(relative))


def test_dataset_spec_from_config_is_honest() -> None:
    config = _config()
    spec = tp.dataset_spec_from_config(config)
    assert spec.dataset_id == config["dataset_id"]
    assert spec.role.value == "target_supervised"
    assert spec.timezone_verified is False
    assert spec.trace_anchor_timestamp == "2016-06-12T06:00:00"
    assert spec.target is not None and spec.target.label_origin is LabelOrigin.COMPONENT_SUM
    assert set(spec.feature_names) == {"outdoor_air_temp_c", "relative_humidity_pct"}
    with pytest.raises(TargetEligibilityError):
        spec.require_verified_timezone("local-time analysis")


# ---------------------------------------------------------------------------
# grid, duplicates, missing values, coverage
# ---------------------------------------------------------------------------


def test_grid_is_complete_and_never_gap_compressed() -> None:
    config = _windowless_config()
    stamps = pd.date_range("2024-01-01 00:00:00", periods=360, freq="1min")
    raw = _raw_minutes("2024-01-01 00:00:00", 360)
    # Remove the raw samples of three consecutive 15-minute intervals (45 minutes).
    keep = ~((stamps >= pd.Timestamp("2024-01-01 02:00")) & (stamps < pd.Timestamp("2024-01-01 02:45")))
    raw = raw.loc[keep].reset_index(drop=True)

    frame, audit = tp.build_target_power_frame(raw, config)
    expected_grid = pd.date_range("2024-01-01 00:00:00", "2024-01-01 05:45:00", freq="15min")
    assert list(frame.index) == list(expected_grid)
    assert frame.index.name == "timestamp"
    assert audit["grid"]["slots"] == 24
    assert audit["grid"]["missing_interval_count"] == 3
    for stamp in ("2024-01-01 02:00", "2024-01-01 02:15", "2024-01-01 02:30"):
        assert not bool(frame.loc[pd.Timestamp(stamp), "power_valid"])
        assert pd.isna(frame.loc[pd.Timestamp(stamp), "power_kw"])
    # Later timestamps did not shift: the slot after the gap is still 02:45.
    assert pd.Timestamp("2024-01-01 02:45") in frame.index
    assert bool(frame.loc[pd.Timestamp("2024-01-01 02:45"), "power_valid"])
    assert audit["grid"]["gap_compression"] is False


def test_duplicate_timestamps_collapse_deterministically() -> None:
    config = _windowless_config()
    raw = _raw_minutes("2024-01-01 00:00:00", 60)
    duplicated = raw.iloc[[10]].copy()
    for component in COMPONENTS:
        duplicated[component] = duplicated[component] + 100.0
    raw_with_duplicate = pd.concat([raw, duplicated], ignore_index=True).sort_values("ts")

    frame_a, audit = tp.build_target_power_frame(raw_with_duplicate, config)
    frame_b, _ = tp.build_target_power_frame(raw_with_duplicate, config)
    assert audit["raw"]["duplicate_timestamp_rows_collapsed"] == 1
    pd.testing.assert_frame_equal(frame_a, frame_b)
    # The duplicate minute was collapsed to the mean of the two rows, not dropped or summed:
    # the component interval mean gains half of the injected +100 kW on that one minute.
    expected_first = (raw["it_power_kw"].iloc[:15].sum() + 50.0) / 15.0
    _, components, _ = tp.build_target_power_frame_with_components(raw_with_duplicate, config)
    assert components["it_power_kw"].iloc[0] == pytest.approx(expected_first, rel=0, abs=1e-9)


def test_non_finite_values_become_invalid_never_zero_filled() -> None:
    config = _windowless_config()
    raw = _raw_minutes("2024-01-01 00:00:00", 60)
    # Interval 00:15-00:29: all samples have a NaN or infinite cooling value.
    mask = (raw["ts"] >= pd.Timestamp("2024-01-01 00:15")) & (raw["ts"] < pd.Timestamp("2024-01-01 00:30"))
    raw.loc[mask, "cooling_kw"] = np.nan
    raw.loc[raw["ts"] == pd.Timestamp("2024-01-01 00:20"), "cooling_kw"] = np.inf
    # Interval 00:45-00:59: every component is missing.
    mask2 = raw["ts"] >= pd.Timestamp("2024-01-01 00:45")
    raw.loc[mask2, COMPONENTS] = np.nan

    frame, audit = tp.build_target_power_frame(raw, config)
    assert pd.isna(frame.loc[pd.Timestamp("2024-01-01 00:15"), "power_kw"])
    assert not bool(frame.loc[pd.Timestamp("2024-01-01 00:15"), "power_valid"])
    assert pd.isna(frame.loc[pd.Timestamp("2024-01-01 00:45"), "power_kw"])
    assert not bool(frame.loc[pd.Timestamp("2024-01-01 00:45"), "power_valid"])
    # No invalid slot was silently turned into 0 kW.
    invalid = frame.loc[~frame["power_valid"], "power_kw"]
    assert invalid.isna().all()
    assert audit["raw"]["infinite_values_replaced_by_nan"]["cooling_kw"] >= 1


def test_coverage_ratio_bounds_and_validity_monotone_in_threshold() -> None:
    stamps = pd.date_range("2024-01-01 00:00:00", periods=60, freq="1min")
    raw = _raw_minutes("2024-01-01 00:00:00", 60)
    # Interval 1 (00:15-00:29): only 3 native samples. Interval 2 (00:30-00:44): none.
    keep = np.ones(len(raw), dtype=bool)
    keep &= ~(
        (stamps >= pd.Timestamp("2024-01-01 00:15")) & (stamps < pd.Timestamp("2024-01-01 00:28"))
    )
    keep &= ~(
        (stamps >= pd.Timestamp("2024-01-01 00:30")) & (stamps < pd.Timestamp("2024-01-01 00:45"))
    )
    raw = raw.loc[keep].reset_index(drop=True)

    low = _windowless_config(**{"coverage.coverage_validity_threshold": 1.0 / 15.0})
    high = _windowless_config(**{"coverage.coverage_validity_threshold": 0.8})
    frame_low, audit_low = tp.build_target_power_frame(raw, low)
    frame_high, _ = tp.build_target_power_frame(raw, high)

    coverage = frame_low["coverage_ratio"].to_numpy(dtype=float)
    assert np.nanmin(coverage) >= 0.0 and np.nanmax(coverage) <= 1.0
    np.testing.assert_allclose(coverage, frame_high["coverage_ratio"].to_numpy(dtype=float))
    valid_low = frame_low["power_valid"].to_numpy(dtype=bool)
    valid_high = frame_high["power_valid"].to_numpy(dtype=bool)
    assert not np.any(valid_high & ~valid_low), "valid high-threshold set must be a subset"
    assert valid_low.sum() > valid_high.sum()
    assert audit_low["coverage_validity_threshold"] == pytest.approx(1.0 / 15.0)
    assert audit_low["native_sampling"]["min_samples_per_interval"] == 0


def test_component_sum_equals_facility_label_exactly() -> None:
    config = _windowless_config()
    raw = _raw_minutes("2024-01-01 00:00:00", 120)
    raw.loc[raw.index[30:45], "hvac_kw"] = np.nan  # whole interval missing one component
    frame, components, audit = tp.build_target_power_frame_with_components(raw, config)

    valid = frame["power_valid"].to_numpy(dtype=bool)
    total = components.sum(axis=1, min_count=len(COMPONENTS))
    np.testing.assert_allclose(
        total[valid].to_numpy(dtype=float),
        frame["power_kw"].to_numpy(dtype=float)[valid],
        rtol=0.0,
        atol=1e-9,
    )
    # An invalid interval with a missing component must be NaN, not a partial sum.
    assert not valid[2] and pd.isna(frame["power_kw"].iloc[2])
    assert audit["label_composition"] == COMPONENTS


def test_build_rejects_selection_without_any_valid_component() -> None:
    config = _windowless_config()
    raw = _raw_minutes("2024-01-01 00:00:00", 30)
    raw[COMPONENTS] = np.nan
    with pytest.raises(ContractError):
        tp.build_target_power_frame(raw, config)


def test_target_frame_has_exact_core_shape() -> None:
    config = _windowless_config()
    frame, _ = tp.build_target_power_frame(_raw_minutes("2024-01-01 00:00:00", 90), config)
    assert list(frame.columns) == ["site_id", "power_kw", "power_valid", "coverage_ratio"]
    assert frame.index.name == "timestamp"
    assert frame.index.tz is None
    assert frame.index.is_monotonic_increasing
    assert (frame["site_id"] == config["site_id"]).all()
    assert frame["power_valid"].dtype == bool
    assert frame["power_kw"].dtype == "float64"
    assert np.isfinite(frame["power_kw"].to_numpy(dtype=float)).all()


# ---------------------------------------------------------------------------
# weather features
# ---------------------------------------------------------------------------


def test_feature_long_format_units_and_sentinel() -> None:
    config = _windowless_config()
    frame, _ = tp.build_target_power_frame(_raw_minutes("2024-01-01 00:00:00", 120), config)
    stamps = pd.date_range("2024-01-01 00:00:00", periods=120, freq="1min")
    weather = pd.DataFrame(
        {
            "ts": stamps,
            "outdoor_air_temp": 68.0,       # 68 degF == 20 degC
            "outdoor_air_humidity": 50.0,
            "unit": None,
        }
    )
    sentinel = pd.DataFrame(
        {
            "ts": [pd.Timestamp("1970-01-01 00:00:00")],
            "outdoor_air_temp": [0.0],
            "outdoor_air_humidity": [0.0],
            "unit": [None],
        }
    )
    weather = pd.concat([sentinel, weather], ignore_index=True)
    # Remove humidity for the last half hour to exercise the observed mask.
    weather.loc[weather["ts"] >= pd.Timestamp("2024-01-01 01:30"), "outdoor_air_humidity"] = np.nan

    features, audit = tp.build_target_feature_frame(weather, frame.index, config)
    assert list(features.columns) == list(tp.FEATURE_COLUMNS)
    assert len(features) == len(frame) * 2
    assert set(features["feature_name"]) == {"outdoor_air_temp_c", "relative_humidity_pct"}
    assert set(features["unit"]) == {"degC", "percent"}
    assert (features["domain_id"] == config["domain_id"]).all()

    first_temp = features.loc[
        (features["feature_name"] == "outdoor_air_temp_c")
        & (features["timestamp"] == pd.Timestamp("2024-01-01 00:00"))
    ]["feature_value"].iloc[0]
    assert first_temp == pytest.approx(20.0, abs=1e-9)
    late_humidity = features.loc[
        (features["feature_name"] == "relative_humidity_pct")
        & (features["timestamp"] == pd.Timestamp("2024-01-01 01:45"))
    ]
    assert not bool(late_humidity["observed_mask"].iloc[0])
    assert pd.isna(late_humidity["feature_value"].iloc[0])
    assert audit["sentinel"]["excluded_rows"] == 1
    assert audit["unit_column"]["present"] is True
    assert audit["unit_column"]["trusted_as_unit_declaration"] is False
    assert (features["coverage_ratio"].between(0.0, 1.0)).all()
    assert audit["fields"]["outdoor_air_temp_c"]["scale"] == pytest.approx(
        0.5555555555555556
    )
    assert audit["fields"]["outdoor_air_temp_c"]["offset"] == pytest.approx(-17.77777777777778)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def test_metadata_is_honest_and_counts_the_excluded_era() -> None:
    config = _config()
    _set(config, "analysis.window_start", "2024-01-01T00:00:00")
    pre_anchor_raw = _raw_minutes("2023-12-31 00:00:00", 1440)  # all-missing day before anchor
    pre_anchor_raw[COMPONENTS] = np.nan
    valid_raw = _raw_minutes("2024-01-01 00:00:00", 120)
    raw = pd.concat([pre_anchor_raw, valid_raw], ignore_index=True)
    frame, audit = tp.build_target_power_frame(raw, config)

    assert audit["raw"]["rows_excluded_before_anchor"] == 1440
    assert audit["raw"]["rows_excluded_before_anchor_with_all_components_nan"] == 1440

    metadata = tp.build_target_metadata(
        frame,
        audit,
        config,
        source_metadata=[{"role": "power_label_source", "path": "synthetic-test"}],
    )
    assert metadata["label_type"] == "facility_derived_kw"
    assert metadata["label"]["label_type"] == "facility_derived_kw"
    assert metadata["timezone_verified"] is False
    assert metadata["timezone_metadata"]["verified"] is False
    assert metadata["timezone_metadata"]["local_time_operations_blocked"] is True
    assert metadata["timezone_metadata"]["dst_rules_applied"] is False
    assert (
        metadata["timezone_metadata"]["anchor_based_trace_days_require_verified_timezone"]
        is False
    )
    assert "future-known" in metadata["label"]["composition_warning"]
    assert metadata["label"]["target_spec"]["label_origin"] == "component_sum"
    assert metadata["data_selection"]["rows_excluded_before_anchor"] == 1440
    assert metadata["data_selection"]["raw_power_first_finite_component_timestamp"].startswith(
        "2024-01-01T00:00"
    )
    assert metadata["anchor_timestamp"] == "2024-01-01T00:00:00"
    assert metadata["aggregation"]["imputation_used"] is False
    assert metadata["aggregation"]["gap_compression"] is False
    assert metadata["aggregation"]["coverage_validity_threshold"] == pytest.approx(1.0 / 15.0)
    # The metadata document is JSON-serialisable as-is.
    json.dumps(metadata)


# ---------------------------------------------------------------------------
# writer / safety boundaries
# ---------------------------------------------------------------------------


def test_write_artifacts_roundtrip(tmp_path: Path) -> None:
    config, frame, components, features, metadata, quality = _write_roundtrip_inputs()
    written = tp.write_target_artifacts(
        frame, components, features, metadata, quality, config, out_dir=tmp_path
    )
    paths = written["paths"]
    for role in ("target_power", "target_components", "target_features",
                 "target_metadata", "target_quality"):
        assert Path(paths[role]).is_file()

    target_table = pd.read_parquet(paths["target_power"])
    assert list(target_table.columns) == list(tp.CORE_TARGET_COLUMNS)
    assert len(target_table) == len(frame)

    component_table = pd.read_parquet(paths["target_components"])
    assert list(component_table.columns) == ["timestamp", *COMPONENTS]

    feature_table = pd.read_parquet(paths["target_features"])
    assert list(feature_table.columns) == list(tp.FEATURE_COLUMNS)
    assert len(feature_table) == len(features)

    metadata_document = json.loads(Path(paths["target_metadata"]).read_text(encoding="utf-8"))
    quality_document = json.loads(Path(paths["target_quality"]).read_text(encoding="utf-8"))
    assert metadata_document["label_type"] == "facility_derived_kw"
    assert metadata_document["artifacts"]["target_components"]["label_composition_only"] is True
    assert "NOT" in metadata_document["artifacts"]["target_components"]["warning"]
    assert "sha256" in metadata_document["artifacts"]["target_power"]
    assert quality_document["target"]["validity"]["valid_slots"] >= 1
    # Re-running regenerates the declared artefact names without touching anything else.
    second = tp.write_target_artifacts(
        frame, components, features, metadata, quality, config, out_dir=tmp_path
    )
    assert Path(second["paths"]["target_power"]).is_file()


def test_writer_refuses_data_raw_destination() -> None:
    config, frame, components, features, metadata, quality = _write_roundtrip_inputs()
    forbidden = "data/raw/stage1_must_not_be_written"
    with pytest.raises(ContractError):
        tp.write_target_artifacts(
            frame, components, features, metadata, quality, config, out_dir=forbidden
        )
    assert not Path(forbidden).exists()


def test_metadata_builder_rejects_wrong_quality_payload() -> None:
    config = _windowless_config()
    frame, _ = tp.build_target_power_frame(_raw_minutes("2024-01-01 00:00:00", 60), config)
    with pytest.raises(ContractError):
        tp.build_target_metadata(frame, ["not-a-mapping"], config)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# raw-data protection and the real bounded raw slice
# ---------------------------------------------------------------------------


def test_raw_files_are_never_modified() -> None:
    raw_path = Path(RAW_POWER)
    before = raw_path.stat()
    raw = tp.load_raw_components(
        RAW_POWER,
        start="2024-01-01T00:00:00",
        end="2024-01-02T00:00:00",
        columns=["ts", *COMPONENTS],
    )
    frame, _ = tp.build_target_power_frame(raw, _windowless_config())
    assert len(frame) == 96
    after = raw_path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns


def test_loader_respects_bounds_and_columns() -> None:
    raw = tp.load_raw_components(
        RAW_POWER,
        start="2024-06-01T00:00:00",
        end="2024-06-02T00:00:00",
        columns=["ts", "it_power_kw"],
    )
    assert list(raw.columns) == ["ts", "it_power_kw"]
    assert len(raw) > 0
    assert raw["ts"].min() >= pd.Timestamp("2024-06-01 00:00:00")
    assert raw["ts"].max() < pd.Timestamp("2024-06-02 00:00:00")
    with pytest.raises(ContractError):
        tp.load_raw_components(RAW_POWER, columns=["ts", "not_a_real_column"])
    with pytest.raises(FileNotFoundError):
        tp.load_raw_components("data/raw/NLR_Dataset/does_not_exist.parquet")


# ---------------------------------------------------------------------------
# real integration: first 31 trace days are complete and 100% valid
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_first_31_days():
    config = _config()
    anchor = pd.Timestamp(config["analysis"]["window_start"])
    raw = tp.load_raw_components(
        RAW_POWER,
        start=anchor,
        end=anchor + pd.to_timedelta(31, unit="D"),
        columns=["ts", *COMPONENTS],
    )
    frame, components, audit = tp.build_target_power_frame_with_components(raw, config)
    return config, frame, components, audit


def test_real_first_31_trace_days_complete_and_valid(real_first_31_days) -> None:
    config, frame, components, audit = real_first_31_days
    points_per_day = int(config["points_per_day"])
    assert len(frame) == 31 * points_per_day == 2976
    assert int(frame["power_valid"].sum()) == 2976
    assert frame["coverage_ratio"].between(0.0, 1.0).all()
    # Every trace day has exactly 96 grid points.
    assert (frame.groupby(np.arange(len(frame)) // points_per_day).size() == points_per_day).all()
    # Verified real-data facts for this window.
    assert float(frame["power_kw"].mean()) == pytest.approx(FIRST_31_MEAN_KW, abs=1.0)
    assert float(frame["power_kw"].min()) == pytest.approx(FIRST_31_MIN_KW, abs=0.01)
    assert float(frame["power_kw"].max()) == pytest.approx(FIRST_31_MAX_KW, abs=0.01)
    # Some intervals have fewer than the expected 15 native samples but are still real
    # observations (coverage_ratio < 1) and remain valid thanks to the declared threshold.
    assert (frame["coverage_ratio"] < 1.0).any()
    assert (frame["coverage_ratio"] >= audit["coverage_validity_threshold"]).all()
    assert audit["anchor"]["anchor_timestamp"] == "2016-06-12T06:00:00"
    assert audit["anchor"]["anchor_matches_first_complete_slot"] is True
    assert audit["raw"]["first_timestamp_with_all_components_finite"] == "2016-06-12T06:00:18"
    # Component sum equals the facility label on every interval.
    total = components.sum(axis=1, min_count=len(COMPONENTS))
    np.testing.assert_allclose(
        total.to_numpy(dtype=float),
        frame["power_kw"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    )
