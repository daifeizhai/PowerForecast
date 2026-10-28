"""Tests for the AcmeTrace Seren target-domain assembly (power-transfer-v2).

The tests use small synthetic fixtures written under ``tmp_path`` so no multi-gigabyte raw
data is needed.  They pin the honesty rules: cluster-IT label only, mask-first aggregation,
no zero fill, explicit 15-minute grid, and refusal to write below ``data/raw/``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import acmetrace_target as target
from src.data.contracts import ContractError

# UTC-naive label timestamps of the synthetic 3-bin label grid.
LABEL_TIMESTAMPS = pd.to_datetime(
    ["2023-07-01 00:00:00", "2023-07-01 00:15:00", "2023-07-01 00:30:00"]
)
# The monitor files write explicit +08:00 offsets.
NODE_TIME_BIN0 = [
    "2023-07-01 08:00:00+08:00",
    "2023-07-01 08:00:15+08:00",
    "2023-07-01 08:00:30+08:00",
    "2023-07-01 08:00:45+08:00",
]
NODE_TIME_BIN1 = [
    "2023-07-01 08:15:00+08:00",
    "2023-07-01 08:15:15+08:00",
    "2023-07-01 08:15:30+08:00",
    "2023-07-01 08:15:45+08:00",
]


def _write_power_parquet(path: Path) -> None:
    rows = []
    for index, timestamp in enumerate(LABEL_TIMESTAMPS):
        rows.append(
            {
                "timestamp": timestamp,
                "cluster_id": "seren",
                "level": "cluster_it",
                "device_id": "seren_cluster_it",
                "power_kw": [100.0, np.nan, 120.0][index],
                "valid": [True, False, True][index],
                "coverage_ratio": [1.0, 0.0, 0.8][index],
                "n_samples": 3,
            }
        )
    # a non-target row must be filtered out by the loader
    rows.append(
        {
            "timestamp": LABEL_TIMESTAMPS[0],
            "cluster_id": "kalos",
            "level": "cluster_gpu_sum",
            "device_id": "kalos_cluster_gpu_sum",
            "power_kw": 9.0,
            "valid": True,
            "coverage_ratio": 1.0,
            "n_samples": 60,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_node_metric(path: Path) -> None:
    rows = []
    for offset, timestamp in enumerate(NODE_TIME_BIN0):
        rows.append({"Time": timestamp, "10.0.0.1": 1.0 + offset, "10.0.0.2": 10.0 * (offset + 1)})
    for timestamp in NODE_TIME_BIN1:
        rows.append({"Time": timestamp, "10.0.0.1": 5.0, "10.0.0.2": np.nan})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_job_trace(path: Path) -> None:
    rows = [
        {
            "job_id": 1,
            "user": "u1",
            "node_num": 1,
            "gpu_num": 8,
            "cpu_num": 128,
            "type": "Training",
            "state": "COMPLETED",
            "submit_time": "2023-07-01 08:00:00+08:00",
            "start_time": "2023-07-01 08:02:00+08:00",
            "end_time": "2023-07-01 08:25:00+08:00",
            "duration": 1380,
            "queue": 120,
            "gpu_time": 11040.0,
        },
        {
            "job_id": 2,
            "user": "u2",
            "node_num": 1,
            "gpu_num": 4,
            "cpu_num": 64,
            "type": "Inference",
            "state": "COMPLETED",
            "submit_time": "2023-07-01 08:35:00+08:00",
            "start_time": "2023-07-01 08:25:00+08:00",
            "end_time": "2023-07-01 08:40:00+08:00",
            "duration": 900,
            "queue": 0,
            "gpu_time": 3600.0,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_config(tmp_path: Path) -> dict:
    """Build a valid target config whose raw files and outputs live under ``tmp_path``."""
    raw_root = tmp_path / "raw"
    (raw_root / "data" / "utilization" / "seren").mkdir(parents=True, exist_ok=True)
    (raw_root / "data" / "job_trace").mkdir(parents=True, exist_ok=True)
    power_parquet = tmp_path / "processed" / "power_15min.parquet"
    _write_power_parquet(power_parquet)
    node_file = raw_root / "data" / "utilization" / "seren" / "NODE_CPU_UTILIZATION.csv"
    _write_node_metric(node_file)
    job_file = raw_root / "data" / "job_trace" / "trace_seren.csv"
    _write_job_trace(job_file)
    processed = tmp_path / "processed"
    return {
        "schema_version": 1,
        "dataset": {
            "dataset_id": "acmetrace_seren_target",
            "site_id": "acmetrace",
            "domain_id": "acmetrace_seren",
            "display_name": "synthetic AcmeTrace Seren target",
            "frequency": "15min",
            "points_per_day": 96,
        },
        "source": {
            "raw_root": str(raw_root),
            "cluster": "seren",
            "timezone": "+08:00",
            "timezone_verified": False,
            "power_parquet": str(power_parquet),
            "label": {
                "cluster_id": "seren",
                "level": "cluster_it",
                "device_id": "seren_cluster_it",
                "label_type": "acmetrace_seren_cluster_it_derived_kw",
                "target_kind": "active_power",
                "target_scope": "it",
                "target_unit": "kW",
                "label_origin": "component_sum",
                "unit_inferred": "W",
                "unit_verified": False,
                "measurement_boundary": "sum of measured IPMI Sys_Total_Power over reporting Seren nodes",
                "sampling_semantics": "300 s per-node samples -> 15-minute means -> sum over reporting nodes",
            },
        },
        "features": {
            "node_metrics": [
                {
                    "metric": "NODE_CPU_UTILIZATION",
                    "path": "data/utilization/seren/NODE_CPU_UTILIZATION.csv",
                    "feature_name": "node_cpu_utilization",
                    "unit": "percent",
                    "semantic": "node CPU utilisation",
                }
            ],
            "gpu_metrics": [],
            "job_trace": {
                "path": "data/job_trace/trace_seren.csv",
                "features": [
                    {
                        "feature_name": "job_active_count",
                        "unit": "count",
                        "semantic": "active jobs",
                    },
                    {
                        "feature_name": "job_active_gpu_num",
                        "unit": "gpu_count",
                        "semantic": "active GPU count",
                    },
                    {
                        "feature_name": "job_submitted_count",
                        "unit": "count",
                        "semantic": "submitted jobs",
                    },
                ],
            },
        },
        "aggregation_15min": {
            "step_seconds": 900,
            "label": "interval_start",
            "timezone_handling": "explicit offsets -> UTC-naive",
            "expected_samples_15s": 4,
            "expected_samples_300s": 3,
            "wide_chunk_rows": 100,
            "narrow_chunk_rows": 100,
            "validity_threshold": 0.5,
            "coverage_definition": "mean over devices of min(samples, expected)/expected",
            "duplicate_policy": "counted, not removed; coverage capped",
            "frame": "label grid",
        },
        "outputs": {
            "processed_dir": str(processed),
            "power_parquet": str(processed / "acmetrace_target_power.parquet"),
            "features_parquet": str(processed / "acmetrace_target_features.parquet"),
            "metadata_json": str(processed / "acmetrace_target_metadata.json"),
            "quality_json": str(processed / "acmetrace_target_quality.json"),
            "results_dir": str(tmp_path / "analysis"),
        },
    }


# ---------------------------------------------------------------------------
# config honesty
# ---------------------------------------------------------------------------


def test_config_rejects_facility_scope(tmp_path: Path) -> None:
    config = copy.deepcopy(make_config(tmp_path))
    config["source"]["label"]["target_scope"] = "facility"
    with pytest.raises(ContractError, match="facility"):
        target.validate_target_config(config)


def test_config_rejects_non_cluster_it_level(tmp_path: Path) -> None:
    config = copy.deepcopy(make_config(tmp_path))
    config["source"]["label"]["level"] = "gpu"
    with pytest.raises(ContractError, match="cluster_it"):
        target.validate_target_config(config)


def test_config_rejects_verified_units_claim(tmp_path: Path) -> None:
    config = copy.deepcopy(make_config(tmp_path))
    config["source"]["label"]["unit_verified"] = True
    with pytest.raises(ContractError, match="unit_verified"):
        target.validate_target_config(config)


def test_config_rejects_output_below_data_raw(tmp_path: Path) -> None:
    config = copy.deepcopy(make_config(tmp_path))
    config["outputs"]["power_parquet"] = "data/raw/acmetrace_target_power.parquet"
    with pytest.raises(ContractError, match="raw"):
        target.validate_target_config(config)


# ---------------------------------------------------------------------------
# timestamp / binning
# ---------------------------------------------------------------------------


def test_explicit_offsets_convert_to_utc_naive() -> None:
    parsed = target.parse_monitor_timestamps(
        ["2023-07-01 08:00:00+08:00", "2023-07-01 00:00:00+00:00"]
    )
    assert parsed.tz is None
    assert list(parsed) == list(LABEL_TIMESTAMPS[:1]) * 2


def test_scan_wide_metric_binning_and_coverage(tmp_path: Path) -> None:
    config = target.validate_target_config(make_config(tmp_path))
    path = Path(config["source"]["raw_root"]) / "data/utilization/seren/NODE_CPU_UTILIZATION.csv"
    payload = target.scan_wide_metric(
        path,
        metric="NODE_CPU_UTILIZATION",
        feature_name="node_cpu_utilization",
        unit="percent",
        semantic="node CPU utilisation",
        config=config,
    )
    bin0 = int(target.bin_index(LABEL_TIMESTAMPS[:1])[0])
    assert payload["bins"].tolist() == [bin0, bin0 + 1]
    values = payload["values"]
    assert values[0] == pytest.approx((1 + 2 + 3 + 4 + 10 + 20 + 30 + 40) / 8.0)
    assert values[1] == pytest.approx(5.0)
    assert payload["coverage"][0] == pytest.approx(1.0)
    assert payload["coverage"][1] == pytest.approx(0.5)  # second node absent -> 1/2 device coverage
    assert payload["audit"]["device_columns"] == 2
    assert payload["audit"]["unit_verified"] is False


def test_job_activity_half_open_lifetimes(tmp_path: Path) -> None:
    config = target.validate_target_config(make_config(tmp_path))
    path = Path(config["source"]["raw_root"]) / "data/job_trace/trace_seren.csv"
    result = target.scan_job_activity(
        path,
        feature_specs=config["features"]["job_trace"]["features"],
        config=config,
        cluster="seren",
    )
    series = {payload["feature_name"]: payload for payload in result["payloads"]}
    assert series["job_active_count"]["values"].tolist() == [1.0, 2.0, 1.0]
    assert series["job_active_gpu_num"]["values"].tolist() == [8.0, 12.0, 4.0]
    assert series["job_submitted_count"]["values"].tolist() == [1.0, 0.0, 1.0]
    assert result["audit"]["jobs_with_lifetime"] == 2
    assert result["audit"]["jobs_missing_start"] == 0


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def test_label_loader_filters_to_cluster_it(tmp_path: Path) -> None:
    config = target.validate_target_config(make_config(tmp_path))
    frame, audit = target.load_target_power(config)
    assert list(frame.columns) == list(target.POWER_COLUMNS)
    assert len(frame) == 3  # the Kalos row is filtered out
    assert audit["level"] == "cluster_it"
    assert audit["valid_slots"] == 2
    assert np.isnan(frame["power_kw"].to_numpy()[1])


def test_full_pipeline_aligns_features_and_never_zero_fills(tmp_path: Path) -> None:
    config = target.validate_target_config(make_config(tmp_path))
    result = target.run_pipeline(config, include_gpu=False)
    features = result["features"]
    assert list(features.columns) == list(target.FEATURE_COLUMNS)
    names = set(features["feature_name"].unique())
    assert names == {
        "node_cpu_utilization",
        "job_active_count",
        "job_active_gpu_num",
        "job_submitted_count",
    }
    node = features[features["feature_name"] == "node_cpu_utilization"].reset_index(drop=True)
    assert node["observed_mask"].tolist() == [True, True, False]  # third label bin has no metric rows
    assert np.isnan(node["feature_value"].to_numpy()[2])  # missing stays NaN, never 0
    assert node["coverage_ratio"].tolist() == [
        pytest.approx(1.0),
        pytest.approx(0.5),
        pytest.approx(0.0),
    ]
    jobs = features[features["feature_name"] == "job_active_count"]
    assert jobs["observed_mask"].all()
    metadata = result["metadata"]
    assert metadata["label"]["never_facility_power"] is True
    assert metadata["label"]["unit_verified"] is False
    assert metadata["timezone_metadata"]["timezone_verified"] is False
    assert metadata["aggregation"]["missing_policy"].startswith("NaN")
    quality = result["quality"]
    assert any("not facility" in warning.lower() for warning in quality["quality_warnings"])


def test_bounded_run_is_marked_debug(tmp_path: Path) -> None:
    config = target.validate_target_config(make_config(tmp_path))
    result = target.run_pipeline(config, include_gpu=False, max_chunks=1)
    assert result["metadata"]["bounded_debug"] is True
    assert "bounded_debug_note" in result["metadata"]


def test_run_pipeline_writes_contract_artifacts(tmp_path: Path) -> None:
    config = target.validate_target_config(make_config(tmp_path))
    result = target.run_pipeline(config, include_gpu=False, write=True)
    for key in ("power_parquet", "features_parquet"):
        assert Path(result["outputs"][key]).is_file()
    power = pd.read_parquet(result["outputs"]["power_parquet"])
    features = pd.read_parquet(result["outputs"]["features_parquet"])
    assert list(power.columns) == list(target.POWER_COLUMNS)
    assert list(features.columns) == list(target.FEATURE_COLUMNS)
    assert pd.notna(power["timestamp"]).all()


# ---------------------------------------------------------------------------
# real config / registry wiring
# ---------------------------------------------------------------------------


def test_real_target_config_is_valid() -> None:
    config = target.load_target_config()
    assert config["source"]["label"]["level"] == "cluster_it"
    assert config["source"]["label"]["target_scope"] == "it"
    assert config["source"]["label"]["unit_verified"] is False
    assert config["dataset"]["frequency"] == "15min"


def test_registry_registers_acmetrace_without_facility_claim() -> None:
    from src.data.registry import load_source_registry

    registry = load_source_registry("configs/datasets/registry.yaml")
    seren = registry["sources"]["acmetrace_seren"]
    assert seren["candidate_role"] == "target_supervised"
    assert seren["label"]["scope"] == "it"
    assert seren["can_produce_facility_kw"] is False
    kalos = registry["sources"]["acmetrace_kalos"]
    assert kalos["label"] is None
    assert kalos["can_produce_facility_kw"] is False
