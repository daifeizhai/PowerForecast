"""Stage-1 source-domain acceptance tests (M100 REAL node telemetry, bounded runs).

These tests run against a bounded slice of the REAL M100 files (two nodes, 288 native
samples) so they stay fast; no data is fabricated and every assertion concerns the
actual processed values.  They cover:

* the long-format panel carries EXACTLY the task-book logical fields and no label/target
  column;
* ``observed_mask`` is False exactly where the value is NaN, missing values are never
  zero-filled and the timeline is not compressed;
* two M100 nodes stay independent domains and are never time-spliced by date (the
  audited files end 15 minutes apart, which the panel must preserve);
* units/semantics/aggregation/availability metadata exist for every feature and power
  fields are marked node-level, not facility;
* ``write_source_artifacts`` refuses ``data/raw/`` and leaves every raw file untouched;
* the UCC bounded audit keeps the native 10-minute frequency instead of faking 15 min.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import source_features as sf
from src.data.contracts import ContractError, IntervalStatistic
from src.utils import PROJECT_ROOT, read_json

BOUNDED_NODES = ["10", "11"]
BOUNDED_ROWS = 288


@pytest.fixture(scope="module")
def config() -> dict:
    return sf.load_source_config()


@pytest.fixture(scope="module")
def bounded_panel(config: dict) -> tuple[pd.DataFrame, dict]:
    return sf.build_source_panel(
        config, nodes=BOUNDED_NODES, max_rows_per_node=BOUNDED_ROWS
    )


@pytest.fixture(scope="module")
def bounded_metadata(
    config: dict, bounded_panel: tuple[pd.DataFrame, dict]
) -> dict:
    panel, summary = bounded_panel
    return sf.build_source_metadata(
        panel,
        config,
        panel_summary=summary,
        hash_files=True,
        include_ucc_audit=False,
    )


@pytest.fixture(scope="module")
def bounded_quality(
    config: dict, bounded_panel: tuple[pd.DataFrame, dict]
) -> dict:
    panel, summary = bounded_panel
    return sf.build_source_quality(panel, config, panel_summary=summary)


# ---------------------------------------------------------------------------
# panel schema / missing semantics
# ---------------------------------------------------------------------------


def test_panel_has_exactly_the_logical_fields(bounded_panel: tuple[pd.DataFrame, dict]) -> None:
    panel, _ = bounded_panel
    assert list(panel.columns) == list(sf.PANEL_COLUMNS)
    assert len(sf.PANEL_COLUMNS) == 7
    assert isinstance(panel["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(panel["timestamp"].dt.tz) == "UTC"
    assert panel["observed_mask"].dtype == bool
    for column in ("domain_id", "feature_name", "unit"):
        assert isinstance(panel[column].dtype, pd.CategoricalDtype)
    # No label/target column may exist in an unlabelled source panel.
    forbidden = [
        column
        for column in panel.columns
        if any(token in column.lower() for token in ("label", "target", "power_kw"))
    ]
    assert forbidden == []
    assert "active_power_kw" not in panel.columns
    assert "facility_power_kw" not in panel.columns


def test_missing_observations_are_masked_and_never_zero_filled(
    bounded_panel: tuple[pd.DataFrame, dict],
) -> None:
    panel, _ = bounded_panel
    missing = ~panel["observed_mask"]
    assert int(missing.sum()) > 0, "bounded slice must contain real telemetry gaps"
    # observed_mask is False exactly where the value is NaN ...
    assert bool((panel["feature_value"].isna() == missing).all())
    # ... and those rows are never encoded as a measured zero.
    assert bool(panel.loc[missing, "feature_value"].isna().all())
    assert bool(np.isfinite(panel.loc[~missing, "feature_value"].to_numpy()).all())
    # coverage_ratio is 1.0 for observed native samples and 0.0 for missing ones.
    assert bool((panel.loc[panel["observed_mask"], "coverage_ratio"] == 1.0).all())
    assert bool((panel.loc[missing, "coverage_ratio"] == 0.0).all())
    # The timeline is not compressed: node 10/11 keep all 288 expected 15-min slots.
    assert int(panel["timestamp"].nunique()) == BOUNDED_ROWS


def test_two_nodes_are_independent_domains_not_time_spliced(config: dict) -> None:
    # Node 13's audited last timestamp is 2022-09-28 22:00, while node 10 ends 21:45;
    # an honest per-domain grid must keep those independent ranges instead of pairing rows.
    panel, summary = sf.build_source_panel(config, nodes=["10", "13"], max_rows_per_node=84000)
    domains = list(panel["domain_id"].cat.categories)
    assert domains == ["m100_node_10", "m100_node_13"]
    assert summary["domain_count"] == 2
    per_domain = {}
    for domain in domains:
        subset = panel[panel["domain_id"] == domain]
        timestamps = subset["timestamp"].drop_duplicates().sort_values()
        per_domain[domain] = (timestamps.iloc[0], timestamps.iloc[-1], len(timestamps))
        # each domain keeps exactly one row per feature per native slot
        assert int(subset.groupby("timestamp", observed=True).size().max()) == 12
    end_10 = per_domain["m100_node_10"][1]
    end_13 = per_domain["m100_node_13"][1]
    assert end_10 == pd.Timestamp("2022-09-28T21:45:00+00:00")
    assert end_13 == pd.Timestamp("2022-09-28T22:00:00+00:00")
    assert end_10 != end_13
    # At node 13's exclusive last timestamp only node 13 rows exist: no cross-domain pairing.
    last_rows = panel[panel["timestamp"] == end_13]
    assert set(last_rows["domain_id"].astype(str)) == {"m100_node_13"}
    # Rows are the sum of independent per-domain grids, not a wide join on dates.
    expected_rows = sum(item[2] for item in per_domain.values()) * 12
    assert len(panel) == expected_rows


def test_load_m100_node_returns_utc_canonical_frame(config: dict) -> None:
    files = sf.m100_files(config, nodes=["10"])
    (domain, path), = files.items()
    assert domain == "m100_node_10"
    frame = sf.load_m100_node(path, columns=["total_power_avg", "ambient_avg"])
    assert list(frame.columns) == ["node_total_power_w", "node_ambient_temp_c"]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert str(frame.index.tz) == "UTC"
    assert frame.index.name == "timestamp"
    assert frame.attrs["domain_id"] == "m100_node_10"
    assert frame.attrs["scope"] == "node"
    assert frame.attrs["source_file"].endswith("10.parquet")
    # bookkeeping / uncurated columns must never be treated as physical features
    with pytest.raises(ContractError, match="not in the curated feature list"):
        sf.load_m100_node(path, columns=["value"])
    with pytest.raises(ContractError, match="not in the curated feature list"):
        sf.load_m100_node(path, columns=["__index_level_0__"])


# ---------------------------------------------------------------------------
# semantic metadata
# ---------------------------------------------------------------------------


def test_units_and_semantics_present_for_every_feature(bounded_metadata: dict) -> None:
    features = bounded_metadata["features"]
    assert set(features) == set(
        [
            "node_total_power_w",
            "cpu0_power_w",
            "cpu1_power_w",
            "cpu0_memory_power_w",
            "cpu0_io_power_w",
            "psu0_input_power_w",
            "psu1_input_power_w",
            "disk_power_w",
            "node_ambient_temp_c",
            "gpu0_core_temp_c",
            "gpu1_core_temp_c",
            "fan0_0_speed_rpm",
        ]
    )
    for name, item in features.items():
        assert item["unit"].strip(), name
        assert item["semantic_type"] in sf.SEMANTIC_TYPES, name
        assert item["aggregation_rule"] in sf.AGGREGATION_RULES, name
        assert item["availability"] in sf.AVAILABILITIES, name
        assert item["physical_range"] is not None or item["semantic_type"] == "other"
        assert item["source_field"] == item["raw_name"]
        assert item["source_dataset"] == "m100_exadata_pretrain"
        assert item["statistic"] in {stat.value for stat in IntervalStatistic}
        assert 0.0 <= item["observed_fraction_overall"] <= 1.0
        assert set(item["observed_fraction_by_domain"]) == {f"m100_node_{stem}" for stem in BOUNDED_NODES}


def test_power_fields_are_node_level_not_facility(bounded_metadata: dict) -> None:
    for name, item in bounded_metadata["features"].items():
        if item["semantic_type"] not in {"power", "temperature", "fan_speed"}:
            continue
        assert item["scope"] == "node", name
        assert item["node_level_not_facility"] is True, name
        boundary = item["measurement_boundary"].lower()
        assert "node" in boundary, name
        assert "facility" in boundary, name
        assert "not facility" in boundary, name
    statement = bounded_metadata["sources"]["m100_exadata"]["explicit_statement"]
    assert "NODE-level" in statement
    assert "NOT facility" in statement
    # M100 native frequency is audited as 15 min and kept natively.
    native = bounded_metadata["native_frequency"]["m100_exadata"]
    assert native["verdict"] == "15min_native_kept"
    assert native["declared_frequency"] == native["target_frequency"] == "15min"
    for domain in ("m100_node_10", "m100_node_11"):
        assert native["domains"][domain]["native_interval_seconds"] == 900.0
        assert native["domains"][domain]["all_deltas_multiples_of_native"] is True


def test_metadata_reports_honest_training_approval(bounded_metadata: dict) -> None:
    approval = bounded_metadata["training_approval"]
    assert approval["approved"] is False
    assert approval["approved_for_facility_kw_training"] is False
    assert approval["reason"].strip()
    assert bounded_metadata["sources"]["m100_exadata"]["approved_for_training"] is False
    assert bounded_metadata["sources"]["m100_exadata"]["audit_status"] == "partial"
    assert bounded_metadata["sources"]["ucc2023"]["approved_for_training"] is False
    alibaba = bounded_metadata["sources"]["alibaba_gpu_v2020"]
    assert alibaba["status"] == "not_ingested"
    assert alibaba["reasons"]
    # file provenance: streamed SHA256 for the consumed M100 files
    files = bounded_metadata["sources"]["m100_exadata"]["files"]
    assert len(files) == len(BOUNDED_NODES)
    for item in files:
        assert item["sha256"] and len(item["sha256"]) == 64
        assert item["bytes"] > 0
        assert item["domain_id"].startswith("m100_node_")


def test_quality_counts_missingness_and_range_policy(bounded_quality: dict) -> None:
    features = bounded_quality["features"]
    for name, item in features.items():
        assert item["rows"] == item["observed"] + item["missing"]
        assert 0.0 <= item["observed_fraction"] <= 1.0
        assert item["range_violations_clipped"] is False
        assert item["distribution_sampled"]
    flags = bounded_quality["quality_flags"]
    assert flags["missing_encoded_as_zero"] is False
    assert flags["range_violations_clipped"] is False
    assert flags["internal_gaps_preserved"] is True
    assert flags["cross_domain_time_splicing"] is False
    assert flags["label_columns_present"] is False
    assert bounded_quality["distribution_summary"]["total_rows"] > 0


# ---------------------------------------------------------------------------
# artifacts / raw protection
# ---------------------------------------------------------------------------


def _raw_snapshot() -> dict[str, tuple[int, int]]:
    root = PROJECT_ROOT / "data" / "raw"
    snapshot: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path)] = (int(stat.st_size), int(stat.st_mtime_ns))
    return snapshot


def test_write_artifacts_roundtrip_and_refuses_raw(
    config: dict,
    bounded_panel: tuple[pd.DataFrame, dict],
    bounded_metadata: dict,
    bounded_quality: dict,
    tmp_path: Path,
) -> None:
    panel, _ = bounded_panel
    before = _raw_snapshot()
    artifacts = sf.write_source_artifacts(
        panel, bounded_metadata, bounded_quality, out_dir=tmp_path
    )
    assert artifacts["panel_rows"] == len(panel)
    assert artifacts["label_columns_present"] is False
    for key in ("panel_parquet", "metadata_json", "quality_json"):
        assert Path(artifacts[key]).is_file()
    roundtrip = pd.read_parquet(artifacts["panel_parquet"])
    assert list(roundtrip.columns) == list(sf.PANEL_COLUMNS)
    assert len(roundtrip) == len(panel)
    assert bool((roundtrip["feature_value"].isna() == ~roundtrip["observed_mask"]).all())
    metadata = read_json(artifacts["metadata_json"])
    assert metadata["panel"]["rows"] == len(panel)
    assert metadata["artifacts"]["panel_bytes"] > 0
    quality = read_json(artifacts["quality_json"])
    assert quality["panel"]["rows"] == len(panel)
    # The raw tree is untouched by the whole bounded build+write.
    assert _raw_snapshot() == before
    # Writing below data/raw/ is refused.
    with pytest.raises(ContractError, match="raw"):
        sf.write_source_artifacts(panel, bounded_metadata, bounded_quality, out_dir="data/raw/processed")


def test_cli_refuses_raw_output_before_building() -> None:
    from scripts import prepare_source_features

    with pytest.raises(SystemExit) as excinfo:
        prepare_source_features.main(
            ["--nodes", "10", "--max-rows-per-node", "48", "--out", "data/raw/processed"]
        )
    assert excinfo.value.code == 1
    assert not (PROJECT_ROOT / "data" / "raw" / "processed").exists()


def test_ucc_audit_preserves_native_ten_minute_frequency(config: dict) -> None:
    bounded_config = {
        **config,
        "ucc2023": {**config["ucc2023"], "bounded_audit_rows": 2000},
    }
    audit = sf.build_ucc_audit(bounded_config, hash_files=False)
    assert audit["native_frequency"] == "10min"
    assert audit["target_frequency"] == "15min"
    assert audit["aggregation"]["status"] == "native_preserved_not_15min"
    assert audit["aggregation"]["verdict"] == "not_defensible_without_assumptions"
    assert audit["aggregation"]["fifteen_minute_product_created"] is False
    assert audit["bounded_audit_rows_per_file"] == 2000
    native_counts = audit["tables"]["primary_file"]["native_delta_seconds_counts"]
    assert "600" in native_counts
    assert int(native_counts["600"]) > 0
    assert audit["tables"]["primary_file"]["all_deltas_multiples_of_native"] is True
    assert audit["limitations"]
    assert audit["approved_for_training"] is False
    assert audit["files"]["primary_file"]["rows_total"] > 0
    assert audit["files"]["primary_file"]["rows_audited"] == 2000
