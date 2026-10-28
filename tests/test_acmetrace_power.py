"""Fast acceptance tests for the AcmeTrace power audit (synthetic fixtures only).

No multi-GB raw file is ever read here: every test builds a tiny fixture tree with the
real AcmeTrace column names and runs the streaming code against it.  Covered rules:

* mixed IPMI timestamp parsing (Chinese-locale CST + ISO +08:00);
* GPU device-column parsing;
* zero/negative power is invalid, never 0 kW;
* 15-minute coverage math (hand-computed), full-grid gap rows, no interpolation;
* cluster aggregation never zero-fills missing nodes and reports online counts;
* PSU metering ratios/correlation are same-server diagnostics, not independent evidence;
* Kalos has no server / cluster_it level (no IPMI);
* unit_verified markers stay false and the raw tree is never modified;
* inventory vs HF-tree comparison (sha/size/missing);
* processed parquet schema and the exact cluster_it / cluster_gpu_sum rows.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import acmetrace_power as acme
from src.data.contracts import ContractError
from src.utils import PROJECT_ROOT

HOST_TZ = "+08:00"


# ---------------------------------------------------------------------------
# fixture tree
# ---------------------------------------------------------------------------


def _write_gpu_file(path: Path, devices: list[str], *, start: str, periods: int, freq: str) -> pd.DataFrame:
    stamps = pd.date_range(start, periods=periods, freq=freq, tz="Asia/Shanghai")
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({device: rng.uniform(100.0, 400.0, periods) for device in devices})
    frame.insert(0, "Time", stamps.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
        r"(\d{2})(\d{2})$", r"\1:\2", regex=True
    ))
    frame.to_csv(path, index=False)
    return frame


def _write_ipmi_cpu_d(path: Path) -> None:
    times = [
        "2023年 05月 15日 星期一 14:15:01 CST",
        "2023年 05月 15日 星期一 14:20:01 CST",
        "2023年 05月 15日 星期一 14:25:01 CST",
        "2023-05-15 14:30:01+08:00",
        "2023-05-15 14:35:01+08:00",
        "2023-05-15 14:40:01+08:00",
    ]
    frame = pd.DataFrame(
        {
            "Time": times,
            "Sys_Total_Power": [500, 520, 480, 510, 530, 490],
            "CPU_Power": [300] * 6,
            "Memory_Power": [50] * 6,
            "PSU1_Pin": [260] * 6,
            "PSU1_Pout": [240] * 6,
            "PSU2_Pin": [270] * 6,
            "PSU2_Pout": [250] * 6,
            "node": ["SH-IDC1-10-140-0-1"] * 6,
            "PSU_IN_SUM": [530, 551, 509, 541, 562, 519],
            "PSU_OUT_SUM": [490, 510, 470, 500, 519, 480],
        }
    )
    frame.to_csv(path, index=False)


def _write_ipmi_model_file(path: Path, *, model: str, node: str, periods: int = 6) -> None:
    stamps = pd.date_range("2023-05-15 14:15:01", periods=periods, freq="5min", tz="Asia/Shanghai")
    times = stamps.strftime("%Y-%m-%d %H:%M:%S+08:00")
    if model == "GPU_AB":
        frame = pd.DataFrame(
            {
                "Time": times,
                "PSU5_PIN": [0] * periods,
                "PSU6_PIN": [0] * periods,
                "PSU1_PIN": [10] * periods,
                "PSU2_PIN": [10] * periods,
                "PSU3_PIN": [10] * periods,
                "PSU4_PIN": [10] * periods,
                "Sys_Total_Power": [5000] * periods,
                "CPU_Power": [400] * periods,
                "Memory_Power": [200] * periods,
                "node": [node] * periods,
                "PSU_IN_SUM": [5300] * periods,
            }
        )
    else:
        frame = pd.DataFrame(
            {
                "Time": times,
                "Sys_Total_Power": [4500] * periods,
                "FAN_Total_Power": [20] * periods,
                "CPU_Power": [300] * periods,
                "Memory_Power": [60] * periods,
                "2U_Total_Power": [700] * periods,
                "HDD_Power": [30] * periods,
                "PSU1_Pin": [400] * periods,
                "PSU1_Pout": [380] * periods,
                "PSU2_Pin": [400] * periods,
                "PSU2_Pout": [380] * periods,
                "PSU3_Pin": [400] * periods,
                "PSU3_Pout": [380] * periods,
                "PSU4_Pin": [400] * periods,
                "PSU4_Pout": [380] * periods,
                "node": [node] * periods,
                "PSU_IN_SUM": [4800] * periods,
                "PSU_OUT_SUM": [4500] * periods,
            }
        )
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def fixture_tree(tmp_path: Path) -> dict:
    """A tiny AcmeTrace-like raw tree + HF tree JSON + derived config."""
    raw = tmp_path / "raw"
    seren = raw / "data/utilization/seren"
    kalos = raw / "data/utilization/kalos"
    ipmi = raw / "data/utilization/ipmi"
    for directory in (seren, kalos, ipmi, raw / "data/job_trace"):
        directory.mkdir(parents=True, exist_ok=True)
    (raw / "README.md").write_text("# fixture acme trace\n", encoding="utf-8")
    (raw / "data/cluster_summary.csv").write_text("id,job_num\nSeren,1\nKalos,2\n", encoding="utf-8")
    (raw / "data/job_trace/trace_kalos.csv").write_text("job_id,user\n1,u1\n", encoding="utf-8")
    # 3 bins: rows 00:00-00:15 (60 samples), gap at 00:15-00:30, rows 00:30-00:45 (60 samples).
    seren_stamps = pd.date_range("2023-05-16 00:00:00", periods=60, freq="15s", tz="Asia/Shanghai").tolist()
    seren_stamps += pd.date_range("2023-05-16 00:30:00", periods=60, freq="15s", tz="Asia/Shanghai").tolist()
    rng = np.random.default_rng(7)
    seren_frame = pd.DataFrame(
        {
            "Time": [stamp.strftime("%Y-%m-%d %H:%M:%S+08:00") for stamp in seren_stamps],
            "10.140.0.1-0": rng.uniform(100.0, 400.0, len(seren_stamps)),
            "10.140.0.1-1": rng.uniform(100.0, 400.0, len(seren_stamps)),
        }
    )
    seren_frame.loc[5, "10.140.0.1-0"] = 0.0      # offline reading -> invalid
    seren_frame.loc[6, "10.140.0.1-1"] = -3.0     # sensor glitch -> invalid
    seren_frame.to_csv(seren / "POWER_USAGE.csv", index=False)
    _write_gpu_file(kalos / "POWER_USAGE.csv", ["172.31.0.1-0"], start="2023-08-15 00:00:00", periods=60, freq="15s")
    for cluster_dir, node in ((seren, "10.140.0.1"), (kalos, "172.31.0.1")):
        stamps = pd.date_range("2023-05-16 00:00:00", periods=3, freq="15min", tz="Asia/Shanghai")
        (cluster_dir / "NODE_CPU_UTILIZATION.csv").write_text(
            "Time," + node + "\n" + "\n".join(f"{stamp.strftime('%Y-%m-%d %H:%M:%S+08:00')},1.0" for stamp in stamps) + "\n",
            encoding="utf-8",
        )
    _write_ipmi_cpu_d(ipmi / "CPU_D_Power.csv")
    _write_ipmi_model_file(ipmi / "GPU_AB_Power.csv", model="GPU_AB", node="SH-IDC1-10-140-0-2")
    _write_ipmi_model_file(ipmi / "GPU_C_Power.csv", model="GPU_C", node="SH-IDC1-10-140-0-3")

    tree = {
        "format_version": 1,
        "files": {
            "README.md": {"size": (raw / "README.md").stat().st_size, "blob_id": "x"},
            "data/utilization/seren/POWER_USAGE.csv": {
                "size": (seren / "POWER_USAGE.csv").stat().st_size,
                "lfs_sha256": "deadbeef",
                "lfs_size": (seren / "POWER_USAGE.csv").stat().st_size,
            },
            "data/utilization/kalos/POWER_USAGE.csv": {
                "size": (kalos / "POWER_USAGE.csv").stat().st_size,
                "lfs_sha256": _sha256(kalos / "POWER_USAGE.csv"),
                "lfs_size": (kalos / "POWER_USAGE.csv").stat().st_size,
            },
            "data/utilization/seren/NODE_CPU_UTILIZATION.csv": {
                "size": 999999,
                "lfs_sha256": _sha256(seren / "NODE_CPU_UTILIZATION.csv"),
                "lfs_size": 999999,
            },
            "LICENSE.txt": {"size": 100, "blob_id": "x"},
        },
    }
    tree_path = tmp_path / "hf_tree.json"
    tree_path.write_text(json.dumps(tree), encoding="utf-8")

    config = copy.deepcopy(acme.load_acmetrace_config())
    config["raw_root"] = str(raw)
    config["hf_tree"] = str(tree_path)
    config["readme"] = str(raw / "README.md")
    for key, directory in (("seren", seren), ("kalos", kalos)):
        config["clusters"][key]["gpu_power_file"] = str(directory / "POWER_USAGE.csv")
        config["clusters"][key]["node_inventory_file"] = str(directory / "NODE_CPU_UTILIZATION.csv")
    config["ipmi"]["files"]["cpu_d"]["path"] = str(ipmi / "CPU_D_Power.csv")
    config["ipmi"]["files"]["gpu_ab"]["path"] = str(ipmi / "GPU_AB_Power.csv")
    config["ipmi"]["files"]["gpu_c"]["path"] = str(ipmi / "GPU_C_Power.csv")
    processed = tmp_path / "processed"
    config["outputs"] = {
        "processed_dir": str(processed),
        "results_dir": str(tmp_path / "results"),
        "inventory_json": str(processed / acme.INVENTORY_FILENAME),
        "power_audit_json": str(processed / acme.POWER_AUDIT_FILENAME),
        "metadata_json": str(processed / acme.METADATA_FILENAME),
        "power_15min_parquet": str(processed / acme.POWER_15MIN_FILENAME),
    }
    return {"raw": raw, "tree": tree_path, "config": config, "processed": processed}


# ---------------------------------------------------------------------------
# parsers / masks
# ---------------------------------------------------------------------------


def test_parse_mixed_ipmi_timestamps() -> None:
    values = [
        "2023年 05月 15日 星期一 14:15:01 CST",
        "2023-05-15 14:15:01+08:00",
        "2023年 05月 15日 星期一 14:20:01 CST",
        "2023-05-15 14:20:01+08:00",
    ]
    stamps = acme.parse_acmetrace_timestamps(values)
    assert str(stamps[0]) == "2023-05-15 06:15:01"
    assert str(stamps[1]) == "2023-05-15 06:15:01"  # both spellings = same instant
    assert str(stamps[2]) == "2023-05-15 06:20:01"
    assert str(stamps[3]) == "2023-05-15 06:20:01"
    with pytest.raises(ValueError, match="Unparseable"):
        acme.parse_acmetrace_timestamps(["not-a-timestamp"])


def test_parse_gpu_device_columns() -> None:
    assert acme.parse_gpu_device_id("10.140.1.10-6") == ("10.140.1.10", 6)
    assert acme.gpu_device_id("10.140.1.10", 6) == "10.140.1.10-6"
    with pytest.raises(ValueError):
        acme.parse_gpu_device_id("10.140.1.10")
    with pytest.raises(ValueError):
        acme.parse_gpu_device_id("not-an-ip-0")


def test_zero_and_negative_power_is_invalid() -> None:
    values = np.array([0.0, -5.0, 100.0, 1500.0, 1500.1, np.nan])
    mask = acme.valid_power_mask(values, max_plausible_w=1500.0)
    assert mask.tolist() == [False, False, True, True, False, False]


def test_node_id_to_ip() -> None:
    assert acme.node_to_ip("SH-IDC1-10-140-0-112") == "10.140.0.112"
    with pytest.raises(ValueError):
        acme.node_to_ip("SH-IDC1-10-140")


# ---------------------------------------------------------------------------
# 15-minute coverage math
# ---------------------------------------------------------------------------


def test_aggregate_device_15min_hand_computed() -> None:
    # 60 valid samples at 15 s = one full 15-min bin with mean 300 W -> 0.3 kW.
    stamps = pd.date_range("2023-05-16 00:00:00", periods=60, freq="15s")
    values = np.full(60, 300.0)
    frame = acme.aggregate_device_15min(
        stamps.to_numpy("datetime64[ns]").astype(np.int64),
        values,
        expected_samples=60,
        max_plausible_w=2000,
        validity_threshold=0.5,
    )
    assert len(frame) == 1
    assert frame["power_kw"].iloc[0] == pytest.approx(0.3)
    assert bool(frame["valid"].iloc[0]) is True
    assert frame["coverage_ratio"].iloc[0] == pytest.approx(1.0)
    assert int(frame["n_samples"].iloc[0]) == 60
    # Zero readings are excluded (not averaged in as 0 kW), lowering coverage.
    values[0:30] = 0.0
    frame = acme.aggregate_device_15min(
        stamps.to_numpy("datetime64[ns]").astype(np.int64),
        values,
        expected_samples=60,
        max_plausible_w=2000,
        validity_threshold=0.5,
    )
    assert frame["coverage_ratio"].iloc[0] == pytest.approx(0.5)
    assert frame["power_kw"].iloc[0] == pytest.approx(0.3)
    assert bool(frame["valid"].iloc[0]) is True  # coverage exactly at the threshold
    values[0:35] = 0.0
    frame = acme.aggregate_device_15min(
        stamps.to_numpy("datetime64[ns]").astype(np.int64),
        values,
        expected_samples=60,
        max_plausible_w=2000,
        validity_threshold=0.5,
    )
    assert frame["coverage_ratio"].iloc[0] == pytest.approx(25 / 60)
    assert bool(frame["valid"].iloc[0]) is False


def test_aggregate_device_15min_keeps_full_grid_gap_rows() -> None:
    first = pd.date_range("2023-05-16 00:00:00", periods=60, freq="15s")
    second = pd.date_range("2023-05-16 00:30:00", periods=60, freq="15s")
    stamps = pd.DatetimeIndex(list(first) + list(second))
    values = np.full(120, 250.0)
    frame = acme.aggregate_device_15min(
        stamps.to_numpy("datetime64[ns]").astype(np.int64),
        values,
        expected_samples=60,
        max_plausible_w=2000,
        validity_threshold=0.5,
    )
    assert len(frame) == 3  # 00:00, 00:15 (gap), 00:30
    rows = frame.set_index("timestamp")
    assert str(frame["timestamp"].iloc[0]) == "2023-05-16 00:00:00"
    assert str(frame["timestamp"].iloc[1]) == "2023-05-16 00:15:00"
    gap = rows.loc[pd.Timestamp("2023-05-16 00:15:00")]
    assert np.isnan(gap["power_kw"])
    assert bool(gap["valid"]) is False
    assert gap["coverage_ratio"] == 0.0
    assert int(gap["n_samples"]) == 0
    assert bool(rows.loc[pd.Timestamp("2023-05-16 00:30:00"), "valid"]) is True


def test_aggregate_cluster_15min_never_zero_fills_missing_nodes() -> None:
    # Three native slots in one bin: two with all 248 nodes reporting, one with 124 online.
    stamps = pd.date_range("2023-05-16 00:00:00", periods=3, freq="5min")
    per_slot_sum = np.array([100_000.0, 100_000.0, 50_000.0])
    per_slot_online = np.array([248.0, 248.0, 124.0])
    frame = acme.aggregate_cluster_15min(
        stamps.to_numpy("datetime64[ns]").astype(np.int64),
        per_slot_sum,
        per_slot_online,
        expected_samples=3,
        expected_members=248,
        validity_threshold=0.5,
        member_kind="ipmi_node",
    )
    assert len(frame) == 1
    assert frame["power_kw"].iloc[0] == pytest.approx((100_000 + 100_000 + 50_000) / 3 / 1000)
    assert frame["coverage_ratio"].iloc[0] == pytest.approx((248 + 248 + 124) / (3 * 248))
    assert int(frame["n_samples"].iloc[0]) == 3
    assert bool(frame["valid"].iloc[0]) is True
    # A partially covered bin is invalid, not a fabricated full-cluster total.
    frame = acme.aggregate_cluster_15min(
        stamps.to_numpy("datetime64[ns]").astype(np.int64),
        per_slot_sum,
        np.array([124.0, 0.0, 0.0]),
        expected_samples=3,
        expected_members=248,
        validity_threshold=0.5,
        member_kind="ipmi_node",
    )
    assert frame["power_kw"].iloc[0] == pytest.approx(100.0)  # only reporting slot
    assert bool(frame["valid"].iloc[0]) is False


def test_build_cluster_slots_uses_reporting_only() -> None:
    ts = np.array([0, 0, 900_000_000_000], dtype=np.int64)
    power = np.array([100.0, 50.0, 200.0])
    members = np.array([True, False, True])
    slot_ts, slot_sum, slot_online = acme.build_cluster_slots(
        [(ts, power, members)], member_mask=np.array([True, True, True])
    )
    assert slot_ts.tolist() == [0, 900_000_000_000]
    assert slot_sum.tolist() == [100.0, 200.0]
    assert slot_online.tolist() == [1.0, 1.0]


# ---------------------------------------------------------------------------
# streaming scans on fixtures
# ---------------------------------------------------------------------------


def test_scan_gpu_file_counts_zero_negative_and_valid(fixture_tree: dict) -> None:
    scan = acme.scan_gpu_power_file(fixture_tree["config"], "seren", progress_every=0)
    assert scan["device_count"] == 2
    assert scan["node_count"] == 1
    assert scan["native_rows"] == 120
    assert scan["native_interval_mode_seconds"] == 15
    stats = scan["device_stats"]
    assert stats["n_zero"].sum() == 1
    assert stats["n_negative"].sum() == 1
    # 120 native rows; device 0 has one zero and device 1 one negative reading
    assert stats["n_valid"].tolist() == [119, 119]
    # zero/negative are not in the valid range
    assert float(stats["min_w"][0]) > 0.0
    assert float(stats["min_w"][1]) > 0.0


def test_scan_ipmi_timestamps_metering_and_zero(fixture_tree: dict) -> None:
    config = fixture_tree["config"]
    scan = acme.scan_ipmi_power_file(config, "cpu_d", scope_nodes={"10.140.0.1"}, progress_every=0)
    assert scan["node_count"] == 1
    assert scan["native_interval_mode_seconds"] == 300
    assert scan["timestamp_min"] == "2023-05-15 06:15:00"
    assert scan["timestamp_max"] == "2023-05-15 06:40:00"
    metering = scan["metering"]
    key = "Sys_Total_Power__PSU_IN_SUM"
    assert metering[key]["n"] == 6
    assert metering[key]["ratio_mean"] == pytest.approx(1.06, rel=1e-3)
    assert "not independent" in metering[key]["note"]
    # CPU_D declares meter columns [Sys_Total_Power, PSU_IN_SUM, PSU_OUT_SUM]; the scan
    # compares Sys_Total to each of the others.
    assert "Sys_Total_Power__PSU_OUT_SUM" in metering
    assert metering["Sys_Total_Power__PSU_OUT_SUM"]["ratio_mean"] == pytest.approx(0.98, rel=1e-3)
    # Cluster slot inputs exist for the in-scope node (2 bins, 3 samples each).
    cluster_ts, cluster_sum, cluster_online = scan["_cluster_arrays"]
    assert cluster_ts.size == 6
    assert np.all(cluster_online == 1.0)
    assert np.isclose(cluster_sum.sum(), 3030.0)


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_inventory_compares_hashes_sizes_and_missing(fixture_tree: dict) -> None:
    inventory = acme.build_inventory(fixture_tree["config"], hash_contents=True, progress=False)
    summary = inventory["summary"]
    assert summary["local_files"] >= 10
    assert summary["files_missing_locally"] == 1  # LICENSE.txt
    assert inventory["missing_from_local"] == ["LICENSE.txt"]
    assert "data/utilization/seren/POWER_USAGE.csv" in summary["sha256_mismatches"]
    assert inventory["integrity"]["size_mismatches"] == ["data/utilization/seren/NODE_CPU_UTILIZATION.csv"]
    seren = inventory["files"]["data/utilization/seren/POWER_USAGE.csv"]
    assert seren["category"] == "dataset_csv"
    assert seren["sha256_match"] is False
    assert seren["lines_total"] == 121
    assert seren["data_rows"] == 120
    assert seren["n_columns"] == 3
    kalos = inventory["files"]["data/utilization/kalos/POWER_USAGE.csv"]
    assert kalos["sha256_match"] is True
    readme = inventory["files"]["README.md"]
    assert readme["category"] == "documentation"
    assert readme["in_hf_tree"] is True


def test_inventory_skip_hash_marks_unverified_not_mismatch(fixture_tree: dict) -> None:
    inventory = acme.build_inventory(fixture_tree["config"], hash_contents=False, progress=False)
    summary = inventory["summary"]
    assert summary["hash_contents"] is False
    assert summary["sha256_verification"] == "skipped"
    assert summary["sha256_mismatches"] == []
    assert summary["local_bytes"] > 0
    assert summary["sha256_unverified_files"] > 0
    assert summary["files_present_locally"] >= 1
    seren = inventory["files"]["data/utilization/seren/POWER_USAGE.csv"]
    assert seren["sha256_match"] is None
    assert str(seren["sha256_verification"]).startswith("skipped")
    assert "SKIPPED" in inventory["integrity"]["note"]


# ---------------------------------------------------------------------------
# configuration honesty rules
# ---------------------------------------------------------------------------


def test_config_rejects_facility_or_kalos_server_claims() -> None:
    config = acme.load_acmetrace_config()
    assert config["levels"]["kalos"] == ["gpu", "cluster_gpu_sum"]
    assert config["clusters"]["kalos"]["has_ipmi_server_power"] is False
    assert config["source"]["units"]["power_unit_verified"] is False
    broken = copy.deepcopy(config)
    broken["levels"]["kalos"].append("server")
    with pytest.raises(ValueError, match="Kalos"):
        acme.validate_acmetrace_config(broken)
    broken = copy.deepcopy(config)
    broken["source"]["units"]["power_unit_verified"] = True
    with pytest.raises(ValueError, match="unit_verified|power_unit_verified"):
        acme.validate_acmetrace_config(broken)
    broken = copy.deepcopy(config)
    broken["aggregation_15min"]["zero_fill"] = True
    with pytest.raises(ValueError, match="zero_fill"):
        acme.validate_acmetrace_config(broken)


# ---------------------------------------------------------------------------
# end-to-end fixture pipeline + parquet contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    module_tmp = tmp_path_factory.mktemp("acmetrace_power")
    tree = _fixture_tree_value(module_tmp)
    result = acme.run_pipeline(tree["config"], hash_contents=True, progress=False)
    return {**tree, "result": result}


def _fixture_tree_value(tmp_path: Path) -> dict:
    # Local copy of the fixture builder for module scope (kept small on purpose).
    raw = tmp_path / "raw"
    seren = raw / "data/utilization/seren"
    kalos = raw / "data/utilization/kalos"
    ipmi = raw / "data/utilization/ipmi"
    for directory in (seren, kalos, ipmi, raw / "data/job_trace"):
        directory.mkdir(parents=True, exist_ok=True)
    (raw / "README.md").write_text("# fixture acme trace\n", encoding="utf-8")
    (raw / "data/cluster_summary.csv").write_text("id,job_num\nSeren,1\nKalos,2\n", encoding="utf-8")
    (raw / "data/job_trace/trace_kalos.csv").write_text("job_id,user\n1,u1\n", encoding="utf-8")
    stamps = pd.date_range("2023-05-16 00:00:00", periods=60, freq="15s", tz="Asia/Shanghai")
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "Time": stamps.strftime("%Y-%m-%d %H:%M:%S+08:00"),
            "10.140.0.1-0": rng.uniform(100.0, 400.0, 60),
            "10.140.0.1-1": rng.uniform(100.0, 400.0, 60),
        }
    )
    frame.to_csv(seren / "POWER_USAGE.csv", index=False)
    _write_gpu_file(kalos / "POWER_USAGE.csv", ["172.31.0.1-0"], start="2023-08-15 00:00:00", periods=60, freq="15s")
    for cluster_dir, node in ((seren, "10.140.0.1"), (kalos, "172.31.0.1")):
        cluster_dir.joinpath("NODE_CPU_UTILIZATION.csv").write_text(
            "Time," + node + "\n" + f"2023-05-16 00:00:00+08:00,1.0\n", encoding="utf-8"
        )
    _write_ipmi_cpu_d(ipmi / "CPU_D_Power.csv")
    _write_ipmi_model_file(ipmi / "GPU_AB_Power.csv", model="GPU_AB", node="SH-IDC1-10-140-0-2")
    _write_ipmi_model_file(ipmi / "GPU_C_Power.csv", model="GPU_C", node="SH-IDC1-10-140-0-3")
    tree_path = tmp_path / "hf_tree.json"
    tree_path.write_text(json.dumps({"format_version": 1, "files": {}}), encoding="utf-8")
    config = copy.deepcopy(acme.load_acmetrace_config())
    config["raw_root"] = str(raw)
    config["hf_tree"] = str(tree_path)
    config["readme"] = str(raw / "README.md")
    for key, directory in (("seren", seren), ("kalos", kalos)):
        config["clusters"][key]["gpu_power_file"] = str(directory / "POWER_USAGE.csv")
        config["clusters"][key]["node_inventory_file"] = str(directory / "NODE_CPU_UTILIZATION.csv")
    config["ipmi"]["files"]["cpu_d"]["path"] = str(ipmi / "CPU_D_Power.csv")
    config["ipmi"]["files"]["gpu_ab"]["path"] = str(ipmi / "GPU_AB_Power.csv")
    config["ipmi"]["files"]["gpu_c"]["path"] = str(ipmi / "GPU_C_Power.csv")
    processed = tmp_path / "processed"
    config["outputs"] = {
        "processed_dir": str(processed),
        "results_dir": str(tmp_path / "results"),
        "inventory_json": str(processed / acme.INVENTORY_FILENAME),
        "power_audit_json": str(processed / acme.POWER_AUDIT_FILENAME),
        "metadata_json": str(processed / acme.METADATA_FILENAME),
        "power_15min_parquet": str(processed / acme.POWER_15MIN_FILENAME),
    }
    return {"raw": raw, "tree": tree_path, "config": config, "processed": processed}


def test_pipeline_levels_and_parquet_contract(pipeline_outputs: dict) -> None:
    tree = pipeline_outputs
    result = tree["result"]
    parquet = pd.read_parquet(tree["processed"] / acme.POWER_15MIN_FILENAME)
    assert list(parquet.columns) == list(acme.POWER_15MIN_COLUMNS)
    levels = set(zip(parquet["cluster_id"].astype(str), parquet["level"].astype(str)))
    assert ("seren", "cluster_it") in levels
    assert ("seren", "cluster_gpu_sum") in levels
    assert ("seren", "server") in levels
    assert ("kalos", "cluster_gpu_sum") in levels
    assert ("kalos", "gpu") in levels
    # Kalos has NO server / cluster_it rows.
    assert ("kalos", "server") not in levels
    assert ("kalos", "cluster_it") not in levels
    # The exact V3.2 window rows are identifiable.
    seren_it = parquet[(parquet["cluster_id"] == "seren") & (parquet["level"] == "cluster_it")]
    assert set(seren_it["device_id"].astype(str)) == {"seren_cluster_it"}
    kalos_sum = parquet[(parquet["cluster_id"] == "kalos") & (parquet["level"] == "cluster_gpu_sum")]
    assert set(kalos_sum["device_id"].astype(str)) == {"kalos_cluster_gpu_sum"}
    # Mask semantics: NaN only where no native sample was observed.
    assert bool((parquet["power_kw"].isna() == (parquet["n_samples"] == 0)).all())
    assert np.isfinite(parquet.loc[parquet["n_samples"] > 0, "power_kw"]).all()
    assert bool(((parquet["coverage_ratio"] >= 0) & (parquet["coverage_ratio"] <= 1)).all())
    # Seren cluster IT mean for the one fixture bin is the mean of the reporting node.
    assert seren_it["power_kw"].iloc[0] == pytest.approx(500 / 1000, rel=1e-6)
    assert seren_it["valid"].iloc[0]
    # Seren cluster_gpu_sum excludes zero/negative readings, so it is not a fixed sum.
    assert kalos_sum["power_kw"].iloc[0] > 0


def test_pipeline_artefacts_and_raw_immutability(pipeline_outputs: dict) -> None:
    tree = pipeline_outputs
    raw = tree["raw"]
    before = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path))
        for path in raw.rglob("*")
        if path.is_file()
    }
    config = copy.deepcopy(tree["config"])
    processed = tree["processed"]
    result = acme.run_pipeline(config, hash_contents=True, progress=False)
    after = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path))
        for path in raw.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert (processed / acme.INVENTORY_FILENAME).is_file()
    assert (processed / acme.POWER_AUDIT_FILENAME).is_file()
    assert (processed / acme.METADATA_FILENAME).is_file()
    metadata = json.loads((processed / acme.METADATA_FILENAME).read_text(encoding="utf-8"))
    audit = json.loads((processed / acme.POWER_AUDIT_FILENAME).read_text(encoding="utf-8"))
    assert metadata["source"]["units"]["power_unit_verified"] is False
    assert audit["units"]["power_unit_verified"] is False
    assert audit["hierarchy"]["facility_power_available"] is False
    assert audit["hierarchy"]["kalos"]["has_server_power"] is False
    assert audit["hierarchy"]["kalos"]["has_cluster_it"] is False
    assert audit["readiness"]["ready_for_training"] is False
    assert result["table_stats"]["rows"] == len(pd.read_parquet(processed / acme.POWER_15MIN_FILENAME))
    # 15-minute contract metadata
    assert metadata["aggregation_15min"]["validity_threshold"] == 0.5
    assert metadata["aggregation_15min"]["interpolation"] is False
    assert metadata["aggregation_15min"]["zero_fill"] is False


def test_writers_and_cli_refuse_data_raw(pipeline_outputs: dict, tmp_path: Path) -> None:
    config = copy.deepcopy(pipeline_outputs["config"])
    raw_root = Path(str(config["raw_root"]))
    config["outputs"] = {
        "processed_dir": str(tmp_path / "processed"),
        "results_dir": str(tmp_path / "results"),
        "inventory_json": str(raw_root / "acmetrace_inventory.json"),
        "power_audit_json": str(tmp_path / "processed/acmetrace_power_audit.json"),
        "metadata_json": str(tmp_path / "processed/acmetrace_power_metadata.json"),
        "power_15min_parquet": str(tmp_path / "processed/acmetrace_power_15min.parquet"),
    }
    with pytest.raises(ContractError, match="raw"):
        acme.run_pipeline(config, hash_contents=False, progress=False, write=True)
    assert not (raw_root / "acmetrace_inventory.json").exists()

    from scripts import audit_acmetrace_power

    with pytest.raises(SystemExit) as excinfo:
        audit_acmetrace_power.main(["--processed-dir", "data/raw/processed"])
    assert excinfo.value.code == 1
    assert not (PROJECT_ROOT / "data" / "raw" / "processed").exists()
