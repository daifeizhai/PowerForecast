"""Fast acceptance tests for the AcmeTrace ops / pairing / job-trace audit.

All tests use small synthetic CSVs in ``tmp_path`` (never the 75 GB raw tree), so they run
in a couple of seconds and cover:

* device-name parsing (``ip-gpu`` and pod-style names) and IPMI node->IP mapping;
* explicit timezone handling (Seren +08:00 vs Kalos +00:00; ``CST`` = +08:00 assumption);
* native interval detection at 15 s and 300 s, gap counting;
* the missing mask semantics of the streaming scanner (NaN stays missing, never zero),
  duplicate/out-of-order timestamps, out-of-range counts, placeholder files;
* IPMI long-format scanning (Chinese locale + ISO rows, zero ``Sys_Total_Power`` invalid);
* pairing only when the same device **and** overlapping timestamps exist, with
  cross-cluster joins rejected;
* job-trace field classification (pre-origin vs post-hoc) and plausibility checks;
* artifact writing to a bounded temporary directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import acmetrace_ops as ops


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seren_times(count: int, *, start: str = "2023-05-16T17:43:15+08:00", step_seconds: int = 15) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=count, freq=f"{step_seconds}s")


def _format_times(times: pd.DatetimeIndex, *, offset: str = "+08:00") -> list[str]:
    local = times.tz_convert("Asia/Shanghai") if offset == "+08:00" else times.tz_convert("UTC")
    suffix = "+08:00" if offset == "+08:00" else "+00:00"
    return [timestamp.strftime("%Y-%m-%d %H:%M:%S") + suffix for timestamp in local]


def _write_wide_csv(path, times: pd.DatetimeIndex, columns: list[str], values: np.ndarray, *, offset: str = "+08:00"):
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "Time", _format_times(times, offset=offset))
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# naming / timezone
# ---------------------------------------------------------------------------


def test_device_name_parsing() -> None:
    assert ops.parse_device("10.140.1.10-6") == ("10.140.1.10", 6)
    assert ops.parse_device("lingjun-pod9-0440-4") == ("lingjun-pod9-0440", 4)
    assert ops.parse_device("10.140.1.10") is None
    assert ops.parse_device("not-a-device") is None
    assert ops.parse_node_device("10.140.1.10") == "10.140.1.10"
    parsed = ops.parse_gpu_device_columns(["10.0.0.1-0", "10.0.0.1-1"])
    assert parsed == [("10.0.0.1", 0), ("10.0.0.1", 1)]
    assert ops.parse_gpu_device_columns(["10.0.0.1-0", "bad"] ) is None


def test_ipmi_node_name_to_ip() -> None:
    assert ops.ipmi_node_to_ip("SH-IDC1-10-140-0-112") == "10.140.0.112"
    assert ops.ipmi_node_to_ip("10.140.0.112") == "10.140.0.112"
    assert ops.ipmi_node_to_ip("unexpected") is None


def test_timezone_offsets_are_explicit_and_not_equated() -> None:
    seren = ops.parse_iso_timestamps(["2023-05-16 17:43:15+08:00"])[0]
    kalos = ops.parse_iso_timestamps(["2023-05-16 17:43:15+00:00"])[0]
    assert seren == pd.Timestamp("2023-05-16T09:43:15Z")
    assert kalos == pd.Timestamp("2023-05-16T17:43:15Z")
    assert seren != kalos
    assert (kalos - seren) == pd.Timedelta(8, unit="h")
    assert ops.observed_offset(["2023-05-16 17:43:15+08:00"]) == "+08:00"
    assert ops.observed_offset(["2023-05-16 17:43:15+00:00"]) == "+00:00"


def test_ipmi_chinese_locale_timestamps_are_cst_plus_08() -> None:
    parsed = ops.parse_ipmi_timestamps(
        ["2023年 05月 15日 星期一 14:15:01 CST", "2023-05-15 14:20:01+08:00"]
    )
    expected = pd.DatetimeIndex(
        [pd.Timestamp("2023-05-15T06:15:01Z"), pd.Timestamp("2023-05-15T06:20:01Z")]
    )
    assert list(parsed) == list(expected)
    assert str(parsed.tz) == "UTC"


# ---------------------------------------------------------------------------
# native interval evidence
# ---------------------------------------------------------------------------


def _timestamps_ns(start: str, count: int, step_seconds: int) -> np.ndarray:
    index = pd.date_range(start=start, periods=count, freq=f"{step_seconds}s", tz="UTC")
    return index.asi8.astype(np.int64)


def test_interval_detection_15_seconds() -> None:
    stats = ops.interval_statistics(
        _timestamps_ns("2023-05-16T00:00:00Z", 10, 15), expected_interval_seconds=15.0
    )
    assert stats["native_interval_seconds"] == 15.0
    assert stats["matches_expected_interval"] is True
    assert stats["gap_count"] == 0
    assert stats["all_deltas_multiples_of_native"] is True


def test_interval_detection_300_seconds_and_gap() -> None:
    base = _timestamps_ns("2023-05-16T00:00:00Z", 5, 300)
    with_gap = np.concatenate([base, [base[-1] + 900 * 10**9]])
    stats = ops.interval_statistics(with_gap, expected_interval_seconds=300.0)
    assert stats["native_interval_seconds"] == 300.0
    assert stats["matches_expected_interval"] is True
    assert stats["gap_count"] == 1
    assert stats["max_gap_seconds"] == 900.0


# ---------------------------------------------------------------------------
# util scanner semantics
# ---------------------------------------------------------------------------


def test_utilization_scanner_missing_and_quality_semantics(tmp_path) -> None:
    times = _seren_times(4)
    values = np.array(
        [
            [10.0, 20.0],
            [110.0, np.nan],
            [-5.0, 30.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    # duplicate + out-of-order timestamp is injected by rewriting the third row's time.
    path = _write_wide_csv(tmp_path / "GPU_UTIL.csv", times, ["10.0.0.1-0", "10.0.0.1-1"], values)
    frame = pd.read_csv(path)
    frame.loc[3, "Time"] = frame.loc[1, "Time"]
    frame.to_csv(path, index=False)

    record, payload = ops.scan_utilization_file(
        path, cluster="seren", metric="GPU_UTIL", chunk_rows=2, pairing=True
    )
    assert record["rows"] == 4
    assert record["device_columns"] == 2
    assert record["kind"] == "gpu_wide"
    assert record["placeholder"] is False
    assert record["duplicate_timestamps"] == 1
    assert record["out_of_order_rows"] == 1
    assert record["missing_fraction_per_device"] == [0.0, 0.25]
    assert record["missing_fraction_aggregate"] == pytest.approx(0.125)
    assert record["fully_missing_devices"] == 0
    assert record["out_of_range"]["negative_values"] == 1
    assert record["out_of_range"]["above_expected_upper"] == 1
    assert record["out_of_range"]["values_clipped"] is False
    assert record["native_interval"]["native_interval_seconds"] == 15.0
    assert record["value_min"] == -5.0
    assert record["value_max"] == 110.0
    assert payload["devices"] == ["10.0.0.1-0", "10.0.0.1-1"]
    # Missing values stay NaN in the source (no zero fill is observed or invented here).
    assert "missing_fraction_per_device" in record


def test_placeholder_file_is_reported_not_fabricated(tmp_path) -> None:
    path = tmp_path / "DRAM_ACTIVE.csv"
    path.write_text("Time\n", encoding="utf-8")
    record, payload = ops.scan_utilization_file(path, cluster="kalos", metric="DRAM_ACTIVE")
    assert record["rows"] == 0
    assert record["device_columns"] == 0
    assert record["placeholder"] is True
    assert payload["kind"] == "placeholder"
    assert "no continuous series fabricated" in record["notes"]


# ---------------------------------------------------------------------------
# IPMI scanner
# ---------------------------------------------------------------------------


def test_ipmi_scanner_counts_zero_power_as_invalid(tmp_path) -> None:
    path = tmp_path / "CPU_D_Power.csv"
    path.write_text(
        "Time,Sys_Total_Power,CPU_Power,node\n"
        "2023年 05月 15日 星期一 14:15:01 CST,500,300,SH-IDC1-10-140-0-112\n"
        "2023年 05月 15日 星期一 14:20:01 CST,0,0,SH-IDC1-10-140-0-112\n"
        "2023-05-15 14:20:01+08:00,300,200,SH-IDC1-10-140-0-113\n",
        encoding="utf-8",
    )
    record, payload = ops.scan_ipmi_file(path, chunk_rows=2, pairing=True)
    assert record["rows"] == 3
    assert record["node_count"] == 2
    assert record["node_ip_count"] == 2
    assert record["zero_sys_total_power_rows"] == 1
    assert record["native_interval"]["native_interval_seconds"] == 300.0
    assert record["timestamp_offset_observed"].startswith("+08:00")
    accumulator = payload["bin_accumulator"]
    assert accumulator is not None
    assert accumulator.devices == ["10.140.0.112", "10.140.0.113"]
    # zero Sys_Total_Power row is excluded from the pairing sum (500 W from node .112 only)
    column = accumulator.devices.index("10.140.0.112")
    node_112_total = sum(float(values[column]) for values in accumulator.sums.values())
    assert node_112_total == 500.0


def test_mixed_iso_and_chinese_timestamps_in_one_ipmi_file(tmp_path) -> None:
    # GPU_C/GPU_AB contain node blocks that switch from ISO rows to Chinese-locale rows
    # mid-file; parsing must not silently produce NaT (which used to break pairing).
    path = tmp_path / "GPU_C_Power.csv"
    path.write_text(
        "Time,Sys_Total_Power,node\n"
        "2023-05-15 14:15:01+08:00,400,SH-IDC1-10-140-1-153\n"
        "2023年 05月 15日 星期一 14:20:01 CST,410,SH-IDC1-10-140-1-153\n"
        "2023-05-15 14:25:01+08:00,420,SH-IDC1-10-140-1-153\n",
        encoding="utf-8",
    )
    record, _ = ops.scan_ipmi_file(path, chunk_rows=2, pairing=False)
    assert record["rows"] == 3
    assert record["nat_timestamps"] == 0
    assert record["native_interval"]["native_interval_seconds"] == 300.0
    assert record["node_count"] == 1


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------


def _payload_for(tmp_path, name: str, values: np.ndarray, *, offset: str = "+08:00"):
    times = _seren_times(values.shape[0])
    path = _write_wide_csv(tmp_path / f"{name}.csv", times, ["10.0.0.1-0", "10.0.0.1-1"], values, offset=offset)
    _, payload = ops.scan_utilization_file(
        path, cluster="seren", metric=name, chunk_rows=2, pairing=True
    )
    return payload


def test_pairing_requires_same_device_and_overlapping_timestamps(tmp_path) -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    base = _payload_for(tmp_path, "POWER_USAGE", values)
    other = _payload_for(tmp_path, "GPU_UTIL", values * 10)
    pairs = ops.pair_gpu_metrics("seren", {"POWER_USAGE": base, "GPU_UTIL": other})
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["common_devices"] == 2
    assert pair["matched_devices"] == 2
    assert pair["matched_device_slots"] > 0
    assert pair["match_rate"] == pytest.approx(1.0)
    assert pair["joint_span_first_utc"] is not None

    # Same device names but a completely disjoint time window -> no matched slots.
    shifted_times = _seren_times(values.shape[0], start="2023-06-16T17:43:15+08:00")
    shifted_path = _write_wide_csv(
        tmp_path / "GPU_TEMP_shift.csv", shifted_times, ["10.0.0.1-0", "10.0.0.1-1"], values
    )
    _, shifted = ops.scan_utilization_file(
        shifted_path, cluster="seren", metric="GPU_TEMP", chunk_rows=2, pairing=True
    )
    disjoint = ops.pair_gpu_metrics("seren", {"POWER_USAGE": base, "GPU_TEMP": shifted})[0]
    assert disjoint["matched_device_slots"] == 0
    assert disjoint["matched_devices"] == 0
    assert disjoint["match_rate"] is None

    # Different device names -> no common devices.
    other_path = _write_wide_csv(
        tmp_path / "FB_USED.csv", _seren_times(4), ["10.0.0.2-0", "10.0.0.2-1"], values
    )
    _, different = ops.scan_utilization_file(
        other_path, cluster="seren", metric="FB_USED", chunk_rows=2, pairing=True
    )
    no_common = ops.pair_gpu_metrics("seren", {"POWER_USAGE": base, "FB_USED": different})[0]
    assert no_common["common_devices"] == 0
    assert no_common["matched_devices"] == 0


def test_cross_cluster_pairing_is_rejected(tmp_path) -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    seren = _payload_for(tmp_path, "POWER_USAGE", values)
    kalos = dict(seren)
    kalos["cluster"] = "kalos"
    with pytest.raises(ValueError, match="cross-cluster"):
        ops.pair_gpu_metrics("seren", {"POWER_USAGE": seren, "GPU_TEMP": kalos})


def test_native_timestamp_intersection_helper() -> None:
    left = np.array([1, 2, 3], dtype=np.int64)
    right = np.array([2, 3, 4], dtype=np.int64)
    result = ops._pair_native_timestamp_sets(
        left, right, cluster="seren", metric_a="POWER_USAGE", metric_b="GPU_UTIL"
    )
    assert result["exact_common_instants"] == 2


# ---------------------------------------------------------------------------
# job trace audit
# ---------------------------------------------------------------------------

SEREN_JOB_HEADER = (
    "job_id,user,node_num,gpu_num,cpu_num,type,state,submit_time,start_time,end_time,duration,queue,gpu_time"
)


def _write_job_trace(path, rows: list[list], *, kalos: bool = False):
    if kalos:
        header = (
            "job_id,user,node_num,gpu_num,cpu_num,mem_per_pod_GB,shared_mem_per_pod,type,state,"
            "submit_time,start_time,end_time,fail_time,stop_time,duration,queue,gpu_time"
        )
        lines = [header]
        for row in rows:
            lines.append(",".join(str(item) for item in row))
    else:
        lines = [SEREN_JOB_HEADER]
        for row in rows:
            lines.append(",".join(str(item) for item in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_job_field_classification_and_plausibility(tmp_path) -> None:
    path = _write_job_trace(
        tmp_path / "trace_seren.csv",
        [
            [1, "u1", 1, 8, 128, "Other", "COMPLETED", "2023-03-01 00:00:00+08:00", "2023-03-01 00:01:00+08:00", "2023-03-01 00:02:00+08:00", 60, 60, 480.0],
            [2, "u1", 1, 0, 8, "CPU", "FAILED", "2023-03-01 00:03:00+08:00", "2023-03-01 00:02:00+08:00", "2023-03-01 00:01:00+08:00", 0, -60, 0.0],
        ],
    )
    audit, frame = ops.audit_job_trace(path, cluster="seren")
    assert audit["rows"] == 2
    assert audit["no_node_identifiers"] is True
    assert "job_id" in audit["pre_origin_fields"]
    assert "submit_time" in audit["pre_origin_fields"]
    for field in ("state", "start_time", "end_time", "duration", "queue", "gpu_time"):
        assert field in audit["post_hoc_fields"], field
    assert audit["plausibility"]["duration_negative_or_zero"] == 1
    assert audit["plausibility"]["end_before_start_rows"] == 1
    assert audit["plausibility"]["negative_queue_rows"] == 1
    assert audit["plausibility"]["gpu_jobs"] == 1
    assert audit["state_distribution"]["COMPLETED"] == 1
    assert "no node IDs" in audit["limitation"]


def test_job_activity_excludes_open_and_invalid_jobs() -> None:
    frame = pd.DataFrame(
        {
            "submit_time": [
                "2023-03-01 00:00:00+08:00",
                "2023-03-01 01:00:00+08:00",
                "2023-03-01 02:00:00+08:00",
            ],
            "start_time": [
                "2023-03-01 00:00:00+08:00",
                "2023-03-01 01:00:00+08:00",
                "2023-03-01 02:00:00+08:00",
            ],
            "end_time": [
                "2023-03-01 00:30:00+08:00",
                np.nan,
                "2023-03-01 01:00:00+08:00",
            ],
            "gpu_num": [8, 8, 8],
        }
    )
    bin_start = pd.Timestamp("2023-03-01 00:00:00+08:00").tz_convert("UTC").value // (900 * 10**9)
    result = ops.job_bin_activity(frame, bin_series=np.array([bin_start], dtype=np.int64))
    assert result["valid_jobs"] == 1
    assert result["open_or_invalid_jobs_excluded"] == 2
    assert result["active_gpu"][0] == pytest.approx(8.0)


def test_kalos_job_extra_fields_are_post_hoc(tmp_path) -> None:
    path = _write_job_trace(
        tmp_path / "trace_kalos.csv",
        [
            ["dlc1", "u1", 8, 64, 960, 1000, 100.0, "Other", "FAILED", "2023-05-17 11:00:58+00:00", "2023-05-17 11:01:08+00:00", "2023-05-17 11:01:16+00:00", "2023-05-17 11:01:16+00:00", "", 18, 10.0, 1152.0],
        ],
        kalos=True,
    )
    audit, _ = ops.audit_job_trace(path, cluster="kalos")
    assert audit["raw_timestamp_offsets"]["submit_time"] == "+00:00"
    assert "mem_per_pod_GB" in audit["pre_origin_fields"]
    assert "fail_time" in audit["post_hoc_fields"]
    assert "stop_time" in audit["post_hoc_fields"]
    assert audit["plausibility"]["duration_negative_or_zero"] == 0


def test_timezone_impact_is_quantified() -> None:
    job_audit = {"raw_timestamp_offsets": {"submit_time": "+00:00"}}
    impact = ops.timezone_impact(
        cluster="kalos", job_audit=job_audit, utilization_offsets={"GPU_UTIL": "+08:00"}
    )
    assert impact["shift_hours_if_naively_compared"] == 8.0
    assert "8.0 hours" in impact["impact_if_ignored"]
    same = ops.timezone_impact(
        cluster="seren", job_audit={"raw_timestamp_offsets": {"submit_time": "+08:00"}},
        utilization_offsets={"GPU_UTIL": "+08:00"},
    )
    assert same["shift_hours_if_naively_compared"] == 0.0


# ---------------------------------------------------------------------------
# artifact writing / raw protection
# ---------------------------------------------------------------------------


def test_write_audit_artifacts_to_bounded_directory(tmp_path) -> None:
    ops_payload = {"clusters": {"seren": {"files": [], "summary": {}, "device_sets": {"groups": {}}, "devices_per_node": {"files": {}}}}}
    pairing_payload = {"clusters": {"seren": {"gpu_pairs": [], "node_pairs": [], "ipmi_files": [], "device_sets": {"groups": {}}, "devices_per_node": {"files": {}}}}}
    job_payload = {"clusters": {}}
    paths = ops.write_audit_artifacts(
        ops_payload,
        pairing_payload,
        job_payload,
        out_dir=tmp_path / "processed",
        analysis_dir=tmp_path / "analysis",
    )
    import os

    for path in paths.values():
        assert os.path.exists(path)
    assert (tmp_path / "processed" / "acmetrace_ops_audit.json").is_file()
    assert (tmp_path / "analysis" / "runtime_file_summary.csv").is_file()


def test_scanners_do_not_modify_raw_files() -> None:
    raw_file = ops.IPMI_ROOT + "/CPU_D_Power.csv"
    from src.utils import resolve_path

    path = resolve_path(raw_file)
    if not path.is_file():
        pytest.skip(
            "AcmeTrace raw snapshot not available at the configured raw root; the read-only "
            "behaviour is covered by the fixture-based scanner tests instead."
        )
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    # Bounded read of the first chunk only; the scanner is read-only by construction.
    ops.scan_ipmi_file(path, chunk_rows=1000, pairing=False, max_chunks=1)
    after = (path.stat().st_size, path.stat().st_mtime_ns)
    assert before == after


def test_linkage_supplement_has_raw_write_guard() -> None:
    from scripts.analyze_acmetrace_linkage import DEFAULT_OUT_DIR, assert_output_dir

    with pytest.raises(ValueError, match="raw"):
        assert_output_dir("data/raw/acmetrace_linkage_out")
    assert DEFAULT_OUT_DIR == "results/analysis/acmetrace"
