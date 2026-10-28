"""Supplementary read-only AcmeTrace power<->runtime evidence for the v2 data layer.

The restricted runtime-metric audit (``python -m scripts.audit_acmetrace_ops`` with a
``--metrics`` whitelist) reports runtime-metric quality and IPMI<->node *coverage* pairing,
but two research-relevant quantities are outside its payload:

* a Pearson correlation between IPMI ``Sys_Total_Power`` and the four Seren ``NODE_*``
  metrics (the audit module intentionally only reports matched device-slots / match rate);
* the cluster-level job-activity vs GPU-utilisation linkage, because ``GPU_UTIL`` is not in
  the restricted metric whitelist (the official ``acmetrace_job_audit.json`` therefore has
  ``cluster_linkage = null``).

This script fills exactly those two gaps by re-using the audit module's OWN streaming scan
and accumulator functions (``src/data/acmetrace_ops.py`` is imported read-only, never
modified).  Nothing below ``E:/Data/AcmeTrace`` is written; only two CSVs plus one JSON are
written under ``results/analysis/acmetrace/``.

Method (identical binning to the audit module):

* every native sample is binned into 900 s bins and reduced to a per-device bin mean
  (sum / observed count); ``Sys_Total_Power == 0`` rows are excluded upstream by the
  module (invalid/offline, never a real 0 W sample);
* a (device, bin) slot is *matched* when both sides observed at least one native sample in
  that bin; Pearson is computed per device over its matched bins only;
* a device enters the correlation summary only when it has >= 96 matched bins (>= 24 h of
  paired 15-minute slots);
* the linkage replicates ``cluster_temporal_linkage``: per-bin mean GPU_UTIL over reporting
  GPUs vs number of active GPU jobs from the Seren job trace (started, not yet ended),
  Pearson over overlapping bins.

Run from the repository root (CPU only, read-only raw tree, ~8 GB streamed)::

    & .\\.venv\\Scripts\\python.exe -m scripts.analyze_acmetrace_linkage

The output is descriptive evidence for ``docs/ACMETRACE_OPS_AUDIT_V2.md``; it is NOT part
of the restricted audit payload and NOT a predictive-value claim.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):  # allow `python scripts/analyze_acmetrace_linkage.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import acmetrace_ops as ops  # noqa: E402
from src.utils import ensure_dir, resolve_path  # noqa: E402

METRICS = (
    "NODE_CPU_UTILIZATION",
    "NODE_MEMORY_UTILIZATION",
    "NODE_IB_RECEIVE",
    "NODE_IB_SEND",
)
MIN_MATCHED_BINS = 96  # >= 24 h of 15-minute slots
DEFAULT_OUT_DIR = "results/analysis/acmetrace"


def _metric_path(metric: str) -> Path:
    matches = [path for path in ops.utilization_files("seren") if path.stem.upper() == metric]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one source file for {metric}, got {matches}")
    return matches[0]


def _device_means(accumulator: ops.BinAccumulator) -> dict[str, Any]:
    """Dense per-device 15-minute means plus the observed mask."""
    dense = accumulator.dense()
    sums = np.asarray(dense["sums"], dtype=np.float64)
    counts = np.asarray(dense["counts"], dtype=np.float64)
    means = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    return {"bins": np.asarray(dense["bins"], dtype=np.int64), "means": means, "counts": counts}


def _align_pair(
    left: dict[str, Any],
    left_devices: list[str],
    right: dict[str, Any],
    right_devices: list[str],
) -> dict[str, Any]:
    """Align two dense accumulators on common bins and common devices."""
    common_bins = np.intersect1d(left["bins"], right["bins"])
    left_rows = np.searchsorted(left["bins"], common_bins)
    right_rows = np.searchsorted(right["bins"], common_bins)
    common_devices = sorted(set(left_devices) & set(right_devices))
    left_index = {name: index for index, name in enumerate(left_devices)}
    right_index = {name: index for index, name in enumerate(right_devices)}
    left_means = left["means"][np.ix_(left_rows, [left_index[name] for name in common_devices])]
    right_means = right["means"][np.ix_(right_rows, [right_index[name] for name in common_devices])]
    return {
        "bins": common_bins,
        "devices": common_devices,
        "left_means": left_means,
        "right_means": right_means,
    }


def _correlation_summary(
    aligned: dict[str, Any], metric: str, min_matched_bins: int = MIN_MATCHED_BINS
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left = aligned["left_means"]
    right = aligned["right_means"]
    both = np.isfinite(left) & np.isfinite(right)
    either = np.isfinite(left) | np.isfinite(right)
    per_device_rows: list[dict[str, Any]] = []
    correlations: list[float] = []
    for column, device in enumerate(aligned["devices"]):
        mask = both[:, column]
        matched = int(mask.sum())
        union = int(either[:, column].sum())
        correlation = None
        if matched >= min_matched_bins:
            x = left[mask, column]
            y = right[mask, column]
            if np.std(x) > 0 and np.std(y) > 0:
                correlation = float(np.corrcoef(x, y)[0, 1])
        if correlation is not None:
            correlations.append(correlation)
        per_device_rows.append(
            {
                "metric": metric,
                "device": device,
                "matched_bins": matched,
                "union_bins": union,
                "pearson": correlation,
            }
        )
    values = np.asarray(correlations, dtype=np.float64)
    summary: dict[str, Any] = {
        "metric": metric,
        "matched_devices": int(np.count_nonzero(both.any(axis=0))),
        "matched_device_slots": int(both.sum()),
        "union_device_slots": int(either.sum()),
        "match_rate": float(both.sum() / either.sum()) if either.sum() else None,
        "common_bins": int(len(aligned["bins"])),
        "joint_span_first_utc": (
            str(np.datetime64(int(aligned["bins"][0]) * ops.BIN_SECONDS, "s")) + "+00:00"
            if len(aligned["bins"])
            else None
        ),
        "joint_span_last_utc": (
            str(np.datetime64(int(aligned["bins"][-1]) * ops.BIN_SECONDS, "s")) + "+00:00"
            if len(aligned["bins"])
            else None
        ),
        "nodes_with_pearson": int(values.size),
        "min_matched_bins_per_node": int(min_matched_bins),
        "pearson_median": float(np.median(values)) if values.size else None,
        "pearson_mean": float(np.mean(values)) if values.size else None,
        "pearson_p10": float(np.quantile(values, 0.10)) if values.size else None,
        "pearson_p90": float(np.quantile(values, 0.90)) if values.size else None,
        "pearson_min": float(values.min()) if values.size else None,
        "pearson_max": float(values.max()) if values.size else None,
        "nodes_positive": int(np.count_nonzero(values > 0)) if values.size else 0,
        "nodes_ge_0_5": int(np.count_nonzero(values >= 0.5)) if values.size else 0,
        "note": (
            "per-device Pearson over matched 15-minute bins (IPMI Sys_Total_Power bin mean vs "
            "NODE_* bin mean); devices need >= 96 matched bins (>=24 h); 0 W IPMI rows excluded "
            "upstream by the audit module"
        ),
    }
    return summary, per_device_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Supplementary read-only evidence: IPMI Sys_Total_Power <-> NODE_* per-device "
            "15-minute Pearson and cluster-level active-GPU-jobs vs GPU_UTIL linkage."
        )
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="override the AcmeTrace raw root (default: $ACMETRACE_RAW_ROOT or E:/Data/AcmeTrace)",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"output directory for the supplementary JSON/CSV (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)
    if args.raw_root:
        import os

        root = str(Path(args.raw_root).resolve())
        os.environ["ACMETRACE_RAW_ROOT"] = root
        ops.RAW_ROOT = root
        ops.UTIL_ROOT = f"{root}/data/utilization"
        ops.IPMI_ROOT = f"{ops.UTIL_ROOT}/ipmi"
        ops.JOB_ROOT = f"{root}/data/job_trace"
    out_dir = ensure_dir(assert_output_dir(args.out_dir))
    if not ops.utilization_files("seren"):
        raise RuntimeError("no Seren utilization files found; check E:/Data/AcmeTrace")

    generated_at = datetime.now(timezone.utc).isoformat()
    print("[1/3] scanning IPMI server power (read-only, streamed) ...", flush=True)
    ipmi_accumulators: list[ops.BinAccumulator] = []
    ipmi_inputs: list[dict[str, Any]] = []
    for path in ops.ipmi_files():
        record, payload = ops.scan_ipmi_file(path, chunk_rows=ops.NARROW_CHUNK_ROWS, pairing=True)
        if payload.get("bin_accumulator") is not None:
            ipmi_accumulators.append(payload["bin_accumulator"])
        ipmi_inputs.append({"path": str(path), "bytes": record["bytes"], "rows": record["rows"]})
        print(f"  {path.name}: rows={record['rows']} nodes={record['node_count']}", flush=True)
    ipmi_accumulator = ops.merge_bin_accumulators(ipmi_accumulators)
    ipmi_dense = _device_means(ipmi_accumulator)

    metric_summaries: list[dict[str, Any]] = []
    per_device_rows: list[dict[str, Any]] = []
    metric_inputs: list[dict[str, Any]] = []
    for metric in METRICS:
        path = _metric_path(metric)
        print(f"[2/3] scanning {metric} (read-only, streamed) ...", flush=True)
        record, payload = ops.scan_utilization_file(
            path,
            cluster="seren",
            metric=metric,
            chunk_rows=ops.FULL_CHUNK_ROWS,
            pairing=True,
        )
        accumulator = payload["accumulator"]
        metric_inputs.append({"metric": metric, "path": str(path), "bytes": record["bytes"], "rows": record["rows"]})
        dense = _device_means(accumulator)
        aligned = _align_pair(dense, accumulator.devices, ipmi_dense, ipmi_accumulator.devices)
        summary, rows = _correlation_summary(aligned, metric)
        metric_summaries.append(summary)
        per_device_rows.extend(rows)
        print(
            f"  {metric}: matched_slots={summary['matched_device_slots']} "
            f"match_rate={summary['match_rate']:.6f} "
            f"median_r={summary['pearson_median']} nodes={summary['nodes_with_pearson']}",
            flush=True,
        )

    print("[3/3] recomputing cluster-level job linkage (GPU_UTIL not in restricted whitelist) ...", flush=True)
    gpu_path = _metric_path("GPU_UTIL")
    gpu_record, gpu_payload = ops.scan_utilization_file(
        gpu_path,
        cluster="seren",
        metric="GPU_UTIL",
        chunk_rows=ops.FULL_CHUNK_ROWS,
        pairing=True,
    )
    job_audit, job_frame = ops.audit_job_trace(ops.job_trace_file("seren"), cluster="seren")
    linkage = ops.cluster_temporal_linkage(
        cluster="seren",
        gpu_util_accumulator=gpu_payload["accumulator"],
        job_frame=job_frame,
        job_audit=job_audit,
    )
    print(
        f"  linkage: bins={linkage.get('n_bins')} overlap={linkage.get('overlap_bins')} "
        f"pearson={linkage.get('pearson_active_gpu_vs_mean_util')}",
        flush=True,
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "purpose": (
            "supplementary evidence for docs/ACMETRACE_OPS_AUDIT_V2.md: (a) per-device Pearson "
            "between IPMI Sys_Total_Power and each NODE_* metric on 15-minute bins; (b) cluster-level "
            "job-activity vs GPU_UTIL linkage, because GPU_UTIL is outside the restricted audit "
            "whitelist and the official acmetrace_job_audit.json therefore has cluster_linkage=null."
        ),
        "script": "scripts/analyze_acmetrace_linkage.py",
        "module": "src/data/acmetrace_ops.py (imported read-only; unmodified)",
        "raw_root": str(Path(ops.RAW_ROOT)),
        "raw_writes": "none (all raw CSV files opened by the audit module in streaming read mode)",
        "bin_seconds": ops.BIN_SECONDS,
        "min_matched_bins_per_node": MIN_MATCHED_BINS,
        "ipmi_inputs": ipmi_inputs,
        "metric_inputs": metric_inputs,
        "ipmi_devices_total": len(ipmi_accumulator.devices),
        "node_power_pearson": metric_summaries,
        "linkage": {
            "metric": "GPU_UTIL",
            "gpu_util_path": str(gpu_path),
            "gpu_util_bytes": gpu_record["bytes"],
            "gpu_util_rows": gpu_record["rows"],
            "job_trace_rows": job_audit["rows"],
            "result": linkage,
        },
        "caveats": [
            "Pearson is a descriptive association over matched 15-minute bins; it is not a causal test.",
            "IPMI Sys_Total_Power is a per-server wall power (W inferred, unit_verified=false); "
            "node metrics are per-server utilisation/util rates (units unverified for NODE_IB_*).",
            "IPMI 300 s vs node-metric 15 s native sampling makes same-instant pairing impossible; "
            "only 15-minute-bin pairing is reported.",
            "The linkage is cluster-level only: the Seren job trace has no node IDs.",
            "These supplementary computations are NOT part of the restricted audit payload; "
            "they re-use the audit module's own scan/accumulator functions.",
        ],
    }
    (out_dir / "v2_ops_supplement.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "v2_node_power_pearson.csv", metric_summaries)
    _write_csv(out_dir / "v2_node_power_pearson_by_node.csv", per_device_rows)
    print(
        f"written to {out_dir}: v2_ops_supplement.json, v2_node_power_pearson.csv, "
        "v2_node_power_pearson_by_node.csv"
    )
    return 0


def assert_output_dir(path: str | Path) -> Path:
    """Refuse to write supplementary evidence into the raw tree."""
    resolved = resolve_path(path)
    lowered = [part.lower() for part in resolved.parts]
    if any(left == "data" and right == "raw" for left, right in zip(lowered, lowered[1:])):
        raise ValueError(f"refusing to write supplementary evidence under a raw-data directory: {resolved}")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
