"""AcmeTrace Seren target-domain assembly for the power-transfer-v2 research line.

This module turns the complete AcmeTrace snapshot (``E:/Data/AcmeTrace``) into a
15-minute target-domain dataset:

* **label** -- Seren ``cluster_it`` power: the sum of MEASURED IPMI server power over the
  nodes that report in each native 300 s slot, adapted to 15-minute interval means.  This
  is an IT-side cluster aggregate and is **never** facility power.
* **features** -- cluster-level runtime metrics (node CPU / memory utilisation, IB receive
  / send, GPU utilisation) and job-trace activity (active jobs, active GPUs, submissions),
  aligned to the label's 15-minute grid and mask-first.

Hard honesty rules (enforced by :func:`validate_target_config`):

* no facility-kW claim: the label level must stay ``cluster_it`` and the scope must not be
  ``facility``;
* power units are inferred as W from magnitude evidence only, so ``unit_verified`` must be
  false and the kW values keep that uncertainty;
* missing observations stay NaN with an explicit ``observed_mask``; nothing is
  interpolated or zero-filled;
* runtime features are means / sums over REPORTING devices, not complete-cluster totals;
  their coverage is reported for every bin;
* the job trace carries no node IDs, so job activity is a cluster-level feature only.

Raw data is read-only: every writer goes through
:func:`src.data.loaders.assert_not_raw_path`.

The companion CLI is ``scripts/prepare_acmetrace_target.py``; the declarative settings live
in ``configs/datasets/acmetrace_target.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.data.contracts import ContractError
from src.data.loaders import assert_not_raw_path
from src.utils import (
    config_digest,
    dependency_versions,
    ensure_dir,
    git_branch,
    git_rev,
    resolve_path,
    sha256_file,
    utc_timestamp,
    write_json,
)

CONFIG_PATH = "configs/datasets/acmetrace_target.yaml"
BIN_SECONDS = 900
BINS_PER_DAY = 96
FEATURE_ROLE = "auxiliary_target_runtime"

#: Column order of the target power parquet (kept close to the Stage-1 target contract).
POWER_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "site_id",
    "power_kw",
    "power_valid",
    "coverage_ratio",
)
#: Column order of the target feature long table (Stage-1 style).
FEATURE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "domain_id",
    "feature_name",
    "feature_value",
    "unit",
    "observed_mask",
    "coverage_ratio",
)

Progress = Callable[[str], None]


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_target_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the target-assembly YAML and apply the ``ACMETRACE_RAW_ROOT`` override."""
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"AcmeTrace target config not found: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ContractError(f"{resolved} must contain a YAML mapping.")
    config = dict(payload)
    raw_override = os.environ.get("ACMETRACE_RAW_ROOT")
    if raw_override:
        source = dict(config.get("source") or {})
        source["raw_root"] = raw_override
        config["source"] = source
    return validate_target_config(config, path=str(resolved))


def _require_keys(payload: Mapping[str, Any], keys: Sequence[str], *, where: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ContractError(f"AcmeTrace target config {where} is missing keys: {missing}.")


def validate_target_config(config: Mapping[str, Any], *, path: str = "<mapping>") -> dict[str, Any]:
    """Validate the declarative contract and refuse dishonest label claims."""
    _require_keys(
        config,
        ("schema_version", "dataset", "source", "features", "aggregation_15min", "outputs"),
        where=path,
    )
    dataset = config["dataset"]
    source = config["source"]
    features = config["features"]
    aggregation = config["aggregation_15min"]
    outputs = config["outputs"]
    for name, block, keys in (
        (
            "dataset",
            dataset,
            ("dataset_id", "site_id", "domain_id", "display_name", "frequency", "points_per_day"),
        ),
        ("source", source, ("raw_root", "cluster", "power_parquet", "label")),
        (
            "outputs",
            outputs,
            (
                "processed_dir",
                "power_parquet",
                "features_parquet",
                "metadata_json",
                "quality_json",
                "results_dir",
            ),
        ),
    ):
        if not isinstance(block, Mapping):
            raise ContractError(f"AcmeTrace target config {name} must be a mapping.")
        _require_keys(block, keys, where=f"{path}:{name}")
    if dataset["frequency"] != "15min" or int(dataset["points_per_day"]) != BINS_PER_DAY:
        raise ContractError(
            "AcmeTrace target dataset must stay on the 15-minute grid (frequency=15min, "
            "points_per_day=96)."
        )
    label = source["label"]
    if not isinstance(label, Mapping):
        raise ContractError("AcmeTrace target source.label must be a mapping.")
    _require_keys(
        label,
        (
            "cluster_id",
            "level",
            "device_id",
            "label_type",
            "target_kind",
            "target_scope",
            "target_unit",
            "label_origin",
            "unit_inferred",
            "unit_verified",
            "measurement_boundary",
            "sampling_semantics",
        ),
        where=f"{path}:source.label",
    )
    if str(label["target_scope"]).strip().lower() == "facility":
        raise ContractError(
            "AcmeTrace has no facility power meter: target_scope='facility' is a dishonest "
            "claim. Keep the label at cluster IT scope."
        )
    if str(label["level"]) != "cluster_it":
        raise ContractError(
            "AcmeTrace target label level must be 'cluster_it' (the only defensible power "
            f"aggregate); got {label['level']!r}."
        )
    if str(label["target_kind"]) != "active_power" or str(label["target_unit"]) != "kW":
        raise ContractError("AcmeTrace target label must be active_power in kW.")
    if str(label["unit_inferred"]).upper() != "W":
        raise ContractError("AcmeTrace power units are inferred as W; unit_inferred must be 'W'.")
    if bool(label["unit_verified"]):
        raise ContractError(
            "AcmeTrace power units are NOT documented by the source; unit_verified must stay false."
        )
    for key in ("measurement_boundary", "sampling_semantics"):
        if not str(label[key]).strip():
            raise ContractError(f"source.label.{key} must state the metering boundary explicitly.")

    _require_keys(aggregation, ("step_seconds", "expected_samples_15s", "validity_threshold"), where=f"{path}:aggregation_15min")
    if int(aggregation["step_seconds"]) != BIN_SECONDS:
        raise ContractError(
            f"AcmeTrace target bins must stay at {BIN_SECONDS} s; got {aggregation['step_seconds']}."
        )
    expected = int(aggregation["expected_samples_15s"])
    if expected < 1:
        raise ContractError("aggregation_15min.expected_samples_15s must be >= 1.")
    threshold = float(aggregation["validity_threshold"])
    if not 0.0 < threshold <= 1.0:
        raise ContractError("aggregation_15min.validity_threshold must be in (0, 1].")

    metric_specs: list[Mapping[str, Any]] = []
    node_metrics = features.get("node_metrics") or []
    gpu_metrics = features.get("gpu_metrics") or []
    if not isinstance(node_metrics, list) or not node_metrics:
        raise ContractError("features.node_metrics must be a non-empty list.")
    for spec in list(node_metrics) + list(gpu_metrics):
        if not isinstance(spec, Mapping):
            raise ContractError("every metric feature spec must be a mapping.")
        _require_keys(spec, ("metric", "path", "feature_name", "unit", "semantic"), where=f"{path}:features")
        metric_specs.append(spec)
    job_trace = features.get("job_trace")
    if not isinstance(job_trace, Mapping):
        raise ContractError("features.job_trace must be a mapping.")
    _require_keys(job_trace, ("path", "features"), where=f"{path}:features.job_trace")
    job_specs = job_trace["features"]
    if not isinstance(job_specs, list) or not job_specs:
        raise ContractError("features.job_trace.features must be a non-empty list.")
    for spec in job_specs:
        if not isinstance(spec, Mapping):
            raise ContractError("every job-trace feature spec must be a mapping.")
        _require_keys(spec, ("feature_name", "unit", "semantic"), where=f"{path}:features.job_trace")

    names = [str(spec["feature_name"]) for spec in metric_specs + list(job_specs)]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ContractError(f"duplicate feature_name entries in AcmeTrace target config: {duplicates}.")

    for key in ("power_parquet", "features_parquet", "metadata_json", "quality_json"):
        assert_not_raw_path(str(outputs[key]))
    if len({str(outputs[key]) for key in ("power_parquet", "features_parquet", "metadata_json", "quality_json")}) != 4:
        raise ContractError("AcmeTrace target output paths must be distinct.")
    return dict(config)


def _raw_root(config: Mapping[str, Any]) -> Path:
    return resolve_path(str(config["source"]["raw_root"]))


def _raw_file(config: Mapping[str, Any], relative: str) -> Path:
    candidate = Path(str(relative))
    if candidate.is_absolute():
        return resolve_path(candidate)
    return _raw_root(config) / candidate


# ---------------------------------------------------------------------------
# timestamp / bin helpers
# ---------------------------------------------------------------------------


def parse_monitor_timestamps(values: Sequence[Any]) -> pd.DatetimeIndex:
    """Parse explicit-offset monitor timestamps into a UTC-naive ``DatetimeIndex``.

    The AcmeTrace monitor files write explicit ``+08:00`` offsets (the IPMI ``CPU_D`` file
    mixes an equivalent Chinese-locale CST spelling; it is not used by this module because
    the IPMI power label is produced by ``acmetrace_power``).  Naive values are excluded
    (``NaT``) rather than guessed.
    """
    series = pd.Series(list(values), dtype="string")
    parsed = pd.to_datetime(series, utc=True, format="mixed", errors="coerce")
    index = pd.DatetimeIndex(parsed)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index


def bin_index(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Absolute 15-minute bin ids (``floor(ns / 900e9)``); ``NaT`` maps to ``INT64_MIN``."""
    return np.asarray(timestamps.asi8, dtype=np.int64) // (BIN_SECONDS * 1_000_000_000)


def bin_timestamps(bins: Sequence[int]) -> pd.DatetimeIndex:
    """Inverse of :func:`bin_index`: absolute bins -> UTC-naive interval-start timestamps."""
    values = np.asarray(bins, dtype=np.int64) * BIN_SECONDS
    return pd.DatetimeIndex(pd.to_datetime(values, unit="s"))


# ---------------------------------------------------------------------------
# wide-metric streaming scan
# ---------------------------------------------------------------------------


def _accumulate_bins(
    store: dict[int, list[np.ndarray]],
    bins: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    for position, bin_id in enumerate(bins.tolist()):
        entry = store.get(bin_id)
        if entry is None:
            store[bin_id] = [sums[position].copy(), counts[position].copy()]
        else:
            entry[0] += sums[position]
            entry[1] += counts[position]


def scan_wide_metric(
    path: str | Path,
    *,
    metric: str,
    feature_name: str,
    unit: str,
    semantic: str,
    config: Mapping[str, Any],
    max_chunks: int | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Stream one wide monitor CSV and aggregate it to 15-minute cluster-level series.

    Returns a payload with ``bins`` / ``values`` / ``coverage`` / ``audit``.  Values are the
    mean over all finite native samples of all REPORTING devices; coverage is the mean over
    expected devices of ``min(samples, expected) / expected``.  A bin with no samples keeps
    ``NaN`` (never zero).  Duplicate timestamps are counted, not silently removed, and
    coverage is capped at 1.0.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"AcmeTrace metric file not found: {resolved}")
    aggregation = config["aggregation_15min"]
    expected_samples = int(aggregation["expected_samples_15s"])
    chunk_rows = int(aggregation.get("wide_chunk_rows", 10_000))
    threshold = float(aggregation["validity_threshold"])

    header = [str(column) for column in pd.read_csv(resolved, nrows=0).columns]
    if not header or header[0] != "Time":
        raise ContractError(f"{resolved}: expected a leading 'Time' column, got {header[:3]!r}.")
    devices = header[1:]
    if not devices:
        raise ContractError(
            f"{resolved} is a placeholder/empty monitor file (no device columns); the AcmeTrace "
            "target features require real per-device series."
        )
    n_devices = len(devices)

    store: dict[int, list[np.ndarray]] = {}
    rows = 0
    nat_timestamps = 0
    duplicate_timestamps = 0
    out_of_order_rows = 0
    seen_timestamps: set[int] = set()
    delta_counts: dict[int, int] = {}
    previous_last_ns: int | None = None
    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    raw_min = np.inf
    raw_max = -np.inf

    reader = pd.read_csv(resolved, chunksize=chunk_rows)
    for chunk_index, chunk in enumerate(reader):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        if "Time" not in chunk.columns:
            raise ContractError(f"{resolved} lost its 'Time' column while streaming.")
        timestamps = parse_monitor_timestamps(chunk["Time"].tolist())
        values = chunk.iloc[:, 1:].to_numpy(dtype=np.float64, copy=True)
        if values.shape[1] != n_devices:
            raise ContractError(
                f"{resolved}: expected {n_devices} device columns, got {values.shape[1]}."
            )
        rows += int(len(chunk))
        good = np.asarray(timestamps.notna())
        nat_timestamps += int(np.count_nonzero(~good))
        if good.any():
            if first_ts is None:
                first_ts = timestamps[good].min()
                last_ts = timestamps[good].max()
            else:
                first_ts = min(first_ts, timestamps[good].min())
                last_ts = max(last_ts, timestamps[good].max())
            ns = timestamps.asi8[good]
            if previous_last_ns is not None:
                boundary = int((ns[0] - previous_last_ns) // 1_000_000_000)
                delta_counts[boundary] = delta_counts.get(boundary, 0) + 1
            deltas = np.diff(ns) // 1_000_000_000
            for value, count in zip(*np.unique(deltas, return_counts=True)):
                delta_counts[int(value)] = delta_counts.get(int(value), 0) + int(count)
            out_of_order_rows += int(np.count_nonzero(deltas < 0))
            previous_last_ns = int(ns[-1])
            for value in ns.tolist():
                if value in seen_timestamps:
                    duplicate_timestamps += 1
                else:
                    seen_timestamps.add(value)
            finite_values = values[good][np.isfinite(values[good])]
            if finite_values.size:
                raw_min = min(raw_min, float(finite_values.min()))
                raw_max = max(raw_max, float(finite_values.max()))
            bins = ns // (BIN_SECONDS * 1_000_000_000)
            chunk_values = values[good]
            finite = np.isfinite(chunk_values)
            zeroed = np.where(finite, chunk_values, 0.0)
            order = np.argsort(bins, kind="stable")
            bin_sorted = bins[order]
            zeroed_sorted = zeroed[order]
            finite_sorted = finite[order]
            unique_bins, first_index = np.unique(bin_sorted, return_index=True)
            sums = np.add.reduceat(zeroed_sorted, first_index, axis=0)
            counts = np.add.reduceat(finite_sorted.astype(np.int64), first_index, axis=0)
            _accumulate_bins(store, unique_bins, sums, counts)
        if progress is not None:
            progress(
                f"{metric}: chunk {chunk_index + 1} rows={rows:,} "
                f"visible_bins={len(store):,}"
            )

    if not store:
        raise ContractError(f"{resolved}: no observable 15-minute bins were produced.")
    ordered = np.asarray(sorted(store), dtype=np.int64)
    sums_matrix = np.stack([store[int(bin_id)][0] for bin_id in ordered]).astype(np.float64)
    counts_matrix = np.stack([store[int(bin_id)][1] for bin_id in ordered]).astype(np.int64)
    del store
    total_counts = counts_matrix.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        values_mean = np.where(total_counts > 0, sums_matrix.sum(axis=1) / np.maximum(total_counts, 1), np.nan)
        per_device_coverage = np.minimum(counts_matrix, expected_samples) / float(expected_samples)
        coverage = per_device_coverage.mean(axis=1)

    native_interval = None
    positive_deltas = {int(key): int(value) for key, value in delta_counts.items() if key > 0}
    if positive_deltas:
        native_interval = float(max(positive_deltas, key=lambda item: positive_deltas[item]))
    audit = {
        "feature_name": feature_name,
        "metric": metric,
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "device_columns": n_devices,
        "rows": rows,
        "nat_timestamps": nat_timestamps,
        "duplicate_timestamps": duplicate_timestamps,
        "out_of_order_rows": out_of_order_rows,
        "first_timestamp_utc": None if first_ts is None else first_ts.isoformat(),
        "last_timestamp_utc": None if last_ts is None else last_ts.isoformat(),
        "native_interval_seconds": native_interval,
        "expected_samples_15min": expected_samples,
        "bins_total": int(len(ordered)),
        "bins_observed": int(np.count_nonzero(total_counts > 0)),
        "coverage_mean": float(np.mean(coverage)),
        "coverage_min": float(np.min(coverage)),
        "coverage_max": float(np.max(coverage)),
        "sample_completeness": float(total_counts.sum() / (len(ordered) * expected_samples * n_devices)),
        "raw_value_min": None if not np.isfinite(raw_min) else float(raw_min),
        "raw_value_max": None if not np.isfinite(raw_max) else float(raw_max),
        "unit": unit,
        "unit_verified": False,
        "semantic": semantic,
        "validity_threshold": threshold,
        "missing_policy": "NaN with observed_mask=False; no interpolation, no zero-fill",
        "notes": [],
    }
    if max_chunks is not None:
        audit["notes"].append(f"bounded debug scan: first {max_chunks} CSV chunks only")
    return {
        "feature_name": feature_name,
        "metric": metric,
        "unit": unit,
        "semantic": semantic,
        "bins": ordered,
        "values": values_mean,
        "coverage": coverage,
        "audit": audit,
    }


# ---------------------------------------------------------------------------
# job-trace activity
# ---------------------------------------------------------------------------

JOB_TRACE_COLUMNS: tuple[str, ...] = (
    "job_id",
    "gpu_num",
    "node_num",
    "state",
    "submit_time",
    "start_time",
    "end_time",
)


def scan_job_activity(
    path: str | Path,
    *,
    feature_specs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    cluster: str = "seren",
    max_chunks: int | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Stream the job trace and derive 15-minute cluster-level activity features.

    15-minute activity uses half-open job lifetimes ``[start_time, end_time)``: a job counts
    in a bin when its recorded execution overlaps that bin.  Jobs with missing start/end
    times are excluded and counted in the audit.  The trace has no node identifiers, so the
    activity is a cluster-level feature (documented, not a device pairing).
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"AcmeTrace job trace not found: {resolved}")
    header = [str(column) for column in pd.read_csv(resolved, nrows=0).columns]
    missing = [column for column in JOB_TRACE_COLUMNS if column not in header]
    if missing:
        raise ContractError(f"{resolved}: job trace is missing columns {missing}.")
    narrow_rows = int(config["aggregation_15min"].get("narrow_chunk_rows", 50_000))
    nrows = None if max_chunks is None else max_chunks * narrow_rows
    frame = pd.read_csv(resolved, usecols=list(JOB_TRACE_COLUMNS), nrows=nrows)
    if frame.empty:
        raise ContractError(f"{resolved}: job trace is empty.")
    if progress is not None:
        progress(f"job trace: read {len(frame):,} rows")

    submit = parse_monitor_timestamps(frame["submit_time"].tolist())
    start = parse_monitor_timestamps(frame["start_time"].tolist())
    end = parse_monitor_timestamps(frame["end_time"].tolist())
    gpu_num = pd.to_numeric(frame["gpu_num"], errors="coerce")
    gpu_values = np.nan_to_num(gpu_num.to_numpy(dtype=np.float64), nan=0.0)
    gpu_values = np.clip(gpu_values, 0.0, None)

    ns_submit = _valid_ns(submit)
    ns_start = _valid_ns(start)
    ns_end = _valid_ns(end)
    all_ns = np.concatenate([ns_submit, ns_start, ns_end])
    if all_ns.size == 0:
        raise ContractError(f"{resolved}: no parseable timestamps in the job trace.")
    bin_ns = BIN_SECONDS * 1_000_000_000
    min_bin = int(all_ns.min() // bin_ns)
    max_bin = int(all_ns.max() // bin_ns)
    n_bins = max_bin - min_bin + 1

    has_lifetime = (~np.isnan(ns_start)) & (~np.isnan(ns_end))
    lifetime = has_lifetime & (ns_end > ns_start)
    start_bin = ((ns_start[lifetime] // bin_ns).astype(np.int64)) - min_bin
    end_bin = (((ns_end[lifetime] - 1.0) // bin_ns).astype(np.int64)) - min_bin
    keep = end_bin >= start_bin
    diff_jobs = np.zeros(n_bins + 1, dtype=np.int64)
    diff_gpu = np.zeros(n_bins + 1, dtype=np.float64)
    np.add.at(diff_jobs, start_bin[keep], 1)
    np.add.at(diff_jobs, end_bin[keep] + 1, -1)
    np.add.at(diff_gpu, start_bin[keep], gpu_values[lifetime][keep])
    np.add.at(diff_gpu, end_bin[keep] + 1, -gpu_values[lifetime][keep])
    active_jobs = np.cumsum(diff_jobs[:-1]).astype(np.float64)
    active_gpu = np.cumsum(diff_gpu[:-1])
    valid_submit = ~np.isnan(ns_submit)
    submitted_bins = (ns_submit[valid_submit] // bin_ns).astype(np.int64) - min_bin
    submitted = np.bincount(submitted_bins, minlength=n_bins).astype(np.float64)

    bins = min_bin + np.arange(n_bins, dtype=np.int64)
    values_by_feature = {
        "job_active_count": active_jobs,
        "job_active_gpu_num": active_gpu,
        "job_submitted_count": submitted,
    }
    coverage = np.ones(n_bins, dtype=np.float64)
    payloads = []
    for spec in feature_specs:
        name = str(spec["feature_name"])
        if name not in values_by_feature:
            raise ContractError(
                f"job-trace feature {name!r} is not produced by this module; known: "
                f"{sorted(values_by_feature)}."
            )
        payloads.append(
            {
                "feature_name": name,
                "unit": str(spec["unit"]),
                "semantic": str(spec["semantic"]),
                "bins": bins,
                "values": values_by_feature[name],
                "coverage": coverage,
            }
        )
    audit = {
        "cluster": cluster,
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "rows": int(len(frame)),
        "bounded_debug": max_chunks is not None,
        "jobs_total": int(len(frame)),
        "jobs_with_lifetime": int(np.count_nonzero(lifetime)),
        "jobs_missing_start": int(np.count_nonzero(np.isnan(ns_start))),
        "jobs_missing_end": int(np.count_nonzero(np.isnan(ns_end))),
        "jobs_nonpositive_duration": int(np.count_nonzero(has_lifetime & ~lifetime)),
        "jobs_zero_gpu": int(np.count_nonzero(gpu_values == 0)),
        "first_timestamp_utc": bin_timestamps([min_bin])[0].isoformat(),
        "last_timestamp_utc": bin_timestamps([max_bin])[0].isoformat(),
        "bins_total": int(n_bins),
        "max_active_jobs": float(active_jobs.max()),
        "max_active_gpu": float(active_gpu.max()),
        "coverage_definition": (
            "1.0 inside the recorded trace span, 0 outside; trace completeness is not "
            "independently verifiable"
        ),
        "node_identifiers": "absent (cluster-level activity only)",
        "missing_policy": "no imputation; jobs with missing lifetimes are excluded and counted",
    }
    return {"payloads": payloads, "audit": audit}


def _valid_ns(timestamps: pd.DatetimeIndex) -> np.ndarray:
    values = np.asarray(timestamps.asi8, dtype=np.float64)
    values[np.asarray(timestamps.isna())] = np.nan
    return values


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def load_target_power(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the Seren ``cluster_it`` label from the 15-minute power parquet."""
    source = config["source"]
    label = source["label"]
    parquet = resolve_path(str(source["power_parquet"]))
    if not parquet.is_file():
        raise FileNotFoundError(
            f"AcmeTrace 15-minute power parquet not found: {parquet}. Run "
            "`python -m scripts.audit_acmetrace_power` first."
        )
    frame = pd.read_parquet(
        parquet,
        filters=[
            ("cluster_id", "==", str(label["cluster_id"])),
            ("level", "==", str(label["level"])),
            ("device_id", "==", str(label["device_id"])),
        ],
    )
    required = {"timestamp", "power_kw", "valid", "coverage_ratio"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractError(f"{parquet}: missing columns {missing}.")
    if frame.empty:
        raise ContractError(
            f"{parquet}: no rows for cluster_id={label['cluster_id']!r}, level={label['level']!r}, "
            f"device_id={label['device_id']!r}."
        )
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert("UTC").tz_localize(None)
    order = np.argsort(timestamps.asi8)
    timestamps = timestamps[order]
    if timestamps.duplicated().any():
        raise ContractError(f"{parquet}: duplicate label timestamps for {label['device_id']}.")
    power_kw = frame["power_kw"].to_numpy(dtype=np.float64)[order]
    power_valid = frame["valid"].to_numpy(dtype=bool)[order]
    coverage = frame["coverage_ratio"].to_numpy(dtype=np.float64)[order]
    power_kw = np.where(power_valid, power_kw, np.nan)
    target = pd.DataFrame(
        {
            "timestamp": timestamps,
            "site_id": str(config["dataset"]["site_id"]),
            "power_kw": power_kw,
            "power_valid": power_valid,
            "coverage_ratio": coverage,
        },
        columns=list(POWER_COLUMNS),
    )
    valid_values = power_kw[power_valid]
    audit = {
        "source_parquet": str(parquet),
        "level": str(label["level"]),
        "device_id": str(label["device_id"]),
        "slots": int(len(target)),
        "valid_slots": int(np.count_nonzero(power_valid)),
        "valid_fraction": float(np.count_nonzero(power_valid) / max(len(target), 1)),
        "first_timestamp_utc": timestamps[0].isoformat() if len(timestamps) else None,
        "last_timestamp_utc": timestamps[-1].isoformat() if len(timestamps) else None,
        "power_kw_mean": None if valid_values.size == 0 else float(np.mean(valid_values)),
        "power_kw_min": None if valid_values.size == 0 else float(np.min(valid_values)),
        "power_kw_max": None if valid_values.size == 0 else float(np.max(valid_values)),
        "missing_slots": int(len(target) - np.count_nonzero(power_valid)),
        "label_type": str(label["label_type"]),
        "never_facility_power": True,
    }
    return target, audit


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _align_payload(
    payload: Mapping[str, Any], label_timestamps: pd.DatetimeIndex, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_bins = bin_index(label_timestamps)
    values = np.full(len(label_bins), np.nan, dtype=np.float64)
    coverage = np.zeros(len(label_bins), dtype=np.float64)
    source_bins = np.asarray(payload["bins"], dtype=np.int64)
    if source_bins.size:
        index = np.searchsorted(source_bins, label_bins)
        inside = index < source_bins.size
        clipped = np.clip(index, 0, max(source_bins.size - 1, 0))
        found = inside & (source_bins[clipped] == label_bins)
        source_values = np.asarray(payload["values"], dtype=np.float64)
        source_coverage = np.asarray(payload["coverage"], dtype=np.float64)
        values = np.where(found, source_values[clipped], np.nan)
        coverage = np.where(found, source_coverage[clipped], 0.0)
    else:
        found = np.zeros(len(label_bins), dtype=bool)
    observed = found & np.isfinite(values) & (coverage >= threshold)
    values = np.where(observed, values, np.nan)
    return values, coverage, observed


def assemble_feature_frame(
    label_timestamps: pd.DatetimeIndex,
    payloads: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Align every feature payload onto the label's 15-minute grid (mask first)."""
    domain_id = str(config["dataset"]["domain_id"])
    threshold = float(config["aggregation_15min"]["validity_threshold"])
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for payload in payloads:
        values, coverage, observed = _align_payload(payload, label_timestamps, threshold)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": label_timestamps,
                    "domain_id": domain_id,
                    "feature_name": str(payload["feature_name"]),
                    "feature_value": values,
                    "unit": str(payload["unit"]),
                    "observed_mask": observed,
                    "coverage_ratio": coverage,
                }
            )
        )
        observed_index = np.flatnonzero(observed)
        audits.append(
            {
                "feature_name": str(payload["feature_name"]),
                "unit": str(payload["unit"]),
                "semantic": str(payload.get("semantic", "")),
                "observed_bins": int(observed_index.size),
                "label_bins": int(len(label_timestamps)),
                "observed_fraction": float(observed_index.size / max(len(label_timestamps), 1)),
                "first_observed_utc": (
                    label_timestamps[observed_index[0]].isoformat() if observed_index.size else None
                ),
                "last_observed_utc": (
                    label_timestamps[observed_index[-1]].isoformat() if observed_index.size else None
                ),
                "coverage_mean_observed": (
                    float(coverage[observed].mean()) if observed_index.size else None
                ),
                "validity_threshold": threshold,
            }
        )
    if not frames:
        raise ContractError("no feature payloads were produced; cannot assemble a target dataset.")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["feature_name", "timestamp"], kind="stable").reset_index(drop=True)
    return frame, audits


# ---------------------------------------------------------------------------
# metadata / quality
# ---------------------------------------------------------------------------


def _file_entry(path: Path, *, hash_contents: bool, role: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": role, "path": str(path)}
    if path.is_file():
        entry["bytes"] = int(path.stat().st_size)
        entry["modified_at_utc"] = pd.Timestamp(
            path.stat().st_mtime, unit="s", tz="UTC"
        ).isoformat()
        entry["sha256"] = sha256_file(path) if hash_contents else None
        if not hash_contents:
            entry["hash_note"] = "not recomputed in this light run (raw snapshot > 75 GB)"
    return entry


def build_metadata(
    config: Mapping[str, Any],
    *,
    power_frame: pd.DataFrame,
    power_audit: Mapping[str, Any],
    feature_audit: Sequence[Mapping[str, Any]],
    job_audit: Mapping[str, Any] | None,
    output_hashes: Mapping[str, str | None],
    max_chunks: int | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Assemble the target-dataset metadata JSON (label semantics, registry, provenance)."""
    dataset = config["dataset"]
    source = config["source"]
    label = source["label"]
    aggregation = config["aggregation_15min"]
    feature_specs: list[dict[str, Any]] = []
    for spec in list(config["features"]["node_metrics"]) + list(config["features"].get("gpu_metrics") or []):
        feature_specs.append(
            {
                "feature_name": str(spec["feature_name"]),
                "metric": str(spec["metric"]),
                "source_file": str(_raw_file(config, str(spec["path"]))),
                "unit": str(spec["unit"]),
                "unit_verified": False,
                "semantic": str(spec["semantic"]),
                "role": FEATURE_ROLE,
                "native_interval_seconds": 15,
                "expected_samples_15min": int(aggregation["expected_samples_15s"]),
                "future_known": False,
                "note": (
                    "cluster-level mean over reporting devices; not a complete-cluster quantity"
                ),
            }
        )
    if job_audit is not None:
        for spec in config["features"]["job_trace"]["features"]:
            feature_specs.append(
                {
                    "feature_name": str(spec["feature_name"]),
                    "metric": str(spec["feature_name"]),
                    "source_file": str(_raw_file(config, str(config["features"]["job_trace"]["path"]))),
                    "unit": str(spec["unit"]),
                    "unit_verified": True,
                    "semantic": str(spec["semantic"]),
                    "role": FEATURE_ROLE,
                    "native_interval_seconds": None,
                    "expected_samples_15min": None,
                    "future_known": False,
                    "note": "cluster-level job-trace activity; the trace has no node IDs",
                }
            )
    output_files = _output_files(config)
    artifacts = {
        "power_parquet": {
            "path": str(output_files["power_parquet"]),
            "rows": int(len(power_frame)),
            "columns": list(power_frame.columns),
            "sha256": output_hashes.get("power_parquet"),
        },
        "features_parquet": {
            "path": str(output_files["features_parquet"]),
            "sha256": output_hashes.get("features_parquet"),
        },
        "metadata_json": {"path": str(output_files["metadata_json"])},
        "quality_json": {"path": str(output_files["quality_json"])},
    }
    raw_files = [_file_entry(p, hash_contents=False, role="raw_monitor_metric") for p in _metric_paths(config)]
    raw_files.append(_file_entry(_raw_root(config) / str(config["features"]["job_trace"]["path"]), hash_contents=False, role="raw_job_trace"))
    raw_files.append(_file_entry(resolve_path(str(source["power_parquet"])), hash_contents=True, role="stage_1_15min_power_audit_product"))
    first = power_frame["timestamp"].iloc[0]
    last = power_frame["timestamp"].iloc[-1]
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "stage": "acmetrace_target_v2",
        "generated_at_utc": utc_timestamp(),
        "dataset_id": str(dataset["dataset_id"]),
        "site_id": str(dataset["site_id"]),
        "domain_id": str(dataset["domain_id"]),
        "display_name": str(dataset["display_name"]),
        "synthetic": False,
        "research_result": False,
        "bounded_debug": max_chunks is not None,
        "dataset": {
            "dataset_id": str(dataset["dataset_id"]),
            "site_id": str(dataset["site_id"]),
            "domain_id": str(dataset["domain_id"]),
            "role": "target_supervised",
            "clusters": [str(source["cluster"])],
            "frequency": "15min",
            "points_per_day": int(dataset["points_per_day"]),
            "synthetic": False,
            "research_result": False,
        },
        "label": {
            "label_type": str(label["label_type"]),
            "label_origin": str(label["label_origin"]),
            "target_kind": str(label["target_kind"]),
            "target_scope": str(label["target_scope"]),
            "target_name": f"{label['cluster_id']}_{label['level']}_kw",
            "unit": str(label["target_unit"]),
            "unit_inferred_from": str(label["unit_inferred"]),
            "unit_verified": bool(label["unit_verified"]),
            "measurement_boundary": str(label["measurement_boundary"]),
            "sampling_semantics": str(label["sampling_semantics"]),
            "never_facility_power": True,
            "composition_warning": (
                "Seren cluster IT power is an IT-side aggregate (sum over reporting server "
                "nodes). It excludes cooling, PDU/transformer losses, lighting and other "
                "facility loads. It is NOT a facility total-meter measurement and must never "
                "be reported as facility kW."
            ),
        },
        "timezone_metadata": {
            "declared_monitor_offset": str(source["timezone"]),
            "verified": bool(source["timezone_verified"]),
            "timezone_verified": bool(source["timezone_verified"]),
            "assumption": "explicit +08:00 offsets are parsed; timestamps are stored as UTC-naive",
            "note": (
                "The AcmeTrace README documents no site timezone. The monitor files write "
                "explicit offsets, which are parsed, but the site timezone stays unverified, "
                "so all prepared timestamps are UTC-naive interval starts."
            ),
        },
        "time": {
            "first_label_timestamp_utc": first.isoformat() if len(power_frame) else None,
            "last_label_timestamp_utc": last.isoformat() if len(power_frame) else None,
            "slots": int(len(power_frame)),
            "interval_step_seconds": BIN_SECONDS,
            "interval_label": "start; [timestamp, timestamp + 15 minutes)",
            "valid_slots": int(power_audit["valid_slots"]),
            "valid_fraction": float(power_audit["valid_fraction"]),
        },
        "aggregation": {
            "frequency": "15min",
            "step_seconds": BIN_SECONDS,
            "label": "interval_start",
            "timezone_handling": str(aggregation.get("timezone_handling", "")),
            "validity_threshold": float(aggregation["validity_threshold"]),
            "coverage_definition": str(aggregation.get("coverage_definition", "")),
            "duplicate_policy": str(aggregation.get("duplicate_policy", "")),
            "frame": str(aggregation.get("frame", "")),
            "missing_policy": "NaN with explicit masks; no interpolation, no zero-fill",
        },
        "features": {
            "role": "target_domain_runtime_auxiliary",
            "alignment": "label 15-minute grid; absent bins stay masked/NaN",
            "registry": feature_specs,
            "audit": list(feature_audit),
        },
        "job_trace": None if job_audit is None else {
            "path": job_audit["path"],
            "rows": job_audit["rows"],
            "cluster_level_only": True,
            "node_identifiers": job_audit["node_identifiers"],
            "audit": dict(job_audit),
        },
        "power_audit": dict(power_audit),
        "source_files": raw_files,
        "artifacts": artifacts,
        "readiness": {
            "ready_for_15min_analysis": True,
            "ready_for_training": False,
            "reasons": list(config.get("readiness", {}).get("reasons", [])),
            "note": (
                "This is a target-domain proxy dataset for the v2 transfer/mapping research; "
                "the label is cluster IT power with inferred units and an unverified site "
                "timezone, so it can support research experiments but not a formal facility-kW "
                "claim."
            ),
        },
        "provenance": {
            "git_branch": git_branch(),
            "git_head": git_rev(),
            "python": _python_version(),
            "dependencies": dependency_versions(),
            "config_digest": config_digest(config),
            "raw_root": str(_raw_root(config)),
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "run_mode": "bounded_debug" if max_chunks is not None else "canonical_light_run",
        },
    }
    if max_chunks is not None:
        metadata["bounded_debug_note"] = (
            f"DEBUG/BOUNDED run: only the first {max_chunks} chunks of each source file were "
            "read. Numbers are pilot figures and MUST NOT be cited as full coverage."
        )
    return metadata


def build_quality(
    *,
    power_frame: pd.DataFrame,
    power_audit: Mapping[str, Any],
    feature_frame: pd.DataFrame,
    feature_audit: Sequence[Mapping[str, Any]],
    metric_audits: Sequence[Mapping[str, Any]],
    job_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Quality report: label stats, feature coverage and power<->feature correlation."""
    power = power_frame.copy()
    correlations: list[dict[str, Any]] = []
    for feature_name in sorted(feature_frame["feature_name"].unique()):
        subset = feature_frame[feature_frame["feature_name"] == feature_name]
        merged = pd.DataFrame(
            {
                "power": power["power_kw"].to_numpy(dtype=np.float64),
                "power_valid": power["power_valid"].to_numpy(dtype=bool),
                "value": subset["feature_value"].to_numpy(dtype=np.float64),
                "observed": subset["observed_mask"].to_numpy(dtype=bool),
            }
        )
        usable = merged["power_valid"] & merged["observed"] & np.isfinite(merged["value"]) & np.isfinite(merged["power"])
        entry: dict[str, Any] = {
            "feature_name": feature_name,
            "overlap_valid_bins": int(np.count_nonzero(usable)),
        }
        if int(np.count_nonzero(usable)) >= 10:
            x = merged.loc[usable, "value"].to_numpy(dtype=np.float64)
            y = merged.loc[usable, "power"].to_numpy(dtype=np.float64)
            if np.std(x) > 0 and np.std(y) > 0:
                entry["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
            else:
                entry["pearson_r"] = None
                entry["note"] = "zero variance on the overlap; correlation undefined"
        else:
            entry["pearson_r"] = None
            entry["note"] = "fewer than 10 overlapping valid bins"
        correlations.append(entry)
    warnings = [
        "Power units are inferred as W from magnitude evidence only (unit_verified=false).",
        "Site timezone is not documented; all timestamps are UTC-naive interval starts.",
        "The label is Seren cluster IT power (sum over reporting servers), NOT facility power.",
        "Runtime features are means over reporting devices; coverage varies by bin.",
        "The job trace has no node IDs; job activity is cluster-level only.",
        "Correlations are descriptive overlap statistics, not evidence of predictive value.",
    ]
    quality: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "dataset_id": None,
        "power": dict(power_audit),
        "features": list(feature_audit),
        "metric_audit": list(metric_audits),
        "job_trace": None if job_audit is None else dict(job_audit),
        "feature_power_correlation": correlations,
        "quality_warnings": warnings,
        "rows": {
            "power": int(len(power_frame)),
            "features": int(len(feature_frame)),
            "feature_names": int(feature_frame["feature_name"].nunique()),
        },
    }
    return quality


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def _metric_specs(config: Mapping[str, Any], *, include_gpu: bool) -> list[Mapping[str, Any]]:
    specs = list(config["features"]["node_metrics"])
    if include_gpu:
        specs.extend(config["features"].get("gpu_metrics") or [])
    return specs


def _metric_paths(config: Mapping[str, Any]) -> list[Path]:
    return [_raw_file(config, str(spec["path"])) for spec in _metric_specs(config, include_gpu=True)]


def _python_version() -> str:
    import platform

    return platform.python_version()


def _output_files(config: Mapping[str, Any]) -> dict[str, Path]:
    outputs = config["outputs"]
    return {key: resolve_path(str(outputs[key])) for key in outputs if key != "results_dir"}


def run_pipeline(
    config: Mapping[str, Any],
    *,
    max_chunks: int | None = None,
    include_gpu: bool = True,
    include_jobs: bool = True,
    write: bool = True,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the target-domain assembly end to end and (optionally) write its artefacts."""
    import time

    started = time.perf_counter()
    config = validate_target_config(config)
    power_frame, power_audit = load_target_power(config)
    if progress is not None:
        progress(
            f"label: {len(power_frame)} slots ({power_audit['valid_slots']} valid) "
            f"{power_audit['first_timestamp_utc']} -> {power_audit['last_timestamp_utc']}"
        )
    payloads: list[dict[str, Any]] = []
    metric_audits: list[dict[str, Any]] = []
    for spec in _metric_specs(config, include_gpu=include_gpu):
        payload = scan_wide_metric(
            _raw_file(config, str(spec["path"])),
            metric=str(spec["metric"]),
            feature_name=str(spec["feature_name"]),
            unit=str(spec["unit"]),
            semantic=str(spec["semantic"]),
            config=config,
            max_chunks=max_chunks,
            progress=progress,
        )
        payloads.append(payload)
        metric_audits.append(payload["audit"])
    job_audit: dict[str, Any] | None = None
    if include_jobs:
        job_payload = scan_job_activity(
            _raw_file(config, str(config["features"]["job_trace"]["path"])),
            feature_specs=config["features"]["job_trace"]["features"],
            config=config,
            cluster=str(config["source"]["cluster"]),
            max_chunks=max_chunks,
            progress=progress,
        )
        payloads.extend(job_payload["payloads"])
        job_audit = job_payload["audit"]
    label_timestamps = pd.DatetimeIndex(power_frame["timestamp"])
    feature_frame, feature_audit = assemble_feature_frame(
        label_timestamps, payloads, config=config
    )
    elapsed = time.perf_counter() - started
    output_files = _output_files(config)
    output_hashes: dict[str, str | None] = {"power_parquet": None, "features_parquet": None}
    if write:
        _write_frames(config, power_frame, feature_frame)
        output_hashes["power_parquet"] = sha256_file(output_files["power_parquet"])
        output_hashes["features_parquet"] = sha256_file(output_files["features_parquet"])
    metadata = build_metadata(
        config,
        power_frame=power_frame,
        power_audit=power_audit,
        feature_audit=feature_audit,
        job_audit=job_audit,
        output_hashes=output_hashes,
        max_chunks=max_chunks,
        elapsed_seconds=elapsed,
    )
    quality = build_quality(
        power_frame=power_frame,
        power_audit=power_audit,
        feature_frame=feature_frame,
        feature_audit=feature_audit,
        metric_audits=metric_audits,
        job_audit=job_audit,
    )
    quality["dataset_id"] = str(config["dataset"]["dataset_id"])
    if write:
        _write_reports(config, metadata, quality)
    return {
        "power": power_frame,
        "features": feature_frame,
        "metadata": metadata,
        "quality": quality,
        "metric_audits": metric_audits,
        "job_audit": job_audit,
        "elapsed_seconds": elapsed,
        "outputs": {key: str(path) for key, path in output_files.items()},
    }


def _write_frames(
    config: Mapping[str, Any],
    power_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> None:
    outputs = _output_files(config)
    if list(power_frame.columns) != list(POWER_COLUMNS):
        raise ContractError(
            f"target power frame columns {list(power_frame.columns)} != expected {list(POWER_COLUMNS)}"
        )
    if list(feature_frame.columns) != list(FEATURE_COLUMNS):
        raise ContractError(
            f"target feature frame columns {list(feature_frame.columns)} != expected "
            f"{list(FEATURE_COLUMNS)}"
        )
    for key in ("power_parquet", "features_parquet"):
        assert_not_raw_path(str(outputs[key]))
    ensure_dir(outputs["power_parquet"].parent)
    power_frame.to_parquet(outputs["power_parquet"], index=False)
    feature_frame.to_parquet(outputs["features_parquet"], index=False)


def _write_reports(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> None:
    outputs = _output_files(config)
    for key in ("metadata_json", "quality_json"):
        assert_not_raw_path(str(outputs[key]))
    ensure_dir(outputs["metadata_json"].parent)
    results_dir = resolve_path(str(config["outputs"]["results_dir"]))
    ensure_dir(results_dir)
    write_json(outputs["metadata_json"], metadata)
    write_json(outputs["quality_json"], quality)
    write_json(
        results_dir / "acmetrace_target_run_summary.json",
        {
            "generated_at_utc": utc_timestamp(),
            "dataset_id": metadata["dataset_id"],
            "elapsed_seconds": metadata["provenance"]["elapsed_seconds"],
            "run_mode": metadata["provenance"]["run_mode"],
            "label": metadata["label"],
            "time": metadata["time"],
            "features": metadata["features"]["audit"],
            "job_trace": metadata["job_trace"],
            "artifacts": {
                "power_parquet": metadata["artifacts"]["power_parquet"],
                "features_parquet": metadata["artifacts"]["features_parquet"],
            },
            "readiness": metadata["readiness"],
        },
    )


__all__ = [
    "BINS_PER_DAY",
    "BIN_SECONDS",
    "CONFIG_PATH",
    "FEATURE_COLUMNS",
    "POWER_COLUMNS",
    "assemble_feature_frame",
    "bin_index",
    "bin_timestamps",
    "build_metadata",
    "build_quality",
    "load_target_config",
    "load_target_power",
    "parse_monitor_timestamps",
    "run_pipeline",
    "scan_job_activity",
    "scan_wide_metric",
    "validate_target_config",
]
