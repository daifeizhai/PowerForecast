"""AcmeTrace runtime-metric, power<->ops pairing and job-trace admission audit.

REAL data only (``E:/Data/AcmeTrace/`` by default, read-only; overridable via
``ACMETRACE_RAW_ROOT`` or ``--raw-root``).  The raw tree is ~80 GB, so every
utilization/IPMI file is streamed in bounded pandas chunks; nothing is loaded whole and
peak memory stays far below the reader budget (the largest chunk is a few hundred MB).

What this module audits (task book: AcmeTrace admission audit):

* **Runtime metrics** -- every ``utilization/<cluster>/*.csv`` file: rows, device columns,
  verified native sampling interval (mode of consecutive-timestamp deltas) and its
  distribution, first/last timestamp, exact device-set differences between files, per-device
  missing fraction (aggregate + min/median/max), magnitude/range evidence for unit inference
  (``unit_verified=false``), out-of-range counts (utilisation outside 0-100, negative
  temperatures / free memory / power / clocks), XID error counts and rates, placeholder/empty
  files, duplicate/out-of-order timestamps, gaps and coverage.
* **Pairing** -- GPU column name -> ``(node_ip, gpu_index)``; 8-GPU/node consistency; node-set
  consistency across metrics, NODE_* files and IPMI nodes; timezone table and conversion rule
  (Seren job trace +08:00, **Kalos job trace +00:00**, utilization +08:00); same-instant
  pairing on a bounded native grid and full-file 15-minute bins at GPU level (DCGM
  ``POWER_USAGE`` vs ``GPU_UTIL``/``GPU_TEMP``/``FB_USED``) and at node level (Seren IPMI
  ``Sys_Total_Power`` vs ``NODE_CPU``/``NODE_MEMORY``/``NODE_IB``); matched devices, matched
  device-slots, effective joint span and match rates.  Unsafe/impossible pairings are listed
  explicitly (Kalos has no IPMI; GPU vs server power are different measurement levels; the
  job traces carry **no node IDs**, so per-server job linking is impossible).
* **Job traces** -- rows, time range, field completeness, state/type distributions,
  duration/queue plausibility, resource-count ranges, and a strict pre-origin vs post-hoc
  field classification (post-hoc fields may never be future inputs), plus cluster-level
  temporal linkage (job activity vs mean GPU utilisation) with the no-node-ID limitation.

Timezone rule enforced everywhere: every timestamp is parsed with its explicit offset (or
``CST`` = +08:00 for the Chinese-locale IPMI timestamps) and converted to UTC before any
comparison.  A UTC string is never equated with a +08:00 string.
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils import ensure_dir, resolve_path, utc_timestamp, write_json

# ---------------------------------------------------------------------------
# constants / paths (derived from the verified raw layout)
# ---------------------------------------------------------------------------

#: Raw AcmeTrace tree (moved out of the repo to ``E:/Data/AcmeTrace``).  The
#: ``ACMETRACE_RAW_ROOT`` environment variable or the CLI ``--raw-root`` flag overrides it.
RAW_ROOT = os.environ.get("ACMETRACE_RAW_ROOT", "E:/Data/AcmeTrace")
UTIL_ROOT = f"{RAW_ROOT}/data/utilization"
IPMI_ROOT = f"{UTIL_ROOT}/ipmi"
JOB_ROOT = f"{RAW_ROOT}/data/job_trace"
CLUSTERS: tuple[str, ...] = ("seren", "kalos")

BIN_SECONDS = 900  # 15-minute analysis bins

FULL_CHUNK_ROWS = 50_000
WIDE_CHUNK_ROWS = 10_000  # >1000 device columns: bounded memory during binning
NARROW_CHUNK_ROWS = 200_000
VALUE_SAMPLE_TARGET = 200_000

#: Native-window pairing is verified on a bounded 24 h window on the common 15 s grid;
#: the full joint span is covered by 15-minute-bin pairing.
NATIVE_WINDOWS: dict[str, tuple[str, str]] = {
    "seren": ("2023-07-10T00:00:00+08:00", "2023-07-11T00:00:00+08:00"),
    "kalos": ("2023-08-16T00:00:00+08:00", "2023-08-17T00:00:00+08:00"),
}

#: Metrics paired at GPU level: DCGM power against utilisation/temperature/framebuffer.
GPU_PAIRING_METRICS: tuple[str, ...] = ("POWER_USAGE", "GPU_UTIL", "GPU_TEMP", "FB_USED")
GPU_PAIRING_BASE = "POWER_USAGE"

#: Metrics paired at node level against IPMI ``Sys_Total_Power`` (Seren only; no Kalos IPMI).
NODE_PAIRING_METRICS: tuple[str, ...] = (
    "NODE_CPU_UTILIZATION",
    "NODE_MEMORY_UTILIZATION",
    "NODE_IB_RECEIVE",
)
IPMI_POWER_COLUMN = "Sys_Total_Power"

#: Per-metric semantics.  ``unit_verified`` is False for every metric: the local files do not
#: document units, so the unit below is an inference backed by the magnitude evidence emitted
#: in the audit (never silently assumed).
METRIC_SPECS: dict[str, dict[str, Any]] = {
    "GPU_UTIL": {"level": "gpu", "semantic": "gpu_utilization", "unit_guess": "percent", "expected_range": (0.0, 100.0)},
    "GPU_TEMP": {"level": "gpu", "semantic": "gpu_core_temperature", "unit_guess": "degC", "expected_range": (-20.0, 150.0)},
    "MEMORY_TEMP": {"level": "gpu", "semantic": "gpu_memory_temperature", "unit_guess": "degC", "expected_range": (-20.0, 150.0)},
    "FB_USED": {"level": "gpu", "semantic": "framebuffer_memory_used", "unit_guess": "bytes_or_MB_unverified", "expected_range": (0.0, None)},
    "FB_FREE": {"level": "gpu", "semantic": "framebuffer_memory_free", "unit_guess": "bytes_or_MB_unverified", "expected_range": (0.0, None)},
    "DRAM_ACTIVE": {"level": "gpu", "semantic": "dram_active", "unit_guess": "unverified", "expected_range": (0.0, None)},
    "MEM_CLOCK": {"level": "gpu", "semantic": "gpu_memory_clock", "unit_guess": "MHz_unverified", "expected_range": (0.0, None)},
    "MEM_COPY_UTIL": {"level": "gpu", "semantic": "memory_copy_utilization", "unit_guess": "percent", "expected_range": (0.0, 100.0)},
    "PIPE_TENSOR_ACTIVE": {"level": "gpu", "semantic": "pipe_tensor_active", "unit_guess": "percent_or_ratio_unverified", "expected_range": (0.0, None)},
    "SM_ACTIVE": {"level": "gpu", "semantic": "sm_active", "unit_guess": "percent", "expected_range": (0.0, 100.0)},
    "SM_OCCUPANCY": {"level": "gpu", "semantic": "sm_occupancy", "unit_guess": "percent", "expected_range": (0.0, 100.0)},
    "XID_ERRORS": {"level": "gpu", "semantic": "xid_error_count", "unit_guess": "count", "expected_range": (0.0, None)},
    "POWER_USAGE": {"level": "gpu", "semantic": "gpu_power", "unit_guess": "W_unverified", "expected_range": (0.0, None)},
    "NODE_CPU_UTILIZATION": {"level": "node", "semantic": "node_cpu_utilization", "unit_guess": "percent", "expected_range": (0.0, 100.0)},
    "NODE_MEMORY_UTILIZATION": {"level": "node", "semantic": "node_memory_utilization", "unit_guess": "percent", "expected_range": (0.0, 100.0)},
    "NODE_IB_RECEIVE": {"level": "node", "semantic": "node_ib_receive", "unit_guess": "bytes_or_bytes_per_s_unverified", "expected_range": (0.0, None)},
    "NODE_IB_SEND": {"level": "node", "semantic": "node_ib_send", "unit_guess": "bytes_or_bytes_per_s_unverified", "expected_range": (0.0, None)},
}

UTILIZATION_METRICS: tuple[str, ...] = tuple(METRIC_SPECS)

#: Job-trace fields strictly known BEFORE a forecast origin vs post-hoc only.  ``queue`` is
#: post-hoc because the file computes it as ``start_time - submit_time``; a queued job's wait
#: is not known at submission.  ``start_time`` is also post-hoc for a job that has not started.
JOB_FIELD_CLASSIFICATION: dict[str, dict[str, Any]] = {
    "job_id": {"known_before_origin": True, "timing": "submission", "reason": "job identity is fixed at submission"},
    "user": {"known_before_origin": True, "timing": "submission", "reason": "submitting user is fixed at submission"},
    "node_num": {"known_before_origin": True, "timing": "submission", "reason": "requested node count is a submission-time resource request"},
    "gpu_num": {"known_before_origin": True, "timing": "submission", "reason": "requested GPU count is a submission-time resource request"},
    "cpu_num": {"known_before_origin": True, "timing": "submission", "reason": "requested CPU count is a submission-time resource request"},
    "type": {"known_before_origin": True, "timing": "submission", "reason": "workload type is declared at submission"},
    "mem_per_pod_GB": {"known_before_origin": True, "timing": "submission", "reason": "pod memory configuration is a submission-time request (Kalos only)"},
    "shared_mem_per_pod": {"known_before_origin": True, "timing": "submission", "reason": "pod shared-memory configuration is a submission-time request (Kalos only)"},
    "submit_time": {"known_before_origin": True, "timing": "submission", "reason": "submission timestamp itself"},
    "state": {"known_before_origin": False, "timing": "post_hoc_completion", "reason": "terminal state is only known after the job ends"},
    "start_time": {"known_before_origin": False, "timing": "post_hoc_start", "reason": "actual start time is unknown before the job starts"},
    "end_time": {"known_before_origin": False, "timing": "post_hoc_completion", "reason": "termination time is only known after the job ends"},
    "duration": {"known_before_origin": False, "timing": "post_hoc_completion", "reason": "computed as end_time - start_time"},
    "queue": {"known_before_origin": False, "timing": "post_hoc_start", "reason": "computed as start_time - submit_time, so it is only known once the job starts"},
    "gpu_time": {"known_before_origin": False, "timing": "post_hoc_completion", "reason": "computed as duration * gpu_num"},
    "fail_time": {"known_before_origin": False, "timing": "post_hoc_failure", "reason": "failure instant is only known after it happens (Kalos only)"},
    "stop_time": {"known_before_origin": False, "timing": "post_hoc_completion", "reason": "stop instant is only known after termination (Kalos only)"},
}

_IP_RE = re.compile(r"^(?P<ip>\d{1,3}(?:\.\d{1,3}){3})$")
_IP_GPU_RE = re.compile(r"^(?P<node>\d{1,3}(?:\.\d{1,3}){3})-(?P<gpu>\d{1,2})$")
_GENERIC_GPU_RE = re.compile(r"^(?P<node>.+)-(?P<gpu>\d{1,2})$")
_OFFSET_RE = re.compile(r"([+-]\d{2}:\d{2})$")
_IPMI_NODE_RE = re.compile(r"^SH-IDC1-(\d+)-(\d+)-(\d+)-(\d+)$")
_IPMI_CHINESE_RE = re.compile(r"^(\d{4})年\s*(\d{2})月\s*(\d{2})日\s*星期.\s*(\d{2}):(\d{2}):(\d{2})")


# ---------------------------------------------------------------------------
# device / node naming
# ---------------------------------------------------------------------------


def parse_device(column: str) -> tuple[str, int] | None:
    """Parse a GPU-wide column name into ``(node, gpu_index)``.

    Handles ``10.140.1.10-6`` (Seren / Kalos IP naming) and ``lingjun-pod9-0440-4``
    (Kalos pod naming).  The GPU index is the final hyphen-separated integer.
    """
    if not isinstance(column, str):
        return None
    match = _IP_GPU_RE.match(column)
    if match:
        return match.group("node"), int(match.group("gpu"))
    match = _GENERIC_GPU_RE.match(column)
    if match and match.group("node"):
        return match.group("node"), int(match.group("gpu"))
    return None


def parse_node_device(column: str) -> str | None:
    """Return the node identifier of a NODE_* column (an IPv4 string here)."""
    if not isinstance(column, str):
        return None
    match = _IP_RE.match(column)
    if match:
        return match.group("ip")
    return None


def parse_gpu_device_columns(columns: Sequence[str]) -> list[tuple[str, int]] | None:
    parsed = [parse_device(column) for column in columns]
    return None if any(item is None for item in parsed) else parsed  # type: ignore[return-value]


def ipmi_node_to_ip(node: str) -> str | None:
    """Map an IPMI node name (``SH-IDC1-10-140-0-112``) to its IPv4 address."""
    if not isinstance(node, str):
        return None
    match = _IPMI_NODE_RE.match(node.strip())
    if match:
        return ".".join(match.groups())
    match = _IP_RE.match(node.strip())
    return match.group("ip") if match else None


def device_node_set(devices: Sequence[tuple[str, int]]) -> set[str]:
    return {node for node, _ in devices}


def devices_per_node(devices: Sequence[tuple[str, int]]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for node, index in devices:
        result.setdefault(node, set()).add(index)
    return result


# ---------------------------------------------------------------------------
# timestamp parsing (explicit offsets only; no naive==aware comparisons)
# ---------------------------------------------------------------------------

_EXTERNAL_IPMI_PARSER = None


def _external_ipmi_parser():
    """Optional reuse of ``src/data/acmetrace_power.py`` if that teammate parser exists."""
    global _EXTERNAL_IPMI_PARSER
    if _EXTERNAL_IPMI_PARSER is None:
        try:  # pragma: no cover - depends on parallel teammate delivery
            from src.data import acmetrace_power  # type: ignore

            candidate = getattr(acmetrace_power, "parse_ipmi_timestamps", None)
            if not callable(candidate):
                candidate = getattr(acmetrace_power, "parse_acmetrace_timestamps", None)
            _EXTERNAL_IPMI_PARSER = candidate if callable(candidate) else False
        except Exception:  # pragma: no cover
            _EXTERNAL_IPMI_PARSER = False
    return _EXTERNAL_IPMI_PARSER or None


def parse_iso_timestamps(values: Sequence[Any]) -> pd.DatetimeIndex:
    """Parse ISO-8601 timestamps with explicit offsets and convert to UTC."""
    series = pd.Series(values, dtype="object")
    parsed = pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    return pd.DatetimeIndex(parsed)


def parse_ipmi_timestamps(values: Sequence[Any]) -> pd.DatetimeIndex:
    """Parse the IPMI timestamps, including the Chinese-locale ``CST`` rows.

    ``CPU_D_Power.csv`` mixes rows such as ``2023年 05月 15日 星期一 14:15:01 CST`` with
    ISO rows; ``CST`` in this local file is China Standard Time (+08:00), which is documented
    as an assumption because the string itself does not carry a numeric offset.
    """
    external = _external_ipmi_parser()
    if external is not None:  # pragma: no cover - reuse teammate parser when available
        try:
            parsed = external(values)
            index = pd.DatetimeIndex(parsed)
            if index.tz is None:
                index = index.tz_localize("UTC")
            return index.tz_convert("UTC")
        except Exception:  # pragma: no cover - fall through to the local parser
            pass
    series = pd.Series(values, dtype="object").astype(str)
    parsed = pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    index = pd.DatetimeIndex(parsed)
    missing_mask = np.asarray(index.isna())
    if not bool(missing_mask.any()):
        return index
    extracted = series.iloc[np.flatnonzero(missing_mask)].str.extract(_IPMI_CHINESE_RE)
    ok = extracted.notna().all(axis=1)
    if not bool(ok.any()):
        return index
    pieces = extracted[ok]
    rebuilt = (
        pieces[0].astype(str)
        + "-"
        + pieces[1].astype(str)
        + "-"
        + pieces[2].astype(str)
        + " "
        + pieces[3].astype(str)
        + ":"
        + pieces[4].astype(str)
        + ":"
        + pieces[5].astype(str)
    )
    local = pd.to_datetime(rebuilt, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    local = local.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    result = list(index)
    positions = np.flatnonzero(missing_mask)
    ok_positions = positions[ok.to_numpy()]
    for position, timestamp in zip(ok_positions.tolist(), local.tolist()):
        result[position] = pd.Timestamp(timestamp)
    return pd.DatetimeIndex(result)


def detect_timestamp_format(sample: Any) -> str:
    text = str(sample)
    if "年" in text:
        return "ipmi_chinese_or_iso"
    if _IPMI_NODE_RE.match(text):
        return "ipmi_node"
    return "iso8601"


def observed_offset(values: Sequence[Any]) -> str | None:
    """Return the explicit UTC offset found in the raw timestamp strings, if any."""
    for value in values:
        text = str(value)
        if "CST" in text:
            return "+08:00 (CST literal, China Standard Time assumption)"
        match = _OFFSET_RE.search(text)
        if match:
            return match.group(1)
    return None


def parse_timestamps(values: Sequence[Any], *, format_kind: str) -> pd.DatetimeIndex:
    """Parse a timestamp column, tolerating mixed ISO + Chinese-locale IPMI rows."""
    if format_kind.startswith("ipmi"):
        return parse_ipmi_timestamps(values)
    index = parse_iso_timestamps(values)
    if bool(index.isna().any()):
        return parse_ipmi_timestamps(values)
    return index


# ---------------------------------------------------------------------------
# interval statistics
# ---------------------------------------------------------------------------


def interval_statistics(
    timestamps_ns: np.ndarray,
    *,
    expected_interval_seconds: float | None = None,
) -> dict[str, Any]:
    """Native-interval evidence from a *snapshot* array of UTC epoch nanoseconds."""
    values = np.asarray(timestamps_ns, dtype=np.int64)
    values = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    if len(values) < 2:
        return {
            "count": int(len(values)),
            "native_interval_seconds": None,
            "native_interval_counts": {},
            "distinct_deltas": 0,
            "negative_or_zero_deltas": 0,
            "gap_count": 0,
            "max_gap_seconds": 0.0,
            "all_deltas_multiples_of_native": True,
            "matches_expected_interval": None,
        }
    deltas = np.diff(values.astype("int64")) / 1e9
    counts = Counter(float(value) for value in deltas)
    native = counts.most_common(1)[0][0]
    gaps = sum(1 for value in counts if value > native)
    positive = [value for value in counts if value > 0]
    all_multiples = bool(
        np.allclose(np.mod(deltas[deltas > 0], native), 0.0)
    ) if len(positive) else True
    return {
        "count": int(len(values)),
        "native_interval_seconds": float(native),
        "native_interval_counts": {str(int(key)): int(value) for key, value in counts.most_common(12)},
        "distinct_deltas": int(len(counts)),
        "negative_or_zero_deltas": int(sum(1 for value in counts if value <= 0)),
        "gap_count": int(gaps),
        "max_gap_seconds": float(max(counts)),
        "all_deltas_multiples_of_native": all_multiples,
        "matches_expected_interval": (
            None if expected_interval_seconds is None else bool(abs(native - expected_interval_seconds) < 1e-9)
        ),
    }


def infer_unit_evidence(
    sample: np.ndarray,
    *,
    metric: str,
    spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Magnitude evidence supporting the (unverified) unit guess of one metric."""
    spec = spec or METRIC_SPECS.get(metric, {})
    sample = np.asarray(sample, dtype="float64")
    sample = sample[np.isfinite(sample)]
    evidence: dict[str, Any] = {
        "unit_guess": spec.get("unit_guess", "unverified"),
        "unit_verified": False,
        "sample_size": int(len(sample)),
        "note": "unit inference from magnitude/range evidence only; local files document no units",
    }
    if len(sample) == 0:
        evidence["inference"] = "no observed values"
        return evidence
    quantiles = np.quantile(sample, [0.01, 0.5, 0.99])
    evidence.update(
        {
            "min": float(sample.min()),
            "p01": float(quantiles[0]),
            "median": float(quantiles[1]),
            "p99": float(quantiles[2]),
            "max": float(sample.max()),
        }
    )
    median = float(quantiles[1])
    metric_upper = metric.upper()
    if metric_upper in {"GPU_UTIL", "MEM_COPY_UTIL", "SM_ACTIVE", "SM_OCCUPANCY", "NODE_CPU_UTILIZATION", "NODE_MEMORY_UTILIZATION"}:
        evidence["inference"] = "values lie in 0-100 -> percent-style utilization"
    elif metric_upper in {"GPU_TEMP", "MEMORY_TEMP"}:
        evidence["inference"] = "values in tens -> degrees Celsius temperature"
    elif metric_upper == "XID_ERRORS":
        evidence["inference"] = "small nonnegative integers -> event count"
    elif metric_upper == "POWER_USAGE":
        evidence["inference"] = "median in ~10^2 -> watts-scale GPU power"
    elif metric_upper in {"FB_USED", "FB_FREE", "DRAM_ACTIVE"}:
        if median > 1e6:
            evidence["inference"] = "median > 10^6 -> byte-scale memory quantity"
        else:
            evidence["inference"] = "median < 10^6 -> MB-scale or other memory quantity; unverified"
    elif metric_upper == "MEM_CLOCK":
        evidence["inference"] = "median in ~10^3 -> MHz-scale clock"
    elif metric_upper in {"NODE_IB_RECEIVE", "NODE_IB_SEND"}:
        evidence["inference"] = "large nonnegative values -> bytes or bytes/s scale; unverified"
    else:
        evidence["inference"] = "magnitude only; unit and semantics unverified"
    return evidence


# ---------------------------------------------------------------------------
# bin / native-window accumulators used for pairing
# ---------------------------------------------------------------------------


class BinAccumulator:
    """Sparse (15-minute bin -> per-device sum/count) accumulator (memory bounded)."""

    def __init__(self) -> None:
        self.sums: dict[int, np.ndarray] = {}
        self.counts: dict[int, np.ndarray] = {}
        self.devices: list[str] = []
        self._dense_cache: dict[str, Any] | None = None

    def add(self, bins: np.ndarray, sums: np.ndarray, counts: np.ndarray) -> None:
        self._dense_cache = None
        for position, bin_id in enumerate(bins.tolist()):
            existing = self.sums.get(bin_id)
            if existing is None:
                self.sums[bin_id] = sums[position].astype(np.float32, copy=True)
                self.counts[bin_id] = counts[position].astype(np.uint16, copy=True)
            else:
                existing += sums[position]
                self.counts[bin_id] += counts[position].astype(np.uint16)

    def dense(self) -> dict[str, Any]:
        if self._dense_cache is not None:
            return self._dense_cache
        bins = np.asarray(sorted(self.sums), dtype=np.int64)
        if len(bins) == 0:
            payload = {
                "bins": bins,
                "sums": np.zeros((0, len(self.devices)), dtype=np.float32),
                "counts": np.zeros((0, len(self.devices)), dtype=np.uint16),
            }
        else:
            sums = np.stack([self.sums[bin_id] for bin_id in bins]).astype(np.float32)
            counts = np.stack([self.counts[bin_id] for bin_id in bins]).astype(np.uint16)
            payload = {"bins": bins, "sums": sums, "counts": counts}
        # Release the sparse per-bin arrays once the dense view exists (the audit only
        # materialises dense views after all chunks of a file have been consumed), keeping
        # peak memory comfortably below the 2 GB budget.
        self.sums = {}
        self.counts = {}
        self._dense_cache = payload
        return payload

    def mean(self, device_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        dense = self.dense()
        counts = dense["counts"].astype(np.float64)
        if device_indices is not None:
            counts = counts[:, device_indices]
            sums = dense["sums"][:, device_indices]
        else:
            sums = dense["sums"]
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
        return dense["bins"], means


class NativeWindowAccumulator:
    """Dense (native-window timestep -> per-device value) accumulator for same-instant pairing."""

    def __init__(self, *, start_ns: int, step_seconds: float, length: int, n_devices: int) -> None:
        self.start_ns = int(start_ns)
        self.step_ns = int(round(step_seconds * 1e9))
        self.length = int(length)
        self.n_devices = int(n_devices)
        self.values: np.ndarray | None = None
        self.observed: np.ndarray | None = None

    def add(self, timestamps_ns: np.ndarray, values: np.ndarray) -> None:
        if len(timestamps_ns) == 0:
            return
        indices = np.floor_divide(timestamps_ns.astype(np.int64) - self.start_ns, self.step_ns)
        valid = (indices >= 0) & (indices < self.length)
        if not bool(valid.any()):
            return
        if self.values is None:
            self.values = np.full((self.length, self.n_devices), np.nan, dtype=np.float32)
            self.observed = np.zeros((self.length, self.n_devices), dtype=bool)
        rows = indices[valid]
        chunk = values[valid]
        self.values[rows] = chunk
        self.observed[rows] = np.isfinite(chunk)


def _bin_indices(timestamps: pd.DatetimeIndex) -> np.ndarray:
    return timestamps.asi8 // (BIN_SECONDS * 1_000_000_000)


def _nan_min_max(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware per-column min/max without all-NaN RuntimeWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(values, axis=0), np.nanmax(values, axis=0)


def _native_window_timestamps_ns(cluster: str) -> tuple[int, int, int]:
    start_text, end_text = NATIVE_WINDOWS[cluster]
    start = pd.Timestamp(start_text).tz_convert("UTC")
    end = pd.Timestamp(end_text).tz_convert("UTC")
    length = int(round((end - start).total_seconds() / 15.0))
    return int(start.value), length, 15


# ---------------------------------------------------------------------------
# per-file streaming scans
# ---------------------------------------------------------------------------


def _header(path: str | Path) -> list[str]:
    frame = pd.read_csv(path, nrows=0)
    return [str(column) for column in frame.columns]


def _metric_from_filename(path: Path) -> str:
    return path.stem.upper()


def _reservoir_sample_add(
    reservoir: list[np.ndarray], total_values: int, flattened: np.ndarray
) -> int:
    """Append a bounded, deterministic sample slice from one chunk."""
    if total_values >= VALUE_SAMPLE_TARGET or flattened.size == 0:
        return total_values
    finite = flattened[np.isfinite(flattened)]
    if finite.size == 0:
        return total_values
    remaining = VALUE_SAMPLE_TARGET - total_values
    if finite.size > remaining:
        stride = max(1, finite.size // remaining)
        finite = finite[::stride][:remaining]
    reservoir.append(finite.astype(np.float32))
    return total_values + int(finite.size)


def scan_utilization_file(
    path: str | Path,
    *,
    cluster: str,
    metric: str,
    chunk_rows: int = FULL_CHUNK_ROWS,
    pairing: bool = False,
    native_window: bool = False,
    max_chunks: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream one utilization CSV and return ``(audit_record, pairing_payload)``."""
    resolved = resolve_path(path)
    columns = _header(resolved)
    device_columns = columns[1:]
    bytes_total = int(resolved.stat().st_size)
    parsed_devices = parse_gpu_device_columns(device_columns)
    is_gpu_wide = parsed_devices is not None
    node_devices = [parse_node_device(column) for column in device_columns]
    is_node_wide = all(item is not None for item in node_devices)
    if is_gpu_wide:
        kind = "gpu_wide"
        device_labels = [f"{node}-{index}" for node, index in parsed_devices]  # type: ignore[misc]
        device_nodes = [node for node, _ in parsed_devices]  # type: ignore[misc]
    elif is_node_wide:
        kind = "node_wide"
        device_labels = [str(item) for item in node_devices]
        device_nodes = list(device_labels)
    else:
        kind = "other_wide"
        device_labels = list(device_columns)
        device_nodes = list(device_columns)

    n_devices = len(device_columns)
    spec = METRIC_SPECS.get(metric, {})
    expected_range = spec.get("expected_range", (None, None))
    lower, upper = expected_range
    if n_devices > 1000:
        chunk_rows = min(chunk_rows, WIDE_CHUNK_ROWS)

    audit: dict[str, Any] = {
        "cluster": cluster,
        "metric": metric,
        "path": str(resolved),
        "bytes": bytes_total,
        "kind": kind,
        "device_columns": n_devices,
        "device_labels": device_labels,
        "device_nodes": sorted(set(device_nodes)),
        "semantic": spec.get("semantic", metric.lower()),
        "unit_guess": spec.get("unit_guess", "unverified"),
        "unit_verified": False,
        "expected_native_interval_seconds": 15.0 if kind in {"gpu_wide", "node_wide"} else None,
        "placeholder": bool(n_devices == 0),
    }
    if n_devices == 0:
        audit.update(
            {
                "rows": 0,
                "first_timestamp_utc": None,
                "last_timestamp_utc": None,
                "timestamp_offset_observed": observed_offset([""]),
                "native_interval": interval_statistics(np.array([], dtype=np.int64)),
                "missing_fraction_aggregate": None,
                "missing_fraction_min": None,
                "missing_fraction_median": None,
                "missing_fraction_max": None,
                "value_evidence": infer_unit_evidence(np.array([]), metric=metric, spec=spec),
                "out_of_range": {},
                "xid": None,
                "notes": "placeholder/empty file: no device columns; no continuous series fabricated",
            }
        )
        return audit, {"kind": "placeholder", "cluster": cluster, "metric": metric}

    nan_counts = np.zeros(n_devices, dtype=np.int64)
    value_min = np.full(n_devices, np.inf, dtype=np.float64)
    value_max = np.full(n_devices, -np.inf, dtype=np.float64)
    below_negative = np.zeros(n_devices, dtype=np.int64)
    above_upper = np.zeros(n_devices, dtype=np.int64)
    below_lower = np.zeros(n_devices, dtype=np.int64)
    nonzero_counts = np.zeros(n_devices, dtype=np.int64) if metric == "XID_ERRORS" else None
    rows = 0
    nat_timestamps = 0
    first_ts = None
    last_ts = None
    offset = None
    seen_timestamps: set[int] = set()
    duplicate_timestamps = 0
    out_of_order_rows = 0
    previous_last = None
    delta_counts: Counter[int] = Counter()
    timestamp_ns_parts: list[np.ndarray] = []  # bounded: only first/last samples matter
    reservoir: list[np.ndarray] = []
    sampled_values = 0
    format_kind = None
    interval_guess = None

    accumulator = BinAccumulator() if pairing else None
    if accumulator is not None:
        accumulator.devices = list(device_labels)
    window = None
    window_ns = None
    if native_window and is_gpu_wide and metric in GPU_PAIRING_METRICS:
        start_ns, length, step_seconds = _native_window_timestamps_ns(cluster)
        window = NativeWindowAccumulator(
            start_ns=start_ns, step_seconds=step_seconds, length=length, n_devices=n_devices
        )
        window_ns = (start_ns, length, step_seconds)

    try:
        reader = pd.read_csv(resolved, chunksize=chunk_rows)
    except (OSError, ValueError) as exc:  # pragma: no cover
        raise RuntimeError(f"cannot stream {resolved}: {exc}") from exc

    for chunk_index, chunk in enumerate(reader):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        if "Time" not in chunk.columns:
            raise RuntimeError(f"{resolved} has no `Time` column")
        if format_kind is None:
            format_kind = detect_timestamp_format(chunk["Time"].iloc[0])
            offset = observed_offset(chunk["Time"].head(5).tolist())
            first_raw = str(chunk["Time"].iloc[0])
            interval_guess = 15.0 if "+" in first_raw or "Z" in first_raw else None
        timestamps = parse_timestamps(chunk["Time"].tolist(), format_kind=format_kind)
        ns = timestamps.asi8.astype(np.int64)
        nat = int(np.count_nonzero(np.asarray(timestamps.isna())))
        nat_timestamps += nat
        valid = ~np.asarray(timestamps.isna())
        ns_valid = ns[valid]
        if len(ns_valid):
            if first_ts is None:
                first_ts = timestamps.min()
                last_ts = timestamps.max()
            else:
                first_ts = min(first_ts, timestamps.min())
                current_max = timestamps.max()
                if current_max > last_ts:
                    last_ts = current_max
            if previous_last is not None:
                # count file-order negatives across the chunk boundary as out-of-order rows
                pass
            deltas = np.diff(ns_valid) // 1_000_000_000
            if previous_last is not None and len(ns_valid):
                boundary = int((ns_valid[0] - previous_last) // 1_000_000_000)
                delta_counts[int(boundary)] += 1
            for value, count in Counter(int(item) for item in deltas).items():
                delta_counts[value] += int(count)
            out_of_order_rows += int(np.count_nonzero(deltas < 0))
            previous_last = int(ns_valid[-1])
            for value in ns_valid.tolist():
                if value in seen_timestamps:
                    duplicate_timestamps += 1
                else:
                    seen_timestamps.add(value)
            timestamp_ns_parts.append(ns_valid[:2])
            timestamp_ns_parts.append(ns_valid[-2:])
        rows += int(len(chunk))
        values = chunk.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
        if values.shape[1] != n_devices:
            raise RuntimeError(f"{resolved}: expected {n_devices} device columns, got {values.shape[1]}")
        nan_mask = np.isnan(values)
        finite_counts = (~nan_mask).sum(axis=0)
        nan_counts += nan_mask.sum(axis=0).astype(np.int64)
        current_min, current_max = _nan_min_max(values)
        value_min = np.fmin(value_min, current_min)
        value_max = np.fmax(value_max, current_max)
        negative_mask = values < 0
        below_negative += negative_mask.sum(axis=0).astype(np.int64)
        if lower is not None:
            below_lower += (values < lower).sum(axis=0).astype(np.int64)
        if upper is not None:
            above_upper += (values > upper).sum(axis=0).astype(np.int64)
        if nonzero_counts is not None:
            nonzero_counts += (values > 0).sum(axis=0).astype(np.int64)
        sampled_values = _reservoir_sample_add(
            reservoir, sampled_values, values[np.isfinite(values)]
        )
        if accumulator is not None:
            bins = _bin_indices(timestamps)
            good = ~np.asarray(timestamps.isna())
            if bool(good.any()):
                bin_ids = bins[good]
                chunk_values = values[good]
                order = np.argsort(bin_ids, kind="stable")
                bin_sorted = bin_ids[order]
                values_sorted = chunk_values[order]
                finite_sorted = np.isfinite(values_sorted)
                values_zero = np.where(finite_sorted, values_sorted, 0.0).astype(np.float32)
                unique_bins, first_index = np.unique(bin_sorted, return_index=True)
                sums = np.add.reduceat(values_zero, first_index, axis=0)
                counts = np.add.reduceat(finite_sorted.astype(np.uint16), first_index, axis=0)
                accumulator.add(unique_bins, sums, counts)
        if window is not None and window_ns is not None:
            start_ns, length, _ = window_ns
            window_mask = (ns >= start_ns) & (ns < start_ns + length * 15 * 1_000_000_000)
            if bool(window_mask.any()):
                window.add(ns[window_mask], values[window_mask])

    del reader
    observed_counts = rows - nan_counts
    missing_fraction = np.where(rows > 0, nan_counts / float(rows), np.nan)
    with np.errstate(invalid="ignore"):
        aggreg = float(np.nanmean(missing_fraction)) if rows else None
        minimum = float(np.nanmin(missing_fraction)) if rows else None
        median = float(np.nanmedian(missing_fraction)) if rows else None
        maximum = float(np.nanmax(missing_fraction)) if rows else None
    sample = np.concatenate(reservoir) if reservoir else np.array([], dtype=np.float32)
    first_text = first_ts.isoformat() if first_ts is not None else None
    last_text = last_ts.isoformat() if last_ts is not None else None
    span_seconds = (
        float((last_ts - first_ts).total_seconds()) if first_ts is not None and last_ts is not None else 0.0
    )
    native_interval: float | None = None
    if delta_counts:
        positive = [(key, value) for key, value in delta_counts.items() if key > 0]
        if positive:
            native_interval = float(max(positive, key=lambda item: item[1])[0])
        gap_count = int(
            sum(value for key, value in delta_counts.items() if native_interval and key > native_interval)
        )
        negative_or_zero = int(sum(value for key, value in delta_counts.items() if key <= 0))
        all_multiples = bool(
            native_interval
            and all(abs(key % native_interval) < 1e-9 for key in delta_counts if key > 0)
        )
        max_gap = float(max(delta_counts))
    else:
        gap_count = 0
        negative_or_zero = 0
        all_multiples = True
        max_gap = 0.0
    expected = 15.0 if kind in {"gpu_wide", "node_wide"} else None
    duplicate_rate = duplicate_timestamps / max(rows - nat_timestamps, 1)
    unique_timestamps = rows - nat_timestamps - duplicate_timestamps
    audit.update(
        {
            "rows": int(rows),
            "first_timestamp_utc": first_text,
            "last_timestamp_utc": last_text,
            "timestamp_offset_observed": offset,
            "duplicate_timestamps": int(duplicate_timestamps),
            "duplicate_timestamp_fraction": float(duplicate_rate),
            "out_of_order_rows": int(out_of_order_rows),
            "nat_timestamps": int(nat_timestamps),
            "unique_timestamps": int(unique_timestamps),
            "observed_span_seconds": span_seconds,
            "coverage_ratio_vs_15s_grid": (
                float((rows - nat_timestamps) * 15.0 / span_seconds)
                if span_seconds > 0 and expected == 15.0
                else None
            ),
            "native_interval": {
                "native_interval_seconds": native_interval,
                "native_interval_counts": {
                    str(int(key)): int(value) for key, value in delta_counts.most_common(12)
                },
                "distinct_deltas": len(delta_counts),
                "negative_or_zero_deltas": negative_or_zero,
                "gap_count": gap_count,
                "max_gap_seconds": max_gap,
                "all_deltas_multiples_of_native": all_multiples,
                "matches_expected_interval": (
                    None if expected is None or native_interval is None else bool(abs(native_interval - expected) < 1e-9)
                ),
                "evidence": "mode of consecutive-timestamp deltas over the full file",
            },
            "missing_fraction_aggregate": aggreg,
            "missing_fraction_min": minimum,
            "missing_fraction_median": median,
            "missing_fraction_max": maximum,
            "fully_missing_devices": int(np.count_nonzero(observed_counts == 0)),
            "rows_with_any_missing_fraction": float(nan_counts.sum() / max(rows * n_devices, 1)),
            "missing_fraction_per_device": [round(float(value), 6) for value in missing_fraction],
            "value_evidence": infer_unit_evidence(sample, metric=metric, spec=spec),
            "value_min": float(np.nanmin(value_min)) if np.isfinite(value_min).any() else None,
            "value_max": float(np.nanmax(value_max)) if np.isfinite(value_max).any() else None,
            "value_min_per_device": [
                None if not np.isfinite(value) else round(float(value), 6) for value in value_min
            ],
            "value_max_per_device": [
                None if not np.isfinite(value) else round(float(value), 6) for value in value_max
            ],
            "out_of_range": {
                "negative_values": int(below_negative.sum()),
                "devices_with_negative": int(np.count_nonzero(below_negative)),
                "below_expected_lower": int(below_lower.sum()) if lower is not None else None,
                "above_expected_upper": int(above_upper.sum()) if upper is not None else None,
                "expected_range": list(expected_range) if expected_range != (None, None) else None,
                "values_clipped": False,
            },
            "xid": (
                {
                    "nonzero_values": int(nonzero_counts.sum()),
                    "devices_with_errors": int(np.count_nonzero(nonzero_counts)),
                    "nonzero_per_device": [int(value) for value in nonzero_counts],
                }
                if nonzero_counts is not None
                else None
            ),
            "placeholder": bool(rows == 0),
            "notes": (
                "placeholder/empty file: header only; no continuous series fabricated"
                if rows == 0
                else ""
            ),
        }
    )
    pairing_payload: dict[str, Any] = {"kind": kind}
    if accumulator is not None:
        pairing_payload = {
            "kind": kind,
            "cluster": cluster,
            "metric": metric,
            "devices": device_labels,
            "device_nodes": sorted(set(device_nodes)),
            "accumulator": accumulator,
            "native_window": window,
            "timestamps_ns_sorted": np.array(sorted(seen_timestamps), dtype=np.int64),
        }
    return audit, pairing_payload


class IpmiNodeAccumulator:
    """Sparse per-(15-minute bin, node IP) IPMI power accumulator."""

    def __init__(self) -> None:
        # (absolute_bin, node_ip) -> [sum, count]; Sys_Total_Power==0 rows excluded upstream.
        self.power: dict[tuple[int, str], list[float]] = {}

    def add_grouped(self, grouped: pd.DataFrame) -> None:
        for (bin_id, node), row in grouped.iterrows():
            key = (int(bin_id), str(node))
            entry = self.power.get(key)
            total = float(row["sum"])
            count = int(row["count"])
            if entry is None:
                self.power[key] = [total, float(count)]
            else:
                entry[0] += total
                entry[1] += count

    def merge(self, other: "IpmiNodeAccumulator") -> None:
        for key, (total, count) in other.power.items():
            entry = self.power.get(key)
            if entry is None:
                self.power[key] = [total, count]
            else:
                entry[0] += total
                entry[1] += count

    def nodes(self) -> list[str]:
        return sorted({node for _, node in self.power})

    def to_bin_accumulator(self, nodes: Sequence[str] | None = None) -> "BinAccumulator":
        ordered = list(nodes) if nodes is not None else self.nodes()
        accumulator = BinAccumulator()
        accumulator.devices = ordered
        index = {name: position for position, name in enumerate(ordered)}
        for (bin_id, node), (total, count) in self.power.items():
            if node in index:
                accumulator_add_columns(accumulator, bin_id, [index[node]], total, int(count))
        return accumulator


def _ipmi_grouped_chunk(
    bins: np.ndarray,
    node_ips: Sequence[str | None],
    power: np.ndarray,
    valid: np.ndarray,
) -> pd.DataFrame:
    good = valid & np.isfinite(power) & (power > 0)
    node_array = np.asarray(node_ips, dtype=object)
    good &= node_array != None  # noqa: E711 - elementwise object comparison
    if not bool(good.any()):
        return pd.DataFrame(columns=["bin", "node", "power"])
    frame = pd.DataFrame(
        {
            "bin": bins[good],
            "node": node_array[good].astype(str),
            "power": power[good].astype(np.float64),
        }
    )
    return frame.groupby(["bin", "node"], sort=False)["power"].agg(["sum", "count"])


def scan_ipmi_file(
    path: str | Path,
    *,
    chunk_rows: int = NARROW_CHUNK_ROWS,
    pairing: bool = True,
    max_chunks: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream one Seren IPMI power CSV (long format: one row per node per 5-min sample)."""
    resolved = resolve_path(path)
    columns = _header(resolved)
    if "node" not in columns or "Time" not in columns:
        raise RuntimeError(f"{resolved} does not look like an IPMI long-format power file")
    numeric_columns = [column for column in columns if column not in {"Time", "node"}]
    power_columns = [
        column
        for column in numeric_columns
        if any(token in column.upper() for token in ("POWER", "PIN", "POUT"))
    ]
    power_index = numeric_columns.index(IPMI_POWER_COLUMN) if IPMI_POWER_COLUMN in numeric_columns else None
    audit: dict[str, Any] = {
        "cluster": "seren",
        "metric": "IPMI_POWER",
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "kind": "ipmi_long",
        "columns": columns,
        "numeric_columns": numeric_columns,
        "power_columns": power_columns,
        "semantic": "server_total_power",
        "unit_guess": "W_unverified",
        "unit_verified": False,
        "expected_native_interval_seconds": 300.0,
    }
    n_numeric = len(numeric_columns)
    rows = 0
    nat = 0
    first_ts = last_ts = None
    offset = None
    format_kind = None
    nodes: set[str] = set()
    node_ips: set[str] = set()
    duplicate_within_chunk = 0
    delta_counter: Counter[int] = Counter()
    last_timestamp_per_node: dict[str, int] = {}
    nan_counts = np.zeros(n_numeric, dtype=np.int64)
    value_min = np.full(n_numeric, np.inf, dtype=np.float64)
    value_max = np.full(n_numeric, -np.inf, dtype=np.float64)
    negative = np.zeros(n_numeric, dtype=np.int64)
    zero_counts = np.zeros(n_numeric, dtype=np.int64)
    sample_parts: list[np.ndarray] = []
    sampled = 0
    accumulator = IpmiNodeAccumulator() if pairing else None
    ipmi_node_index: dict[str, int] = {}
    ipmi_bin_sums: dict[int, np.ndarray] = {}
    ipmi_bin_counts: dict[int, np.ndarray] = {}

    for chunk_index, chunk in enumerate(pd.read_csv(resolved, chunksize=chunk_rows)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        if format_kind is None:
            format_kind = detect_timestamp_format(chunk["Time"].iloc[0])
            offset = observed_offset(chunk["Time"].head(5).tolist())
        timestamps = parse_timestamps(chunk["Time"].tolist(), format_kind=format_kind)
        ns = timestamps.asi8.astype(np.int64)
        nat += int(np.count_nonzero(np.asarray(timestamps.isna())))
        valid = ~np.asarray(timestamps.isna())
        if bool(valid.any()):
            current_min = timestamps.min()
            current_max = timestamps.max()
            first_ts = current_min if first_ts is None else min(first_ts, current_min)
            last_ts = current_max if last_ts is None else max(last_ts, current_max)
        node_names = chunk["node"].astype(str).tolist()
        nodes.update(node_names)
        mapped_nodes = [ipmi_node_to_ip(name) for name in node_names]
        node_ips.update(name for name in mapped_nodes if name)
        values = chunk[numeric_columns].to_numpy(dtype=np.float32)
        nan_counts += np.isnan(values).sum(axis=0).astype(np.int64)
        chunk_min, chunk_max = _nan_min_max(values)
        value_min = np.fmin(value_min, chunk_min)
        value_max = np.fmax(value_max, chunk_max)
        negative += (values < 0).sum(axis=0).astype(np.int64)
        zero_counts += (values == 0).sum(axis=0).astype(np.int64)
        sampled = _reservoir_sample_add(sample_parts, sampled, values[np.isfinite(values)])
        # Per-node native interval evidence (carry one tail row per node across chunks).
        row_indices = np.flatnonzero(np.asarray(valid))
        current = pd.DataFrame(
            {
                "node": np.asarray(node_names, dtype=object)[row_indices],
                "ts": ns[row_indices],
            }
        )
        if len(current):
            if last_timestamp_per_node:
                tail_rows = pd.DataFrame(
                    {
                        "node": list(last_timestamp_per_node),
                        "ts": [last_timestamp_per_node[node] for node in last_timestamp_per_node],
                    }
                )
                combined = pd.concat([tail_rows, current], ignore_index=True)
            else:
                combined = current
            deltas = combined.groupby("node", sort=False)["ts"].diff().dropna()
            for value, count in Counter(int(item) for item in (deltas // 1_000_000_000)).items():
                delta_counter[value] += int(count)
            last_timestamp_per_node = {
                str(node): int(group["ts"].iloc[-1]) for node, group in current.groupby("node", sort=False)
            }
            duplicate_within_chunk += int(current.duplicated(subset=["node", "ts"]).sum())
        if pairing and power_index is not None:
            bins = _bin_indices(timestamps)
            node_codes, node_names = pd.factorize(
                np.asarray(mapped_nodes, dtype=object), sort=False, use_na_sentinel=True
            )
            for name in node_names.tolist():
                if name is None:
                    continue
                if str(name) not in ipmi_node_index:
                    ipmi_node_index[str(name)] = len(ipmi_node_index)
            global_codes = np.array(
                [ipmi_node_index.get(str(name), -1) for name in node_names.tolist()], dtype=np.int64
            )[node_codes]
            power_values = values[:, power_index]
            good = valid & np.isfinite(power_values) & (power_values > 0) & (node_codes >= 0)
            if bool(good.any()):
                good_bins = bins[good]
                good_codes = global_codes[good]
                unique_bins, inverse = np.unique(good_bins, return_inverse=True)
                width = len(ipmi_node_index)
                sums_chunk = np.zeros((len(unique_bins), width), dtype=np.float64)
                counts_chunk = np.zeros((len(unique_bins), width), dtype=np.int64)
                np.add.at(sums_chunk, (inverse, good_codes), power_values[good].astype(np.float64))
                np.add.at(counts_chunk, (inverse, good_codes), 1)
                for row_index, bin_id in enumerate(unique_bins.tolist()):
                    total_array = ipmi_bin_sums.get(int(bin_id))
                    count_array = ipmi_bin_counts.get(int(bin_id))
                    if total_array is None or len(total_array) != width:
                        padded = np.zeros(width, dtype=np.float64)
                        if total_array is not None:
                            padded[: len(total_array)] = total_array
                        padded_counts = np.zeros(width, dtype=np.int64)
                        if count_array is not None:
                            padded_counts[: len(count_array)] = count_array
                        total_array, count_array = padded, padded_counts
                        ipmi_bin_sums[int(bin_id)] = total_array
                        ipmi_bin_counts[int(bin_id)] = count_array
                    total_array[: len(sums_chunk[row_index])] += sums_chunk[row_index]
                    count_array[: len(counts_chunk[row_index])] += counts_chunk[row_index]
        rows += int(len(chunk))
    sample = np.concatenate(sample_parts) if sample_parts else np.array([], dtype=np.float32)
    span_seconds = (
        float((last_ts - first_ts).total_seconds()) if first_ts is not None and last_ts is not None else 0.0
    )
    nan_fraction = nan_counts / max(rows, 1)
    native_interval = None
    gap_count = 0
    all_multiples = True
    max_gap = 0.0
    negative_or_zero = 0
    if delta_counter:
        positive = [(key, value) for key, value in delta_counter.items() if key > 0]
        if positive:
            native_interval = float(max(positive, key=lambda item: item[1])[0])
        gap_count = int(
            sum(value for key, value in delta_counter.items() if native_interval and key > native_interval)
        )
        negative_or_zero = int(sum(value for key, value in delta_counter.items() if key <= 0))
        all_multiples = bool(
            native_interval
            and all(abs(key % native_interval) < 1e-9 for key in delta_counter if key > 0)
        )
        max_gap = float(max(delta_counter))
    coverage = (
        float((rows - nat) * 300.0 / (span_seconds * max(len(nodes), 1)))
        if span_seconds > 0 and len(nodes)
        else None
    )
    audit.update(
        {
            "rows": int(rows),
            "node_count": len(nodes),
            "node_ip_count": len(node_ips),
            "node_names": sorted(nodes),
            "node_ips": sorted(node_ips),
            "first_timestamp_utc": first_ts.isoformat() if first_ts is not None else None,
            "last_timestamp_utc": last_ts.isoformat() if last_ts is not None else None,
            "timestamp_offset_observed": offset,
            "duplicate_node_timestamp_pairs_within_chunks": int(duplicate_within_chunk),
            "nat_timestamps": int(nat),
            "observed_span_seconds": span_seconds,
            "coverage_ratio_vs_300s_grid": coverage,
            "native_interval": {
                "native_interval_seconds": native_interval,
                "native_interval_counts": {
                    str(int(key)): int(value) for key, value in delta_counter.most_common(8)
                },
                "gap_count": gap_count,
                "max_gap_seconds": max_gap,
                "negative_or_zero_deltas": negative_or_zero,
                "all_deltas_multiples_of_native": all_multiples,
                "matches_expected_interval": (
                    None if native_interval is None else bool(abs(native_interval - 300.0) < 1e-9)
                ),
                "evidence": "mode of consecutive per-node timestamp deltas over the full file",
            },
            "missing_fraction_by_field": {
                column: float(value) for column, value in zip(numeric_columns, nan_fraction)
            },
            "negative_values_by_field": {
                column: int(value) for column, value in zip(numeric_columns, negative)
            },
            "zero_values_by_field": {
                column: int(value) for column, value in zip(numeric_columns, zero_counts)
            },
            "value_evidence": {
                IPMI_POWER_COLUMN: infer_unit_evidence(sample, metric="IPMI_POWER")
            },
            "value_min_by_field": {
                column: (None if not np.isfinite(value) else float(value))
                for column, value in zip(numeric_columns, value_min)
            },
            "value_max_by_field": {
                column: (None if not np.isfinite(value) else float(value))
                for column, value in zip(numeric_columns, value_max)
            },
            "zero_sys_total_power_rows": (
                int(zero_counts[numeric_columns.index(IPMI_POWER_COLUMN)])
                if IPMI_POWER_COLUMN in numeric_columns
                else None
            ),
            "invalid_power_note": (
                "Sys_Total_Power == 0 is invalid/offline: counted, excluded from power statistics "
                "and from pairing sums; never treated as a real 0 W observation."
            ),
        }
    )
    payload: dict[str, Any] = {"kind": "ipmi_long", "cluster": "seren", "metric": "IPMI_POWER"}
    bin_accumulator = None
    if pairing and ipmi_node_index:
        final_width = len(ipmi_node_index)
        ordered_names = [name for name, _ in sorted(ipmi_node_index.items(), key=lambda item: item[1])]
        bin_accumulator = BinAccumulator()
        bin_accumulator.devices = list(ordered_names)
        for bin_id, total_array in ipmi_bin_sums.items():
            padded_sum = np.zeros(final_width, dtype=np.float32)
            padded_sum[: len(total_array)] = total_array
            padded_count = np.zeros(final_width, dtype=np.uint16)
            raw_counts = ipmi_bin_counts[bin_id]
            padded_count[: len(raw_counts)] = raw_counts.astype(np.uint16)
            bin_accumulator.sums[int(bin_id)] = padded_sum
            bin_accumulator.counts[int(bin_id)] = padded_count
    if pairing:
        payload.update(
            {
                "devices": list(bin_accumulator.devices) if bin_accumulator is not None else sorted(node_ips),
                "bin_accumulator": bin_accumulator,
                "bounded_debug": max_chunks is not None,
            }
        )
    return audit, payload


def merge_bin_accumulators(accumulators: Sequence["BinAccumulator"]) -> "BinAccumulator":
    """Merge several per-file bin accumulators onto the union of their devices."""
    devices: list[str] = []
    index: dict[str, int] = {}
    for accumulator in accumulators:
        if accumulator is None:
            continue
        for name in accumulator.devices:
            if name not in index:
                index[name] = len(devices)
                devices.append(name)
    merged = BinAccumulator()
    merged.devices = devices
    width = len(devices)
    for accumulator in accumulators:
        if accumulator is None or not accumulator.devices:
            continue
        column_index = np.array([index[name] for name in accumulator.devices], dtype=np.int64)
        for bin_id, sums in accumulator.sums.items():
            total = merged.sums.get(bin_id)
            if total is None:
                total = np.zeros(width, dtype=np.float32)
                merged.sums[bin_id] = total
                merged.counts[bin_id] = np.zeros(width, dtype=np.uint16)
            total[column_index] += sums
            merged.counts[bin_id][column_index] += accumulator.counts[bin_id]
    return merged


def accumulator_device_index(accumulator: BinAccumulator, names: Sequence[str]) -> dict[str, int]:
    existing = {name: index for index, name in enumerate(accumulator.devices)}
    for name in names:
        if name not in existing:
            existing[name] = len(accumulator.devices)
            accumulator.devices.append(name)
    return existing


def accumulator_add_columns(
    accumulator: BinAccumulator, bin_id: int, column_indices: Sequence[int], total: float, count: int
) -> None:
    width = len(accumulator.devices)
    sums = accumulator.sums.get(bin_id)
    counts = accumulator.counts.get(bin_id)
    if sums is None or len(sums) != width:
        padded = np.zeros(width, dtype=np.float32)
        if sums is not None:
            padded[: len(sums)] = sums
        sums = padded
        padded_counts = np.zeros(width, dtype=np.uint16)
        if counts is not None:
            padded_counts[: len(counts)] = counts
        accumulator.sums[bin_id] = sums
        accumulator.counts[bin_id] = padded_counts
    for column in column_indices:
        sums[column] += float(total)
        accumulator.counts[bin_id][column] += int(count)


def _pair_dense_accumulators(
    *,
    cluster: str,
    level: str,
    metric_a: str,
    accumulator_a: BinAccumulator,
    metric_b: str,
    accumulator_b: BinAccumulator,
    native_a: NativeWindowAccumulator | None = None,
    native_b: NativeWindowAccumulator | None = None,
) -> dict[str, Any]:
    devices_a = list(accumulator_a.devices)
    devices_b = list(accumulator_b.devices)
    index_a = {name: index for index, name in enumerate(devices_a)}
    index_b = {name: index for index, name in enumerate(devices_b)}
    common = sorted(set(devices_a) & set(devices_b))
    report: dict[str, Any] = {
        "cluster": cluster,
        "level": level,
        "metric_a": metric_a,
        "metric_b": metric_b,
        "devices_a": len(devices_a),
        "devices_b": len(devices_b),
        "common_devices": len(common),
        "unmatched_devices_a": len(set(devices_a) - set(devices_b)),
        "unmatched_devices_b": len(set(devices_b) - set(devices_a)),
        "matched_devices": 0,
        "matched_device_slots": 0,
        "union_device_slots": 0,
        "match_rate": None,
        "joint_span_first_utc": None,
        "joint_span_last_utc": None,
        "native_window": None,
    }
    dense_a = accumulator_a.dense()
    dense_b = accumulator_b.dense()
    if len(common) and len(dense_a["bins"]) and len(dense_b["bins"]):
        ia = np.array([index_a[name] for name in common], dtype=np.int64)
        ib = np.array([index_b[name] for name in common], dtype=np.int64)
        bins_common = np.intersect1d(dense_a["bins"], dense_b["bins"])
        if len(bins_common):
            rows_a = np.searchsorted(dense_a["bins"], bins_common)
            rows_b = np.searchsorted(dense_b["bins"], bins_common)
            observed_a = dense_a["counts"][rows_a][:, ia] > 0
            observed_b = dense_b["counts"][rows_b][:, ib] > 0
            matched = observed_a & observed_b
            union = observed_a | observed_b
            per_device = matched.sum(axis=0)
            any_match = matched.any(axis=1)
            matched_bins = bins_common[any_match]
            report.update(
                {
                    "matched_devices": int(np.count_nonzero(per_device > 0)),
                    "matched_device_slots": int(matched.sum()),
                    "union_device_slots": int(union.sum()),
                    "match_rate": float(matched.sum() / union.sum()) if union.sum() else None,
                    "matched_slots_per_device_median": float(np.median(per_device)),
                    "common_bins": int(len(bins_common)),
                    "joint_span_first_utc": (
                        pd.Timestamp(int(matched_bins[0]) * BIN_SECONDS, unit="s", tz="UTC").isoformat()
                        if len(matched_bins)
                        else None
                    ),
                    "joint_span_last_utc": (
                        pd.Timestamp(int(matched_bins[-1]) * BIN_SECONDS, unit="s", tz="UTC").isoformat()
                        if len(matched_bins)
                        else None
                    ),
                }
            )
    if native_a is not None and native_b is not None and common:
        ia = np.array([index_a[name] for name in common], dtype=np.int64)
        ib = np.array([index_b[name] for name in common], dtype=np.int64)
        if native_a.observed is not None and native_b.observed is not None:
            obs_a = native_a.observed[:, ia]
            obs_b = native_b.observed[:, ib]
            matched_native = obs_a & obs_b
            union_native = obs_a | obs_b
            per_device = matched_native.sum(axis=0)
            report["native_window"] = {
                "window": list(NATIVE_WINDOWS[cluster]),
                "native_step_seconds": 15.0,
                "matched_devices": int(np.count_nonzero(per_device > 0)),
                "matched_device_instants": int(matched_native.sum()),
                "union_device_instants": int(union_native.sum()),
                "match_rate": float(matched_native.sum() / union_native.sum()) if union_native.sum() else None,
                "note": "bounded 24 h same-instant check on the common 15 s native grid",
            }
    return report


def _pair_native_timestamp_sets(
    times_a: np.ndarray,
    times_b: np.ndarray,
    *,
    cluster: str,
    metric_a: str,
    metric_b: str,
) -> dict[str, Any]:
    left = np.asarray(times_a, dtype=np.int64)
    right = np.asarray(times_b, dtype=np.int64)
    common = np.intersect1d(left, right)
    return {
        "cluster": cluster,
        "metric_a": metric_a,
        "metric_b": metric_b,
        "timestamps_a": int(len(left)),
        "timestamps_b": int(len(right)),
        "exact_common_instants": int(len(common)),
        "first_common_utc": (
            pd.Timestamp(int(common[0]), tz="UTC").isoformat() if len(common) else None
        ),
        "last_common_utc": (
            pd.Timestamp(int(common[-1]), tz="UTC").isoformat() if len(common) else None
        ),
    }


# ---------------------------------------------------------------------------
# job trace audit
# ---------------------------------------------------------------------------


def _parse_job_times(series: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce"))


def audit_job_trace(path: str | Path, *, cluster: str) -> tuple[dict[str, Any], pd.DataFrame]:
    resolved = resolve_path(path)
    frame = pd.read_csv(resolved)
    audit: dict[str, Any] = {
        "cluster": cluster,
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(key): str(value) for key, value in frame.dtypes.items()},
        "field_completeness": {},
        "time_ranges_utc": {},
        "raw_timestamp_offsets": {},
        "state_distribution": {},
        "type_distribution": {},
        "plausibility": {},
        "prediction_field_classification": {},
        "no_node_identifiers": not any(
            column.lower() in {"node_id", "node", "nodes", "node_name"} for column in frame.columns
        ),
        "limitation": (
            "The job traces carry no node IDs (only requested node_num / gpu_num), so a job "
            "cannot be linked to an individual server or GPU. Only cluster-level temporal "
            "linkage is defensible."
        ),
    }
    for column in list(frame.columns):
        if column.endswith("_time"):
            parsed = _parse_job_times(frame[column])
            non_null = int(parsed.notna().sum())
            raw = frame[column].dropna().astype(str)
            offset = None
            if len(raw):
                match = _OFFSET_RE.search(raw.iloc[0])
                offset = match.group(1) if match else ("CST" if "CST" in raw.iloc[0] else None)
            audit["raw_timestamp_offsets"][column] = offset
            audit["time_ranges_utc"][column] = {
                "non_null": non_null,
                "min": parsed.min().isoformat() if non_null else None,
                "max": parsed.max().isoformat() if non_null else None,
            }
        audit["field_completeness"][column] = {
            "non_null": int(frame[column].notna().sum()),
            "null_fraction": float(frame[column].isna().mean()),
        }
    for column in ("state", "type"):
        if column in frame:
            counts = frame[column].value_counts(dropna=False)
            audit[f"{column}_distribution"] = {str(key): int(value) for key, value in counts.items()}
    submit = _parse_job_times(frame["submit_time"]) if "submit_time" in frame else None
    start = _parse_job_times(frame["start_time"]) if "start_time" in frame else None
    end = _parse_job_times(frame["end_time"]) if "end_time" in frame else None
    duration = pd.to_numeric(frame.get("duration"), errors="coerce")
    gpu_time = pd.to_numeric(frame.get("gpu_time"), errors="coerce")
    gpu_num = pd.to_numeric(frame.get("gpu_num"), errors="coerce")
    cpu_num = pd.to_numeric(frame.get("cpu_num"), errors="coerce")
    node_num = pd.to_numeric(frame.get("node_num"), errors="coerce")
    reported_queue = pd.to_numeric(frame.get("queue"), errors="coerce")
    computed_queue = (start - submit).total_seconds() if start is not None and submit is not None else None
    derived_duration = (end - start).total_seconds() if end is not None and start is not None else None
    derived_gpu_time = duration * gpu_num if duration is not None and gpu_num is not None else None
    audit["plausibility"] = {
        "duration_negative_or_zero": int((duration <= 0).sum()) if duration is not None else None,
        "duration_negative": int((duration < 0).sum()) if duration is not None else None,
        "duration_zero": int((duration == 0).sum()) if duration is not None else None,
        "end_before_start_rows": int((end < start).sum()) if end is not None and start is not None else None,
        "negative_queue_rows": int((start < submit).sum()) if start is not None and submit is not None else None,
        "queue_recomputation_mismatch_rows": (
            int(((computed_queue - reported_queue).abs() > 1.0).sum())
            if computed_queue is not None and reported_queue is not None
            else None
        ),
        "duration_recomputation_mismatch_rows": (
            int(((derived_duration - duration).abs() > 1.0).sum())
            if derived_duration is not None and duration is not None
            else None
        ),
        "gpu_time_recomputation_mismatch_rows": (
            int(((derived_gpu_time - gpu_time).abs() > 1.0).sum())
            if derived_gpu_time is not None and gpu_time is not None
            else None
        ),
        "gpu_jobs": int((gpu_num > 0).sum()) if gpu_num is not None else None,
        "cpu_only_jobs": int((gpu_num == 0).sum()) if gpu_num is not None else None,
        "node_num_min": int(node_num.min()) if node_num is not None else None,
        "node_num_max": int(node_num.max()) if node_num is not None else None,
        "gpu_num_min": int(gpu_num.min()) if gpu_num is not None else None,
        "gpu_num_max": int(gpu_num.max()) if gpu_num is not None else None,
        "cpu_num_max": int(cpu_num.max()) if cpu_num is not None else None,
        "queue_seconds_max": float(reported_queue.max()) if reported_queue is not None else None,
        "duration_seconds_max": float(duration.max()) if duration is not None else None,
    }
    for column in frame.columns:
        entry = JOB_FIELD_CLASSIFICATION.get(
            column,
            {
                "known_before_origin": False,
                "timing": "unclassified_review",
                "reason": "not in the audited whitelist: must not be used as a prediction input before review",
            },
        )
        audit["prediction_field_classification"][column] = dict(entry)
    audit["pre_origin_fields"] = sorted(
        column
        for column, entry in audit["prediction_field_classification"].items()
        if entry["known_before_origin"]
    )
    audit["post_hoc_fields"] = sorted(
        column
        for column, entry in audit["prediction_field_classification"].items()
        if not entry["known_before_origin"]
    )
    return audit, frame


def job_bin_activity(frame: pd.DataFrame, *, bin_series: np.ndarray) -> dict[str, Any]:
    nat_sentinel = np.iinfo(np.int64).min
    start = _parse_job_times(frame["start_time"]).asi8 // (BIN_SECONDS * 1_000_000_000)
    end = _parse_job_times(frame["end_time"]).asi8 // (BIN_SECONDS * 1_000_000_000)
    submit = _parse_job_times(frame["submit_time"]).asi8 // (BIN_SECONDS * 1_000_000_000)
    weights = pd.to_numeric(frame["gpu_num"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    valid = (start != nat_sentinel) & (end != nat_sentinel) & (end >= start)
    start = start[valid].astype(np.int64)
    end = end[valid].astype(np.int64)
    weights = weights[valid]
    bins = np.asarray(bin_series, dtype=np.int64)
    low = int(min(start.min(), bins.min())) if len(start) else int(bins.min())
    high = int(max(end.max(), bins.max())) if len(end) else int(bins.max())
    span = high - low + 2
    difference = np.zeros(span + 1, dtype=np.float64)
    np.add.at(difference, start - low, weights)
    np.add.at(difference, end - low, -weights)
    active = np.cumsum(difference)[:span]
    submitted = np.zeros(span, dtype=np.int64)
    submit_ok = (submit != nat_sentinel)
    submit_valid = submit[submit_ok].astype(np.int64)
    submit_valid = submit_valid[(submit_valid >= low) & (submit_valid < high)]
    np.add.at(submitted, submit_valid - low, 1)
    return {
        "active_gpu": active[bins - low],
        "submitted_jobs": submitted[bins - low],
        "job_span_first_utc": pd.Timestamp(int(low) * BIN_SECONDS, unit="s", tz="UTC").isoformat(),
        "job_span_last_utc": pd.Timestamp(int(high) * BIN_SECONDS, unit="s", tz="UTC").isoformat(),
        "valid_jobs": int(valid.sum()),
        "open_or_invalid_jobs_excluded": int((~valid).sum()),
    }


def cluster_temporal_linkage(
    *,
    cluster: str,
    gpu_util_accumulator: BinAccumulator,
    job_frame: pd.DataFrame,
    job_audit: Mapping[str, Any],
) -> dict[str, Any]:
    bins, means = gpu_util_accumulator.mean()
    observed = np.isfinite(means).any(axis=1) if means.size else np.zeros(0, dtype=bool)
    util_bins = bins[observed]
    util_mean = np.nanmean(means[observed], axis=1) if bool(observed.any()) else np.array([])
    if len(util_bins) == 0:
        return {"cluster": cluster, "available": False, "reason": "no observed GPU utilization bins"}
    activity = job_bin_activity(job_frame, bin_series=util_bins)
    active = activity["active_gpu"]
    valid = np.isfinite(active)
    correlation = None
    if valid.sum() > 2 and np.nanstd(util_mean[valid]) > 0 and np.nanstd(active[valid]) > 0:
        correlation = float(np.corrcoef(util_mean[valid], active[valid])[0, 1])
    return {
        "cluster": cluster,
        "available": True,
        "level": "cluster",
        "no_node_ids": True,
        "n_bins": int(len(util_bins)),
        "overlap_bins": int(valid.sum()),
        "util_span_utc": [
            pd.Timestamp(int(util_bins[0]) * BIN_SECONDS, unit="s", tz="UTC").isoformat(),
            pd.Timestamp(int(util_bins[-1]) * BIN_SECONDS, unit="s", tz="UTC").isoformat(),
        ],
        "job_span_utc": [
            job_audit["time_ranges_utc"]["submit_time"]["min"],
            job_audit["time_ranges_utc"]["end_time"]["max"],
        ],
        "mean_gpu_utilization": float(np.nanmean(util_mean)),
        "active_gpu_jobs_mean": float(np.nanmean(active)),
        "active_gpu_jobs_max": float(np.nanmax(active)),
        "pearson_active_gpu_vs_mean_util": correlation,
        "limitation": (
            "cluster-level temporal linkage only: job traces have no node IDs, so no per-server "
            "job<->metric pairing is possible."
        ),
    }


def _offset_hours(text: str | None) -> float | None:
    if not text:
        return None
    if "CST" in text:
        return 8.0
    match = re.match(r"([+-])(\d{2}):(\d{2})", text)
    if not match:
        return None
    sign = 1.0 if match.group(1) == "+" else -1.0
    return sign * (int(match.group(2)) + int(match.group(3)) / 60.0)


def timezone_impact(
    *, cluster: str, job_audit: Mapping[str, Any], utilization_offsets: Mapping[str, str | None]
) -> dict[str, Any]:
    job_offsets = {
        column: value
        for column, value in job_audit.get("raw_timestamp_offsets", {}).items()
        if value is not None
    }
    job_hours = sorted(
        {_offset_hours(value) for value in job_offsets.values() if _offset_hours(value) is not None}
    )
    util_hours = sorted(
        {_offset_hours(value) for value in utilization_offsets.values() if _offset_hours(value) is not None}
    )
    result: dict[str, Any] = {
        "cluster": cluster,
        "job_trace_offsets": job_offsets,
        "utilization_offsets": dict(utilization_offsets),
        "job_offset_hours": job_hours,
        "utilization_offset_hours": util_hours,
        "rule": "parse every timestamp with its explicit offset, then convert to UTC before comparison",
    }
    if job_hours and util_hours:
        shift = util_hours[0] - job_hours[0]
        result["shift_hours_if_naively_compared"] = float(shift)
        result["impact_if_ignored"] = (
            f"dropping the offsets would misalign the {cluster} job trace against utilization "
            f"files by {abs(shift):.1f} hours"
            if abs(shift) > 1e-9
            else "no shift: job trace and utilization share the same explicit offset"
        )
    return result


def pair_gpu_metrics(
    cluster: str, payloads: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Pair DCGM ``POWER_USAGE`` with the other GPU metrics **within one cluster**.

    A cross-cluster join is rejected outright: the two clusters are separate systems and
    their device sets / sampling windows are not comparable.
    """
    base = payloads.get(GPU_PAIRING_BASE)
    if base is None or base.get("accumulator") is None:
        return []
    if base.get("cluster") != cluster:
        raise ValueError(
            f"cross-cluster pairing rejected: base metric belongs to {base.get('cluster')!r}, "
            f"expected {cluster!r}"
        )
    pairs: list[dict[str, Any]] = []
    for metric in GPU_PAIRING_METRICS:
        if metric == GPU_PAIRING_BASE:
            continue
        other = payloads.get(metric)
        if other is None or other.get("accumulator") is None:
            continue
        if other.get("cluster") != cluster:
            raise ValueError(
                f"cross-cluster pairing rejected: {metric} belongs to {other.get('cluster')!r}, "
                f"expected {cluster!r}"
            )
        pair = _pair_dense_accumulators(
            cluster=cluster,
            level="gpu",
            metric_a=GPU_PAIRING_BASE,
            accumulator_a=base["accumulator"],
            metric_b=metric,
            accumulator_b=other["accumulator"],
            native_a=base.get("native_window"),
            native_b=other.get("native_window"),
        )
        pair["exact_native_timestamp_intersection"] = _pair_native_timestamp_sets(
            base["timestamps_ns_sorted"],
            other["timestamps_ns_sorted"],
            cluster=cluster,
            metric_a=GPU_PAIRING_BASE,
            metric_b=metric,
        )
        pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _cluster_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sum(int(record.get("rows") or 0) for record in records)
    nan_values = sum(int(record.get("nan_values") or 0) for record in records)
    values_total = sum(int(record.get("values_total") or 0) for record in records)
    placeholders = [record["metric"] for record in records if record.get("placeholder")]
    intervals: dict[str, list[str]] = {}
    for record in records:
        interval = (record.get("native_interval") or {}).get("native_interval_seconds")
        intervals.setdefault(str(interval), []).append(record["metric"])
    return {
        "files": len(records),
        "rows": rows,
        "placeholder_files": sorted(placeholders),
        "nan_values": nan_values,
        "values_total": values_total,
        "missing_fraction_overall": (nan_values / values_total) if values_total else None,
        "native_interval_seconds_groups": {key: sorted(value) for key, value in sorted(intervals.items())},
        "xid_nonzero_values": sum(
            int((record.get("xid") or {}).get("nonzero_values") or 0) for record in records
        ),
        "range_violation_negative_values": sum(
            int((record.get("out_of_range") or {}).get("negative_values") or 0) for record in records
        ),
    }


def run_audit(
    *,
    clusters: Sequence[str] = CLUSTERS,
    chunk_rows: int | None = None,
    include_ipmi: bool = True,
    include_jobs: bool = True,
    skip_pairing: bool = False,
    max_chunks: int | None = None,
    metrics: Sequence[str] | None = None,
    progress: Any = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the AcmeTrace ops/pairing/job audit and return the three payloads.

    ``metrics`` optionally restricts the utilization-metric scan to an explicit whitelist
    (upper-case metric names such as ``NODE_CPU_UTILIZATION``).  The default scans every
    file.  A restricted run is a *partial* audit: metrics outside the whitelist are not
    audited in that invocation and this limitation is recorded in the payload notes.
    """
    ops: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "dataset": "AcmeTrace",
        "raw_root": str(resolve_path(RAW_ROOT)),
        "clusters": {},
        "notes": [
            "REAL data only: every file below E:/Data/AcmeTrace is streamed read-only in chunks.",
            "unit_verified is false for every metric: local files do not document units; the "
            "reported unit is an inference backed by magnitude/range evidence.",
            "missing values stay missing (NaN); nothing is imputed or fabricated.",
        ],
    }
    pairing: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "clusters": {},
        "unsafe_pairings": [
            "Kalos has no IPMI power files -> no server-power vs node-metric pairing exists for Kalos.",
            "GPU power (DCGM POWER_USAGE) and server power (IPMI Sys_Total_Power) are different "
            "measurement levels and are never fused into one quantity.",
            "Job traces carry no node IDs -> job<->server/GPU pairing is impossible; only "
            "cluster-level temporal linkage is reported.",
            "Kalos PIPE_TENSOR_ACTIVE / SM_ACTIVE / SM_OCCUPANCY use pod-style device names "
            "(e.g. lingjun-pod9-0440-4) while Kalos POWER_USAGE uses IP-style names; the two "
            "naming schemes cannot be paired without an audited mapping.",
            "IPMI samples every 300 s and node metrics every 15 s: native same-instant pairing "
            "between them is impossible; only 15-minute-bin pairing is reported.",
        ],
        "timezone_table": {},
    }
    jobs: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "no_node_identifiers": True,
        "clusters": {},
    }
    metric_filter: set[str] | None = None
    if metrics is not None:
        metric_filter = {str(item).strip().upper() for item in metrics if str(item).strip()}
        if not metric_filter:
            raise ValueError("metrics filter is empty; pass at least one metric name or omit it")
        available = {
            path.stem.upper()
            for cluster_name in clusters
            for path in utilization_files(cluster_name)
        }
        unknown = sorted(metric_filter - available)
        if unknown:
            raise ValueError(
                f"unknown utilization metrics for clusters {sorted(clusters)}: {unknown}; "
                f"available: {sorted(available)}"
            )
        ops["notes"].append(
            "PARTIAL util-metric scan: only " + ", ".join(sorted(metric_filter)) +
            " were audited in this invocation (explicit --metrics whitelist); all other "
            "utilization metrics were NOT read and are not represented in this payload."
        )
    for cluster in clusters:
        records: list[Mapping[str, Any]] = []
        payloads: dict[str, Mapping[str, Any]] = {}
        files = [
            path
            for path in utilization_files(cluster)
            if metric_filter is None or path.stem.upper() in metric_filter
        ]
        for path in files:
            metric = path.stem.upper()
            is_gpu_pairing = metric in GPU_PAIRING_METRICS
            is_node_pairing = cluster == "seren" and metric in NODE_PAIRING_METRICS
            do_pairing = (not skip_pairing) and (is_gpu_pairing or is_node_pairing)
            do_native = (not skip_pairing) and is_gpu_pairing and cluster in NATIVE_WINDOWS
            record, payload = scan_utilization_file(
                path,
                cluster=cluster,
                metric=metric,
                chunk_rows=chunk_rows or FULL_CHUNK_ROWS,
                pairing=do_pairing,
                native_window=do_native,
                max_chunks=max_chunks,
            )
            if record.get("missing_fraction_per_device"):
                fractions = np.asarray(record["missing_fraction_per_device"], dtype=np.float64)
                record["nan_values"] = int(round(float(fractions.sum()) * int(record.get("rows") or 0)))
            else:
                record["nan_values"] = 0
            record["values_total"] = int(record.get("rows") or 0) * int(record.get("device_columns") or 0)
            records.append(record)
            if "accumulator" in payload:
                payloads[metric] = payload
            if progress is not None:
                progress(
                    f"{cluster}/{metric}: rows={record['rows']} devices={record['device_columns']} "
                    f"interval={((record.get('native_interval') or {}).get('native_interval_seconds'))}"
                )
        ops["clusters"][cluster] = {
            "files": records,
            "summary": _cluster_summary(records),
            "device_sets": device_set_report(cluster=cluster, records=records),
            "devices_per_node": devices_per_node_report(cluster, records),
        }
        pairing["timezone_table"][cluster] = {
            record["metric"]: record.get("timestamp_offset_observed") for record in records
        }
        gpu_pairs: list[dict[str, Any]] = []
        node_pairs: list[dict[str, Any]] = []
        if not skip_pairing:
            gpu_pairs = pair_gpu_metrics(cluster, payloads)
        ipmi_records: list[Mapping[str, Any]] = []
        if include_ipmi and cluster == "seren":
            ipmi_accumulators: list[BinAccumulator] = []
            for path in ipmi_files():
                record, payload = scan_ipmi_file(
                    path,
                    chunk_rows=chunk_rows or NARROW_CHUNK_ROWS,
                    pairing=not skip_pairing,
                    max_chunks=max_chunks,
                )
                record["nan_values"] = int(
                    sum(
                        float(value) * record["rows"]
                        for value in record["missing_fraction_by_field"].values()
                    )
                )
                record["values_total"] = int(record["rows"] * len(record["numeric_columns"]))
                ipmi_records.append(record)
                if payload.get("bin_accumulator") is not None:
                    ipmi_accumulators.append(payload["bin_accumulator"])
                if progress is not None:
                    progress(
                        f"seren/{path.stem}: rows={record['rows']} nodes={record['node_count']} "
                        f"interval={((record.get('native_interval') or {}).get('native_interval_seconds'))}"
                    )
            if not skip_pairing and ipmi_accumulators:
                ipmi_bin = merge_bin_accumulators(ipmi_accumulators)
                for metric in NODE_PAIRING_METRICS:
                    other = payloads.get(metric)
                    if other is None or other.get("accumulator") is None:
                        continue
                    pair = _pair_dense_accumulators(
                        cluster=cluster,
                        level="node",
                        metric_a="IPMI_Sys_Total_Power",
                        accumulator_a=ipmi_bin,
                        metric_b=metric,
                        accumulator_b=other["accumulator"],
                    )
                    pair["native_note"] = (
                        "IPMI 300 s vs node 15 s: no native same-instant pairing is possible"
                    )
                    node_pairs.append(pair)
        pairing["clusters"][cluster] = {
            "gpu_pairs": gpu_pairs,
            "node_pairs": node_pairs,
            "ipmi_files": ipmi_records,
            "device_sets": ops["clusters"][cluster]["device_sets"],
            "devices_per_node": ops["clusters"][cluster]["devices_per_node"],
        }
        if include_jobs:
            job_audit, frame = audit_job_trace(job_trace_file(cluster), cluster=cluster)
            gpu_util = payloads.get("GPU_UTIL")
            if not skip_pairing and gpu_util is not None and gpu_util.get("accumulator") is not None:
                job_audit["cluster_linkage"] = cluster_temporal_linkage(
                    cluster=cluster,
                    gpu_util_accumulator=gpu_util["accumulator"],
                    job_frame=frame,
                    job_audit=job_audit,
                )
            job_audit["timezone_impact"] = timezone_impact(
                cluster=cluster,
                job_audit=job_audit,
                utilization_offsets=pairing["timezone_table"][cluster],
            )
            jobs["clusters"][cluster] = job_audit
        payloads.clear()
    return ops, pairing, jobs


# ---------------------------------------------------------------------------
# artifact writing
# ---------------------------------------------------------------------------


def _runtime_rows(ops: Mapping[str, Any], pairing: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster, payload in ops["clusters"].items():
        for record in payload["files"]:
            interval = record.get("native_interval") or {}
            rows.append(
                {
                    "cluster": cluster,
                    "metric": record["metric"],
                    "kind": record["kind"],
                    "rows": record.get("rows"),
                    "device_columns": record.get("device_columns"),
                    "bytes": record.get("bytes"),
                    "first_timestamp_utc": record.get("first_timestamp_utc"),
                    "last_timestamp_utc": record.get("last_timestamp_utc"),
                    "native_interval_seconds": interval.get("native_interval_seconds"),
                    "matches_expected_interval": interval.get("matches_expected_interval"),
                    "gap_count": interval.get("gap_count"),
                    "duplicate_timestamps": record.get("duplicate_timestamps"),
                    "missing_fraction_aggregate": record.get("missing_fraction_aggregate"),
                    "missing_fraction_min": record.get("missing_fraction_min"),
                    "missing_fraction_max": record.get("missing_fraction_max"),
                    "fully_missing_devices": record.get("fully_missing_devices"),
                    "value_min": record.get("value_min"),
                    "value_max": record.get("value_max"),
                    "unit_guess": record.get("unit_guess"),
                    "placeholder": record.get("placeholder"),
                }
            )
        for record in (pairing.get("clusters", {}).get(cluster, {}).get("ipmi_files") or []):
            rows.append(
                {
                    "cluster": cluster,
                    "metric": "IPMI_" + Path(record["path"]).stem,
                    "kind": record["kind"],
                    "rows": record.get("rows"),
                    "device_columns": record.get("node_count"),
                    "bytes": record.get("bytes"),
                    "first_timestamp_utc": record.get("first_timestamp_utc"),
                    "last_timestamp_utc": record.get("last_timestamp_utc"),
                    "native_interval_seconds": (record.get("native_interval") or {}).get("native_interval_seconds"),
                    "matches_expected_interval": (record.get("native_interval") or {}).get("matches_expected_interval"),
                    "gap_count": (record.get("native_interval") or {}).get("gap_count"),
                    "duplicate_timestamps": record.get("duplicate_node_timestamp_pairs_within_chunks"),
                    "missing_fraction_aggregate": (
                        float(np.mean(list(record.get("missing_fraction_by_field", {}).values())))
                        if record.get("missing_fraction_by_field")
                        else None
                    ),
                    "missing_fraction_min": None,
                    "missing_fraction_max": None,
                    "fully_missing_devices": None,
                    "value_min": (record.get("value_min_by_field") or {}).get(IPMI_POWER_COLUMN),
                    "value_max": (record.get("value_max_by_field") or {}).get(IPMI_POWER_COLUMN),
                    "unit_guess": record.get("unit_guess"),
                    "placeholder": False,
                }
            )
    return rows


def _device_set_rows(pairing: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster, payload in pairing["clusters"].items():
        for kind, group in payload["device_sets"]["groups"].items():
            for entry in group["files"]:
                rows.append(
                    {
                        "cluster": cluster,
                        "kind": kind,
                        "metric": entry["metric"],
                        "devices": entry["devices"],
                        "reference_metric": entry["reference_metric"],
                        "missing_vs_reference": len(entry["missing_vs_reference"]),
                        "extra_vs_reference": len(entry["extra_vs_reference"]),
                    }
                )
        for metric, entry in payload["devices_per_node"]["files"].items():
            rows.append(
                {
                    "cluster": cluster,
                    "kind": "gpus_per_node",
                    "metric": metric,
                    "devices": entry["devices"],
                    "reference_metric": None,
                    "missing_vs_reference": None,
                    "extra_vs_reference": None,
                }
            )
    return rows


def _pairing_rows(pairing: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster, payload in pairing["clusters"].items():
        for pair in list(payload["gpu_pairs"]) + list(payload["node_pairs"]):
            native = pair.get("native_window") or {}
            rows.append(
                {
                    "cluster": cluster,
                    "level": pair["level"],
                    "metric_a": pair["metric_a"],
                    "metric_b": pair["metric_b"],
                    "devices_a": pair["devices_a"],
                    "devices_b": pair["devices_b"],
                    "common_devices": pair["common_devices"],
                    "matched_devices": pair["matched_devices"],
                    "matched_device_slots": pair["matched_device_slots"],
                    "union_device_slots": pair["union_device_slots"],
                    "match_rate": pair["match_rate"],
                    "joint_span_first_utc": pair["joint_span_first_utc"],
                    "joint_span_last_utc": pair["joint_span_last_utc"],
                    "native_matched_device_instants": native.get("matched_device_instants"),
                    "native_match_rate": native.get("match_rate"),
                }
            )
    return rows


def _job_rows(jobs: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster, payload in jobs["clusters"].items():
        for field, entry in payload["prediction_field_classification"].items():
            rows.append(
                {
                    "cluster": cluster,
                    "field": field,
                    "known_before_origin": entry["known_before_origin"],
                    "timing": entry["timing"],
                    "reason": entry["reason"],
                    "non_null": payload["field_completeness"][field]["non_null"],
                    "null_fraction": payload["field_completeness"][field]["null_fraction"],
                }
            )
    return rows


def _linkage_rows(jobs: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster, payload in jobs["clusters"].items():
        linkage = payload.get("cluster_linkage") or {}
        if not linkage.get("available"):
            continue
        rows.append(
            {
                "cluster": cluster,
                "n_bins": linkage["n_bins"],
                "overlap_bins": linkage["overlap_bins"],
                "mean_gpu_utilization": linkage["mean_gpu_utilization"],
                "active_gpu_jobs_mean": linkage["active_gpu_jobs_mean"],
                "pearson_active_gpu_vs_mean_util": linkage["pearson_active_gpu_vs_mean_util"],
                "util_span_utc": " .. ".join(linkage["util_span_utc"]),
                "job_span_utc": " .. ".join(str(item) for item in linkage["job_span_utc"]),
                "limitation": linkage["limitation"],
            }
        )
    return rows


def write_audit_artifacts(
    ops_audit: Mapping[str, Any],
    pairing_audit: Mapping[str, Any],
    job_audit: Mapping[str, Any],
    *,
    out_dir: str | Path = "data/processed",
    analysis_dir: str | Path = "results/analysis/acmetrace",
) -> dict[str, Any]:
    destination = ensure_dir(resolve_path(out_dir))
    analysis = ensure_dir(resolve_path(analysis_dir))
    paths: dict[str, Any] = {}
    for name, payload in (
        ("acmetrace_ops_audit.json", ops_audit),
        ("acmetrace_pairing_audit.json", pairing_audit),
        ("acmetrace_job_audit.json", job_audit),
    ):
        paths[name] = str(write_json(destination / name, payload))
    tables = {
        "runtime_file_summary.csv": _runtime_rows(ops_audit, pairing_audit),
        "device_sets.csv": _device_set_rows(pairing_audit),
        "pairing_summary.csv": _pairing_rows(pairing_audit),
        "job_field_classification.csv": _job_rows(job_audit),
        "cluster_linkage.csv": _linkage_rows(job_audit),
    }
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        target = analysis / name
        frame.to_csv(target, index=False)
        paths[name] = str(target)
    return paths


def utilization_files(cluster: str) -> list[Path]:
    pattern = str(resolve_path(f"{UTIL_ROOT}/{cluster}/*.csv"))
    return sorted(Path(path) for path in glob.glob(pattern))


def ipmi_files() -> list[Path]:
    pattern = str(resolve_path(f"{IPMI_ROOT}/*.csv"))
    return sorted(Path(path) for path in glob.glob(pattern))


def job_trace_file(cluster: str) -> Path:
    return resolve_path(f"{JOB_ROOT}/trace_{cluster}.csv")


def device_set_report(
    *, cluster: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Exact device-set differences relative to the first file of the same kind."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["kind"], []).append(record)
    report: dict[str, Any] = {"cluster": cluster, "groups": {}}
    for kind, group in sorted(grouped.items()):
        reference = group[0]
        reference_set = set(reference["device_labels"])
        entries = []
        for record in group:
            current = set(record["device_labels"])
            entries.append(
                {
                    "metric": record["metric"],
                    "devices": len(current),
                    "reference_metric": reference["metric"],
                    "missing_vs_reference": sorted(reference_set - current),
                    "extra_vs_reference": sorted(current - reference_set),
                }
            )
        report["groups"][kind] = {
            "reference_metric": reference["metric"],
            "reference_device_count": len(reference_set),
            "files": entries,
        }
    return report


def devices_per_node_report(cluster: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gpu_records = [
        record for record in records if record["kind"] == "gpu_wide" and record.get("device_columns")
    ]
    report: dict[str, Any] = {"cluster": cluster, "files": {}}
    for record in gpu_records:
        parsed = parse_gpu_device_columns(record["device_labels"])
        if parsed is None:
            continue
        per_node = devices_per_node(parsed)
        counts = {node: len(indices) for node, indices in per_node.items()}
        report["files"][record["metric"]] = {
            "nodes": len(per_node),
            "devices": len(parsed),
            "gpus_per_node_counts": {
                str(key): int(value) for key, value in Counter(counts.values()).most_common()
            },
            "nodes_with_out_of_range_gpu_index": sorted(
                node for node, indices in per_node.items() if any(index < 0 or index > 7 for index in indices)
            ),
        }
    return report


__all__ = [
    "RAW_ROOT",
    "UTIL_ROOT",
    "IPMI_ROOT",
    "JOB_ROOT",
    "CLUSTERS",
    "BIN_SECONDS",
    "METRIC_SPECS",
    "JOB_FIELD_CLASSIFICATION",
    "GPU_PAIRING_METRICS",
    "NODE_PAIRING_METRICS",
    "parse_device",
    "parse_node_device",
    "ipmi_node_to_ip",
    "parse_iso_timestamps",
    "parse_ipmi_timestamps",
    "parse_timestamps",
    "detect_timestamp_format",
    "observed_offset",
    "interval_statistics",
    "infer_unit_evidence",
    "scan_utilization_file",
    "scan_ipmi_file",
    "pair_gpu_metrics",
    "audit_job_trace",
    "cluster_temporal_linkage",
    "run_audit",
    "write_audit_artifacts",
]
