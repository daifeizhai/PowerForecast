"""AcmeTrace admission audit: inventory, power audit, hierarchy and 15-minute adaptation.

Scope (V3.2 admission audit, Task 2 for the ``ucc-data`` teammate):

* **inventory/integrity** of every local file under ``E:/Data/AcmeTrace/`` (streamed SHA256,
  true line counts, columns, first/last timestamps, readability) against the HuggingFace
  tree JSON of revision ``a484f77...``;
* **power audit** of the two clusters: DCGM ``POWER_USAGE`` at GPU level for Seren (1992
  GPU columns) and Kalos (2344), and IPMI server power for Seren (3 server-model files);
* **hierarchy**: GPU power (per GPU) vs server power (IPMI per node) vs Seren cluster IT
  (sum of *reporting* IPMI nodes, never zero-filled) vs a clearly-labelled
  ``cluster_gpu_sum`` aggregate; **no facility power exists in AcmeTrace**;
* **15-minute adaptation**: native samples aggregated to 15-minute interval means on a
  full grid with explicit ``valid``/``coverage_ratio``/``n_samples`` masks, no
  interpolation and no zero fill.

Honesty rules: power units are inferred as W from magnitude evidence and stay
``unit_verified=false``; zero/negative readings are invalid/offline, never real 0 kW;
Kalos has **no server power**, hence no Kalos ``server`` or ``cluster_it`` level; GPU sums
are never called cluster IT or facility power; raw files are read-only.

Memory discipline: all raw files are streamed in chunks (the largest CSV is 5.7 GB); the
biggest in-memory object is the per-cluster bin × device accumulator, well below 2 GB.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.contracts import ContractError
from src.data.loaders import assert_not_raw_path
from src.utils import (
    ensure_dir,
    environment_report,
    resolve_path,
    utc_timestamp,
    write_json,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

CONFIG_PATH: str = "configs/datasets/acmetrace.yaml"
DEFAULT_PROCESSED_DIR: str = "data/processed"
DEFAULT_RESULTS_DIR: str = "results/analysis/acmetrace"

INVENTORY_FILENAME: str = "acmetrace_inventory.json"
POWER_AUDIT_FILENAME: str = "acmetrace_power_audit.json"
METADATA_FILENAME: str = "acmetrace_power_metadata.json"
POWER_15MIN_FILENAME: str = "acmetrace_power_15min.parquet"

BIN_SECONDS: int = 900
BIN_NS: int = BIN_SECONDS * 1_000_000_000

POWER_15MIN_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "cluster_id",
    "level",
    "device_id",
    "power_kw",
    "valid",
    "coverage_ratio",
    "n_samples",
)

LEVEL_ORDER: tuple[str, ...] = ("gpu", "server", "cluster_it", "cluster_gpu_sum")

_GPU_COLUMN_RE = re.compile(r"^(?P<ip>\d{1,3}(?:\.\d{1,3}){3})-(?P<index>\d+)$")
_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

#: Forbidden tokens in the processed power table: no facility/cluster-total claim may exist.
_FORBIDDEN_TOKENS: tuple[str, ...] = ("facility", "cluster_total", "site_total")

#: CSV chunks are bounded so the wide GPU chunks stay well below 2 GB.
_MAX_CHUNK_ROWS_WIDE = 20000


def load_acmetrace_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and structurally validate ``configs/datasets/acmetrace.yaml``.

    The raw tree lives outside the repository (moved to ``E:/Data/AcmeTrace``); the
    ``ACMETRACE_RAW_ROOT`` environment variable overrides the configured ``raw_root``
    without editing the YAML, and every root-relative raw path is resolved against it.
    """
    from src.utils import load_config

    config = load_config(path)
    override = os.environ.get("ACMETRACE_RAW_ROOT")
    if override:
        config = dict(config)
        config["raw_root"] = override
    return validate_acmetrace_config(config, path=str(path))


def _require(config: Mapping[str, Any], keys: Sequence[str], *, where: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"{where} is missing keys: {missing}.")


def validate_acmetrace_config(config: Mapping[str, Any], *, path: str = "<mapping>") -> dict[str, Any]:
    """Validate the AcmeTrace config and its honesty rules."""
    if not isinstance(config, Mapping):
        raise ValueError(f"{path}: config must be a mapping.")
    _require(
        config,
        (
            "schema_version",
            "raw_root",
            "hf_tree",
            "source",
            "clusters",
            "ipmi",
            "levels",
            "aggregation_15min",
            "streaming",
            "outputs",
            "readiness",
        ),
        where=path,
    )
    if int(config["schema_version"]) != 1:
        raise ValueError(f"{path}: schema_version must be 1.")
    clusters = config["clusters"]
    if list(clusters) != ["seren", "kalos"]:
        raise ValueError(f"{path}: clusters must be exactly ['seren', 'kalos'], got {list(clusters)}.")
    for key, section in clusters.items():
        _require(
            section,
            ("gpu_power_file", "native_interval_seconds", "expected_samples_15min", "has_ipmi_server_power"),
            where=f"{path}.clusters[{key}]",
        )
        if int(section["native_interval_seconds"]) != 15:
            raise ValueError(f"{path}.clusters[{key}]: GPU native interval must stay 15 s.")
        if int(section["expected_samples_15min"]) * int(section["native_interval_seconds"]) != BIN_SECONDS:
            raise ValueError(
                f"{path}.clusters[{key}]: expected 15-min samples must be BIN_SECONDS/native "
                f"({BIN_SECONDS // int(section["native_interval_seconds"])}), got "
                f"{section['expected_samples_15min']}."
            )
    if bool(clusters["kalos"]["has_ipmi_server_power"]):
        raise ValueError(
            f"{path}: Kalos has no IPMI/server power; has_ipmi_server_power must stay false."
        )
    if not bool(clusters["seren"]["has_ipmi_server_power"]):
        raise ValueError(f"{path}: Seren IPMI server power exists and must be declared.")
    levels = config["levels"]
    if list(levels["seren"]) != ["gpu", "server", "cluster_it", "cluster_gpu_sum"]:
        raise ValueError(f"{path}: Seren levels must be gpu/server/cluster_it/cluster_gpu_sum.")
    if list(levels["kalos"]) != ["gpu", "cluster_gpu_sum"]:
        raise ValueError(
            f"{path}: Kalos has no server power, so its levels must be exactly "
            "['gpu', 'cluster_gpu_sum'] (no server / cluster_it)."
        )
    ag = config["aggregation_15min"]
    _require(
        ag,
        ("step_seconds", "validity_threshold", "strict_coverage_threshold", "interpolation", "zero_fill"),
        where=f"{path}.aggregation_15min",
    )
    if int(ag["step_seconds"]) != BIN_SECONDS:
        raise ValueError(f"{path}: aggregation step must be {BIN_SECONDS} s.")
    if not (0.0 < float(ag["validity_threshold"]) < 1.0):
        raise ValueError(f"{path}: validity_threshold must be in (0, 1).")
    if not (float(ag["validity_threshold"]) < float(ag["strict_coverage_threshold"]) <= 1.0):
        raise ValueError(f"{path}: strict_coverage_threshold must exceed validity_threshold and be <= 1.")
    if bool(ag["interpolation"]) or bool(ag["zero_fill"]):
        raise ValueError(f"{path}: interpolation and zero_fill must stay false.")
    source = config["source"]
    units = source["units"]
    if str(units["power_unit_inferred"]) != "W":
        raise ValueError(f"{path}: power unit inference must stay W (magnitude evidence).")
    if bool(units["power_unit_verified"]):
        raise ValueError(f"{path}: power units are not documented; power_unit_verified must stay false.")
    clusters_claim = config.get("cluster_gpu_sum", {})
    if not bool(clusters_claim.get("never_facility_power", False)):
        raise ValueError(f"{path}: cluster_gpu_sum must declare never_facility_power.")
    if not bool(clusters_claim.get("never_cluster_it", False)):
        raise ValueError(f"{path}: cluster_gpu_sum must declare never_cluster_it.")
    if bool(config["readiness"]["ready_for_training"]):
        raise ValueError(f"{path}: AcmeTrace power is not approved for training.")
    return _resolve_raw_paths(dict(config))


def _resolve_raw_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Make configured raw paths absolute relative to ``raw_root`` (root-relative YAML)."""
    raw_root = resolve_path(str(config["raw_root"]))
    config["raw_root"] = str(raw_root)

    def resolve(value: str) -> str:
        candidate = Path(str(value))
        return str(candidate if candidate.is_absolute() else raw_root / candidate)

    for key in ("readme", "hf_tree"):
        if key in config:
            config[key] = resolve(config[key])
    for cluster in config["clusters"].values():
        for key in ("gpu_power_file", "node_inventory_file"):
            if key in cluster:
                cluster[key] = resolve(cluster[key])
    for ipmi_file in config["ipmi"]["files"].values():
        if "path" in ipmi_file:
            ipmi_file["path"] = resolve(ipmi_file["path"])
    return config


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def parse_gpu_device_id(column: str) -> tuple[str, int]:
    """Split a GPU-wide column name ``<node_ip>-<gpu_index>`` into its two parts."""
    match = _GPU_COLUMN_RE.match(str(column).strip())
    if not match:
        raise ValueError(f"GPU device column {column!r} does not match <node_ip>-<gpu_index>.")
    ip = match.group("ip")
    if not _IP_RE.match(ip):
        raise ValueError(f"GPU device column {column!r} carries an invalid node IP.")
    return ip, int(match.group("index"))


def gpu_device_id(node_ip: str, gpu_index: int) -> str:
    return f"{node_ip}-{int(gpu_index)}"


def node_to_ip(node: str, *, pattern: str = r"(\d+)-(\d+)-(\d+)-(\d+)$") -> str:
    """Map a node id such as ``SH-IDC1-10-140-0-112`` to its IPv4 address."""
    match = re.search(pattern, str(node).strip())
    if not match:
        raise ValueError(f"Node id {node!r} does not end with four numeric groups.")
    return ".".join(match.groups())


def _parse_chinese_timestamp(value: Any, pattern: str) -> pd.Timestamp | None:
    match = re.match(pattern, str(value).strip())
    if not match:
        return None
    parts = {key: int(match.group(key)) for key in ("year", "month", "day", "hour", "minute", "second")}
    stamp = pd.Timestamp(
        year=parts["year"],
        month=parts["month"],
        day=parts["day"],
        hour=parts["hour"],
        minute=parts["minute"],
        second=parts["second"],
        tz=ZoneInfo("Asia/Shanghai"),
    )
    return stamp


def parse_acmetrace_timestamps(
    values: Iterable[Any], *, strict: bool = True
) -> pd.DatetimeIndex:
    """Parse AcmeTrace monitor timestamps (ISO ``+08:00`` and mixed Chinese-locale CST).

    Returns a UTC-naive :class:`pandas.DatetimeIndex` (project convention).  The Chinese
    locale spelling appears in ``CPU_D_Power.csv``; both spellings denote the same +08:00
    zone.
    """
    series = pd.Series(np.asarray(list(values), dtype=object))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    missing = parsed.isna()
    if bool(missing.any()):
        chinese_pattern = r"(?P<year>\d{4})年\s*(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日\s*星期[一二三四五六日天]\s*(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})\s*CST"
        cache: dict[str, pd.Timestamp] = {}
        for position in np.flatnonzero(missing.to_numpy()):
            raw = str(series.iloc[int(position)])
            if raw not in cache:
                stamp = _parse_chinese_timestamp(raw, chinese_pattern)
                if stamp is None:
                    raise ValueError(f"Unparseable AcmeTrace timestamp {raw!r}.")
                cache[raw] = stamp
            parsed.iloc[int(position)] = cache[raw]
    if bool(parsed.isna().any()):
        raise ValueError("AcmeTrace timestamps contain unparseable values.")
    result = pd.DatetimeIndex(parsed).tz_convert(None)
    return result


def valid_power_mask(values: np.ndarray, *, max_plausible_w: float) -> np.ndarray:
    """A native power reading is valid iff finite, strictly positive and plausible.

    Zero and negative readings are offline/missing-sensor artefacts, never 0 kW power.
    """
    array = np.asarray(values, dtype=np.float64)
    return np.isfinite(array) & (array > 0.0) & (array <= float(max_plausible_w))


def _sanitise(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_sanitise(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value


def _json_ready(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _sanitise(dict(payload))


def _coverage_bands(values: np.ndarray, bands: Sequence[float]) -> dict[str, int]:
    counts: dict[str, int] = {}
    array = np.asarray(values, dtype=float)
    for low, high in zip(bands[:-1], bands[1:]):
        label = f"[{low:.2f},{high:.2f})" if high <= 1.0 else f"[{low:.2f},1.00]"
        counts[label] = int(np.count_nonzero((array >= low) & (array < high)))
    return counts


# ---------------------------------------------------------------------------
# inventory / integrity
# ---------------------------------------------------------------------------


def fingerprint_stream(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Stream one file once, returning SHA256, byte count and true newline count."""
    resolved = resolve_path(path)
    digest = hashlib.sha256()
    size = 0
    newlines = 0
    last_byte = b""
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_bytes))
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            newlines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    lines = newlines + (1 if size and last_byte != b"\n" else 0)
    return {
        "bytes": int(size),
        "sha256": digest.hexdigest(),
        "newline_count": int(newlines),
        "lines_total": int(lines),
        "last_byte_newline": bool(size and last_byte == b"\n"),
    }


def read_csv_edges(
    path: str | Path, *, tail_scan_bytes: int = 2 * 1024 * 1024
) -> dict[str, Any]:
    """Read a CSV's header and first/last data line (no full scan)."""
    resolved = resolve_path(path)
    try:
        with resolved.open("rb") as handle:
            header_line = handle.readline()
            first_line = handle.readline()
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - int(tail_scan_bytes)))
            tail = handle.read()
    except OSError as exc:  # pragma: no cover - defensive
        return {"readable": False, "error": str(exc)}
    header_text = header_line.decode("utf-8", errors="replace").strip()
    columns = header_text.split(",") if header_text else []
    tail_lines = [line for line in tail.decode("utf-8", errors="replace").splitlines() if line.strip()]
    last_text = tail_lines[-1] if tail_lines else ""
    first_text = first_line.decode("utf-8", errors="replace").strip()
    first_timestamp = _edge_timestamp(first_text)
    last_timestamp = _edge_timestamp(last_text)
    return {
        "readable": True,
        "n_columns": int(len(columns)),
        "column_names": columns,
        "first_data_line_present": bool(first_text),
        "first_timestamp": None if first_timestamp is None else str(first_timestamp),
        "last_timestamp": None if last_timestamp is None else str(last_timestamp),
        "last_line_present": bool(last_text),
    }


def _edge_timestamp(line: str) -> pd.Timestamp | None:
    if not line:
        return None
    field = line.split(",", 1)[0].strip()
    if not field or field == "Time":
        return None
    try:
        return parse_acmetrace_timestamps([field])[0]
    except (ValueError, TypeError):
        return None


def build_inventory(
    config: Mapping[str, Any], *, hash_contents: bool = True, progress: bool = True
) -> dict[str, Any]:
    """Build the local-file inventory and compare it with the HF tree JSON."""
    from src.utils import read_json

    raw_root = resolve_path(str(config["raw_root"]))
    if not raw_root.is_dir():
        raise FileNotFoundError(f"AcmeTrace raw root not found: {raw_root}.")
    tree_payload = read_json(str(config["hf_tree"]))
    tree_files: Mapping[str, Mapping[str, Any]] = tree_payload.get("files", {})
    streaming = config["streaming"]
    chunk_bytes = int(streaming["hash_chunk_bytes"])
    local_paths = sorted(path for path in raw_root.rglob("*") if path.is_file())
    files: dict[str, Any] = {}
    total_bytes = 0
    total_lines = 0
    sha_checked = 0
    sha_mismatch: list[str] = []
    missing_local: list[str] = []
    present_not_in_tree: list[str] = []
    for index, path in enumerate(local_paths, start=1):
        rel = path.relative_to(raw_root).as_posix()
        entry: dict[str, Any] = {"path": str(path), "relative_path": rel, "bytes": int(path.stat().st_size)}
        if rel.startswith(".cache/"):
            entry["category"] = "hf_cache_bookkeeping"
        elif rel.endswith(".md"):
            entry["category"] = "documentation"
        elif rel.endswith(".csv"):
            entry["category"] = "dataset_csv"
        else:
            entry["category"] = "other"
        if hash_contents:
            fingerprint = fingerprint_stream(path, chunk_bytes=chunk_bytes)
            entry.update(fingerprint)
            total_bytes += int(fingerprint["bytes"])
            total_lines += int(fingerprint["lines_total"])
            sha_checked += 1
        else:
            # Skip-hash mode: sizes are still inventoried (cheap stat), but no file content
            # is streamed.  Never report an unverified hash as a mismatch.
            entry["sha256"] = None
            entry["lines_total"] = None
            total_bytes += int(entry["bytes"])
        if entry["category"] == "dataset_csv":
            edges = read_csv_edges(path, tail_scan_bytes=int(streaming["tail_scan_bytes"]))
            entry.update(
                {
                    "n_columns": int(edges.get("n_columns", 0)),
                    "column_names_preview": list(edges.get("column_names", []))[:5]
                    if int(edges.get("n_columns", 0)) > 60
                    else list(edges.get("column_names", [])),
                    "readable": bool(edges.get("readable", False)),
                    "first_timestamp": edges.get("first_timestamp"),
                    "last_timestamp": edges.get("last_timestamp"),
                    "time_column": bool(edges.get("column_names")) and edges["column_names"][0] == "Time",
                    "data_rows": int(entry.get("lines_total", 0)) - 1
                    if entry.get("lines_total")
                    else None,
                }
            )
        tree_entry = tree_files.get(rel)
        entry["in_hf_tree"] = tree_entry is not None
        if tree_entry is None:
            present_not_in_tree.append(rel)
        else:
            entry["tree_size"] = tree_entry.get("size")
            entry["tree_size_match"] = bool(tree_entry.get("size") == entry["bytes"])
            entry["tree_lfs_sha256"] = tree_entry.get("lfs_sha256")
            if tree_entry.get("lfs_sha256") is not None and hash_contents:
                entry["sha256_match"] = bool(entry.get("sha256") == tree_entry.get("lfs_sha256"))
                if not entry["sha256_match"]:
                    sha_mismatch.append(rel)
            elif tree_entry.get("lfs_sha256") is None:
                entry["sha256_match"] = None
                entry["tree_hash_type"] = "blob_id_only"
            else:
                entry["sha256_match"] = None
                entry["sha256_verification"] = "skipped (--skip-hash: content not streamed)"
        files[rel] = entry
        if progress and index % 10 == 0:
            print(f"  inventory: {index}/{len(local_paths)} files ({total_bytes/1e9:.2f} GB streamed)", flush=True)
    tree_paths = set(tree_files)
    local_set = set(files)
    missing_local = sorted(tree_paths - local_set)
    summary = {
        "local_files": int(len(files)),
        "dataset_csv_files": int(sum(1 for item in files.values() if item["category"] == "dataset_csv")),
        "local_bytes": int(total_bytes),
        "local_lines_streamed": int(total_lines),
        "tree_files": int(len(tree_files)),
        "tree_total_bytes": int(sum(int(item.get("size", 0)) for item in tree_files.values())),
        "files_present_locally": int(len(local_set & tree_paths)),
        "files_missing_locally": int(len(missing_local)),
        "missing_local_bytes": int(
            sum(int(tree_files[rel].get("size", 0)) for rel in missing_local)
        ),
        "local_files_not_in_tree": int(len(present_not_in_tree)),
        "sha256_streamed_files": int(sha_checked),
        "sha256_mismatches": sha_mismatch,
        "sha256_verification": "full" if hash_contents else "skipped",
        "sha256_unverified_files": int(
            sum(1 for item in files.values() if str(item.get("sha256_verification", "")).startswith("skipped"))
        ),
        "tree_files_with_lfs_sha256": int(
            sum(1 for item in tree_files.values() if item.get("lfs_sha256") is not None)
        ),
        "tree_files_blob_id_only": int(
            sum(1 for item in tree_files.values() if item.get("lfs_sha256") is None)
        ),
        "hash_contents": bool(hash_contents),
    }
    return {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "generated_by": "src.data.acmetrace_power.build_inventory",
        "dataset_id": "acmetrace",
        "raw_root": str(raw_root),
        "hf_tree": {
            "path": str(resolve_path(str(config["hf_tree"]))),
            "revision": str(config["source"]["revision"]),
            "repository": str(config["source"]["repository"]),
            "tree_file_count": len(tree_files),
            "tree_total_bytes": summary["tree_total_bytes"],
        },
        "summary": summary,
        "integrity": {
            "all_local_files_readable": all(
                item.get("readable", True) for item in files.values() if item.get("category") == "dataset_csv"
            ),
            "sha256_mismatches": sha_mismatch,
            "size_mismatches": sorted(
                rel
                for rel, item in files.items()
                if item.get("in_hf_tree") and not item.get("tree_size_match", True)
            ),
            "note": (
                "SHA256 is compared only for tree entries that carry an lfs_sha256; small "
                "non-LFS files in the tree expose a git blob_id only, so their local SHA256 "
                "cannot be compared and is reported as null. "
                + (
                    "Content hashing was SKIPPED in this run (--skip-hash): lfs_sha256 entries "
                    "are marked unverified rather than mismatched."
                    if not hash_contents
                    else "Content hashing was performed for every local file."
                )
            ),
        },
        "missing_from_local": missing_local,
        "local_files_not_in_tree": sorted(present_not_in_tree),
        "files": files,
    }


# ---------------------------------------------------------------------------
# power scanning (streaming)
# ---------------------------------------------------------------------------


def _read_header(path: str | Path) -> list[str]:
    with resolve_path(path).open("rb") as handle:
        return handle.readline().decode("utf-8", errors="replace").strip().split(",")


def _node_columns(path: str | Path) -> list[str]:
    header = _read_header(path)
    if not header or header[0] != "Time":
        raise ValueError(f"Node inventory {path} does not start with a Time column.")
    nodes: list[str] = []
    for column in header[1:]:
        name = str(column).strip()
        if not name:
            continue
        if _IP_RE.match(name):
            nodes.append(name)
        else:
            try:
                nodes.append(node_to_ip(name))
            except ValueError as exc:
                raise ValueError(f"Node inventory {path}: cannot interpret column {name!r}.") from exc
    return sorted(set(nodes))


def _iter_frame_chunks(
    path: str | Path,
    *,
    usecols: Sequence[str] | None,
    dtype: Mapping[str, Any] | None,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    reader = pd.read_csv(
        resolve_path(path),
        usecols=list(usecols) if usecols is not None else None,
        dtype=dict(dtype) if dtype is not None else None,
        chunksize=int(chunksize),
        na_values=[""],
        keep_default_na=True,
    )
    for chunk in reader:
        yield chunk


def _update_min_max(
    valid: np.ndarray, values: np.ndarray, current_min: np.ndarray, current_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    chunk_min = np.min(np.where(valid, values, np.inf), axis=0)
    chunk_max = np.max(np.where(valid, values, -np.inf), axis=0)
    current_min = np.minimum(current_min, chunk_min)
    current_max = np.maximum(current_max, chunk_max)
    return current_min, current_max


def scan_gpu_power_file(
    config: Mapping[str, Any],
    cluster_key: str,
    *,
    max_chunks: int | None = None,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Stream ONE cluster's wide GPU ``POWER_USAGE`` CSV and accumulate audit statistics."""
    from src.utils import load_config  # noqa: F401  (documented shared loader)

    cluster = config["clusters"][cluster_key]
    path = resolve_path(str(cluster["gpu_power_file"]))
    if not path.is_file():
        raise FileNotFoundError(f"GPU power file not found for {cluster_key}: {path}.")
    header = _read_header(path)
    if not header or header[0] != "Time":
        raise ValueError(f"{path}: GPU power file must start with a Time column.")
    device_columns = [str(column).strip() for column in header[1:]]
    devices = [parse_gpu_device_id(column) for column in device_columns]
    if len(set(devices)) != len(devices):
        raise ValueError(f"{path}: duplicate GPU device columns detected.")
    device_ids = [gpu_device_id(ip, index) for ip, index in devices]
    nodes = sorted({ip for ip, _ in devices})
    device_count = len(device_ids)
    expected_samples = int(cluster["expected_samples_15min"])
    native_interval = int(cluster["native_interval_seconds"])
    max_w = float(config["source"]["units"]["plausible_ranges_w"]["gpu"][1])
    max_samples_w = float(config["source"]["units"]["plausible_ranges_w"]["gpu"][1])

    n_valid = np.zeros(device_count, dtype=np.int64)
    n_zero = np.zeros(device_count, dtype=np.int64)
    n_negative = np.zeros(device_count, dtype=np.int64)
    n_out_of_range = np.zeros(device_count, dtype=np.int64)
    total_valid_w = np.zeros(device_count, dtype=np.float64)
    min_w = np.full(device_count, np.inf, dtype=np.float64)
    max_w_array = np.full(device_count, -np.inf, dtype=np.float64)
    first_valid_ns = np.full(device_count, np.iinfo(np.int64).max, dtype=np.int64)
    last_valid_ns = np.full(device_count, np.iinfo(np.int64).min, dtype=np.int64)
    bins: dict[int, dict[str, Any]] = {}
    delta_counter: Counter[int] = Counter()
    row_count = 0
    negative_deltas = 0
    duplicate_timestamps = 0
    previous_last_ns: int | None = None
    chunk_count = 0
    dtype_map = {"Time": str, **{column: np.float32 for column in device_columns}}
    chunksize = int(config["streaming"]["csv_chunksize"])
    for chunk in _iter_frame_chunks(
        path, usecols=["Time", *device_columns], dtype=dtype_map, chunksize=chunksize
    ):
        chunk_count += 1
        if max_chunks is not None and chunk_count > int(max_chunks):
            break
        stamps = parse_acmetrace_timestamps(chunk["Time"].to_numpy(object))
        ts_ns = stamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        values = chunk[device_columns].to_numpy(dtype=np.float32)
        row_count += int(values.shape[0])
        if previous_last_ns is not None:
            boundary = np.array([previous_last_ns, ts_ns[0]], dtype=np.int64)
            deltas = np.diff(boundary) / 1e9
            for delta in deltas:
                delta_counter[int(round(float(delta)))] += 1
        previous_last_ns = int(ts_ns[-1])
        if ts_ns.size > 1:
            diff = np.diff(ts_ns) // 1_000_000_000
            negative_deltas += int((diff < 0).sum())
            duplicate_timestamps += int((diff == 0).sum())
            for value, count in Counter(int(item) for item in diff).items():
                delta_counter[value] += int(count)
        finite = np.isfinite(values)
        valid = finite & (values > 0.0) & (values <= max_w)
        zeros = finite & (values == 0.0)
        negatives = finite & (values < 0.0)
        out_of_range = finite & (values > max_w)
        n_valid += valid.sum(axis=0)
        n_zero += zeros.sum(axis=0)
        n_negative += negatives.sum(axis=0)
        n_out_of_range += out_of_range.sum(axis=0)
        total_valid_w += np.where(valid, values, 0.0).sum(axis=0, dtype=np.float64)
        min_w, max_w_array = _update_min_max(valid, values, min_w, max_w_array)
        any_valid = valid.any(axis=0)
        if bool(any_valid.any()):
            first_index = np.argmax(valid, axis=0)
            last_index = valid.shape[0] - 1 - np.argmax(valid[::-1], axis=0)
            chunk_first = np.where(any_valid, ts_ns[first_index], np.iinfo(np.int64).max)
            chunk_last = np.where(any_valid, ts_ns[last_index], np.iinfo(np.int64).min)
            first_valid_ns = np.minimum(first_valid_ns, chunk_first)
            last_valid_ns = np.maximum(last_valid_ns, chunk_last)
        # Per-native-timestamp sums (for the cluster GPU-sum aggregate) and bin accumulators.
        per_sample_sum = np.where(valid, values, 0.0).sum(axis=1, dtype=np.float64)
        per_sample_has_data = valid.any(axis=1)
        bin_ids = ts_ns // BIN_NS
        if bin_ids.size and np.all(bin_ids[1:] >= bin_ids[:-1]):
            boundaries = np.flatnonzero(np.diff(bin_ids)) + 1
            starts = np.concatenate(([0], boundaries))
            stops = np.concatenate((boundaries, [bin_ids.size]))
        else:
            starts = np.array([0], dtype=np.int64)
            stops = np.array([bin_ids.size], dtype=np.int64)
            unique_bins = np.unique(bin_ids)
            starts = np.array([int(np.flatnonzero(bin_ids == item)[0]) for item in unique_bins])
            stops = np.array([int(np.flatnonzero(bin_ids == item)[-1]) + 1 for item in unique_bins])
        for start, stop in zip(starts.tolist(), stops.tolist()):
            bin_id = int(bin_ids[start])
            accumulator = bins.get(bin_id)
            if accumulator is None:
                accumulator = {
                    "sum_w": np.zeros(device_count, dtype=np.float32),
                    "count": np.zeros(device_count, dtype=np.int32),
                    "native_rows": 0,
                    "sum_per_sample": 0.0,
                    "samples_with_data": 0,
                }
                bins[bin_id] = accumulator
            block_valid = valid[start:stop]
            accumulator["sum_w"] += np.where(
                block_valid, values[start:stop], 0.0
            ).sum(axis=0, dtype=np.float64).astype(np.float32)
            accumulator["count"] += block_valid.sum(axis=0).astype(np.int32)
            accumulator["native_rows"] += int(stop - start)
            accumulator["sum_per_sample"] += float(per_sample_sum[start:stop].sum())
            accumulator["samples_with_data"] += int(per_sample_has_data[start:stop].sum())
        del values, finite, valid, zeros, negatives, out_of_range, per_sample_sum
        if progress_every and chunk_count % int(progress_every) == 0:
            last_bin = int(ts_ns[-1]) if ts_ns.size else 0
            print(
                f"  {cluster_key} GPU scan: chunk {chunk_count} rows={row_count:,} "
                f"last_ts={pd.Timestamp(last_bin)} bins={len(bins)}",
                flush=True,
            )
    min_valid_ns = first_valid_ns[first_valid_ns != np.iinfo(np.int64).max]
    max_valid_ns = last_valid_ns[last_valid_ns != np.iinfo(np.int64).min]
    return {
        "cluster_key": cluster_key,
        "path": str(path),
        "native_interval_seconds": native_interval,
        "expected_samples_15min": expected_samples,
        "device_count": device_count,
        "node_count": len(nodes),
        "nodes": nodes,
        "devices": device_ids,
        "device_columns": device_columns,
        "native_rows": row_count,
        "timestamp_min": str(pd.Timestamp(int(min_valid_ns.min()))) if min_valid_ns.size else None,
        "timestamp_max": str(pd.Timestamp(int(max_valid_ns.max()))) if max_valid_ns.size else None,
        "delta_seconds_histogram": {str(key): int(value) for key, value in delta_counter.most_common(10)},
        "native_interval_mode_seconds": int(delta_counter.most_common(1)[0][0]) if delta_counter else None,
        "negative_timestamp_deltas": negative_deltas,
        "duplicate_timestamps": duplicate_timestamps,
        "bins": bins,
        "bin_id_min": int(min(bins)) if bins else None,
        "bin_id_max": int(max(bins)) if bins else None,
        "device_stats": {
            "n_valid": n_valid,
            "n_zero": n_zero,
            "n_negative": n_negative,
            "n_out_of_range": n_out_of_range,
            "total_valid_w": total_valid_w,
            "min_w": min_w,
            "max_w": max_w_array,
            "first_valid_ns": first_valid_ns,
            "last_valid_ns": last_valid_ns,
        },
        "plausible_max_w": max_w,
    }


def scan_ipmi_power_file(
    config: Mapping[str, Any],
    file_key: str,
    *,
    scope_nodes: set[str] | None = None,
    max_chunks: int | None = None,
    progress_every: int = 20,
) -> dict[str, Any]:
    """Stream ONE IPMI server-power CSV and accumulate node-level audit statistics."""
    ipmi = config["ipmi"]
    section = ipmi["files"][file_key]
    path = resolve_path(str(section["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"IPMI file not found for {file_key}: {path}.")
    header = _read_header(path)
    node_column = str(ipmi["node_column"])
    total_column = str(section["total_power_column"])
    meter_columns = [str(item) for item in section["meter_columns"]]
    required = ["Time", node_column, total_column, *[m for m in meter_columns if m != total_column]]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{path}: IPMI file is missing columns {missing}; header is {header}.")
    dtype_map = {"Time": str, node_column: str}
    for column in header[1:]:
        if column not in dtype_map:
            dtype_map[column] = np.float64
    max_w = float(config["source"]["units"]["plausible_ranges_w"]["server"][1])
    node_index: dict[str, int] = {}
    node_ids: list[str] = []
    node_stats: dict[str, dict[str, Any]] = {}
    bins: dict[int, dict[str, Any]] = {}
    #: Per-native-slot cluster inputs for nodes inside ``scope_nodes``:
    #: ``(bin_id, timestamp_ns) -> (sum_w_of_reporting_scope_nodes, n_reporting_scope_nodes)``.
    cluster_slots: dict[tuple[int, int], tuple[float, float]] = {}
    meter_values: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        f"{total_column}__{meter}": [] for meter in meter_columns if meter != total_column
    }
    delta_counter: Counter[int] = Counter()
    negative_deltas = 0
    duplicate_timestamps = 0
    duplicated_node_timestamp = 0
    duplicated_node_slot_rows = 0
    row_count = 0
    chunk_count = 0
    previous_last_ns: int | None = None
    chunksize = int(config["streaming"]["csv_chunksize"])
    role = re.compile(str(ipmi["node_id_pattern"]))
    for chunk in _iter_frame_chunks(path, usecols=required, dtype=dtype_map, chunksize=chunksize):
        chunk_count += 1
        if max_chunks is not None and chunk_count > int(max_chunks):
            break
        stamps = parse_acmetrace_timestamps(chunk["Time"].to_numpy(object))
        ts_ns = stamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        row_count += int(ts_ns.size)
        if previous_last_ns is not None:
            boundary = np.array([previous_last_ns, ts_ns[0]], dtype=np.int64)
            for delta in np.diff(boundary) / 1e9:
                delta_counter[int(round(float(delta)))] += 1
        previous_last_ns = int(ts_ns[-1])
        if ts_ns.size > 1:
            diff = np.diff(ts_ns) // 1_000_000_000
            negative_deltas += int((diff < 0).sum())
            duplicate_timestamps += int((diff == 0).sum())
            for value, count in Counter(int(item) for item in diff).items():
                delta_counter[value] += int(count)
        nodes = [str(value).strip() for value in chunk[node_column].to_numpy(object)]
        ips = [node_to_ip(node, pattern=str(ipmi["node_id_pattern"])) for node in nodes]
        for ip in ips:
            if ip not in node_index:
                node_index[ip] = len(node_ids)
                node_ids.append(ip)
                node_stats[ip] = {
                    "first_valid_ns": np.iinfo(np.int64).max,
                    "last_valid_ns": np.iinfo(np.int64).min,
                }
        positions = np.fromiter((node_index[ip] for ip in ips), dtype=np.int64, count=len(ips))
        total_values = chunk[total_column].to_numpy(dtype=np.float64)
        meter_arrays = {
            meter: chunk[meter].to_numpy(dtype=np.float64)
            for meter in meter_columns
            if meter != total_column
        }
        finite = np.isfinite(total_values)
        valid = finite & (total_values > 0.0) & (total_values <= max_w)
        node_count = len(node_ids)
        native_step_ns = int(ipmi["native_interval_seconds"]) * 1_000_000_000
        slot_ns = (ts_ns // native_step_ns) * native_step_ns
        # Duplicate raw rows (node, timestamp) are counted for the audit.
        if ts_ns.size:
            duplicated_node_timestamp += int(
                pd.DataFrame({"node": positions, "timestamp": ts_ns}).duplicated().sum()
            )
        # Multiple raw rows of the same node inside one 5-minute native slot are averaged
        # (duplicate sampling rows) and count as ONE native sample for coverage.
        dedupe_frame = pd.DataFrame(
            {
                "node": positions,
                "slot": slot_ns,
                "power": total_values,
                **meter_arrays,
            }
        )
        grouped = dedupe_frame.groupby(["node", "slot"], sort=False, observed=True)
        deduped = grouped.mean(numeric_only=True)
        group_sizes = grouped.size()
        duplicated_node_slot_rows += int((group_sizes - 1).sum())
        ded_positions = deduped.index.get_level_values(0).to_numpy(dtype=np.int64)
        ded_slots = deduped.index.get_level_values(1).to_numpy(dtype=np.int64)
        ded_power = deduped["power"].to_numpy(dtype=np.float64)
        ded_valid = valid_power_mask(ded_power, max_plausible_w=max_w)
        ded_meters = {
            meter: deduped[meter].to_numpy(dtype=np.float64) for meter in meter_arrays
        }
        podeduped = dedupe_frame.drop(columns=["power", *meter_arrays])
        del podeduped
        # Per-node statistics: anomaly counts from RAW rows, valid-slot stats from the
        # deduplicated native slots.
        for position in np.unique(ded_positions):
            mask = ded_positions == position
            values = ded_power[mask]
            valid_mask = ded_valid[mask]
            stats = node_stats[node_ids[int(position)]]
            stats["n_valid"] = int(stats.get("n_valid", 0)) + int(valid_mask.sum())
            raw_mask = positions == position
            raw_values = total_values[raw_mask]
            stats["n_zero"] = int(stats.get("n_zero", 0)) + int(
                (np.isfinite(raw_values) & (raw_values == 0.0)).sum()
            )
            stats["n_negative"] = int(stats.get("n_negative", 0)) + int(
                (raw_values < 0.0).sum()
            )
            stats["n_out_of_range"] = int(stats.get("n_out_of_range", 0)) + int(
                (raw_values > max_w).sum()
            )
            finite_values = values[valid_mask]
            if finite_values.size:
                stats["sum_w"] = float(stats.get("sum_w", 0.0)) + float(finite_values.sum())
                stats["min_w"] = min(float(stats.get("min_w", np.inf)), float(finite_values.min()))
                stats["max_w"] = max(float(stats.get("max_w", -np.inf)), float(finite_values.max()))
                chunk_slots = ded_slots[mask][valid_mask]
                stats["first_valid_ns"] = min(int(stats["first_valid_ns"]), int(chunk_slots.min()))
                stats["last_valid_ns"] = max(int(stats["last_valid_ns"]), int(chunk_slots.max()))
        # Per-node 15-minute bin accumulators (one sample per node per native slot).
        bin_ids = ded_slots // BIN_NS
        file_bins = bins
        for position, bin_id, value, is_valid in zip(
            ded_positions.tolist(), bin_ids.tolist(), ded_power.tolist(), ded_valid.tolist()
        ):
            accumulator = file_bins.get(int(bin_id))
            if accumulator is None:
                accumulator = {
                    "sum_w": np.zeros(node_count, dtype=np.float32),
                    "count": np.zeros(node_count, dtype=np.int32),
                }
                file_bins[int(bin_id)] = accumulator
            elif accumulator["sum_w"].size < node_count:
                padded = np.zeros(node_count, dtype=np.float32)
                padded[: accumulator["sum_w"].size] = accumulator["sum_w"]
                accumulator["sum_w"] = padded
                padded_count = np.zeros(node_count, dtype=np.int32)
                padded_count[: accumulator["count"].size] = accumulator["count"]
                accumulator["count"] = padded_count
            if is_valid:
                accumulator["sum_w"][position] += float(value)
                accumulator["count"][position] += 1
        # Cluster-IT inputs: per-native-slot sum over *reporting* in-scope nodes only.
        if scope_nodes is not None and ded_slots.size:
            scope_by_node = np.array([node in scope_nodes for node in node_ids], dtype=bool)
            scope_valid = ded_valid & scope_by_node[ded_positions]
            if bool(scope_valid.any()):
                order = np.argsort(ded_slots, kind="stable")
                sorted_slots = ded_slots[order]
                boundaries = np.flatnonzero(np.diff(sorted_slots)) + 1
                starts = np.concatenate(([0], boundaries))
                slot_values = sorted_slots[starts]
                slot_power = np.add.reduceat(
                    np.where(scope_valid[order], ded_power[order], 0.0), starts
                )
                slot_online = np.add.reduceat(scope_valid[order].astype(np.float64), starts)
                for slot_ns_value, slot_sum, slot_online_count in zip(
                    slot_values.tolist(), slot_power.tolist(), slot_online.tolist()
                ):
                    if slot_online_count <= 0.0:
                        continue
                    key = (int(slot_ns_value) // BIN_NS, int(slot_ns_value))
                    previous = cluster_slots.get(key)
                    if previous is None:
                        cluster_slots[key] = (float(slot_sum), float(slot_online_count))
                    else:
                        cluster_slots[key] = (
                            previous[0] + float(slot_sum),
                            previous[1] + float(slot_online_count),
                        )
        # Metering pairs on the same deduplicated native slots (same server, different meters).
        for meter in meter_arrays:
            meter_values_key = f"{total_column}__{meter}"
            meter_arr = ded_meters[meter]
            meter_valid = valid_power_mask(meter_arr, max_plausible_w=max_w)
            pair_mask = ded_valid & meter_valid
            meter_values[meter_values_key].append(
                (ded_power[pair_mask].astype(np.float32), meter_arr[pair_mask].astype(np.float32))
            )
        if progress_every and chunk_count % int(progress_every) == 0:
            print(
                f"  IPMI {file_key} scan: chunk {chunk_count} rows={row_count:,} nodes={len(node_ids)}",
                flush=True,
            )
    if cluster_slots:
        ordered_keys = sorted(cluster_slots)
        cluster_ts = np.array([key[1] for key in ordered_keys], dtype=np.int64)
        cluster_power = np.array([cluster_slots[key][0] for key in ordered_keys], dtype=np.float64)
        cluster_seren = np.array([cluster_slots[key][1] for key in ordered_keys], dtype=np.float64)
    else:
        cluster_ts = np.array([], dtype=np.int64)
        cluster_power = np.array([], dtype=np.float64)
        cluster_seren = np.array([], dtype=np.float64)
    meters_payload: dict[str, Any] = {}
    for meter_key, pairs in meter_values.items():
        if not pairs:
            meters_payload[meter_key] = {"n": 0}
            continue
        x = np.concatenate([item[0] for item in pairs])
        y = np.concatenate([item[1] for item in pairs])
        ratio = y / x
        meters_payload[meter_key] = {
            "n": int(x.size),
            "ratio_mean": float(ratio.mean()),
            "ratio_median": float(np.median(ratio)),
            "ratio_p05": float(np.quantile(ratio, 0.05)),
            "ratio_p95": float(np.quantile(ratio, 0.95)),
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "x_mean": float(x.mean()),
            "y_mean": float(y.mean()),
            "pearson_r": _pearson(x, y),
            "note": "same server, different meters; not independent observations",
        }
        del x, y, ratio, pairs
    first_valid = [value["first_valid_ns"] for value in node_stats.values() if value["first_valid_ns"] != np.iinfo(np.int64).max]
    last_valid = [value["last_valid_ns"] for value in node_stats.values() if value["last_valid_ns"] != np.iinfo(np.int64).min]
    payload_nodes: dict[str, Any] = {}
    for ip in node_ids:
        stats = node_stats[ip]
        n_valid = int(stats.get("n_valid", 0))
        payload_nodes[ip] = {
            "n_valid": n_valid,
            "n_zero": int(stats.get("n_zero", 0)),
            "n_negative": int(stats.get("n_negative", 0)),
            "n_out_of_range": int(stats.get("n_out_of_range", 0)),
            "min_w": stats.get("min_w"),
            "max_w": stats.get("max_w"),
            "mean_w": stats.get("sum_w", 0.0) / n_valid if n_valid else None,
            "first_valid": str(pd.Timestamp(int(stats["first_valid_ns"])))
            if stats["first_valid_ns"] != np.iinfo(np.int64).max
            else None,
            "last_valid": str(pd.Timestamp(int(stats["last_valid_ns"])))
            if stats["last_valid_ns"] != np.iinfo(np.int64).min
            else None,
        }
    return {
        "file_key": file_key,
        "server_model": str(section["server_model"]),
        "path": str(path),
        "native_interval_seconds": int(ipmi["native_interval_seconds"]),
        "native_rows": row_count,
        "node_count": len(node_ids),
        "nodes": node_ids,
        "node_stats": payload_nodes,
        "timestamp_min": str(pd.Timestamp(int(min(first_valid)))) if first_valid else None,
        "timestamp_max": str(pd.Timestamp(int(max(last_valid)))) if last_valid else None,
        "delta_seconds_histogram": {str(key): int(value) for key, value in delta_counter.most_common(10)},
        "native_interval_mode_seconds": int(delta_counter.most_common(1)[0][0]) if delta_counter else None,
        "negative_timestamp_deltas": negative_deltas,
        "duplicate_timestamps": duplicate_timestamps,
        "duplicated_node_timestamp_rows": duplicated_node_timestamp,
        "duplicated_node_slot_rows_averaged": duplicated_node_slot_rows,
        "bins": bins,
        "bin_id_min": int(min(bins)) if bins else None,
        "bin_id_max": int(max(bins)) if bins else None,
        "metering": meters_payload,
        "_cluster_slots": cluster_slots,
        "_cluster_arrays": (cluster_ts, cluster_power, cluster_seren),
        "plausible_max_w": max_w,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    dy = y - y_mean
    denominator = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    if denominator == 0.0:
        return None
    return float(np.sum(dx * dy) / denominator)


# ---------------------------------------------------------------------------
# 15-minute aggregation (pure functions used by the pipeline and the tests)
# ---------------------------------------------------------------------------


def aggregate_device_15min(
    timestamps_ns: np.ndarray,
    values_w: np.ndarray,
    *,
    expected_samples: int,
    max_plausible_w: float,
    validity_threshold: float,
    native_interval_seconds: int = 15,
) -> pd.DataFrame:
    """Aggregate one device's native samples to the 15-minute interval mean on a full grid.

    Rows are the complete 15-minute grid from the first to the last *native* timestamp
    (UTC-naive interval start).  Bins with no valid native sample stay as rows with
    ``power_kw=NaN``, ``valid=False``, ``coverage_ratio=0`` and ``n_samples=0``.  A zero or
    negative native value is invalid and is never turned into a 0 kW measurement.
    """
    stamps = np.asarray(timestamps_ns, dtype=np.int64)
    values = np.asarray(values_w, dtype=np.float64)
    if stamps.shape != values.shape:
        raise ValueError("timestamps_ns and values_w must have the same shape.")
    if stamps.size == 0:
        return pd.DataFrame(columns=["timestamp", "power_kw", "valid", "coverage_ratio", "n_samples"])
    valid_native = valid_power_mask(values, max_plausible_w=max_plausible_w)
    bin_ids = stamps // BIN_NS
    min_bin = int(bin_ids.min())
    max_bin = int(bin_ids.max())
    n_bins = max_bin - min_bin + 1
    sum_w = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)
    positions = bin_ids - min_bin
    np.add.at(sum_w, positions[valid_native], values[valid_native])
    np.add.at(count, positions[valid_native], 1)
    timestamps = pd.to_datetime((np.arange(min_bin, max_bin + 1) * BIN_NS), unit="ns")
    coverage = np.divide(count, expected_samples, dtype=np.float64)
    valid_bin = (count > 0) & (coverage >= float(validity_threshold))
    power_w = np.divide(sum_w, count, out=np.full(n_bins, np.nan), where=count > 0)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "power_kw": np.where(valid_bin | (count > 0), power_w / 1000.0, np.nan),
            "valid": valid_bin,
            "coverage_ratio": coverage.astype(np.float64),
            "n_samples": count.astype(np.int32),
        }
    )


def aggregate_cluster_15min(
    timestamps_ns: np.ndarray,
    per_slot_sum_w: np.ndarray,
    per_slot_online: np.ndarray,
    *,
    expected_samples: int,
    expected_members: int,
    validity_threshold: float,
    member_kind: str,
) -> pd.DataFrame:
    """Aggregate a cluster-level per-slot sum to the 15-minute interval mean.

    ``per_slot_sum_w`` is the sum over *reporting* members at that native slot (missing
    members are excluded, never zero-filled) and ``per_slot_online`` is the number of
    reporting members.  ``coverage_ratio = online member-samples / (expected samples ×
    expected members)`` so a partially reporting cluster is marked low-coverage instead of
    pretending to be a complete cluster total.
    """
    stamps = np.asarray(timestamps_ns, dtype=np.int64)
    sums = np.asarray(per_slot_sum_w, dtype=np.float64)
    online = np.asarray(per_slot_online, dtype=np.float64)
    if stamps.size == 0:
        return pd.DataFrame(
            columns=["timestamp", "power_kw", "valid", "coverage_ratio", "n_samples"]
        )
    if not (stamps.shape == sums.shape == online.shape):
        raise ValueError("cluster aggregation arrays must share one shape.")
    keep = online > 0
    stamps = stamps[keep]
    sums = sums[keep]
    online = online[keep]
    if stamps.size == 0:
        return pd.DataFrame(
            columns=["timestamp", "power_kw", "valid", "coverage_ratio", "n_samples"]
        )
    bin_ids = stamps // BIN_NS
    min_bin = int(bin_ids.min())
    max_bin = int(bin_ids.max())
    n_bins = max_bin - min_bin + 1
    positions = bin_ids - min_bin
    sum_per_bin = np.zeros(n_bins, dtype=np.float64)
    samples_per_bin = np.zeros(n_bins, dtype=np.int64)
    online_per_bin = np.zeros(n_bins, dtype=np.float64)
    np.add.at(sum_per_bin, positions, sums)
    np.add.at(samples_per_bin, positions, 1)
    np.add.at(online_per_bin, positions, online)
    timestamps = pd.to_datetime((np.arange(min_bin, max_bin + 1) * BIN_NS), unit="ns")
    denominator = float(expected_samples) * float(expected_members)
    coverage = np.divide(online_per_bin, denominator, dtype=np.float64)
    valid_bin = (samples_per_bin > 0) & (coverage >= float(validity_threshold))
    power_w = np.divide(
        sum_per_bin, samples_per_bin, out=np.full(n_bins, np.nan), where=samples_per_bin > 0
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "power_kw": np.where(samples_per_bin > 0, power_w / 1000.0, np.nan),
            "valid": valid_bin,
            "coverage_ratio": coverage,
            "n_samples": samples_per_bin.astype(np.int32),
        }
    )


def build_cluster_slots(
    cluster_frames: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    member_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn per-file row arrays into per-native-slot (sum, online count) arrays.

    Input rows are ``(timestamps_ns, power_w, is_member)``.  Rows must already be the
    reporting rows for the cluster (valid, in-scope members); ``member_mask`` optionally
    filters them further.  Duplicate (timestamp, member) rows are not deduplicated here —
    the audit reports duplicates separately.
    """
    stamps = np.concatenate([item[0] for item in cluster_frames]) if cluster_frames else np.array([], dtype=np.int64)
    powers = np.concatenate([item[1] for item in cluster_frames]) if cluster_frames else np.array([], dtype=np.float64)
    members = np.concatenate([item[2] for item in cluster_frames]) if cluster_frames else np.array([], dtype=bool)
    if member_mask is not None:
        members = members & np.asarray(member_mask, dtype=bool)
    if stamps.size == 0:
        return stamps, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    order = np.argsort(stamps, kind="stable")
    stamps = stamps[order]
    powers = powers[order]
    members = members[order]
    boundaries = np.flatnonzero(np.diff(stamps)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [stamps.size]))
    slot_stamps = stamps[starts]
    member_power = np.where(members, powers, 0.0)
    sum_w = np.add.reduceat(member_power, starts)
    online = np.add.reduceat(members.astype(np.float64), starts)
    return slot_stamps, sum_w, online


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def _device_stats_payload(scan: Mapping[str, Any]) -> dict[str, Any]:
    stats = scan["device_stats"]
    n_valid = stats["n_valid"]
    devices = scan["devices"]
    first_ns = stats["first_valid_ns"]
    last_ns = stats["last_valid_ns"]
    coverage = np.divide(n_valid, max(int(scan["native_rows"]), 1), dtype=np.float64)
    payload: dict[str, Any] = {}
    for index, device in enumerate(devices):
        total = int(n_valid[index])
        min_value = float(stats["min_w"][index])
        max_value = float(stats["max_w"][index])
        payload[device] = {
            "n_valid": total,
            "n_zero": int(stats["n_zero"][index]),
            "n_negative": int(stats["n_negative"][index]),
            "n_out_of_range": int(stats["n_out_of_range"][index]),
            "mean_w": float(stats["total_valid_w"][index] / total) if total else None,
            "min_w": min_value if math.isfinite(min_value) else None,
            "max_w": max_value if math.isfinite(max_value) else None,
            "native_valid_fraction": float(coverage[index]),
            "first_valid": str(pd.Timestamp(int(first_ns[index])))
            if int(first_ns[index]) != np.iinfo(np.int64).max
            else None,
            "last_valid": str(pd.Timestamp(int(last_ns[index])))
            if int(last_ns[index]) != np.iinfo(np.int64).min
            else None,
        }
    return payload


def _scan_summary(scan: Mapping[str, Any], *, expected_native: int) -> dict[str, Any]:
    stats = scan["device_stats"]
    n_valid = stats["n_valid"]
    valid_total = int(n_valid.sum())
    native_total = int(scan["native_rows"]) * int(scan["device_count"])
    coverage = np.divide(n_valid, max(int(scan["native_rows"]), 1), dtype=np.float64)
    return {
        "device_count": int(scan["device_count"]),
        "node_count": int(scan["node_count"]),
        "native_rows": int(scan["native_rows"]),
        "native_interval_seconds": int(scan["native_interval_seconds"]),
        "native_interval_mode_seconds": scan["native_interval_mode_seconds"],
        "expected_native_interval_seconds": int(expected_native),
        "delta_seconds_histogram": scan["delta_seconds_histogram"],
        "negative_timestamp_deltas": int(scan["negative_timestamp_deltas"]),
        "duplicate_timestamps": int(scan["duplicate_timestamps"]),
        "timestamp_min": scan["timestamp_min"],
        "timestamp_max": scan["timestamp_max"],
        "device_series_cells": int(native_total),
        "valid_series_cells": valid_total,
        "zero_series_cells": int(stats["n_zero"].sum()),
        "negative_series_cells": int(stats["n_negative"].sum()),
        "out_of_range_series_cells": int(stats["n_out_of_range"].sum()),
        "valid_cell_fraction": float(valid_total / native_total) if native_total else None,
        "devices_with_zero_valid_cells": int((n_valid == 0).sum()),
        "devices_with_any_missing_cells": int((n_valid < int(scan["native_rows"])).sum()),
        "per_device_valid_fraction_min": float(coverage.min()) if coverage.size else None,
        "per_device_valid_fraction_p50": float(np.median(coverage)) if coverage.size else None,
        "per_device_valid_fraction_max": float(coverage.max()) if coverage.size else None,
        "plausible_max_w": float(scan["plausible_max_w"]),
    }


def _gpu_level_matrice(scan: Mapping[str, Any], expected_samples: int) -> dict[str, Any]:
    bins = scan["bins"]
    if not bins:
        raise ValueError(f"{scan['cluster_key']}: no GPU bins were accumulated.")
    min_bin = int(scan["bin_id_min"])
    max_bin = int(scan["bin_id_max"])
    n_bins = max_bin - min_bin + 1
    device_count = int(scan["device_count"])
    sums = np.zeros((n_bins, device_count), dtype=np.float32)
    counts = np.zeros((n_bins, device_count), dtype=np.int32)
    native_rows = np.zeros(n_bins, dtype=np.int32)
    cluster_sum = np.zeros(n_bins, dtype=np.float64)
    cluster_samples = np.zeros(n_bins, dtype=np.int32)
    for bin_id, accumulator in bins.items():
        index = bin_id - min_bin
        sums[index] = accumulator["sum_w"]
        counts[index] = accumulator["count"]
        native_rows[index] = accumulator["native_rows"]
        cluster_sum[index] = accumulator["sum_per_sample"]
        cluster_samples[index] = accumulator["samples_with_data"]
    timestamps = pd.to_datetime(np.arange(min_bin, max_bin + 1) * BIN_NS, unit="ns")
    coverage = counts.astype(np.float64) / float(expected_samples)
    return {
        "timestamps": timestamps,
        "sums": sums,
        "counts": counts,
        "coverage": coverage,
        "native_rows": native_rows,
        "cluster_sum_w": cluster_sum,
        "cluster_samples": cluster_samples,
        "devices": list(scan["devices"]),
        "device_count": device_count,
        "min_bin": min_bin,
        "max_bin": max_bin,
    }


def _iter_gpu_15min_blocks(
    matrice: Mapping[str, Any], *, cluster_id: str, expected_samples: int, threshold: float, device_chunk: int = 128
) -> Iterable[pd.DataFrame]:
    timestamps = matrice["timestamps"]
    n_bins = len(timestamps)
    devices = matrice["devices"]
    counts = matrice["counts"]
    sums = matrice["sums"]
    coverage_all = matrice["coverage"]
    for start in range(0, matrice["device_count"], int(device_chunk)):
        stop = min(start + int(device_chunk), matrice["device_count"])
        chunk_counts = counts[:, start:stop]
        chunk_coverage = coverage_all[:, start:stop]
        chunk_sums = sums[:, start:stop]
        power_w = np.divide(
            chunk_sums,
            chunk_counts,
            out=np.full(chunk_sums.shape, np.nan, dtype=np.float64),
            where=chunk_counts > 0,
        )
        valid = (chunk_counts > 0) & (chunk_coverage >= float(threshold))
        block = pd.DataFrame(
            {
                "timestamp": np.repeat(timestamps.to_numpy(), stop - start),
                "cluster_id": cluster_id,
                "level": "gpu",
                "device_id": np.tile(devices[start:stop], n_bins),
                "power_kw": power_w.reshape(-1) / 1000.0,
                "valid": valid.reshape(-1),
                "coverage_ratio": chunk_coverage.reshape(-1),
                "n_samples": chunk_counts.reshape(-1).astype(np.int32),
            }
        )
        yield block


def _iter_device_15min_blocks(
    *,
    timestamps: pd.DatetimeIndex,
    device_ids: Sequence[str],
    sums: np.ndarray,
    counts: np.ndarray,
    native_rows: int,
    expected_samples: int,
    threshold: float,
    cluster_id: str,
    level: str,
    device_chunk: int = 128,
) -> Iterable[pd.DataFrame]:
    n_bins = len(timestamps)
    coverage_all = counts.astype(np.float64) / float(expected_samples)
    for start in range(0, len(device_ids), int(device_chunk)):
        stop = min(start + int(device_chunk), len(device_ids))
        chunk_counts = counts[:, start:stop]
        chunk_coverage = coverage_all[:, start:stop]
        chunk_sums = sums[:, start:stop]
        power_w = np.divide(
            chunk_sums,
            chunk_counts,
            out=np.full(chunk_sums.shape, np.nan, dtype=np.float64),
            where=chunk_counts > 0,
        )
        valid = (chunk_counts > 0) & (chunk_coverage >= float(threshold))
        block = pd.DataFrame(
            {
                "timestamp": np.repeat(timestamps.to_numpy(), stop - start),
                "cluster_id": cluster_id,
                "level": level,
                "device_id": np.tile(list(device_ids[start:stop]), n_bins),
                "power_kw": power_w.reshape(-1) / 1000.0,
                "valid": valid.reshape(-1),
                "coverage_ratio": chunk_coverage.reshape(-1),
                "n_samples": chunk_counts.reshape(-1).astype(np.int32),
            }
        )
        yield block


def _iter_cluster_15min_blocks(
    frame: pd.DataFrame, *, cluster_id: str, level: str, device_id: str
) -> Iterable[pd.DataFrame]:
    if frame.empty:
        return
    block = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            "cluster_id": cluster_id,
            "level": level,
            "device_id": device_id,
            "power_kw": frame["power_kw"].to_numpy(dtype=np.float64),
            "valid": frame["valid"].to_numpy(dtype=bool),
            "coverage_ratio": frame["coverage_ratio"].to_numpy(dtype=np.float64),
            "n_samples": frame["n_samples"].to_numpy(dtype=np.int32),
        }
    )
    yield block


_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ns")),
        pa.field("cluster_id", pa.string()),
        pa.field("level", pa.string()),
        pa.field("device_id", pa.string()),
        pa.field("power_kw", pa.float32()),
        pa.field("valid", pa.bool_()),
        pa.field("coverage_ratio", pa.float32()),
        pa.field("n_samples", pa.int32()),
    ]
)


class ParquetBlockWriter:
    """Incremental zstd parquet writer so the 15-minute table never has to fit in memory."""

    def __init__(self, path: str | Path):
        resolved = assert_not_raw_path(path)
        ensure_dir(resolved.parent)
        self.path = resolved
        self._writer = pq.ParquetWriter(str(resolved), _PARQUET_SCHEMA, compression="zstd")
        self.rows = 0

    def write(self, blocks: Iterable[pd.DataFrame]) -> None:
        for block in blocks:
            if block.empty:
                continue
            table = pa.Table.from_pandas(block, schema=_PARQUET_SCHEMA, preserve_index=False)
            self._writer.write_table(table)
            self.rows += int(table.num_rows)

    def close(self) -> Path:
        self._writer.close()
        return self.path


def _gpu_scan_frame(scan: Mapping[str, Any], expected_samples: int) -> dict[str, Any]:
    return _gpu_level_matrice(scan, expected_samples)


def _merge_cluster_slot_arrays(arrays: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stamps: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    online: list[np.ndarray] = []
    for item in arrays:
        if item[0].size == 0:
            continue
        stamps.append(item[0])
        sums.append(item[1])
        online.append(item[2])
    if not stamps:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    ts = np.concatenate(stamps)
    pw = np.concatenate(sums)
    on = np.concatenate(online)
    order = np.argsort(ts, kind="stable")
    ts, pw, on = ts[order], pw[order], on[order]
    boundaries = np.flatnonzero(np.diff(ts)) + 1
    starts = np.concatenate(([0], boundaries))
    return ts[starts], np.add.reduceat(pw, starts), np.add.reduceat(on, starts)


def _device_coverage_summary(
    n_valid_per_device: np.ndarray, *, slots_total: int, bands: Sequence[float]
) -> dict[str, Any]:
    devices = int(n_valid_per_device.size)
    if devices == 0:
        return {"devices": 0}
    coverage = n_valid_per_device.astype(np.float64) / max(int(slots_total), 1)
    zero = int(np.count_nonzero(n_valid_per_device == 0))
    return {
        "devices": devices,
        "slots_total": int(slots_total),
        "device_slots_total": int(devices * slots_total),
        "device_slots_valid": int(n_valid_per_device.sum()),
        "device_valid_fraction": float(n_valid_per_device.sum() / (devices * slots_total))
        if slots_total
        else None,
        "devices_with_zero_valid_slots": zero,
        "devices_with_zero_valid_slots_ids": None,
        "per_device_valid_slots_min": int(n_valid_per_device.min()),
        "per_device_valid_slots_p50": float(np.median(n_valid_per_device)),
        "per_device_valid_slots_max": int(n_valid_per_device.max()),
        "per_device_coverage_bands": _coverage_bands(coverage, bands),
    }


def _matrix_valid_stats(
    matrices: Sequence[Mapping[str, Any]],
    *,
    level: str,
    cluster_id: str,
    expected_samples: int,
    threshold: float,
    strict: float,
    bands: Sequence[float],
    device_ids: Sequence[str],
) -> dict[str, Any]:
    total_counts = np.hstack([np.asarray(item["counts"]) for item in matrices])
    total_sums = np.hstack([np.asarray(item["sums"]) for item in matrices])
    coverage = total_counts.astype(np.float64) / float(expected_samples)
    valid = (total_counts > 0) & (coverage >= float(threshold))
    bins_with_data_per_device = (total_counts > 0).sum(axis=0)
    stats = _device_coverage_summary(
        bins_with_data_per_device, slots_total=total_counts.shape[0], bands=bands
    )
    stats.update(
        {
            "cluster_id": cluster_id,
            "level": level,
            "rows_total": int(total_counts.size),
            "rows_valid": int(valid.sum()),
            "rows_with_samples": int((total_counts > 0).sum()),
            "coverage_bands": _coverage_bands(coverage.reshape(-1), bands),
            "rows_strict_coverage": int(
                ((total_counts > 0) & (coverage >= float(strict))).sum()
            ),
            "n_samples_total": int(total_counts.sum()),
            "native_samples_per_device_max": int(total_counts.sum(axis=0).max())
            if total_counts.size
            else 0,
            "devices_with_zero_valid_slots_ids": [
                str(device_ids[index])
                for index in np.flatnonzero(total_counts.sum(axis=0) == 0).tolist()
            ][:2000],
        }
    )
    del total_counts, total_sums
    return stats


def _cluster_aggregate_stats(frame: pd.DataFrame, *, strict: float, bands: Sequence[float]) -> dict[str, Any]:
    if frame.empty:
        return {"slots_total": 0, "slots_valid": 0}
    valid = frame["valid"].to_numpy(dtype=bool)
    coverage = frame["coverage_ratio"].to_numpy(dtype=np.float64)
    invalid_run = 0
    current_run = 0
    for flag in valid.tolist():
        if flag:
            current_run = 0
        else:
            current_run += 1
            invalid_run = max(invalid_run, current_run)
    return {
        "slots_total": int(len(frame)),
        "slots_valid": int(valid.sum()),
        "slots_with_samples": int((frame["n_samples"].to_numpy() > 0).sum()),
        "slots_invalid": int((~valid).sum()),
        "max_consecutive_invalid_slots": int(invalid_run),
        "coverage_mean": float(np.mean(coverage)),
        "coverage_bands": _coverage_bands(coverage, bands),
        "slots_strict_coverage": int((coverage >= float(strict)).sum()),
        "timestamp_min": str(pd.Timestamp(frame["timestamp"].iloc[0])),
        "timestamp_max": str(pd.Timestamp(frame["timestamp"].iloc[-1])),
    }


def run_pipeline(
    config: Mapping[str, Any] | None = None,
    *,
    config_path: str | Path = CONFIG_PATH,
    hash_contents: bool = True,
    max_chunks_per_file: int | None = None,
    inventory_payload: Mapping[str, Any] | None = None,
    max_chunks_gpu: int | None = None,
    max_chunks_ipmi: int | None = None,
    write: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the AcmeTrace admission audit end-to-end (chunked, bounded memory)."""
    resolved = dict(config) if config is not None else load_acmetrace_config(config_path)
    outputs = resolved["outputs"]
    aggregation = resolved["aggregation_15min"]
    threshold = float(aggregation["validity_threshold"])
    strict = float(aggregation["strict_coverage_threshold"])
    bands = [float(item) for item in aggregation["coverage_bands"]]
    clusters = resolved["clusters"]
    levels = resolved["levels"]
    inventory_path = resolve_path(str(outputs["inventory_json"]))
    power_audit_path = resolve_path(str(outputs["power_audit_json"]))
    metadata_path = resolve_path(str(outputs["metadata_json"]))
    parquet_path = resolve_path(str(outputs["power_15min_parquet"]))
    raw_root = resolve_path(str(resolved["raw_root"]))
    for path in (inventory_path, power_audit_path, metadata_path, parquet_path):
        assert_not_raw_path(path)
        if path == raw_root or raw_root in path.parents:
            raise ContractError(
                f"refusing to write {path}: outputs must never be written inside the raw "
                f"AcmeTrace root {raw_root}."
            )

    if inventory_payload is not None:
        inventory = dict(inventory_payload)
        if progress:
            print("reusing the existing streamed inventory document (raw files unchanged).", flush=True)
    else:
        inventory = build_inventory(resolved, hash_contents=hash_contents, progress=progress)

    ipmi_node_scope = set(
        _node_columns(clusters[str(resolved["ipmi"]["cluster"])]["node_inventory_file"])
    )
    gpu_scans: dict[str, dict[str, Any]] = {}
    for cluster_key in clusters:
        if progress:
            print(f"scanning {cluster_key} GPU POWER_USAGE (streamed) ...", flush=True)
        chunks = max_chunks_gpu if max_chunks_gpu is not None else max_chunks_per_file
        gpu_scans[cluster_key] = scan_gpu_power_file(
            resolved,
            cluster_key,
            max_chunks=chunks,
            progress_every=int(resolved["streaming"]["progress_every_chunks"]),
        )
    ipmi_scans: dict[str, dict[str, Any]] = {}
    for file_key in resolved["ipmi"]["files"]:
        if progress:
            print(f"scanning IPMI {file_key} (streamed, mixed timestamps) ...", flush=True)
        chunks = max_chunks_ipmi if max_chunks_ipmi is not None else max_chunks_per_file
        ipmi_scans[file_key] = scan_ipmi_power_file(
            resolved,
            file_key,
            scope_nodes=ipmi_node_scope,
            max_chunks=chunks,
            progress_every=int(resolved["streaming"]["progress_every_chunks"]),
        )

    # ---- node-set analysis -------------------------------------------------
    seren_gpu_nodes = set(gpu_scans["seren"]["nodes"])
    ipmi_nodes = sorted({node for item in ipmi_scans.values() for node in item["nodes"]})
    ipmi_node_set = set(ipmi_nodes)
    seren_scope_nodes = sorted(ipmi_node_scope & ipmi_node_set)
    node_sets = {
        "seren_node_inventory_nodes": len(ipmi_node_scope),
        "seren_gpu_nodes": len(seren_gpu_nodes),
        "ipmi_nodes": len(ipmi_node_set),
        "ipmi_intersect_seren_inventory": len(seren_scope_nodes),
        "ipmi_only_nodes": len(ipmi_node_set - ipmi_node_scope),
        "seren_inventory_only_nodes": len(ipmi_node_scope - ipmi_node_set),
        "seren_gpu_nodes_without_ipmi": sorted(seren_gpu_nodes - ipmi_node_set),
        "ipmi_only_node_list": sorted(ipmi_node_set - ipmi_node_scope),
        "seren_scope_node_list": seren_scope_nodes,
    }

    # ---- server-level matrix (union of IPMI nodes) --------------------------
    server_nodes = ipmi_nodes
    server_index = {node: index for index, node in enumerate(server_nodes)}
    min_bin = min(int(item["bin_id_min"]) for item in ipmi_scans.values() if item["bin_id_min"] is not None)
    max_bin = max(int(item["bin_id_max"]) for item in ipmi_scans.values() if item["bin_id_max"] is not None)
    server_bins = max_bin - min_bin + 1
    server_sums = np.zeros((server_bins, len(server_nodes)), dtype=np.float32)
    server_counts = np.zeros((server_bins, len(server_nodes)), dtype=np.int32)
    for item in ipmi_scans.values():
        mapping = np.array([server_index[node] for node in item["nodes"]], dtype=np.int64)
        for bin_id, accumulator in item["bins"].items():
            row = int(bin_id) - min_bin
            width = accumulator["sum_w"].size
            server_sums[row, mapping[:width]] += accumulator["sum_w"]
            server_counts[row, mapping[:width]] += accumulator["count"]
    server_timestamps = pd.to_datetime(np.arange(min_bin, max_bin + 1) * BIN_NS, unit="ns")
    expected_server_samples = int(resolved["ipmi"]["expected_samples_15min"])
    server_valid_slots = (server_counts > 0) & (
        server_counts.astype(np.float64) / expected_server_samples >= threshold
    )

    # ---- Seren cluster IT (reporting in-scope IPMI nodes only) -------------
    cluster_ts, cluster_slot_sum, cluster_slot_online = _merge_cluster_slot_arrays(
        [item["_cluster_arrays"] for item in ipmi_scans.values()]
    )
    expected_cluster_members = len(seren_scope_nodes)
    cluster_it_frame = aggregate_cluster_15min(
        cluster_ts,
        cluster_slot_sum,
        cluster_slot_online,
        expected_samples=expected_server_samples,
        expected_members=expected_cluster_members,
        validity_threshold=threshold,
        member_kind="ipmi_node",
    )
    cluster_it_stats = _cluster_aggregate_stats(cluster_it_frame, strict=strict, bands=bands)
    cluster_it_stats.update(
        {
            "cluster_id": "seren",
            "level": "cluster_it",
            "device_id": "seren_cluster_it",
            "definition": str(resolved["cluster_it"]["rule"]),
            "expected_nodes": expected_cluster_members,
            "native_interval_seconds": int(resolved["ipmi"]["native_interval_seconds"]),
            "expected_samples_15min": expected_server_samples,
            "native_slots": int(cluster_ts.size),
            "node_online_mean": float(np.mean(cluster_slot_online)) if cluster_slot_online.size else None,
            "node_online_min": float(np.min(cluster_slot_online)) if cluster_slot_online.size else None,
            "node_online_max": float(np.max(cluster_slot_online)) if cluster_slot_online.size else None,
            "never_zero_filled_nodes": True,
            "facility_power_claim": False,
            "note": (
                "Sum over reporting server nodes only; missing/offline nodes are excluded and "
                "reduce coverage_ratio. This is Seren cluster IT power, NOT facility power and "
                "NOT a complete-cluster total when coverage is below 1."
            ),
        }
    )

    # ---- GPU-level matrices and cluster GPU sums ---------------------------
    gpu_matrices: dict[str, dict[str, Any]] = {}
    cluster_gpu_sum_frames: dict[str, pd.DataFrame] = {}
    gpu_level_stats: dict[str, dict[str, Any]] = {}
    for cluster_key in clusters:
        scan = gpu_scans[cluster_key]
        expected = int(clusters[cluster_key]["expected_samples_15min"])
        matrice = _gpu_level_matrice(scan, expected)
        gpu_matrices[cluster_key] = matrice
        device_count = matrice["device_count"]
        counts_sum = matrice["counts"].sum(axis=1)
        coverage = counts_sum.astype(np.float64) / float(expected * device_count)
        samples = matrice["cluster_samples"]
        power_w = np.divide(
            matrice["cluster_sum_w"],
            samples,
            out=np.full(samples.shape, np.nan, dtype=np.float64),
            where=samples > 0,
        )
        valid = (samples > 0) & (coverage >= threshold)
        frame = pd.DataFrame(
            {
                "timestamp": matrice["timestamps"],
                "power_kw": power_w / 1000.0,
                "valid": valid,
                "coverage_ratio": coverage,
                "n_samples": samples.astype(np.int32),
            }
        )
        cluster_gpu_sum_frames[cluster_key] = frame
        stats = _cluster_aggregate_stats(frame, strict=strict, bands=bands)
        stats.update(
            {
                "cluster_id": cluster_key,
                "level": "cluster_gpu_sum",
                "device_id": f"{cluster_key}_cluster_gpu_sum",
                "gpu_devices": device_count,
                "native_interval_seconds": int(clusters[cluster_key]["native_interval_seconds"]),
                "expected_samples_15min": expected,
                "definition": str(resolved["cluster_gpu_sum"]["definition"]),
                "never_cluster_it": True,
                "never_facility_power": True,
                "note": "GPU-level aggregate only; excludes CPU/memory/fan/PSU losses.",
            }
        )
        gpu_level_stats[cluster_key] = stats

    # Per-device audit payloads (full per-device coverage/missingness/ranges).
    gpu_device_payloads = {
        cluster_key: _device_stats_payload(gpu_scans[cluster_key]) for cluster_key in clusters
    }
    ipmi_device_payload: dict[str, Any] = {}
    for item in ipmi_scans.values():
        for node, stats in item["node_stats"].items():
            ipmi_device_payload[node] = {
                **stats,
                "server_model": item["server_model"],
                "file_key": item["file_key"],
            }

    # ---- 15-minute parquet (incremental, bounded memory) -------------------
    table_stats: dict[str, Any] = {}
    level_summary: dict[str, Any] = {}
    if write:
        writer = ParquetBlockWriter(parquet_path)
        try:
            for cluster_key in clusters:
                for level in levels[cluster_key]:
                    if level == "gpu":
                        matrice = gpu_matrices[cluster_key]
                        expected = int(clusters[cluster_key]["expected_samples_15min"])
                        coverage = matrice["counts"].astype(np.float64) / float(expected)
                        valid = (matrice["counts"] > 0) & (coverage >= threshold)
                        n_valid_per_device = valid.sum(axis=0)
                        stats = _device_coverage_summary(
                            n_valid_per_device, slots_total=len(matrice["timestamps"]), bands=bands
                        )
                        stats.update(
                            {
                                "cluster_id": cluster_key,
                                "level": level,
                                "rows_total": int(matrice["counts"].size),
                                "rows_valid": int(valid.sum()),
                                "coverage_bands": _coverage_bands(coverage.reshape(-1), bands),
                                "rows_strict_coverage": int(
                                    ((matrice["counts"] > 0) & (coverage >= strict)).sum()
                                ),
                                "devices_with_zero_valid_slots_ids": [
                                    matrice["devices"][index]
                                    for index in np.flatnonzero(n_valid_per_device == 0).tolist()
                                ][:2000],
                            }
                        )
                        writer.write(
                            _iter_gpu_15min_blocks(
                                matrice,
                                cluster_id=cluster_key,
                                expected_samples=expected,
                                threshold=threshold,
                            )
                        )
                        level_summary[f"{cluster_key}/{level}"] = stats
                    elif level == "server":
                        stats = _matrix_valid_stats(
                            [{"counts": server_counts, "sums": server_sums}],
                            level=level,
                            cluster_id=cluster_key,
                            expected_samples=expected_server_samples,
                            threshold=threshold,
                            strict=strict,
                            bands=bands,
                            device_ids=server_nodes,
                        )
                        writer.write(
                            _iter_device_15min_blocks(
                                timestamps=server_timestamps,
                                device_ids=server_nodes,
                                sums=server_sums,
                                counts=server_counts,
                                native_rows=server_bins,
                                expected_samples=expected_server_samples,
                                threshold=threshold,
                                cluster_id=cluster_key,
                                level=level,
                            )
                        )
                        level_summary[f"{cluster_key}/{level}"] = stats
                    elif level == "cluster_it":
                        frame = cluster_it_frame
                        writer.write(
                            _iter_cluster_15min_blocks(
                                frame,
                                cluster_id=cluster_key,
                                level=level,
                                device_id=f"{cluster_key}_cluster_it",
                            )
                        )
                        level_summary[f"{cluster_key}/{level}"] = cluster_it_stats
                    elif level == "cluster_gpu_sum":
                        frame = cluster_gpu_sum_frames[cluster_key]
                        writer.write(
                            _iter_cluster_15min_blocks(
                                frame,
                                cluster_id=cluster_key,
                                level=level,
                                device_id=f"{cluster_key}_cluster_gpu_sum",
                            )
                        )
                        level_summary[f"{cluster_key}/{level}"] = gpu_level_stats[cluster_key]
                    else:  # pragma: no cover - config validation forbids unknown levels
                        raise ValueError(f"Unsupported AcmeTrace level {level!r}.")
        finally:
            writer.close()
        table_stats = {
            "path": str(parquet_path),
            "rows": int(writer.rows),
            "bytes": int(parquet_path.stat().st_size),
            "columns": list(POWER_15MIN_COLUMNS),
            "compression": "zstd",
            "levels": level_summary,
        }

    # ---- audit documents ---------------------------------------------------
    cluster_payload: dict[str, Any] = {}
    for cluster_key in clusters:
        scan = gpu_scans[cluster_key]
        summary = _scan_summary(scan, expected_native=int(clusters[cluster_key]["native_interval_seconds"]))
        summary.update(
            {
                "cluster_id": cluster_key,
                "gpu_power_file": str(scan["path"]),
                "units": {
                    "unit_inferred": str(resolved["source"]["units"]["power_unit_inferred"]),
                    "unit_verified": bool(resolved["source"]["units"]["power_unit_verified"]),
                },
                "devices": scan["devices"],
                "devices_detail": gpu_device_payloads[cluster_key],
                "levels": list(levels[cluster_key]),
            }
        )
        cluster_payload[cluster_key] = summary
    ipmi_payload = {
        "cluster": str(resolved["ipmi"]["cluster"]),
        "native_interval_seconds": int(resolved["ipmi"]["native_interval_seconds"]),
        "expected_samples_15min": expected_server_samples,
        "timestamp_parser": {
            "iso": str(resolved["ipmi"]["timestamp_formats"]["iso"]),
            "chinese_locale": str(resolved["ipmi"]["timestamp_formats"]["chinese_locale"]),
            "note": "CPU_D mixes ISO +08:00 and Chinese-locale CST strings for the same timezone.",
        },
        "files": {},
        "nodes": ipmi_device_payload,
    }
    for file_key, item in ipmi_scans.items():
        ipmi_payload["files"][file_key] = {
            "server_model": item["server_model"],
            "path": item["path"],
            "native_rows": item["native_rows"],
            "nodes": item["node_count"],
            "node_list": item["nodes"],
            "timestamp_min": item["timestamp_min"],
            "timestamp_max": item["timestamp_max"],
            "native_interval_mode_seconds": item["native_interval_mode_seconds"],
            "delta_seconds_histogram": item["delta_seconds_histogram"],
            "negative_timestamp_deltas": item["negative_timestamp_deltas"],
            "duplicate_timestamps": item["duplicate_timestamps"],
            "duplicated_node_timestamp_rows": item["duplicated_node_timestamp_rows"],
            "duplicated_node_slot_rows_averaged": item["duplicated_node_slot_rows_averaged"],
            "metering": item["metering"],
        }
    hierarchy = {
        "levels": {
            "gpu": "per-GPU DCGM POWER_USAGE (15 s native)",
            "server": "per-node IPMI Sys_Total_Power (300 s native, Seren only)",
            "cluster_it": "sum of reporting Seren IPMI nodes per native slot, then 15-min mean",
            "cluster_gpu_sum": "sum of valid per-GPU power per native slot, then 15-min mean",
        },
        "seren": {
            "has_server_power": True,
            "has_cluster_it": True,
            "cluster_it": cluster_it_stats,
            "cluster_gpu_sum": gpu_level_stats["seren"],
            "gpu_vs_server": {
                "note": (
                    "GPU sum is a GPU-level lower bound and is NOT comparable to cluster IT: it "
                    "excludes CPU/memory/fan/PSU losses and any node without GPUs."
                ),
                "seren_gpu_nodes": len(seren_gpu_nodes),
                "seren_server_nodes": len(seren_scope_nodes),
            },
        },
        "kalos": {
            "has_server_power": False,
            "has_cluster_it": False,
            "cluster_gpu_sum": gpu_level_stats["kalos"],
            "note": (
                "Kalos has NO IPMI server power in the local copy; only a GPU-level aggregate "
                "is available. No Kalos server or cluster-IT series is produced."
            ),
        },
        "facility_power_available": False,
        "facility_power_statement": (
            "AcmeTrace contains no facility total power meter. GPU sums and Seren cluster IT "
            "are IT-side values and must never be reported as facility power."
        ),
    }
    power_audit = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "generated_by": "src.data.acmetrace_power.run_pipeline",
        "dataset_id": "acmetrace",
        "revision": str(resolved["source"]["revision"]),
        "scope": {
            "inventory_files": int(inventory["summary"]["local_files"]),
            "inventory_bytes": int(inventory["summary"]["local_bytes"]),
            "clusters": list(clusters),
            "streamed": True,
            "max_chunks_per_file": max_chunks_per_file,
        },
        "units": {
            "power_unit_inferred": str(resolved["source"]["units"]["power_unit_inferred"]),
            "power_unit_verified": bool(resolved["source"]["units"]["power_unit_verified"]),
            "unit_evidence": str(resolved["source"]["units"]["unit_evidence"]),
            "conversion": str(resolved["source"]["units"]["conversion"]),
            "zero_policy": str(resolved["source"]["units"]["zero_policy"]),
            "plausible_ranges_w": dict(resolved["source"]["units"]["plausible_ranges_w"]),
        },
        "clusters": cluster_payload,
        "ipmi": ipmi_payload,
        "node_sets": node_sets,
        "hierarchy": hierarchy,
        "aggregation_15min": {
            "frequency": str(aggregation["frequency"]),
            "step_seconds": BIN_SECONDS,
            "label": str(aggregation["label"]),
            "validity_threshold": threshold,
            "strict_coverage_threshold": strict,
            "coverage_bands": bands,
            "interpolation": False,
            "zero_fill": False,
            "missing_policy": str(aggregation["missing_policy"]),
            "level_summary": level_summary,
        },
        "readiness": {
            "ready_for_training": False,
            "reasons": [str(item) for item in resolved["readiness"]["reasons"]],
            "ready_for_15min_analysis": bool(resolved["readiness"]["ready_for_15min_analysis"]),
            "ready_for_15min_analysis_note": str(
                resolved["readiness"]["ready_for_15min_analysis_note"]
            ),
        },
        "limitations": _limitations(resolved),
    }
    metadata = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "generated_by": "src.data.acmetrace_power.run_pipeline",
        "dataset_id": "acmetrace",
        "source": {
            "repository": str(resolved["source"]["repository"]),
            "revision": str(resolved["source"]["revision"]),
            "raw_root": str(resolve_path(str(resolved["raw_root"]))),
            "hf_tree": str(resolve_path(str(resolved["hf_tree"]))),
            "readme": str(resolve_path(str(resolved["readme"]))),
            "clusters": list(clusters),
            "timezone": str(resolved["source"]["timezone"]),
            "timezone_verified": bool(resolved["source"]["timezone_verified"]),
            "timestamp_semantics": str(resolved["source"]["timestamp_semantics"]),
            "units": power_audit["units"],
        },
        "clusters": {
            cluster_key: {
                "display_name": str(clusters[cluster_key]["display_name"]),
                "native_interval_seconds": int(clusters[cluster_key]["native_interval_seconds"]),
                "expected_samples_15min": int(clusters[cluster_key]["expected_samples_15min"]),
                "gpu_power_file": str(clusters[cluster_key]["gpu_power_file"]),
                "has_ipmi_server_power": bool(clusters[cluster_key]["has_ipmi_server_power"]),
                "levels_available": list(levels[cluster_key]),
                "gpu_devices": int(gpu_scans[cluster_key]["device_count"]),
                "gpu_nodes": int(gpu_scans[cluster_key]["node_count"]),
            }
            for cluster_key in clusters
        },
        "node_sets": node_sets,
        "hierarchy": hierarchy,
        "aggregation_15min": power_audit["aggregation_15min"],
        "inventory": {
            "path": str(inventory_path),
            "summary": inventory["summary"],
            "integrity": inventory["integrity"],
        },
        "readiness": power_audit["readiness"],
        "limitations": power_audit["limitations"],
        "artefacts": {},
        "environment": environment_report(),
    }
    if write:
        write_json(inventory_path, _json_ready(inventory))
        write_json(power_audit_path, _json_ready(power_audit))
        metadata["artefacts"] = {
            "inventory_json": {
                "path": str(inventory_path),
                "bytes": int(inventory_path.stat().st_size),
            },
            "power_audit_json": {
                "path": str(power_audit_path),
                "bytes": int(power_audit_path.stat().st_size),
            },
            "power_15min_parquet": table_stats,
            "metadata_json": {"path": str(metadata_path)},
        }
        write_json(metadata_path, _json_ready(metadata))
        metadata["artefacts"]["metadata_json"]["bytes"] = int(metadata_path.stat().st_size)
        write_json(metadata_path, _json_ready(metadata))
    return {
        "inventory": inventory,
        "power_audit": power_audit,
        "metadata": metadata,
        "table_stats": table_stats,
        "node_sets": node_sets,
    }


def _limitations(config: Mapping[str, Any]) -> list[str]:
    return [
        "Power units are not documented by the AcmeTrace README; the W inference is magnitude-based "
        "and every field keeps unit_verified=false.",
        "The CPU_D IPMI file mixes Chinese-locale CST and ISO +08:00 timestamps; the parser is "
        "explicit, but the original timezone documentation is absent (timezone_verified=false).",
        "Server power (IPMI) exists for Seren only. Kalos has no server power, so no Kalos "
        "server-level or cluster-IT series exists; only a GPU-level aggregate is produced.",
        "GPU-level sums are NOT cluster IT power and NOT facility power: they exclude CPU, memory, "
        "fan and PSU losses. AcmeTrace contains no facility power meter at all.",
        "Zero/negative power readings are treated as invalid/offline and never as 0 kW; missing "
        "nodes/GPUs are never zero-filled, so cluster sums are reporting-node sums with explicit "
        "coverage below 1.",
        "Native sampling differs by source (15 s DCGM GPU power, 300 s IPMI server power) and is "
        "not resampled except by the explicit 15-minute interval-mean adaptation.",
        "The 15-minute validity threshold (0.5) and stricter 0.9 coverage threshold are analysis "
        "choices, not properties of the data; both are reported so downstream work can re-select.",
        "Kalos GPU-wide files cover 2023-08-15 .. 2023-08-31 only, while Seren GPU power covers "
        "2023-05-16 .. 2023-08-18; the two clusters do not share a full common window.",
        "The local copy does not include 36 HF-tree files (previous-work traces, util_pkl, figures, "
        "notebooks, license); the inventory lists them as missing and does not fabricate them.",
    ]
