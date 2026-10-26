"""Stage-1 source-domain data pipeline: M100 node telemetry, UCC audit, alignment.

Task-book scope (V2.0 section 3.1-3.6, section 9.5, Milestone 3, section 13.7):

* read the REAL primary source (M100 ExaData node hardware telemetry) without ever
  modifying ``data/raw/``;
* build the long-format source-panel contract with EXACTLY the logical fields
  ``timestamp, domain_id, feature_name, feature_value, unit, observed_mask,
  coverage_ratio`` -- no label/target column exists;
* keep every missing observation explicit (``observed_mask=False`` + NaN), never
  zero-filled, and never compress the time axis by deleting internal gaps;
* declare field semantics (semantic type, unit, aggregation rule, availability,
  physical range, scope/measurement boundary, audit status) in metadata;
* audit the secondary real source (UCC 2023) at its native 10-minute frequency
  instead of fabricating a 15-minute product;
* register candidate sources (Alibaba GPU v2020) with explicit reasons instead of
  aggregating event tables into fake continuous series;
* classify source / target / shared / incomparable fields through
  :class:`src.data.registry.FieldRegistry` (never by raw column-name matching) and
  emit the tracked ``configs/datasets/feature_registry.yaml`` deliverable.

Nothing in this module trains a model or executes any source->target mapping.
The global fusion weight stays unresolved (``None``); the only defined value is
the target-only *local* routing weight ``1.0`` (a Stage-4 semantic reservation).
"""

from __future__ import annotations

import glob as _glob
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.data.contracts import (
    AuditStatus,
    ContractError,
    DatasetRole,
    DatasetSpec,
    FeatureSpec,
    IntervalStatistic,
    LabelOrigin,
    TargetKind,
    TargetScope,
    TargetSpec,
)
from src.data.loaders import assert_not_raw_path, read_table, write_table
from src.data.registry import (
    TARGET_ONLY_LOCAL_WEIGHT,
    UNRESOLVED_GLOBAL_FUSION_WEIGHT,
    AlignmentResult,
    FeatureRelation,
    FieldRegistry,
)
from src.utils import (
    ensure_dir,
    environment_report,
    file_metadata,
    load_config,
    resolve_path,
    sha256_file,
    utc_timestamp,
    write_json,
)

# ---------------------------------------------------------------------------
# module-level contract constants (data contract, not dataset constants)
# ---------------------------------------------------------------------------

SOURCES_CONFIG: str = "configs/datasets/sources.yaml"
FEATURE_REGISTRY_PATH: str = "configs/datasets/feature_registry.yaml"
DEFAULT_PROCESSED_DIR: str = "data/processed"

PANEL_FILENAME: str = "processed_source_features.parquet"
METADATA_FILENAME: str = "source_metadata.json"
QUALITY_FILENAME: str = "source_quality.json"

#: The logical long-format panel fields from task book section 3.2.  Exactly these
#: columns are written, in this order, and no label/target column may be added.
PANEL_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "domain_id",
    "feature_name",
    "feature_value",
    "unit",
    "observed_mask",
    "coverage_ratio",
)

#: Semantic type vocabulary from task book section 3.2.
SEMANTIC_TYPES: frozenset[str] = frozenset(
    {"power", "temperature", "fan_speed", "utilization", "event_count", "memory", "network", "other"}
)

#: Aggregation-rule vocabulary from task book section 3.3.
AGGREGATION_RULES: frozenset[str] = frozenset(
    {"interval_mean", "interval_sum", "instantaneous_mean", "event_count", "state", "native_kept"}
)

#: Availability vocabulary from task book section 3.2 (leakage protection).
AVAILABILITIES: frozenset[str] = frozenset(
    {"historical_observed", "future_known", "future_unknown"}
)

#: Output column/fragment tokens that must never appear in the source panel.
_FORBIDDEN_PANEL_TOKENS: tuple[str, ...] = ("label", "target", "active_power_kw", "targets_kw")


# ---------------------------------------------------------------------------
# configuration loading / validation
# ---------------------------------------------------------------------------


def _parse_duration(value: str | pd.Timedelta) -> pd.Timedelta:
    """Parse a pandas frequency string without the numpy generic-timedelta warning."""
    if isinstance(value, pd.Timedelta):
        return value
    return pd.Timedelta(pd.tseries.frequencies.to_offset(str(value)).nanos, unit="ns")


def load_source_config(path: str | Path = SOURCES_CONFIG) -> dict[str, Any]:
    """Load and structurally validate ``configs/datasets/sources.yaml``."""
    config = load_config(path)
    required = (
        "frequency",
        "points_per_day",
        "timezone",
        "panel",
        "m100_exadata",
        "ucc2023",
        "alibaba_gpu_v2020",
        "target_reference",
        "field_registry",
        "analysis",
        "paths",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ContractError(f"Source config {path} is missing top-level keys: {missing}.")
    try:
        native = _parse_duration(str(config["frequency"]))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"config frequency {config['frequency']!r} is not a duration.") from exc
    if native <= pd.Timedelta(0):
        raise ContractError("config frequency must be positive.")
    if not isinstance(config["points_per_day"], int) or int(config["points_per_day"]) < 1:
        raise ContractError("config points_per_day must be a positive integer.")
    panel_columns = tuple(config["panel"].get("columns", ()))
    if panel_columns != PANEL_COLUMNS:
        raise ContractError(
            f"panel.columns in {path} must equal the logical contract {list(PANEL_COLUMNS)}, "
            f"got {list(panel_columns)}."
        )
    sources = config["panel"].get("sources")
    if not isinstance(sources, list) or not sources or sources != ["m100_exadata"]:
        raise ContractError(
            "Stage 1 implements the 15-minute source panel for `m100_exadata` only; UCC is "
            "kept at its native 10-minute frequency and is audited rather than panelised."
        )
    _feature_entries(config["m100_exadata"], "m100_exadata")
    _feature_entries(config["target_reference"], "target_reference")
    _feature_entries(config["ucc2023"], "ucc2023")
    return config


def _feature_entries(section: Mapping[str, Any], source_id: str) -> list[dict[str, Any]]:
    """Validate and return one config section's feature table."""
    entries = section.get("features")
    if not isinstance(entries, list) or not entries:
        raise ContractError(f"{source_id}: `features` must be a nonempty list.")
    seen_canonical: set[str] = set()
    seen_raw: set[str] = set()
    required = (
        "canonical_name",
        "raw_name",
        "unit",
        "semantic_type",
        "statistic",
        "aggregation_rule",
        "availability",
        "physical_range",
        "measurement_point",
    )
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ContractError(f"{source_id}: every feature entry must be a mapping.")
        missing = [key for key in required if key not in entry]
        if missing:
            raise ContractError(f"{source_id}: feature entry missing {missing}: {dict(entry)}")
        canonical = str(entry["canonical_name"]).strip()
        raw = str(entry["raw_name"]).strip()
        if not canonical or not raw:
            raise ContractError(f"{source_id}: feature names must be nonempty strings.")
        if canonical in seen_canonical:
            raise ContractError(f"{source_id}: duplicate canonical_name {canonical!r}.")
        if raw in seen_raw:
            raise ContractError(f"{source_id}: duplicate raw_name {raw!r}.")
        seen_canonical.add(canonical)
        seen_raw.add(raw)
        if entry["semantic_type"] not in SEMANTIC_TYPES:
            raise ContractError(
                f"{source_id}.{canonical}: semantic_type={entry['semantic_type']!r} is not one of "
                f"{sorted(SEMANTIC_TYPES)}."
            )
        if entry["aggregation_rule"] not in AGGREGATION_RULES:
            raise ContractError(
                f"{source_id}.{canonical}: aggregation_rule={entry['aggregation_rule']!r} is not "
                f"one of {sorted(AGGREGATION_RULES)}."
            )
        if entry["availability"] not in AVAILABILITIES:
            raise ContractError(
                f"{source_id}.{canonical}: availability={entry['availability']!r} is not one of "
                f"{sorted(AVAILABILITIES)}."
            )
        try:
            IntervalStatistic(str(entry["statistic"]))
        except ValueError as exc:
            raise ContractError(
                f"{source_id}.{canonical}: statistic={entry['statistic']!r} is not a valid "
                f"IntervalStatistic."
            ) from exc
        if not str(entry["unit"]).strip():
            raise ContractError(f"{source_id}.{canonical}: unit must be nonempty.")
        physical_range = entry["physical_range"]
        if physical_range is not None:
            if not isinstance(physical_range, (list, tuple)) or len(physical_range) != 2:
                raise ContractError(f"{source_id}.{canonical}: physical_range must be [low, high] or null.")
            low, high = float(physical_range[0]), float(physical_range[1])
            if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
                raise ContractError(
                    f"{source_id}.{canonical}: physical_range must be finite and increasing."
                )
    return [dict(entry) for entry in entries]


# ---------------------------------------------------------------------------
# M100 file discovery / loading
# ---------------------------------------------------------------------------


def _sort_key(stem: str) -> tuple[int, Any]:
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def m100_files(
    config: Mapping[str, Any], *, nodes: Sequence[str] | None = None
) -> dict[str, Path]:
    """Map ``domain_id -> parquet path`` for the configured M100 files.

    ``nodes`` restricts the set to the given file stems (``"10"`` or ``10``);
    a requested node that does not exist raises instead of being silently skipped.
    """
    section = config["m100_exadata"]
    pattern = str(resolve_path(str(section["glob"])))
    template = str(section["domain_id_template"])
    matches = sorted(
        (Path(item) for item in _glob.glob(pattern)),
        key=lambda path: _sort_key(path.stem),
    )
    if not matches:
        raise FileNotFoundError(
            f"No M100 parquet files matched {pattern!r}; raw data is missing. This pipeline "
            "never fabricates source data."
        )
    discovered = {path.stem: path for path in matches}
    if nodes is not None:
        wanted = [str(node) for node in nodes]
        unknown = [node for node in wanted if node not in discovered]
        if unknown:
            raise ContractError(
                f"Requested M100 nodes {unknown} are not present; available stems: "
                f"{sorted(discovered, key=_sort_key)}."
            )
        selected_stems = wanted
    else:
        selected_stems = sorted(discovered, key=_sort_key)
    result: dict[str, Path] = {}
    for stem in selected_stems:
        domain_id = template.format(stem=stem)
        if domain_id in result:
            raise ContractError(f"Duplicate M100 domain_id {domain_id!r} from template {template!r}.")
        result[domain_id] = discovered[stem]
    return result


def _feature_spec_from_entry(
    entry: Mapping[str, Any],
    *,
    source: str,
    audit_status: AuditStatus,
    native_frequency: str | None,
) -> FeatureSpec:
    physical_range = entry.get("physical_range")
    notes = str(entry.get("notes", ""))
    measurement_point = str(entry.get("measurement_point", ""))
    if measurement_point:
        notes = f"measurement_point={measurement_point}. {notes}".strip()
    return FeatureSpec(
        raw_name=str(entry["raw_name"]),
        canonical_name=str(entry["canonical_name"]),
        unit=str(entry["unit"]),
        statistic=IntervalStatistic(str(entry["statistic"])),
        physical_range=(float(physical_range[0]), float(physical_range[1]))
        if physical_range is not None
        else None,
        native_frequency=native_frequency,
        availability_lag=pd.Timedelta(0),
        future_known=str(entry["availability"]) == "future_known",
        missing_policy="mask",
        source=source,
        audit_status=audit_status,
        notes=notes,
    )


def _load_m100_node(
    path: str | Path, config: Mapping[str, Any], *, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Implementation of :func:`load_m100_node` with an explicit already-loaded config."""
    section = config["m100_exadata"]
    entries = _feature_entries(section, "m100_exadata")
    by_raw = {str(entry["raw_name"]): entry for entry in entries}
    excluded = [str(item) for item in section.get("excluded_columns", [])]
    if columns is None:
        selected = [str(entry["raw_name"]) for entry in entries]
    else:
        selected = [str(column) for column in columns]
        unknown = [column for column in selected if column not in by_raw]
        if unknown:
            hint = (
                f" Bookkeeping/excluded columns {sorted(set(unknown) & set(excluded))} are not "
                "physical features."
                if set(unknown) & set(excluded)
                else ""
            )
            raise ContractError(
                f"Requested M100 columns {unknown} are not in the curated feature list. "
                f"Curated columns: {sorted(by_raw)}.{hint}"
            )
        if not selected:
            raise ContractError("columns must not be empty.")
    resolved = resolve_path(path)
    frame = read_table(resolved, columns=["timestamp", *selected])
    if "timestamp" not in frame.columns:
        raise ContractError(f"M100 file {resolved} has no `timestamp` column.")
    index = pd.DatetimeIndex(frame["timestamp"])
    if index.tz is None:
        raise ContractError(
            f"M100 file {resolved} has timezone-naive timestamps; the audited files are tz-aware "
            "UTC, so a missing timezone must be fixed rather than assumed."
        )
    index = index.tz_convert("UTC")
    if bool(index.duplicated().any()):
        raise ContractError(
            f"M100 file {resolved} contains duplicate timestamps; duplicate handling must be "
            "explicit, not silent."
        )
    order = np.argsort(index.view("int64"), kind="stable")
    if not bool(np.all(order == np.arange(len(order)))):
        frame = frame.iloc[order].reset_index(drop=True)
        index = index[order]
    data = {
        str(entry["canonical_name"]): pd.to_numeric(
            frame[str(entry["raw_name"])], errors="coerce"
        ).to_numpy(dtype="float64")
        for entry in entries
        if str(entry["raw_name"]) in selected
    }
    result = pd.DataFrame(data, index=index)
    result.index.name = "timestamp"
    result.attrs.update(
        {
            "domain_id": str(section["domain_id_template"]).format(stem=resolved.stem),
            "source_file": str(resolved),
            "source_dataset": str(section["dataset_id"]),
            "native_frequency": str(section["native_frequency"]),
            "scope": str(section["scope"]),
            "measurement_boundary": str(section["measurement_boundary"]),
        }
    )
    return result


def load_m100_node(path: str | Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load ONE M100 node file as a canonical, UTC-indexed feature frame.

    Parameters
    ----------
    path:
        Path to one ``data/raw/M100/extracted/<stem>.parquet`` file.
    columns:
        Optional subset of *raw* metric names (e.g. ``["total_power_avg"]``).  The
        default reads the curated metric list from ``configs/datasets/sources.yaml``
        only; the other ~340 telemetry columns (and the ``value`` /
        ``__index_level_0__`` bookkeeping columns) are never loaded as features.

    Returns
    -------
    pandas.DataFrame
        Indexed by tz-aware UTC ``DatetimeIndex`` named ``timestamp``, columns are
        canonical feature names, values ``float64``.  ``attrs`` carries the domain id,
        source file and node-level measurement boundary.
    """
    return _load_m100_node(path, load_source_config(), columns=columns)


# ---------------------------------------------------------------------------
# long-format source panel
# ---------------------------------------------------------------------------


def build_source_panel(
    config: Mapping[str, Any],
    *,
    nodes: Sequence[str] | None = None,
    max_rows_per_node: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the memory-bounded long-format source panel for M100.

    Each M100 file is one independent node/domain.  Every domain is reindexed onto
    its OWN complete 15-minute UTC grid from its own first to last native timestamp:
    internal gaps become NaN rows with ``observed_mask=False`` (the time axis is not
    compressed) and domains are never cross-spliced by date.

    ``nodes`` / ``max_rows_per_node`` are optional bounded-run overrides used by the
    CLI and tests; they do not change the semantics of the panel.

    Returns ``(panel, summary)``.  The panel is written with compact dtypes
    (categorical domain/feature/unit, float32 values and coverage) so a full run of
    16 nodes x 12 features stays well below 1 GB in memory.
    """
    section = config["m100_exadata"]
    entries = _feature_entries(section, "m100_exadata")
    features = [str(entry["canonical_name"]) for entry in entries]
    raw_names = [str(entry["raw_name"]) for entry in entries]
    units = [str(entry["unit"]) for entry in entries]
    unit_list = list(dict.fromkeys(units))
    unit_index = np.asarray([unit_list.index(unit) for unit in units], dtype=np.int16)

    files = m100_files(config, nodes=nodes)
    frequency = str(config["frequency"])
    if max_rows_per_node is None:
        max_rows_per_node = section.get("default_max_rows_per_node")
    if max_rows_per_node is not None:
        if isinstance(max_rows_per_node, bool) or not isinstance(max_rows_per_node, int):
            raise ContractError(
                f"max_rows_per_node must be a positive integer or None, got {max_rows_per_node!r}."
            )
        if max_rows_per_node < 1:
            raise ContractError("max_rows_per_node must be >= 1.")

    feature_count = len(features)
    prepared: list[tuple[str, pd.DatetimeIndex, np.ndarray]] = []
    domain_detail: dict[str, dict[str, Any]] = {}
    total_rows = 0
    for domain_id, path in files.items():
        wide = _load_m100_node(path, config, columns=raw_names)
        raw_rows = int(len(wide))
        if max_rows_per_node is not None and raw_rows > max_rows_per_node:
            wide = wide.iloc[:max_rows_per_node]
        if wide.empty:
            raise ContractError(f"M100 domain {domain_id!r} has no rows after truncation.")
        grid = pd.date_range(
            start=wide.index.min(), end=wide.index.max(), freq=frequency, tz=wide.index.tz
        )
        wide = wide.reindex(grid)
        values = wide.to_numpy(dtype=np.float32, copy=True)
        non_finite = int(np.isinf(values).sum())
        values[~np.isfinite(values)] = np.nan
        mask = ~np.isnan(values)
        observed_slots = int(mask.any(axis=1).sum())
        prepared.append((domain_id, grid, values))
        total_rows += int(len(grid)) * feature_count
        domain_detail[domain_id] = {
            "source_file": str(resolve_path(path)),
            "raw_rows_used": int(min(raw_rows, max_rows_per_node or raw_rows)),
            "grid_slots": int(len(grid)),
            "grid_start": grid[0].isoformat(),
            "grid_end": grid[-1].isoformat(),
            "observed_slots": observed_slots,
            "fully_missing_slots": int(len(grid)) - observed_slots,
            "native_rows_observed": int(np.count_nonzero(mask)),
            "non_finite_values_converted_to_nan": non_finite,
            "duplicate_timestamps": 0,
        }

    if total_rows < 1:
        raise ContractError("The source panel would be empty.")

    timestamp = np.empty(total_rows, dtype="datetime64[ns]")
    domain_codes = np.empty(total_rows, dtype=np.int16)
    feature_codes = np.empty(total_rows, dtype=np.int16)
    values_flat = np.empty(total_rows, dtype=np.float32)
    mask_flat = np.empty(total_rows, dtype=bool)
    coverage_flat = np.empty(total_rows, dtype=np.float32)

    domains = list(files)
    offset = 0
    for domain_code, (domain_id, grid, values) in enumerate(prepared):
        block = int(len(grid)) * feature_count
        stop = offset + block
        naive = grid.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        timestamp[offset:stop] = np.repeat(naive, feature_count)
        domain_codes[offset:stop] = np.int16(domain_code)
        feature_codes[offset:stop] = np.tile(np.arange(feature_count, dtype=np.int16), len(grid))
        block_values = values.reshape(-1)
        values_flat[offset:stop] = block_values
        block_mask = np.isfinite(block_values)
        mask_flat[offset:stop] = block_mask
        coverage_flat[offset:stop] = block_mask.astype(np.float32)
        offset = stop
    if offset != total_rows:
        raise ContractError(f"Panel fill mismatch: {offset} rows filled, {total_rows} expected.")
    del prepared

    panel = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamp).tz_localize(str(config["timezone"])),
            "domain_id": pd.Categorical.from_codes(domain_codes, categories=domains),
            "feature_name": pd.Categorical.from_codes(feature_codes, categories=features),
            "feature_value": values_flat,
            "unit": pd.Categorical.from_codes(unit_index[feature_codes], categories=unit_list),
            "observed_mask": mask_flat,
            "coverage_ratio": coverage_flat,
        }
    )

    # -- contract assertions -------------------------------------------------
    if list(panel.columns) != list(PANEL_COLUMNS):
        raise ContractError(f"Panel columns must be exactly {list(PANEL_COLUMNS)}.")
    forbidden = [
        column
        for column in panel.columns
        if any(token in column.lower() for token in _FORBIDDEN_PANEL_TOKENS)
    ]
    if forbidden:
        raise ContractError(f"Source panel must carry no label/target column, found {forbidden}.")
    if not bool((panel["feature_value"].isna() == ~panel["observed_mask"]).all()):
        raise ContractError(
            "observed_mask must be False exactly where feature_value is NaN: missing values are "
            "never encoded as a real zero."
        )
    if bool(panel.loc[panel["observed_mask"], "feature_value"].isna().any()):
        raise ContractError("observed_mask=True rows must carry a finite feature_value.")
    if bool(np.isinf(panel["feature_value"].to_numpy()).any()):
        raise ContractError("feature_value still contains inf; non-finite values must be masked.")

    summary: dict[str, Any] = {
        "rows": int(len(panel)),
        "columns": list(PANEL_COLUMNS),
        "domains": domains,
        "domain_count": len(domains),
        "features": features,
        "feature_count": feature_count,
        "frequency": frequency,
        "points_per_day": int(config["points_per_day"]),
        "max_rows_per_node": max_rows_per_node,
        "nodes_filter": None if nodes is None else [str(node) for node in nodes],
        "memory_bytes": int(panel.memory_usage(deep=True).sum()),
        "domains_detail": domain_detail,
        "timeline_policy": str(config["panel"].get("timeline_policy", "")),
        "missing_semantics": {
            "nan_means": "missing observation (never a measured zero)",
            "observed_mask": "True exactly where feature_value is finite",
            "zero_filled": False,
            "timeline_compressed": False,
        },
    }
    return panel, summary


# ---------------------------------------------------------------------------
# native-frequency audit
# ---------------------------------------------------------------------------
# native-frequency audit
# ---------------------------------------------------------------------------


def audit_m100_timestamps(
    config: Mapping[str, Any], *, files: Mapping[str, Path] | None = None
) -> dict[str, Any]:
    """Audit the native sampling of the REAL M100 files (timestamp column only).

    ``files`` optionally restricts/renames the audited domains (a ``domain_id ->
    path`` mapping); by default every configured M100 file is audited.

    Returns per-domain native-interval statistics, gap counts and the number of raw
    samples per 15-minute slot.  This is a read-only audit: nothing is resampled.
    """
    targets = dict(files) if files is not None else m100_files(config)
    native_seconds = float(_parse_duration(str(config["frequency"])).total_seconds())
    per_domain: dict[str, Any] = {}
    delta_counter: Counter[float] = Counter()
    samples_per_slot: Counter[int] = Counter()
    for domain_id, path in targets.items():
        resolved = resolve_path(path)
        frame = read_table(resolved, columns=["timestamp"])
        index = pd.DatetimeIndex(frame["timestamp"])
        if index.tz is None:
            raise ContractError(f"{resolved} has timezone-naive timestamps; cannot audit UTC grid.")
        index = index.tz_convert("UTC")
        deltas = index.to_series().diff().dropna().dt.total_seconds()
        delta_counts = Counter(float(value) for value in deltas.to_numpy())
        delta_counter.update(delta_counts)
        native_mode = float(delta_counts.most_common(1)[0][0]) if delta_counts else None
        all_multiples = bool(
            np.allclose(np.mod(deltas.to_numpy(dtype="float64"), native_seconds), 0.0)
        ) if len(deltas) else True
        slots = index.floor(pd.Timedelta(int(native_seconds), unit="s"))
        per_slot = slots.value_counts()
        grid = pd.date_range(index.min(), index.max(), freq=str(config["frequency"]), tz=index.tz)
        slot_counts = Counter(int(count) for count in per_slot.to_numpy())
        slot_counts[0] = len(grid) - len(per_slot)
        samples_per_slot.update(slot_counts)
        per_domain[domain_id] = {
            "source_file": str(resolved),
            "raw_rows": int(len(index)),
            "first_timestamp": index.min().isoformat(),
            "last_timestamp": index.max().isoformat(),
            "duplicate_timestamps": int(index.duplicated().sum()),
            "monotonic_increasing": bool(index.is_monotonic_increasing),
            "native_interval_seconds": native_mode,
            "native_interval_mode_count": int(delta_counts[native_mode]) if native_mode else 0,
            "native_interval_matches_config": bool(native_mode == native_seconds),
            "all_deltas_multiples_of_native": all_multiples,
            "delta_seconds_counts": {str(int(key)): int(value) for key, value in delta_counts.most_common(20)},
            "distinct_delta_values": int(len(delta_counts)),
            "gap_count": int(sum(1 for value in delta_counts if value > native_seconds)),
            "max_gap_seconds": float(deltas.max()) if len(deltas) else 0.0,
            "expected_15min_slots": int(len(grid)),
            "observed_15min_slots": int(len(per_slot)),
            "samples_per_15min_slot_counts": {str(key): int(value) for key, value in sorted(slot_counts.items())},
        }
    verdict = "native_interval_audit_failed"
    if per_domain and all(
        item["native_interval_seconds"] == native_seconds
        and item["native_interval_matches_config"]
        and item["all_deltas_multiples_of_native"]
        for item in per_domain.values()
    ):
        verdict = "15min_native_kept"
    return {
        "dataset_id": str(config["m100_exadata"]["dataset_id"]),
        "declared_frequency": str(config["frequency"]),
        "target_frequency": str(config["frequency"]),
        "verdict": verdict,
        "aggregation_status": "native_preserved",
        "domains": per_domain,
        "global_delta_seconds_counts": {str(int(key)): int(value) for key, value in delta_counter.most_common(20)},
        "global_samples_per_15min_slot_counts": {
            str(key): int(value) for key, value in sorted(samples_per_slot.items())
        },
    }


# ---------------------------------------------------------------------------
# UCC bounded audit (native frequency preserved; no 15-minute fabrication)
# ---------------------------------------------------------------------------


def _count_file_rows(path: str | Path) -> int:
    with resolve_path(path).open("rb") as handle:
        return int(sum(1 for _ in handle)) - 1


def build_ucc_audit(
    config: Mapping[str, Any], *, hash_files: bool = True
) -> dict[str, Any]:
    """Bounded, honest audit of the UCC 2023 secondary real source.

    Only the first ``ucc2023.bounded_audit_rows`` rows of each file are inspected;
    unit/timezone/full-file coverage are therefore explicitly *not* verified.  The
    function never writes data and never resamples UCC to 15 minutes.
    """
    section = config["ucc2023"]
    bounded = int(section.get("bounded_audit_rows", 300000))
    if bounded < 1:
        raise ContractError("ucc2023.bounded_audit_rows must be positive.")
    native = _parse_duration(str(section["native_frequency"]))
    native_seconds = float(native.total_seconds())
    target = str(config["frequency"])

    tables: dict[str, Any] = {}
    files: dict[str, Any] = {}
    for key in ("primary_file", "secondary_file"):
        path = str(section[key])
        resolved = resolve_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"UCC file not found: {resolved}")
        frame = pd.read_csv(resolved, nrows=bounded)
        if "Time" not in frame.columns or "Series" not in frame.columns:
            raise ContractError(
                f"UCC file {resolved} lacks expected `Series`/`Time` columns; audit cannot proceed."
            )
        timestamps = pd.to_datetime(frame["Time"], errors="coerce")
        unparsed = int(timestamps.isna().sum())
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_convert("UTC")
        deltas = timestamps.groupby(frame["Series"]).diff().dt.total_seconds()
        numeric = frame.drop(columns=["Series", "Time"])
        numeric = numeric.apply(pd.to_numeric, errors="coerce")
        missing = numeric.isna().mean()
        non_finite = np.isinf(numeric.to_numpy(dtype="float64"))
        per_series = (
            pd.DataFrame({"series": frame["Series"], "timestamp": timestamps})
            .groupby("series", observed=True)["timestamp"]
            .agg(["count", "min", "max"])
        )
        tables[key] = {
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "bounded_rows": int(len(frame)),
            "series_in_subset": int(frame["Series"].nunique()),
            "time_span_subset": [
                str(timestamps.min()) if not timestamps.empty else None,
                str(timestamps.max()) if not timestamps.empty else None,
            ],
            "unparsed_timestamps": unparsed,
            "duplicate_series_time_pairs": int(
                pd.DataFrame({"s": frame["Series"], "t": timestamps}).duplicated().sum()
            ),
            "native_delta_seconds_counts": {
                str(int(key)): int(value)
                for key, value in Counter(
                    float(value) for value in deltas.dropna().to_numpy()
                ).most_common(12)
            },
            "all_deltas_multiples_of_native": bool(
                np.allclose(np.mod(deltas.dropna().to_numpy(dtype="float64"), native_seconds), 0.0)
            ),
            "missing_fraction_by_column": {
                str(column): float(value) for column, value in missing.items()
            },
            "non_finite_numeric_values": int(non_finite.sum()),
            "series_rows_in_subset": {
                str(series): int(row["count"]) for series, row in per_series.iterrows()
            },
        }
        files[key] = {
            "path": str(resolved),
            "bytes": int(resolved.stat().st_size),
            "rows_total": _count_file_rows(resolved),
            "rows_audited": int(len(frame)),
            "sha256": sha256_file(resolved) if hash_files else None,
            "sha256_status": "streamed" if hash_files else "not_hashed_in_this_run",
        }

    primary = tables["primary_file"]
    aggregation = {
        "target_frequency": target,
        "native_frequency": str(section["native_frequency"]),
        "status": str(section["aggregation"]["status"]),
        "verdict": str(section["aggregation"]["verdict"]),
        "reasons": list(section["aggregation"]["reasons"]),
        "conclusion": str(section["aggregation"]["conclusion"]),
        "fifteen_minute_product_created": False,
    }
    return {
        "dataset_id": str(section["dataset_id"]),
        "display_name": str(section["display_name"]),
        "role": str(section["role"]),
        "audit_status": str(section["audit_status"]),
        "native_frequency": str(section["native_frequency"]),
        "target_frequency": target,
        "timezone": section.get("timezone"),
        "timezone_verified": bool(section.get("timezone_verified", False)),
        "bounded_audit_rows_per_file": bounded,
        "tables": tables,
        "files": files,
        "aggregation": aggregation,
        "limitations": [
            "Bounded audit only: the first `bounded_audit_rows` rows of each file were read; "
            "full-file series count, span and gaps are NOT verified.",
            "No unit/timezone documentation accompanies the local CSV files; field semantics "
            "stay unverified and no field may be merged with another source on name alone.",
            "Native 10-minute sampling is preserved; no 15-minute product is fabricated.",
        ],
        "approved_for_training": False,
        "approval_reason": str(section["training_approval"]["reason"]),
        "audited_at_utc": utc_timestamp(),
    }


# ---------------------------------------------------------------------------
# dataset specifications (used by alignment; target specs are reference-only)
# ---------------------------------------------------------------------------


def build_source_dataset_spec(config: Mapping[str, Any]) -> DatasetSpec:
    """Pretraining ``DatasetSpec`` for M100 (unlabelled: target is None)."""
    section = config["m100_exadata"]
    audit = AuditStatus(str(section["audit_status"]))
    entries = _feature_entries(section, "m100_exadata")
    features = {
        str(entry["canonical_name"]): _feature_spec_from_entry(
            entry,
            source=str(section["dataset_id"]),
            audit_status=audit,
            native_frequency=str(section["native_frequency"]),
        )
        for entry in entries
    }
    return DatasetSpec(
        dataset_id=str(section["dataset_id"]),
        site_id=str(section["site_id"]),
        domain_id=str(section["domain_id"]),
        role=DatasetRole.UNLABELED_PRETRAIN,
        native_frequency=str(section["native_frequency"]),
        trace_anchor_timestamp=str(section["trace_anchor_timestamp"]),
        timezone=str(section["timezone"]),
        timezone_verified=bool(section["timezone_verified"]),
        interval_semantics=str(section["interval_semantics"]),
        availability_semantics=str(section["availability_semantics"]),
        features=features,
        target=None,
        source_version=str(config.get("schema_version", "")),
        provenance={
            "scope": str(section["scope"]),
            "measurement_boundary": str(section["measurement_boundary"]),
            "node_level_not_facility": True,
            "approved_for_training": False,
            "approval_reason": str(section["training_approval"]["reason"]),
        },
        audit_status=audit,
    )


def build_target_dataset_spec(config: Mapping[str, Any]) -> DatasetSpec:
    """Reference-only ``DatasetSpec`` for the NLR ESIF target domain.

    The authoritative processed target files are owned by the ``target-power``
    teammate; this object only encodes documented facts so Stage 1 can align field
    semantics without importing the target pipeline.  It is NOT a target adapter.
    """
    section = config["target_reference"]
    label = section["label"]
    audit = AuditStatus(str(section["audit_status"]))
    target = TargetSpec(
        kind=TargetKind(str(label["target_kind"])),
        scope=TargetScope(str(label["target_scope"])),
        unit=str(label["target_unit"]),
        label_origin=LabelOrigin(str(label["label_origin"])),
        measurement_boundary=str(label["measurement_boundary"]),
        sampling_semantics=str(label["sampling_semantics"]),
        synthetic=bool(label["synthetic"]),
        notes=str(
            "facility_derived_kw from the five audited component fields; the component columns "
            "compose the label and are NOT candidate model inputs."
        ),
    )
    features = {
        str(entry["canonical_name"]): _feature_spec_from_entry(
            entry,
            source=str(section["dataset_id"]),
            audit_status=audit,
            native_frequency=str(section["native_frequency"]),
        )
        for entry in _feature_entries(section, "target_reference")
    }
    return DatasetSpec(
        dataset_id=str(section["dataset_id"]),
        site_id=str(section["site_id"]),
        domain_id=str(section["domain_id"]),
        role=DatasetRole.TARGET_SUPERVISED,
        native_frequency=str(section["native_frequency"]),
        trace_anchor_timestamp=str(section["trace_anchor_timestamp"]),
        timezone=str(section["timezone"]),
        timezone_verified=bool(section["timezone_verified"]),
        interval_semantics=str(section["interval_semantics"]),
        availability_semantics=str(section["availability_semantics"]),
        features=features,
        target=target,
        provenance={
            "owner": "target-power",
            "reference_only": True,
            "authoritative_processed_files_owned_by": "target-power",
            "label_type": str(label["label_type"]),
            "label_composition_fields": list(label["composition_fields"]),
            "label_composition_input_eligible": False,
        },
        audit_status=audit,
    )


def build_ucc_dataset_spec(config: Mapping[str, Any]) -> DatasetSpec:
    """Unlabelled ``DatasetSpec`` for UCC fields (semantics partially unverified)."""
    section = config["ucc2023"]
    audit = AuditStatus(str(section["audit_status"]))
    features = {
        str(entry["canonical_name"]): _feature_spec_from_entry(
            entry,
            source=str(section["dataset_id"]),
            audit_status=audit,
            native_frequency=str(section["native_frequency"]),
        )
        for entry in _feature_entries(section, "ucc2023")
    }
    return DatasetSpec(
        dataset_id=str(section["dataset_id"]),
        site_id=str(section["site_id"]),
        domain_id=str(section["domain_id"]),
        role=DatasetRole.UNLABELED_PRETRAIN,
        native_frequency=str(section["native_frequency"]),
        timezone=section.get("timezone"),
        timezone_verified=bool(section.get("timezone_verified", False)),
        interval_semantics="native 10-minute series captured at its native sampling; 15-minute aggregation not performed",
        availability_semantics="historical observations only; no future-known field; bounded audit",
        features=features,
        target=None,
        provenance={
            "aggregation_status": str(section["aggregation"]["status"]),
            "bounded_audit_rows_per_file": int(section.get("bounded_audit_rows", 0)),
            "approved_for_training": False,
        },
        audit_status=audit,
    )


def build_m100_alignment_target_spec(config: Mapping[str, Any]) -> DatasetSpec:
    """Comparison-only M100 spec declaring its REAL node power as a node-level label.

    ``FieldRegistry.align`` compares a pretraining source against a non-unlabelled
    domain.  To compare M100 fields with UCC field semantics we declare M100's
    genuinely measured node power as an auxiliary node-scope label.  This is never a
    facility-kW claim and it is never used by the Stage-2 main model.
    """
    section = config["m100_exadata"]
    label = section["alignment_node_label"]
    base = build_source_dataset_spec(config)
    target = TargetSpec(
        kind=TargetKind(str(label["target_kind"])),
        scope=TargetScope(str(label["target_scope"])),
        unit=str(label["target_unit"]),
        label_origin=LabelOrigin(str(label["label_origin"])),
        measurement_boundary=str(label["measurement_boundary"]),
        sampling_semantics=str(label["sampling_semantics"]),
        synthetic=bool(label["synthetic"]),
        notes=str(label["notes"]),
    )
    return DatasetSpec(
        dataset_id=str(label["dataset_id"]),
        site_id=base.site_id,
        domain_id=base.domain_id,
        role=DatasetRole.TARGET_SUPERVISED,
        native_frequency=base.native_frequency,
        trace_anchor_timestamp=str(section["trace_anchor_timestamp"]),
        timezone=str(section["timezone"]),
        timezone_verified=bool(section["timezone_verified"]),
        interval_semantics=str(section["interval_semantics"]),
        availability_semantics=str(section["availability_semantics"]),
        features=dict(base.features),
        target=target,
        source_version=base.source_version,
        provenance={
            "comparison_only": True,
            "node_level_auxiliary_label": True,
            "facility_kw_claim": False,
            "note": "used only so FieldRegistry.align can compare M100 vs UCC field semantics",
        },
        audit_status=base.audit_status,
    )


def build_alignment_registry(config: Mapping[str, Any]) -> FieldRegistry:
    """Register the four datasets involved in the Stage-1 field alignment."""
    registry = FieldRegistry()
    registry.register_many(
        [
            build_source_dataset_spec(config),
            build_target_dataset_spec(config),
            build_ucc_dataset_spec(config),
            build_m100_alignment_target_spec(config),
        ]
    )
    return registry


def build_alignment_results(config: Mapping[str, Any]) -> dict[str, AlignmentResult]:
    """Align M100 vs the NLR target domain, and M100 vs UCC, via ``FieldRegistry``."""
    registry = build_alignment_registry(config)
    source = build_source_dataset_spec(config)
    target = build_target_dataset_spec(config)
    ucc = build_ucc_dataset_spec(config)
    m100_label = build_m100_alignment_target_spec(config)
    return {
        "m100_vs_nlr": registry.align(source_id=source.dataset_id, target_id=target.dataset_id),
        "m100_vs_ucc": registry.align(source_id=ucc.dataset_id, target_id=m100_label.dataset_id),
    }


def _alignment_summary(result: AlignmentResult) -> dict[str, Any]:
    payload = result.to_dict()
    payload["counts"] = {
        "shared": len(result.shared_features),
        "source_only": len(result.source_only_features),
        "target_only": len(result.target_only_features),
        "incomparable": len(result.incomparable_features),
    }
    return payload


def _route_reasons(result: AlignmentResult, field_reasons: Mapping[str, str]) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    for route in result.routes:
        collected: list[str] = [field_reasons.get(route.canonical_name, "")]
        collected.extend(str(note) for note in route.notes)
        reasons[route.canonical_name] = [item for item in collected if item]
    return reasons


# ---------------------------------------------------------------------------
# tracked field registry payload (configs/datasets/feature_registry.yaml)
# ---------------------------------------------------------------------------


def _field_entry_payload(
    entry: Mapping[str, Any],
    *,
    native_frequency: str | None,
    scope: str | None,
    feature_role: str,
    role_context: str,
    role_reason: str,
    future_mapping: Mapping[str, Any] | None,
    source_dataset: str,
    audit_status: str,
) -> dict[str, Any]:
    return {
        "canonical_name": str(entry["canonical_name"]),
        "raw_name": str(entry["raw_name"]),
        "unit": str(entry["unit"]),
        "semantic_type": str(entry["semantic_type"]),
        "statistic": str(entry["statistic"]),
        "aggregation_rule": str(entry["aggregation_rule"]),
        "time_granularity": native_frequency,
        "availability": str(entry["availability"]),
        "future_known": str(entry["availability"]) == "future_known",
        "physical_range": list(entry["physical_range"]) if entry["physical_range"] is not None else None,
        "missing_policy": "mask",
        "measurement_point": str(entry["measurement_point"]),
        "scope": scope,
        "source_dataset": source_dataset,
        "audit_status": audit_status,
        "feature_role": feature_role,
        "role_context": role_context,
        "role_reason": role_reason,
        "future_mapping": dict(future_mapping) if future_mapping is not None else None,
        "notes": str(entry.get("notes", "")),
    }


def build_feature_registry_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the tracked field-registry document from config + registry alignment.

    The document is deterministic (no timestamp) so it can be reviewed as a tracked
    deliverable and regenerated by ``scripts/prepare_source_features.py``.
    """
    results = build_alignment_results(config)
    nlr_result = results["m100_vs_nlr"]
    ucc_result = results["m100_vs_ucc"]
    m100 = config["m100_exadata"]
    ucc = config["ucc2023"]
    target = config["target_reference"]
    m100_reason = (
        f"present only in the M100 pretraining source; the target domain "
        f"{nlr_result.target_id!r} does not carry a comparable field (it carries only derived "
        "facility power label components and site-outdoor weather), so no shared representation "
        "exists yet."
    )
    m100_fields: list[dict[str, Any]] = []
    for entry in _feature_entries(m100, "m100_exadata"):
        name = str(entry["canonical_name"])
        if name in nlr_result.shared_features:
            role = "source_shared"
            reason = "semantics, unit and statistic are comparable across source and target."
        else:
            role = "source_only"
            reason = m100_reason
        m100_fields.append(
            _field_entry_payload(
                entry,
                native_frequency=str(m100["native_frequency"]),
                scope=str(m100["scope"]),
                feature_role=role,
                role_context="m100_exadata__vs__nlr_esif",
                role_reason=reason,
                future_mapping=None,
                source_dataset=str(m100["dataset_id"]),
                audit_status=str(m100["audit_status"]),
            )
        )

    ucc_reasons_by_field: dict[str, str] = {}
    for route in ucc_result.routes:
        if route.relation is FeatureRelation.INCOMPARABLE:
            joined = "; ".join(route.notes) if route.notes else "not comparable"
            ucc_reasons_by_field[route.canonical_name] = (
                f"same canonical hypothesis as the M100 field but NOT merged: {joined}. "
                "Comparability requires audited semantics, equal units and a compatible statistic."
            )
    ucc_fields: list[dict[str, Any]] = []
    for entry in _feature_entries(ucc, "ucc2023"):
        name = str(entry["canonical_name"])
        if name in ucc_result.incomparable_features:
            role = "incomparable"
            reason = ucc_reasons_by_field.get(name, "not comparable to the M100 field.")
        elif name in ucc_result.shared_features:
            role = "source_shared"
            reason = "comparable to the M100 field on audited semantics, unit and statistic."
        else:
            role = "source_only"
            reason = (
                "present only in UCC among the registered sources; no comparable M100 or target "
                "field exists, and UCC field units/semantics are unverified."
            )
        ucc_fields.append(
            _field_entry_payload(
                entry,
                native_frequency=str(ucc["native_frequency"]),
                scope=None,
                feature_role=role,
                role_context="m100_exadata__vs__ucc2023",
                role_reason=reason,
                future_mapping=None,
                source_dataset=str(ucc["dataset_id"]),
                audit_status=str(ucc["audit_status"]),
            )
        )

    target_fields: list[dict[str, Any]] = []
    reservation = {
        "local_weight": TARGET_ONLY_LOCAL_WEIGHT,
        "global_fusion_weight": UNRESOLVED_GLOBAL_FUSION_WEIGHT,
        "status": "semantic_reservation_only",
        "execution_status": "not_executed_in_stage_1_or_2",
    }
    for entry in _feature_entries(target, "target_reference"):
        name = str(entry["canonical_name"])
        role = "target_only" if name in nlr_result.target_only_features else "incomparable"
        target_fields.append(
            _field_entry_payload(
                entry,
                native_frequency=str(target["native_frequency"]),
                scope=str(target["label"]["target_scope"]),
                feature_role=role,
                role_context="m100_exadata__vs__nlr_esif",
                role_reason=(
                    "target-only observed field; kept with its raw value, mask and source "
                    "description for a future local branch. local_weight=1.0 is only a Stage-4 "
                    "semantic reservation and the global fusion weight is unresolved."
                ),
                future_mapping=reservation,
                source_dataset=str(target["dataset_id"]),
                audit_status=str(target["audit_status"]),
            )
        )

    label = config["target_reference"]["label"]
    label_payload = {
        "field": str(label["field"]),
        "target_kind": str(label["target_kind"]),
        "target_scope": str(label["target_scope"]),
        "target_unit": str(label["target_unit"]),
        "label_origin": str(label["label_origin"]),
        "label_type": str(label["label_type"]),
        "measurement_boundary": str(label["measurement_boundary"]),
        "sampling_semantics": str(label["sampling_semantics"]),
        "synthetic": bool(label["synthetic"]),
        "composition_fields": list(label["composition_fields"]),
        "composition_policy": str(label["composition_policy"]),
        "input_eligible": False,
        "owner": "target-power",
    }

    alignment_payload = {
        "m100_exadata__vs__nlr_esif": {
            **_alignment_summary(nlr_result),
            "field_reasons": _route_reasons(
                nlr_result,
                {
                    **(  # every target field carries its reservation reason
                        {field["canonical_name"]: field["role_reason"] for field in target_fields}
                    ),
                    **{field["canonical_name"]: field["role_reason"] for field in m100_fields},
                },
            ),
            "direction_note": "source=m100_exadata_pretrain, target=nlr_esif (reference facts only)",
        },
        "m100_exadata__vs__ucc2023": {
            **_alignment_summary(ucc_result),
            "field_reasons": _route_reasons(
                ucc_result,
                {
                    **{field["canonical_name"]: field["role_reason"] for field in m100_fields},
                    **{field["canonical_name"]: field["role_reason"] for field in ucc_fields},
                },
            ),
            "direction_note": (
                "FieldRegistry call direction is source=ucc2023, target=m100_exadata_node_label "
                "(the comparison-only node-level M100 label). The relation names therefore follow "
                "that call: `source_only` = UCC-only, `target_only` = M100-only. Direction does "
                "not change the shared/only/comparability classification."
            ),
        },
    }

    sources_payload = {
        "m100_exadata": {
            "dataset_id": str(m100["dataset_id"]),
            "display_name": str(m100["display_name"]),
            "role": str(m100["role"]),
            "native_frequency": str(m100["native_frequency"]),
            "timezone": str(m100["timezone"]),
            "scope": str(m100["scope"]),
            "measurement_boundary": str(m100["measurement_boundary"]),
            "audit_status": str(m100["audit_status"]),
            "approved_for_training": bool(m100["training_approval"]["approved"]),
            "approval_reason": str(m100["training_approval"]["reason"]),
            "fields": m100_fields,
        },
        "ucc2023": {
            "dataset_id": str(ucc["dataset_id"]),
            "display_name": str(ucc["display_name"]),
            "role": str(ucc["role"]),
            "native_frequency": str(ucc["native_frequency"]),
            "target_frequency": str(ucc["aggregation"]["target_frequency"]),
            "aggregation_status": str(ucc["aggregation"]["status"]),
            "aggregation_verdict": str(ucc["aggregation"]["verdict"]),
            "aggregation_reasons": list(ucc["aggregation"]["reasons"]),
            "audit_status": str(ucc["audit_status"]),
            "approved_for_training": bool(ucc["training_approval"]["approved"]),
            "approval_reason": str(ucc["training_approval"]["reason"]),
            "fields": ucc_fields,
        },
        "alibaba_gpu_v2020": {
            "dataset_id": str(config["alibaba_gpu_v2020"]["dataset_id"]),
            "display_name": str(config["alibaba_gpu_v2020"]["display_name"]),
            "role": str(config["alibaba_gpu_v2020"]["role"]),
            "status": str(config["alibaba_gpu_v2020"]["status"]),
            "audit_status": str(config["alibaba_gpu_v2020"]["audit_status"]),
            "approved_for_training": bool(config["alibaba_gpu_v2020"]["training_approval"]["approved"]),
            "approval_reason": str(config["alibaba_gpu_v2020"]["training_approval"]["reason"]),
            "reasons": list(config["alibaba_gpu_v2020"]["reasons"]),
            "fields": [],
        },
        "nlr_esif": {
            "dataset_id": str(target["dataset_id"]),
            "display_name": str(target["display_name"]),
            "role": str(target["role"]),
            "native_frequency": str(target["native_frequency"]),
            "timezone": str(target["timezone"]),
            "timezone_verified": bool(target["timezone_verified"]),
            "audit_status": str(target["audit_status"]),
            "owner": "target-power",
            "reference_only": True,
            "label": label_payload,
            "fields": target_fields,
        },
    }

    registry_block = config["field_registry"]
    return {
        "schema_version": 1,
        "generated_by": (
            "scripts/prepare_source_features.py from configs/datasets/sources.yaml and "
            "src/data/registry.py FieldRegistry.align"
        ),
        "stage": "Stage 1 field registry: source/target/shared/incomparable classification only",
        "execution_status": (
            "No source->target mapping is executed in Stages 1-2. Target-only fields stay out of "
            "the Stage-2 main model A. The global fusion weight is unresolved."
        ),
        "fusion": {
            "target_only_local_weight": float(TARGET_ONLY_LOCAL_WEIGHT),
            "global_fusion_weight": UNRESOLVED_GLOBAL_FUSION_WEIGHT,
            "global_fusion_status": str(registry_block["global_fusion_status"]),
            "target_only_status": str(registry_block["target_only"]["future_mapping"]["status"]),
            "execution_status": "not_executed_in_stage_1_or_2",
            "semantics": (
                "local_weight=1.0 means a target-only field is not additionally suppressed inside "
                "its own local branch; it is NOT a physical regression coefficient, NOT a global "
                "fusion share, and it is never set to 1 for the global weight."
            ),
        },
        "approval": dict(registry_block["approval"]),
        "sources": sources_payload,
        "alignment": alignment_payload,
        "alignment_status_counts": {
            key: {
                "status": result.status.value,
                "is_transferable": result.is_transferable,
                **{
                    "shared": len(result.shared_features),
                    "source_only": len(result.source_only_features),
                    "target_only": len(result.target_only_features),
                    "incomparable": len(result.incomparable_features),
                },
            }
            for key, result in results.items()
        },
    }


def write_feature_registry(
    payload: Mapping[str, Any], *, path: str | Path = FEATURE_REGISTRY_PATH
) -> Path:
    """Write the tracked field-registry YAML (never below ``data/raw/``)."""
    destination = assert_not_raw_path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(
            "# AUTO-GENERATED by scripts/prepare_source_features.py -- do not hand-edit.\n"
            "# Source of truth: configs/datasets/sources.yaml + src/data/registry.py alignment.\n"
        )
        yaml.safe_dump(
            _plain(payload),
            handle,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=10000,
        )
    return destination


def load_feature_registry(path: str | Path = FEATURE_REGISTRY_PATH) -> dict[str, Any]:
    with resolve_path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or "sources" not in payload:
        raise ContractError(f"{path} does not contain a field-registry document.")
    return payload


# ---------------------------------------------------------------------------
# metadata / quality / artifacts
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Convert numpy/pandas scalars into plain Python values for JSON/YAML."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _feature_section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return config[key]


def build_source_metadata(
    panel: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    panel_summary: Mapping[str, Any] | None = None,
    ucc_audit: Mapping[str, Any] | None = None,
    hash_files: bool = True,
    include_ucc_audit: bool = True,
) -> dict[str, Any]:
    """Build the source metadata JSON (semantics, coverage, native frequency, files).

    ``hash_files`` streams SHA256 for the consumed raw files; tests use ``False`` on a
    bounded run.  ``include_ucc_audit=False`` skips the (bounded) UCC audit for tests
    that only exercise the M100 panel.
    """
    m100 = config["m100_exadata"]
    entries = _feature_entries(m100, "m100_exadata")
    domains = [str(item) for item in panel["domain_id"].cat.categories] if isinstance(
        panel["domain_id"].dtype, pd.CategoricalDtype
    ) else sorted(set(str(item) for item in panel["domain_id"]))
    features = [str(item) for item in panel["feature_name"].cat.categories] if isinstance(
        panel["feature_name"].dtype, pd.CategoricalDtype
    ) else sorted(set(str(item) for item in panel["feature_name"]))

    observed_overall = (
        panel.groupby("feature_name", observed=True)["observed_mask"].mean().to_dict()
    )
    observed_by_domain = (
        panel.groupby(["feature_name", "domain_id"], observed=True)["observed_mask"]
        .mean()
        .to_dict()
    )
    files = m100_files(config)
    selected_files = {domain: path for domain, path in files.items() if domain in domains}
    native_audit = audit_m100_timestamps(config, files=selected_files)

    source_files = []
    for domain, path in selected_files.items():
        entry = (
            file_metadata(path, hash_contents=True)
            if hash_files
            else {
                "path": str(resolve_path(path)),
                "bytes": int(resolve_path(path).stat().st_size),
                "modified_at": None,
                "sha256": None,
            }
        )
        entry["domain_id"] = domain
        entry["sha256_status"] = "streamed" if hash_files else "not_hashed_in_this_run"
        source_files.append(entry)

    entries_by_name = {str(entry["canonical_name"]): entry for entry in entries}
    feature_metadata: dict[str, Any] = {}
    for name in features:
        entry = entries_by_name[name]
        raw_name = str(entry["raw_name"])
        feature_metadata[name] = {
            "canonical_name": name,
            "raw_name": raw_name,
            "source_field": raw_name,
            "source_dataset": str(m100["dataset_id"]),
            "unit": str(entry["unit"]),
            "semantic_type": str(entry["semantic_type"]),
            "statistic": str(entry["statistic"]),
            "aggregation_rule": str(entry["aggregation_rule"]),
            "availability": str(entry["availability"]),
            "future_known": str(entry["availability"]) == "future_known",
            "physical_range": list(entry["physical_range"])
            if entry["physical_range"] is not None
            else None,
            "missing_policy": "mask",
            "scope": str(m100["scope"]),
            "measurement_point": str(entry["measurement_point"]),
            "measurement_boundary": str(m100["measurement_boundary"]),
            "node_level_not_facility": str(entry["semantic_type"]) in {"power", "temperature", "fan_speed"},
            "native_frequency": str(m100["native_frequency"]),
            "audit_status": str(m100["audit_status"]),
            "observed_fraction_overall": float(observed_overall.get(name, float("nan"))),
            "observed_fraction_by_domain": {
                domain: float(observed_by_domain.get((name, domain), float("nan")))
                for domain in domains
            },
            "notes": str(entry.get("notes", "")),
        }

    domain_metadata: dict[str, Any] = {}
    detail = dict(panel_summary.get("domains_detail", {})) if panel_summary else {}
    for domain in domains:
        item = {
            "domain_id": domain,
            "source_file": str(resolve_path(selected_files[domain])) if domain in selected_files else None,
            "native_audit": native_audit["domains"].get(domain),
            "panel_summary": detail.get(domain),
        }
        domain_metadata[domain] = item

    if include_ucc_audit:
        audit_payload = dict(ucc_audit) if ucc_audit is not None else build_ucc_audit(
            config, hash_files=hash_files
        )
    else:
        section = config["ucc2023"]
        audit_payload = {
            "dataset_id": str(section["dataset_id"]),
            "audit_status": str(section["audit_status"]),
            "native_frequency": str(section["native_frequency"]),
            "aggregation": dict(section["aggregation"]),
            "approved_for_training": False,
            "approval_reason": str(section["training_approval"]["reason"]),
            "audit_skipped": True,
        }

    alibaba = config["alibaba_gpu_v2020"]
    alibaba_files = []
    for name in alibaba.get("files", []):
        path = resolve_path(str(Path(str(alibaba["directory"])) / str(name)))
        alibaba_files.append(
            {
                "path": str(path),
                "bytes": int(path.stat().st_size) if path.is_file() else None,
                "exists": bool(path.is_file()),
                "sha256": None,
                "sha256_status": "not_hashed_not_ingested",
            }
        )

    panel_rows = int(len(panel))
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "generated_by": "src.data.source_features.build_source_metadata",
        "stage": "Stage 1 source-domain processing and field audit (no training)",
        "dataset_id": str(m100["dataset_id"]),
        "frequency": str(config["frequency"]),
        "points_per_day": int(config["points_per_day"]),
        "timezone": str(config["timezone"]),
        "panel": {
            "rows": panel_rows,
            "columns": list(PANEL_COLUMNS),
            "domains": domains,
            "domain_count": len(domains),
            "features": features,
            "feature_count": len(features),
            "memory_bytes": int(panel.memory_usage(deep=True).sum()),
            "contains_label_columns": False,
            "label_columns_found": [],
            "title": "long-format source operational-feature table (task book section 3.2)",
        },
        "missing_semantics": {
            "nan_means": "missing observation, NOT a measured zero",
            "observed_mask": "True exactly where feature_value is finite",
            "coverage_ratio": str(config["panel"]["coverage_ratio_definition"]),
            "zero_filled": False,
            "timeline_compressed": False,
            "internal_gaps_preserved": True,
        },
        "timeline_policy": {
            "per_domain_grid": str(config["panel"]["timeline_policy"]),
            "cross_domain_splicing": False,
            "duplicate_timestamp_policy": str(config["panel"]["duplicate_timestamp_policy"]),
            "non_finite_policy": str(config["panel"]["non_finite_policy"]),
        },
        "native_frequency": {
            "m100_exadata": native_audit,
            "ucc2023": audit_payload,
        },
        "features": feature_metadata,
        "domains": domain_metadata,
        "sources": {
            "m100_exadata": {
                "dataset_id": str(m100["dataset_id"]),
                "display_name": str(m100["display_name"]),
                "role": str(m100["role"]),
                "site_id": str(m100["site_id"]),
                "scope": str(m100["scope"]),
                "measurement_boundary": str(m100["measurement_boundary"]),
                "explicit_statement": (
                    "M100 ExaData is NODE-level hardware telemetry (per-server power, CPU/GPU "
                    "power and temperature, fan speed). It is NOT facility-level power and must "
                    "never be reported as a facility total."
                ),
                "node_level_not_facility": True,
                "audit_status": str(m100["audit_status"]),
                "approved_for_training": bool(m100["training_approval"]["approved"]),
                "approval_reason": str(m100["training_approval"]["reason"]),
                "interval_semantics": str(m100["interval_semantics"]),
                "availability_semantics": str(m100["availability_semantics"]),
                "excluded_columns": [str(item) for item in m100.get("excluded_columns", [])],
                "excluded_metrics": dict(m100.get("excluded_metrics", {})),
                "files": source_files,
            },
            "ucc2023": audit_payload,
            "alibaba_gpu_v2020": {
                "dataset_id": str(alibaba["dataset_id"]),
                "display_name": str(alibaba["display_name"]),
                "role": str(alibaba["role"]),
                "status": str(alibaba["status"]),
                "audit_status": str(alibaba["audit_status"]),
                "approved_for_training": False,
                "approval_reason": str(alibaba["training_approval"]["reason"]),
                "reasons": list(alibaba["reasons"]),
                "files": alibaba_files,
            },
        },
        "training_approval": {
            "approved": bool(config["field_registry"]["approval"]["approved_for_pretraining"]),
            "approved_for_stage1_analysis": bool(
                config["field_registry"]["approval"]["approved_for_stage1_analysis"]
            ),
            "approved_for_facility_kw_training": bool(
                config["field_registry"]["approval"]["approved_for_facility_kw_training"]
            ),
            "reason": str(config["field_registry"]["approval"]["reason"]),
        },
        "field_registry": {
            "path": str(config["paths"]["feature_registry"]),
            "note": "configs/datasets/feature_registry.yaml is the tracked field-registry deliverable.",
        },
        "environment": environment_report(),
    }
    return metadata


def build_source_quality(
    panel: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    panel_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the source quality JSON (coverage, missingness, range violations)."""
    entries = _feature_entries(config["m100_exadata"], "m100_exadata")
    entries_by_name = {str(entry["canonical_name"]): entry for entry in entries}
    features = [str(item) for item in panel["feature_name"].cat.categories] if isinstance(
        panel["feature_name"].dtype, pd.CategoricalDtype
    ) else sorted(set(str(item) for item in panel["feature_name"]))
    domains = [str(item) for item in panel["domain_id"].cat.categories] if isinstance(
        panel["domain_id"].dtype, pd.CategoricalDtype
    ) else sorted(set(str(item) for item in panel["domain_id"]))

    overall = (
        panel.groupby("feature_name", observed=True)["observed_mask"]
        .agg(["size", "sum", "mean"])
        .to_dict(orient="index")
    )
    by_domain = (
        panel.groupby(["domain_id", "feature_name"], observed=True)["observed_mask"]
        .agg(["size", "sum", "mean"])
        .to_dict(orient="index")
    )

    total_rows = int(len(panel))
    sample_target = 200_000
    sampled_rows_total = 0

    feature_quality: dict[str, Any] = {}
    for name in features:
        entry = entries_by_name[name]
        stats = overall.get(name, {})
        values = panel.loc[panel["feature_name"] == name, "feature_value"].dropna()
        physical_range = entry["physical_range"]
        violations = 0
        if physical_range is not None and len(values):
            low, high = float(physical_range[0]), float(physical_range[1])
            array = values.to_numpy(dtype="float64")
            violations = int(np.count_nonzero((array < low) | (array > high)))
        values_array = values.to_numpy(dtype="float64")
        stride = max(1, len(values_array) // sample_target)
        sampled = values_array[::stride]
        sampled_rows_total += int(len(sampled))
        description = (
            pd.Series(sampled).describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            if len(sampled)
            else pd.Series(dtype="float64")
        )
        feature_quality[name] = {
            "rows": int(stats.get("size", 0)),
            "observed": int(stats.get("sum", 0)),
            "missing": int(stats.get("size", 0)) - int(stats.get("sum", 0)),
            "observed_fraction": float(stats.get("mean", float("nan"))),
            "physical_range": list(physical_range) if physical_range is not None else None,
            "range_violations": violations,
            "range_violations_clipped": False,
            "value_min": float(values_array.min()) if len(values_array) else None,
            "value_max": float(values_array.max()) if len(values_array) else None,
            "distribution_sample_stride": int(stride),
            "distribution_sampled": {
                str(key): (int(item) if key == "count" else float(item))
                for key, item in description.to_dict().items()
            },
        }

    domain_quality: dict[str, Any] = {}
    detail = dict(panel_summary.get("domains_detail", {})) if panel_summary else {}
    for domain in domains:
        per_feature = {
            name: float(by_domain[(domain, name)]["mean"])
            for name in features
            if (domain, name) in by_domain
        }
        domain_quality[domain] = {
            "rows": int(sum(int(by_domain[(domain, name)]["size"]) for name in features if (domain, name) in by_domain)),
            "feature_count": len(per_feature),
            "observed_fraction_by_feature": per_feature,
            "mean_observed_fraction": float(np.mean(list(per_feature.values()))) if per_feature else None,
            "timeline": detail.get(domain),
        }

    return {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "generated_by": "src.data.source_features.build_source_quality",
        "dataset_id": str(config["m100_exadata"]["dataset_id"]),
        "panel": {
            "rows": total_rows,
            "domains": domains,
            "domain_count": len(domains),
            "features": features,
            "feature_count": len(features),
            "columns": list(PANEL_COLUMNS),
        },
        "features": feature_quality,
        "domains": domain_quality,
        "distribution_summary": {
            "method": "deterministic per-feature stride sample of observed values",
            "sampled_rows": sampled_rows_total,
            "total_rows": total_rows,
            "note": (
                "Counts and observed fractions are exact over the whole panel; the reported "
                "distribution quantiles are computed on a deterministic per-feature stride "
                "sample (documented per feature as distribution_sample_stride)."
            ),
        },
        "non_finite": {
            "policy": str(config["panel"]["non_finite_policy"]),
            "converted_to_nan_by_domain": {
                domain: detail.get(domain, {}).get("non_finite_values_converted_to_nan")
                for domain in domains
            },
        },
        "quality_flags": {
            "missing_encoded_as_zero": False,
            "range_violations_clipped": False,
            "internal_gaps_preserved": True,
            "cross_domain_time_splicing": False,
            "duplicate_timestamps": 0,
            "label_columns_present": False,
        },
        "variables_outside_physical_range": {
            name: item["range_violations"] for name, item in feature_quality.items()
        },
    }


def write_source_artifacts(
    panel: pd.DataFrame,
    metadata: Mapping[str, Any],
    quality: Mapping[str, Any],
    *,
    out_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, Any]:
    """Write the panel parquet + metadata/quality JSON; refuses any ``data/raw`` path."""
    destination = ensure_dir(assert_not_raw_path(out_dir))
    panel_path = write_table(panel, destination / PANEL_FILENAME)
    panel_bytes = int(resolve_path(panel_path).stat().st_size)
    artifacts = {
        "panel_parquet": str(resolve_path(panel_path)),
        "panel_bytes": panel_bytes,
        "panel_rows": int(len(panel)),
        "metadata_json": str(destination / METADATA_FILENAME),
        "quality_json": str(destination / QUALITY_FILENAME),
        "label_columns_present": False,
    }
    metadata_payload = dict(metadata)
    metadata_payload["artifacts"] = artifacts
    quality_payload = dict(quality)
    quality_payload["artifacts"] = artifacts
    metadata_json = write_json(destination / METADATA_FILENAME, metadata_payload)
    quality_json = write_json(destination / QUALITY_FILENAME, quality_payload)
    artifacts["metadata_bytes"] = int(resolve_path(metadata_json).stat().st_size)
    artifacts["quality_bytes"] = int(resolve_path(quality_json).stat().st_size)
    return artifacts


__all__ = [
    "SOURCES_CONFIG",
    "FEATURE_REGISTRY_PATH",
    "PANEL_COLUMNS",
    "PANEL_FILENAME",
    "METADATA_FILENAME",
    "QUALITY_FILENAME",
    "SEMANTIC_TYPES",
    "AGGREGATION_RULES",
    "AVAILABILITIES",
    "load_source_config",
    "m100_files",
    "load_m100_node",
    "build_source_panel",
    "audit_m100_timestamps",
    "build_ucc_audit",
    "build_source_dataset_spec",
    "build_target_dataset_spec",
    "build_ucc_dataset_spec",
    "build_m100_alignment_target_spec",
    "build_alignment_registry",
    "build_alignment_results",
    "build_feature_registry_payload",
    "write_feature_registry",
    "load_feature_registry",
    "build_source_metadata",
    "build_source_quality",
    "write_source_artifacts",
]
