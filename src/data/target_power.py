"""Stage 1 target-domain pipeline for the NLR ESIF five-component facility power.

What this module does (task book §3, §5.1-5.2, §9.2/9.5, Milestone 2):

* reads the raw NLR ESIF influx parquet files *without modifying them*, bounded by
  a declarative time window and pyarrow push-down filters;
* resamples the five ~60 s component power channels onto a complete, anchored
  15-minute grid and derives the facility label
  ``facility_power_kw = it + cooling + hvac + plug_and_light + pump``;
* keeps every 15-minute grid position (no gap compression, no back-filling, no
  imputation): missing intervals stay in the grid with ``power_kw=NaN`` and
  ``power_valid=False``;
* records the label type honestly as ``facility_derived_kw`` (a component sum,
  not an independent facility total meter);
* keeps the timezone uncertainty explicit (``timezone_verified: false``) and
  makes all day/slot arithmetic anchor-based on the 96-slot trace grid, so no
  local timezone or DST rule is ever applied;
* writes the Stage 1 artefacts (three parquet files plus a metadata and a quality
  JSON) under ``data/processed/`` and refuses any destination below
  ``data/raw/``.

All site-specific values (paths, field names, unit conversion, expected native
sampling, coverage threshold, output names, physical ranges) are read from
``configs/datasets/nlr_target.yaml``; this module contains no dataset constants
beyond the declared data-contract column names.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

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
from src.data.labels import classify_target, require_main_model_label
from src.data.loaders import assert_not_raw_path, inspect_table, write_table
from src.utils import (
    config_digest,
    ensure_dir,
    environment_report,
    file_metadata,
    jsonable,
    resolve_path,
    sha256_file,
    utc_timestamp,
    write_json,
)

# ---------------------------------------------------------------------------
# declared data-contract column names (task book §3.2); dataset values come from YAML
# ---------------------------------------------------------------------------

#: Canonical name of the 15-minute interval start (index in memory, column on disk).
TIMESTAMP_COLUMN: str = "timestamp"

#: Core target-domain columns (task book §3.2): exactly these five are written.
CORE_TARGET_COLUMNS: tuple[str, ...] = (
    TIMESTAMP_COLUMN,
    "site_id",
    "power_kw",
    "power_valid",
    "coverage_ratio",
)

#: Long-format feature columns (task book §3.2).
FEATURE_COLUMNS: tuple[str, ...] = (
    TIMESTAMP_COLUMN,
    "domain_id",
    "feature_name",
    "feature_value",
    "unit",
    "observed_mask",
    "coverage_ratio",
)

#: Batch size for the streaming pyarrow reader: memory is bounded by one batch.
_READ_BATCH_SIZE: int = 200_000


# ---------------------------------------------------------------------------
# configuration helpers
# ---------------------------------------------------------------------------


def _require_mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _cfg(config: Mapping, dotted: str) -> Any:
    """Read a required dotted key from the dataset config."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ContractError(f"Dataset config is missing required key {dotted!r}.")
        node = node[part]
    return node


def _cfg_optional(config: Mapping, dotted: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def _frequency(config: Mapping) -> str:
    frequency = str(_cfg(config, "frequency"))
    try:
        offset = pd.tseries.frequencies.to_offset(frequency)
    except (TypeError, ValueError) as exc:  # pragma: no cover - config guard
        raise ContractError(f"Unsupported frequency {frequency!r}: {exc}") from exc
    delta = pd.Timedelta(offset)
    points = int(_cfg(config, "points_per_day"))
    if points < 1 or delta <= pd.Timedelta(0):
        raise ContractError("frequency and points_per_day must be positive.")
    if delta != pd.to_timedelta(1, unit="D") / points:
        raise ContractError(
            f"frequency={frequency!r} and points_per_day={points} do not describe one day."
        )
    return frequency


def _frequency_delta(config: Mapping) -> pd.Timedelta:
    return pd.Timedelta(pd.tseries.frequencies.to_offset(_frequency(config)))


def _components(config: Mapping) -> list[str]:
    columns = [str(column) for column in _cfg(config, "label.derived_columns")]
    if not columns:
        raise ContractError("label.derived_columns must declare at least one component.")
    source_components = [
        str(column) for column in _cfg(config, "sources.power.component_fields")
    ]
    if columns != source_components:
        raise ContractError(
            "label.derived_columns and sources.power.component_fields must declare the same "
            f"components in the same order, got {columns} vs {source_components}."
        )
    return columns


def _timestamp_field(config: Mapping, source: str) -> str:
    return str(_cfg(config, f"sources.{source}.timestamp_field"))


def _weather_fields(config: Mapping) -> list[Mapping]:
    fields = _cfg(config, "sources.weather.fields")
    if not isinstance(fields, Mapping) or not fields:
        raise ContractError("sources.weather.fields must declare at least one field.")
    result: list[Mapping] = []
    for key, spec in fields.items():
        spec = _require_mapping(spec, f"sources.weather.fields.{key}")
        for required in ("source_field", "feature_name", "unit", "scale", "offset"):
            if required not in spec:
                raise ContractError(
                    f"sources.weather.fields.{key} is missing required key {required!r}."
                )
        result.append(spec)
    return result


def _validate_config(config: Mapping) -> Mapping:
    """Cheap structural validation of the dataset config (no I/O)."""
    config = _require_mapping(config, "config")
    for key in ("dataset_id", "site_id", "domain_id", "frequency", "points_per_day"):
        _cfg(config, key)
    _frequency(config)
    _components(config)
    expected = int(_cfg(config, "coverage.expected_native_samples_per_interval"))
    if expected < 1:
        raise ContractError("coverage.expected_native_samples_per_interval must be >= 1.")
    if int(_cfg(config, "native_sampling.expected_samples_per_interval")) != expected:
        raise ContractError(
            "native_sampling.expected_samples_per_interval and "
            "coverage.expected_native_samples_per_interval must agree."
        )
    threshold = float(_cfg(config, "coverage.coverage_validity_threshold"))
    if not (0.0 <= threshold <= 1.0):
        raise ContractError(
            f"coverage.coverage_validity_threshold must be in [0, 1], got {threshold}."
        )
    _timestamp_field(config, "power")
    _timestamp_field(config, "weather")
    _weather_fields(config)
    for dotted in (
        "analysis.window_start",
        "label.measurement_boundary",
        "label.sampling_semantics",
        "label.composition_warning",
        "timezone_note",
        "raw_power_start_note",
        "outputs.out_dir",
        "outputs.target_power_file",
        "outputs.target_components_file",
        "outputs.target_features_file",
        "outputs.metadata_file",
        "outputs.quality_file",
    ):
        _cfg(config, dotted)
    return config


def _coverage_threshold(config: Mapping) -> float:
    return float(_cfg(config, "coverage.coverage_validity_threshold"))


def _expected_samples(config: Mapping) -> int:
    return int(_cfg(config, "coverage.expected_native_samples_per_interval"))


def _parse_naive_utc(value: Any, *, name: str) -> pd.Timestamp:
    """Parse a timestamp and return it as a timezone-naive UTC ``Timestamp``."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name}={value!r} cannot be parsed as a timestamp: {exc}") from exc
    if stamp is pd.NaT:
        raise ContractError(f"{name}={value!r} is NaT; a real timestamp is required.")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def _window_start(config: Mapping) -> pd.Timestamp | None:
    raw = _cfg_optional(config, "analysis.window_start")
    if raw is None:
        return None
    return _parse_naive_utc(raw, name="analysis.window_start")


def _window_end(config: Mapping) -> pd.Timestamp | None:
    raw = _cfg_optional(config, "analysis.window_end")
    if raw is None:
        return None
    return _parse_naive_utc(raw, name="analysis.window_end")


def _parse_utc_naive_array(values: Any) -> np.ndarray:
    """Parse a raw timestamp series into timezone-naive UTC ``datetime64[ns]`` values."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        parsed = series.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        parsed = pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)
    return parsed.to_numpy(dtype="datetime64[ns]")


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if stamp is pd.NaT:
        return None
    return stamp.isoformat()


def _bool_array(values: Any) -> np.ndarray:
    array = np.asarray(values)
    return array.astype(bool, copy=False)


def _native_sampling_audit(
    sample_count: pd.Series, spacing_seconds: pd.Series, expected: int
) -> dict[str, Any]:
    counts = sample_count.to_numpy(dtype=float)
    spacing = spacing_seconds.to_numpy(dtype=float) if len(spacing_seconds) else np.array([])
    return {
        "expected_samples_per_interval": int(expected),
        "mean_samples_per_interval": float(counts.mean()) if counts.size else 0.0,
        "median_samples_per_interval": float(np.median(counts)) if counts.size else 0.0,
        "p05_samples_per_interval": float(np.percentile(counts, 5)) if counts.size else 0.0,
        "p95_samples_per_interval": float(np.percentile(counts, 95)) if counts.size else 0.0,
        "min_samples_per_interval": int(counts.min()) if counts.size else 0,
        "max_samples_per_interval": int(counts.max()) if counts.size else 0,
        "intervals_with_expected_samples": int((counts >= expected).sum()),
        "intervals_with_expected_samples_share": float((counts >= expected).mean())
        if counts.size
        else 0.0,
        "intervals_with_at_least_one_sample": int((counts >= 1).sum()),
        "intervals_with_at_least_one_sample_share": float((counts >= 1).mean())
        if counts.size
        else 0.0,
        "spacing_seconds_p05": float(np.percentile(spacing, 5)) if spacing.size else None,
        "spacing_seconds_median": float(np.median(spacing)) if spacing.size else None,
        "spacing_seconds_p95": float(np.percentile(spacing, 95)) if spacing.size else None,
        "spacing_seconds_min": float(spacing.min()) if spacing.size else None,
        "spacing_seconds_max": float(spacing.max()) if spacing.size else None,
    }


def _bin_indices(
    stamps_ns: np.ndarray, anchor: pd.Timestamp, delta: pd.Timedelta
) -> np.ndarray:
    """Integer 15-minute bin index of each timestamp relative to ``anchor``."""
    delta_ns = int(delta.value)
    if delta_ns <= 0:  # pragma: no cover - guarded by config validation
        raise ContractError("frequency delta must be positive.")
    offsets = stamps_ns.astype("datetime64[ns]") - np.datetime64(anchor, "ns")
    nanoseconds = offsets.astype("timedelta64[ns]").astype("int64")
    return nanoseconds // delta_ns


def _binned_mean(
    indices: np.ndarray, values: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin count of finite values and per-bin mean of the finite values.

    ``np.bincount`` is used instead of a pandas groupby so the 4.5M-row raw file is
    binned in linear time without per-group Python objects. Missing values are
    excluded from each bin mean (never treated as zero) and a bin without any
    finite value gets ``NaN``.
    """
    finite = np.isfinite(values)
    counts = np.bincount(
        indices, weights=finite.astype("float64"), minlength=n_bins
    ).astype("int64")
    sums = np.bincount(
        indices, weights=np.where(finite, values, 0.0), minlength=n_bins
    )
    means = np.full(n_bins, np.nan, dtype="float64")
    np.divide(sums, counts, out=means, where=counts > 0)
    return counts, means


# ---------------------------------------------------------------------------
# raw reading
# ---------------------------------------------------------------------------


def _detect_timestamp_field(schema: pa.Schema) -> str:
    timestamp_fields = [field.name for field in schema if pa.types.is_timestamp(field.type)]
    if len(timestamp_fields) != 1:
        raise ContractError(
            "Expected exactly one timestamp-typed column in the parquet schema, got "
            f"{timestamp_fields!r}."
        )
    return timestamp_fields[0]


def _arrow_scalar_for_field(field: pa.Field, stamp: pd.Timestamp) -> pa.Scalar:
    if pa.types.is_timestamp(field.type) and field.type.tz is not None:
        moment = stamp.tz_localize("UTC").to_pydatetime()
    else:
        moment = stamp.to_pydatetime()
    return pa.scalar(moment, type=field.type)


def _combine_filters(filters: Sequence[Any]) -> Any:
    """Fold pyarrow filter expressions into one ``&`` expression (or ``None``)."""
    if not filters:
        return None
    combined = filters[0]
    for expression in filters[1:]:
        combined = combined & expression
    return combined


def load_raw_components(
    path: str | Path,
    *,
    start: Any = None,
    end: Any = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load raw parquet rows bounded by ``start``/``end`` (never writes the file).

    Parameters
    ----------
    path:
        Raw parquet file path (read-only).
    start, end:
        Optional time bounds parsed as naive UTC (timezone-aware input is converted
        to UTC). ``start`` is inclusive, ``end`` is exclusive. Bounding the read is
        the supported way to keep memory small on the 4.5M-row file: pyarrow
        streams row groups in batches of at most 200k rows and applies the filter
        inside the scanner, so the loader never materialises the whole file unless
        the caller asks for the whole range.
    columns:
        Optional column selection; the file's timestamp column is always included.
        ``None`` reads every column of the file.

    Returns
    -------
    pandas.DataFrame
        Raw rows in file order. The loader does not sort, deduplicate or modify
        values; those steps are explicit and audited in
        :func:`build_target_power_frame`.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Raw parquet file not found: {resolved}")
    dataset = ds.dataset(resolved, format="parquet")
    schema = dataset.schema
    timestamp_field = _detect_timestamp_field(schema)

    if columns is None:
        selected = list(schema.names)
    else:
        if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
            raise TypeError("columns must be a sequence of column names or None.")
        selected = [str(column) for column in columns]
        if len(set(selected)) != len(selected):
            raise ContractError(f"columns contains duplicates: {selected!r}.")
    missing = [column for column in selected if column not in schema.names]
    if missing:
        raise ContractError(f"Requested columns are missing from {resolved}: {missing}.")
    if timestamp_field not in selected:
        selected.insert(0, timestamp_field)

    filters: list[Any] = []
    start_stamp: pd.Timestamp | None = None
    if start is not None:
        start_stamp = _parse_naive_utc(start, name="start")
        filters.append(
            ds.field(timestamp_field)
            >= _arrow_scalar_for_field(schema.field(timestamp_field), start_stamp)
        )
    if end is not None:
        end_stamp = _parse_naive_utc(end, name="end")
        if start_stamp is not None and end_stamp <= start_stamp:
            raise ContractError("end must be after start.")
        filters.append(
            ds.field(timestamp_field)
            < _arrow_scalar_for_field(schema.field(timestamp_field), end_stamp)
        )

    scanner = dataset.scanner(
        columns=selected,
        filter=_combine_filters(filters),
        batch_size=_READ_BATCH_SIZE,
    )
    parts = [batch.to_pandas() for batch in scanner.to_batches() if batch.num_rows]
    if not parts:
        return schema.empty_table().select(selected).to_pandas()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# target frame construction
# ---------------------------------------------------------------------------


def _resample_target_power(
    raw: pd.DataFrame, config: Mapping
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Internal worker: build the target frame, the component frame and the audit."""
    _validate_config(config)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"raw must be a pandas DataFrame, got {type(raw).__name__}.")
    if raw.empty:
        raise ContractError("raw selection is empty; cannot build a target frame.")

    components = _components(config)
    timestamp_field = _timestamp_field(config, "power")
    required = [timestamp_field, *components]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ContractError(f"Raw power frame is missing required columns: {missing}.")

    stamps = _parse_utc_naive_array(raw[timestamp_field])
    nat_count = int(np.count_nonzero(pd.isna(stamps)))
    component_values = np.column_stack(
        [
            pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype="float64")
            for column in components
        ]
    )
    infinity_counts = {
        column: int(np.count_nonzero(np.isinf(component_values[:, index])))
        for index, column in enumerate(components)
    }
    component_values[np.isinf(component_values)] = np.nan

    keep = ~pd.isna(stamps)
    stamps = stamps[keep]
    component_values = component_values[keep]
    if stamps.size == 0:
        raise ContractError("raw selection contains no parseable timestamps.")

    work = pd.DataFrame({TIMESTAMP_COLUMN: stamps, **{
        column: component_values[:, index] for index, column in enumerate(components)
    }})

    complete_rows = work[components].notna().all(axis=1).to_numpy(dtype=bool)
    if not bool(complete_rows.any()):
        raise ContractError(
            "No raw row in the selection has all five power components finite; the label "
            "cannot be built from this selection."
        )
    first_complete_raw = pd.Timestamp(work.loc[complete_rows, TIMESTAMP_COLUMN].min())
    raw_first = pd.Timestamp(work[TIMESTAMP_COLUMN].min())
    raw_last = pd.Timestamp(work[TIMESTAMP_COLUMN].max())

    declared_start = _window_start(config)
    delta = _frequency_delta(config)
    if declared_start is not None and declared_start != declared_start.floor(_frequency(config)):
        raise ContractError(
            f"analysis.window_start={declared_start.isoformat()} is not on the 15-minute grid."
        )
    frequency = _frequency(config)
    anchor = declared_start if declared_start is not None else first_complete_raw.floor(frequency)
    window_end = _window_end(config)

    before_anchor = (work[TIMESTAMP_COLUMN] < anchor).to_numpy(dtype=bool)
    rows_before_anchor = int(before_anchor.sum())
    any_component = work[components].notna().any(axis=1).to_numpy(dtype=bool)
    rows_before_anchor_all_nan = int((before_anchor & ~any_component).sum())
    if window_end is not None:
        after_end = (work[TIMESTAMP_COLUMN] >= window_end).to_numpy(dtype=bool)
    else:
        after_end = np.zeros(len(work), dtype=bool)
    rows_after_end = int(after_end.sum())
    rows_excluded_before_dedup = int((~before_anchor & ~after_end).sum())

    selection = work.loc[~before_anchor & ~after_end].reset_index(drop=True)
    if selection.empty:
        raise ContractError(
            f"No raw rows fall inside the declared analysis window "
            f"[{anchor.isoformat()}, {_iso(window_end) if window_end is not None else 'end of file'})."
        )

    duplicate_rows = int(selection[TIMESTAMP_COLUMN].duplicated().sum())
    if duplicate_rows:
        duplicate_groups = int(
            (selection.groupby(TIMESTAMP_COLUMN, sort=False).size() > 1).sum()
        )
        deduplicated = (
            selection.groupby(TIMESTAMP_COLUMN, sort=True, as_index=False)[components]
            .mean()
            .sort_values(TIMESTAMP_COLUMN, kind="stable")
            .reset_index(drop=True)
        )
    else:
        duplicate_groups = 0
        deduplicated = selection

    stamps_ns = deduplicated[TIMESTAMP_COLUMN].to_numpy(dtype="datetime64[ns]")
    indices = _bin_indices(stamps_ns, anchor, delta)
    if indices.size and int(indices.min()) < 0:  # pragma: no cover - defensive
        raise ContractError("Internal binning error: a negative interval index was produced.")
    n_bins = int(indices.max()) + 1
    grid = pd.date_range(anchor, periods=n_bins, freq=frequency)
    if len(grid) == 0:
        raise ContractError("Computed the 15-minute grid as empty; check window_start/window_end.")

    interval_values = deduplicated[components].to_numpy(dtype="float64")
    sample_count_values = np.bincount(indices, minlength=n_bins).astype("int64")
    complete_values = np.isfinite(interval_values).all(axis=1)
    complete_count_values = np.bincount(
        indices, weights=complete_values.astype("float64"), minlength=n_bins
    ).astype("int64")
    interval_matrix = np.full((n_bins, len(components)), np.nan, dtype="float64")
    for index, column in enumerate(components):
        _per_bin_counts, per_bin_mean = _binned_mean(
            indices, interval_values[:, index], n_bins
        )
        interval_matrix[:, index] = per_bin_mean
    interval_means = pd.DataFrame(interval_matrix, index=grid, columns=components)
    interval_means.index.name = TIMESTAMP_COLUMN
    sample_count = pd.Series(sample_count_values, index=grid)
    complete_count = pd.Series(complete_count_values, index=grid)

    expected_samples = _expected_samples(config)
    threshold = _coverage_threshold(config)
    coverage_ratio = (complete_count.astype("float64") / float(expected_samples)).clip(upper=1.0)
    coverage_ratio.index.name = TIMESTAMP_COLUMN
    power_kw = interval_means.sum(axis=1, min_count=len(components))
    power_kw.index.name = TIMESTAMP_COLUMN

    all_components_finite = interval_means.notna().all(axis=1).to_numpy(dtype=bool)
    coverage_values = coverage_ratio.to_numpy(dtype="float64")
    power_values = power_kw.to_numpy(dtype="float64")
    power_valid = (
        all_components_finite
        & (coverage_values >= threshold)
        & np.isfinite(power_values)
    )

    site_id = str(_cfg(config, "site_id"))
    target = pd.DataFrame(
        {
            "site_id": site_id,
            "power_kw": power_values,
            "power_valid": _bool_array(power_valid),
            "coverage_ratio": coverage_values,
        },
        index=grid,
    )
    target.index.name = TIMESTAMP_COLUMN
    components_frame = interval_means.copy()
    components_frame.index.name = TIMESTAMP_COLUMN

    # --- physical-range diagnostics (never clipping) ---------------------------------
    physical_ranges: dict[str, tuple[float, float]] = {}
    for name, bounds in (_cfg_optional(config, "analysis.physical_ranges", {}) or {}).items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ContractError(f"analysis.physical_ranges.{name} must be a [low, high] pair.")
        physical_ranges[str(name)] = (float(bounds[0]), float(bounds[1]))

    def _out_of_range(series: pd.Series, name: str) -> int:
        bounds = physical_ranges.get(name)
        if bounds is None:
            return 0
        values = series.to_numpy(dtype="float64")
        finite = np.isfinite(values)
        return int(np.count_nonzero(finite & ((values < bounds[0]) | (values > bounds[1]))))

    component_audit: dict[str, Any] = {}
    for index, column in enumerate(components):
        raw_column = component_values[:, index]
        interval_series = interval_means[column]
        raw_finite = np.isfinite(raw_column)
        component_audit[column] = {
            "raw_samples_finite": int(raw_finite.sum()),
            "raw_samples_missing": int((~raw_finite).sum()),
            "raw_samples_non_negative": int(np.count_nonzero(raw_finite & (raw_column >= 0))),
            "raw_samples_negative": int(np.count_nonzero(raw_finite & (raw_column < 0))),
            "interval_means_finite": int(interval_series.notna().sum()),
            "interval_means_missing": int(interval_series.isna().sum()),
            "interval_mean_min_kw": float(interval_series.min())
            if interval_series.notna().any()
            else None,
            "interval_mean_max_kw": float(interval_series.max())
            if interval_series.notna().any()
            else None,
            "interval_mean_mean_kw": float(interval_series.mean())
            if interval_series.notna().any()
            else None,
            "out_of_declared_physical_range": _out_of_range(interval_series, column),
        }

    valid_power = power_kw.loc[power_valid]
    capacity_audit = {
        "power_kw_finite_intervals": int(power_kw.notna().sum()),
        "power_kw_missing_intervals": int(power_kw.isna().sum()),
        "power_kw_valid_mean": float(valid_power.mean()) if len(valid_power) else None,
        "power_kw_valid_std": float(valid_power.std()) if len(valid_power) else None,
        "power_kw_valid_min": float(valid_power.min()) if len(valid_power) else None,
        "power_kw_valid_max": float(valid_power.max()) if len(valid_power) else None,
        "power_kw_valid_median": float(valid_power.median()) if len(valid_power) else None,
        "power_kw_out_of_declared_physical_range": _out_of_range(power_kw, "facility_power_kw"),
    }

    # --- audit assembly ---------------------------------------------------------------
    spacing_seconds = (
        pd.Series(deduplicated[TIMESTAMP_COLUMN]).diff().dt.total_seconds().dropna()
    )
    slots = len(grid)
    slots_with_samples = int((sample_count >= 1).sum())
    slots_without_samples = int((sample_count == 0).sum())
    valid_count = int(power_valid.sum())
    native_audit = _native_sampling_audit(sample_count, spacing_seconds, expected_samples)

    warnings: list[str] = []
    if rows_before_anchor:
        warnings.append(
            f"{rows_before_anchor} raw rows before the declared analysis window start were "
            f"excluded ({rows_before_anchor_all_nan} of them had all five components NaN, i.e. "
            "0% label coverage); this is a documented data-selection boundary, not gap compression."
        )
    if rows_after_end:
        warnings.append(
            f"{rows_after_end} raw rows at/after analysis.window_end were excluded from this run."
        )
    non_zero_infinity = {
        key: value for key, value in infinity_counts.items() if value
    }
    if non_zero_infinity:
        warnings.append(
            "Infinite component values were replaced by NaN (never treated as observations): "
            f"{non_zero_infinity}."
        )
    if duplicate_rows:
        warnings.append(
            f"{duplicate_rows} duplicate raw timestamps were collapsed by taking the component-"
            "wise arithmetic mean over the duplicate rows."
        )
    negative_components = {
        column: audit["raw_samples_negative"]
        for column, audit in component_audit.items()
        if audit["raw_samples_negative"] > 0
    }
    if negative_components:
        warnings.append(
            "Some raw component samples are slightly negative sensor readings; they are kept "
            f"(not clipped): {negative_components}."
        )
    if slots_without_samples:
        warnings.append(
            f"{slots_without_samples} of {slots} grid slots have no native sample at all; they "
            "remain in the grid with power_kw=NaN and power_valid=false."
        )
    if valid_count < slots:
        warnings.append(
            f"{slots - valid_count} of {slots} grid slots are not valid labels "
            f"(coverage_validity_threshold={threshold}); no value was imputed."
        )
    warnings.append(
        "facility_power_kw is a component-sum derived label (label type facility_derived_kw), "
        "not an independent facility total-meter measurement."
    )

    audit: dict[str, Any] = {
        "stage": "stage1_target_power",
        "dataset_id": str(_cfg(config, "dataset_id")),
        "site_id": site_id,
        "domain_id": str(_cfg(config, "domain_id")),
        "frequency": frequency,
        "points_per_day": int(_cfg(config, "points_per_day")),
        "label_type": classify_target(target_spec_from_config(config)).value,
        "label_unit": "kW",
        "label_composition": components,
        "coverage_validity_threshold": threshold,
        "expected_native_samples_per_interval": expected_samples,
        "raw": {
            "rows_loaded": int(len(raw)),
            "rows_unparseable_timestamp": nat_count,
            "rows_excluded_before_anchor": rows_before_anchor,
            "rows_excluded_before_anchor_with_all_components_nan": rows_before_anchor_all_nan,
            "rows_excluded_at_or_after_window_end": rows_after_end,
            "rows_used_before_deduplication": rows_excluded_before_dedup,
            "duplicate_timestamp_rows_collapsed": duplicate_rows,
            "duplicate_timestamp_groups": duplicate_groups,
            "infinite_values_replaced_by_nan": infinity_counts,
            "first_timestamp_loaded": _iso(raw_first),
            "last_timestamp_loaded": _iso(raw_last),
            "first_timestamp_with_all_components_finite": _iso(first_complete_raw),
        },
        "anchor": {
            "anchor_timestamp": _iso(grid[0]),
            "configured_window_start": _iso(declared_start),
            "detected_first_complete_slot": _iso(first_complete_raw.floor(frequency)),
            "anchor_matches_first_complete_slot": bool(
                declared_start is None or declared_start == first_complete_raw.floor(frequency)
            ),
            "anchor_basis": str(_cfg_optional(config, "analysis.anchor_basis", "")),
            "window_end_exclusive": _iso(window_end),
        },
        "grid": {
            "start": _iso(grid[0]),
            "end": _iso(grid[-1]),
            "slots": slots,
            "interval_step_seconds": float(delta.total_seconds()),
            "slots_with_at_least_one_native_sample": slots_with_samples,
            "slots_without_any_native_sample": slots_without_samples,
            "missing_interval_count": slots_without_samples,
            "gap_compression": False,
            "index_timezone": "UTC (timezone-naive values)",
        },
        "native_sampling": native_audit,
        "coverage": {
            "mean": float(np.nanmean(coverage_values)) if slots else 0.0,
            "median": float(np.nanmedian(coverage_values)) if slots else 0.0,
            "min": float(np.nanmin(coverage_values)) if slots else 0.0,
            "max": float(np.nanmax(coverage_values)) if slots else 0.0,
            "share_full_coverage": float(np.mean(coverage_values >= 1.0)) if slots else 0.0,
            "share_at_least_0_8": float(np.mean(coverage_values >= 0.8)) if slots else 0.0,
            "share_zero": float(np.mean(coverage_values == 0.0)) if slots else 0.0,
        },
        "validity": {
            "valid_slots": valid_count,
            "invalid_slots": int(slots - valid_count),
            "valid_fraction": float(valid_count / slots) if slots else 0.0,
        },
        "components": component_audit,
        "power_kw": capacity_audit,
        "physical_ranges": {name: list(bounds) for name, bounds in physical_ranges.items()},
        "quality_warnings": warnings,
    }
    return target, components_frame, audit


def build_target_power_frame(raw: pd.DataFrame, config: Mapping) -> tuple[pd.DataFrame, dict]:
    """Build the Stage 1 target frame from raw component rows.

    Returns ``(frame, audit)``. ``frame`` is indexed by the timezone-naive UTC
    interval-start ``DatetimeIndex`` named ``timestamp`` and has exactly the core
    columns ``site_id, power_kw, power_valid, coverage_ratio``.

    The 15-minute grid is complete from the anchor to the last observed interval
    start: internal gaps keep their rows (``power_kw=NaN``, ``power_valid=False``)
    and later timestamps never shift. Nothing is back-filled or imputed.
    ``audit`` is a JSON-serialisable quality report, written to
    ``target_quality.json`` by :func:`write_target_artifacts`.
    """
    frame, _components_frame, audit = _resample_target_power(raw, config)
    return frame, audit


def build_target_power_frame_with_components(
    raw: pd.DataFrame, config: Mapping
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Like :func:`build_target_power_frame` but also return the component interval means.

    The component frame is indexed like the target frame and holds one column per
    declared component (kW). It is written to ``target_components.parquet`` and is
    explicitly flagged as label composition only, never as features.
    """
    return _resample_target_power(raw, config)


# ---------------------------------------------------------------------------
# target features (weather) construction
# ---------------------------------------------------------------------------


def _unit_column_audit(raw: pd.DataFrame, config: Mapping) -> dict[str, Any]:
    column = _cfg_optional(config, "sources.weather.raw_unit_column")
    if not column or column not in raw.columns:
        return {"column": column, "present": False, "value_counts": {}}
    values = raw[column]
    counts: dict[str, int] = {}
    for value, count in values.value_counts(dropna=False).items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            key = "<missing/null>"
        else:
            key = str(value)
        counts[key] = int(count)
    return {
        "column": str(column),
        "present": True,
        "value_counts": counts,
        "trusted_as_unit_declaration": False,
        "note": (
            "The raw unit column is mostly null; the declared Fahrenheit unit comes from the "
            "documented historical conversion and is verified against the legacy processed file."
        ),
    }


def build_target_feature_frame(
    weather_raw: pd.DataFrame, grid: pd.DatetimeIndex, config: Mapping
) -> tuple[pd.DataFrame, dict]:
    """Resample raw weather observations onto the target 15-minute grid (LONG format).

    Returns ``(features, audit)``. ``features`` has exactly the long-format columns
    ``timestamp, domain_id, feature_name, feature_value, unit, observed_mask,
    coverage_ratio``. Features are target-domain covariates registered with role
    ``target_only`` for the future auxiliary path; they are *not* future-known and
    Stage 1 never uses them for forecasting. Missing observations keep their grid
    rows with ``observed_mask=False`` and no fabricated value.
    """
    _validate_config(config)
    if not isinstance(weather_raw, pd.DataFrame) or weather_raw.empty:
        raise ContractError("weather_raw must be a non-empty pandas DataFrame.")
    if not isinstance(grid, pd.DatetimeIndex) or grid.tz is not None:
        raise ContractError("grid must be a timezone-naive DatetimeIndex (naive UTC convention).")
    if len(grid) < 1:
        raise ContractError("grid must contain at least one interval.")
    if not grid.is_monotonic_increasing or grid.has_duplicates:
        raise ContractError("grid must be strictly increasing without duplicates.")

    frequency = _frequency(config)
    delta = _frequency_delta(config)
    if grid[0] != grid[0].floor(frequency):
        raise ContractError(f"grid start {grid[0]} is not on the {frequency} grid.")
    if len(grid) > 1 and grid[1] - grid[0] != delta:
        raise ContractError(f"grid is not consecutive at the {frequency} frequency.")
    expected_samples = _expected_samples(config)
    threshold = _coverage_threshold(config)
    fields = _weather_fields(config)
    timestamp_field = _timestamp_field(config, "weather")
    sentinel_before = _parse_naive_utc(
        _cfg(config, "sources.weather.sentinel_before"),
        name="sources.weather.sentinel_before",
    )

    source_fields = [str(field["source_field"]) for field in fields]
    required = [timestamp_field, *source_fields]
    missing = [column for column in required if column not in weather_raw.columns]
    if missing:
        raise ContractError(f"Weather frame is missing required columns: {missing}.")

    stamps = _parse_utc_naive_array(weather_raw[timestamp_field])
    nat_count = int(np.count_nonzero(pd.isna(stamps)))
    keep = ~pd.isna(stamps)
    stamps = stamps[keep]

    sentinel_mask = stamps < np.datetime64(sentinel_before, "ns")
    sentinel_rows = stamps[sentinel_mask]
    sentinel_audit = {
        "policy": "exclude_rows_before_sentinel_before",
        "sentinel_before": _iso(sentinel_before),
        "excluded_rows": int(sentinel_mask.sum()),
        "excluded_timestamps": [_iso(value) for value in pd.Series(sentinel_rows).head(10)],
    }
    after_sentinel = ~sentinel_mask

    last_interval_end = pd.Timestamp(grid[-1]) + delta
    first_interval_start = pd.Timestamp(grid[0])
    before_grid = stamps < np.datetime64(first_interval_start, "ns")
    after_grid = stamps >= np.datetime64(last_interval_end, "ns")
    rows_before_grid = int(before_grid.sum())
    rows_after_grid = int(after_grid.sum())
    usable = after_sentinel & ~before_grid & ~after_grid

    values_frame = pd.DataFrame({TIMESTAMP_COLUMN: stamps[usable]})
    converted_columns: dict[str, np.ndarray] = {}
    conversion_audit: dict[str, Any] = {}
    for index, field in enumerate(fields):
        source_field = str(field["source_field"])
        scale = float(field["scale"])
        offset = float(field["offset"])
        raw_values = (
            pd.to_numeric(weather_raw[source_field], errors="coerce")
            .to_numpy(dtype="float64")[keep][usable]
        )
        infinity_count = int(np.count_nonzero(np.isinf(raw_values)))
        raw_values[np.isinf(raw_values)] = np.nan
        converted = raw_values * scale + offset
        converted_columns[str(field["feature_name"])] = converted
        conversion_audit[str(field["feature_name"])] = {
            "source_field": source_field,
            "scale": scale,
            "offset": offset,
            "infinity_values_replaced_by_nan": infinity_count,
        }

    work = pd.DataFrame({TIMESTAMP_COLUMN: stamps[usable], **converted_columns})
    duplicate_rows = int(work[TIMESTAMP_COLUMN].duplicated().sum())
    if duplicate_rows:
        deduplicated = (
            work.groupby(TIMESTAMP_COLUMN, sort=True, as_index=False)[
                list(converted_columns)
            ]
            .mean()
            .sort_values(TIMESTAMP_COLUMN, kind="stable")
            .reset_index(drop=True)
        )
    else:
        deduplicated = work

    feature_names = list(converted_columns)
    usable_stamps = deduplicated[TIMESTAMP_COLUMN].to_numpy(dtype="datetime64[ns]")
    indices = _bin_indices(usable_stamps, pd.Timestamp(grid[0]), delta)
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= len(grid)):
        raise ContractError(
            "weather samples fall outside the target grid after the declared filtering."
        )
    n_bins = len(grid)
    feature_matrix = np.full((n_bins, len(feature_names)), np.nan, dtype="float64")
    count_matrix = np.zeros((n_bins, len(feature_names)), dtype="int64")
    for index, feature_name in enumerate(feature_names):
        per_bin_counts, per_bin_mean = _binned_mean(
            indices, deduplicated[feature_name].to_numpy(dtype="float64"), n_bins
        )
        feature_matrix[:, index] = per_bin_mean
        count_matrix[:, index] = per_bin_counts
    interval_means = pd.DataFrame(feature_matrix, index=grid, columns=feature_names)
    counts = pd.DataFrame(count_matrix, index=grid, columns=feature_names)

    feature_parts: list[pd.DataFrame] = []
    feature_audit: dict[str, Any] = {}
    domain_id = str(_cfg(config, "domain_id"))
    role = str(_cfg(config, "sources.weather.field_role"))
    for field in fields:
        feature_name = str(field["feature_name"])
        unit = str(field["unit"])
        bounds = field.get("physical_range")
        mean_series = interval_means[feature_name]
        count_series = counts[feature_name]
        coverage_ratio = (count_series.astype("float64") / float(expected_samples)).clip(
            upper=1.0
        )
        observed_mask = mean_series.notna().to_numpy(dtype=bool) & (
            coverage_ratio.to_numpy(dtype="float64") >= threshold
        )
        feature_parts.append(
            pd.DataFrame(
                {
                    TIMESTAMP_COLUMN: grid,
                    "domain_id": domain_id,
                    "feature_name": feature_name,
                    "feature_value": mean_series.to_numpy(dtype="float64"),
                    "unit": unit,
                    "observed_mask": observed_mask,
                    "coverage_ratio": coverage_ratio.to_numpy(dtype="float64"),
                }
            )
        )
        converted = converted_columns[feature_name]
        finite_converted = converted[np.isfinite(converted)]
        finite_interval = mean_series.to_numpy(dtype="float64")
        finite_interval = finite_interval[np.isfinite(finite_interval)]
        out_of_range = 0
        if bounds is not None and finite_interval.size:
            low, high = float(bounds[0]), float(bounds[1])
            out_of_range = int(
                np.count_nonzero((finite_interval < low) | (finite_interval > high))
            )
        feature_audit[feature_name] = {
            "source_field": str(field["source_field"]),
            "semantic_type": str(field.get("semantic_type", "")),
            "role": role,
            "unit": unit,
            "source_unit": str(field.get("source_unit", "")),
            **conversion_audit[feature_name],
            "conversion_formula": "converted = raw * scale + offset",
            "raw_samples_finite": int(np.isfinite(converted).sum()),
            "raw_samples_missing": int((~np.isfinite(converted)).sum()),
            "duplicate_timestamps_collapsed": duplicate_rows,
            "converted_sample_min": float(finite_converted.min())
            if finite_converted.size
            else None,
            "converted_sample_max": float(finite_converted.max())
            if finite_converted.size
            else None,
            "converted_sample_mean": float(finite_converted.mean())
            if finite_converted.size
            else None,
            "interval_means_finite": int(mean_series.notna().sum()),
            "interval_means_missing": int(mean_series.isna().sum()),
            "interval_mean_min": float(finite_interval.min())
            if finite_interval.size
            else None,
            "interval_mean_max": float(finite_interval.max())
            if finite_interval.size
            else None,
            "observed_mask_true": int(observed_mask.sum()),
            "observed_mask_false": int((~observed_mask).sum()),
            "coverage_mean": float(coverage_ratio.mean()),
            "coverage_share_zero": float((coverage_ratio == 0).mean()),
            "out_of_declared_physical_range": out_of_range,
            "physical_range": list(bounds) if bounds is not None else None,
        }

    features = pd.concat(feature_parts, ignore_index=True).sort_values(
        [TIMESTAMP_COLUMN, "feature_name"], kind="stable", ignore_index=True
    )

    warnings = [
        "Target-domain weather features are registered with role 'target_only' for the future "
        "auxiliary path; they are NOT known-future values and Stage 1 does not use them.",
        "observed_mask=False keeps the grid row with feature_value=NaN; nothing is imputed.",
    ]
    if sentinel_audit["excluded_rows"]:
        warnings.append(
            f"{sentinel_audit['excluded_rows']} sentinel row(s) before "
            f"{sentinel_audit['sentinel_before']} were excluded from the weather series."
        )

    audit: dict[str, Any] = {
        "stage": "stage1_target_features",
        "domain_id": domain_id,
        "frequency": frequency,
        "grid_start": _iso(grid[0]),
        "grid_end": _iso(grid[-1]),
        "grid_slots": int(len(grid)),
        "expected_native_samples_per_interval": expected_samples,
        "coverage_validity_threshold": threshold,
        "timestamp_field": timestamp_field,
        "rows_loaded": int(len(weather_raw)),
        "rows_unparseable_timestamp": nat_count,
        "rows_before_grid_start": rows_before_grid,
        "rows_at_or_after_grid_end": rows_after_grid,
        "rows_used": int(usable.sum()),
        "unit_column": _unit_column_audit(weather_raw, config),
        "sentinel": sentinel_audit,
        "fields": feature_audit,
        "role": role,
        "availability": str(_cfg(config, "sources.weather.availability")),
        "future_mapping": dict(_cfg(config, "sources.weather.future_mapping")),
        "quality_warnings": warnings,
    }
    return features, audit


# ---------------------------------------------------------------------------
# specs / metadata
# ---------------------------------------------------------------------------


def target_spec_from_config(config: Mapping) -> TargetSpec:
    """Build the Stage 1 :class:`TargetSpec` (active_power / facility / kW / component_sum)."""
    _validate_config(config)
    label = _require_mapping(_cfg(config, "label"), "label")
    spec = TargetSpec(
        kind=TargetKind(str(label["kind"])),
        scope=TargetScope(str(label["scope"])),
        unit=str(label["unit"]),
        label_origin=LabelOrigin(str(label["label_origin"])),
        measurement_boundary=str(label["measurement_boundary"]),
        sampling_semantics=str(label["sampling_semantics"]),
        available_at_lag=pd.to_timedelta(
            float(label.get("available_at_lag_seconds", 0)), unit="s"
        ),
        synthetic=False,
        notes=str(label.get("notes", "")),
    )
    label_type = classify_target(spec)
    expected = label.get("expected_label_type")
    if expected is not None and label_type.value != str(expected):
        raise ContractError(
            f"Config expects label type {expected!r} but the declared target classifies as "
            f"{label_type.value!r}."
        )
    require_main_model_label(label_type, dataset_id=str(_cfg(config, "dataset_id")))
    return spec


def dataset_spec_from_config(config: Mapping) -> DatasetSpec:
    """Build the :class:`DatasetSpec` Stage 2 can use for ``ForecastInputs`` wiring."""
    _validate_config(config)
    target = target_spec_from_config(config)
    weather_native_seconds = int(
        _cfg_optional(config, "sources.weather.expected_native_interval_seconds", 60)
    )
    feature_specs: dict[str, FeatureSpec] = {}
    weather_path = str(_cfg(config, "sources.weather.path"))
    for field in _weather_fields(config):
        feature_name = str(field["feature_name"])
        bounds = field.get("physical_range")
        feature_specs[feature_name] = FeatureSpec(
            raw_name=str(field["source_field"]),
            canonical_name=feature_name,
            unit=str(field["unit"]),
            statistic=IntervalStatistic.INTERVAL_MEAN,
            physical_range=(float(bounds[0]), float(bounds[1])) if bounds is not None else None,
            native_frequency=f"{weather_native_seconds}s",
            availability_lag=pd.Timedelta(0),
            future_known=False,
            missing_policy="mask",
            source=weather_path,
            audit_status=AuditStatus.PARTIAL,
            notes=(
                "Target-domain weather covariate: historical observation only, not known in the "
                "future. Registered with role target_only for the future auxiliary path; Stage 1 "
                "does not use it and Stage 2's main power model must not read it."
            ),
        )
    anchor = _window_start(config)
    if anchor is None:
        raise ContractError(
            "dataset_spec_from_config requires analysis.window_start to declare the trace anchor."
        )
    return DatasetSpec(
        dataset_id=str(_cfg(config, "dataset_id")),
        site_id=str(_cfg(config, "site_id")),
        domain_id=str(_cfg(config, "domain_id")),
        role=DatasetRole(str(_cfg(config, "role"))),
        native_frequency=_frequency(config),
        trace_anchor_timestamp=anchor.isoformat(),
        timezone=str(_cfg(config, "timezone")),
        timezone_verified=bool(_cfg(config, "timezone_verified")),
        interval_semantics=(
            "15-minute interval arithmetic mean, labelled with the interval start "
            "[t, t+15min); timezone-naive UTC by convention (site local timezone unverified)"
        ),
        availability_semantics=(
            "labels are available at the end of their own interval (available_at_lag=0); all "
            "weather covariates are historical observations only, never known in the future"
        ),
        features=feature_specs,
        target=target,
        source_version="nlr-esif-influx-export-audited-stage1",
        provenance={
            "stage": "stage1_target_power",
            "config_digest": config_digest(config),
            "label_type": classify_target(target).value,
            "timezone_verified": False,
        },
        audit_status=AuditStatus.PARTIAL,
    )


def _source_file_metadata(path: str | Path, *, role: str) -> dict[str, Any]:
    metadata = file_metadata(path)
    summary = inspect_table(path)
    return {
        "role": role,
        "path": metadata["path"],
        "bytes": metadata["bytes"],
        "modified_at": metadata["modified_at"],
        "sha256": metadata["sha256"],
        "rows": summary["rows"],
        "schema": summary["schema"],
        "raw_data_modified": False,
    }


def _feature_registry_entries(config: Mapping) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    weather_path = str(_cfg(config, "sources.weather.path"))
    availability = str(_cfg(config, "sources.weather.availability"))
    future_mapping = dict(_cfg(config, "sources.weather.future_mapping"))
    role = str(_cfg(config, "sources.weather.field_role"))
    for field in _weather_fields(config):
        bounds = field.get("physical_range")
        entries.append(
            {
                "feature_name": str(field["feature_name"]),
                "semantic_type": str(field.get("semantic_type", "")),
                "aggregation_rule": "interval_mean",
                "source_field": str(field["source_field"]),
                "source_dataset": weather_path,
                "source_unit": str(field.get("source_unit", "")),
                "unit": str(field["unit"]),
                "physical_range": list(bounds) if bounds is not None else None,
                "availability": availability,
                "future_known": False,
                "role": role,
                "observed_mask_semantics": (
                    "True = at least one finite native observation and "
                    "coverage_ratio >= configured coverage threshold"
                ),
                "coverage_ratio_semantics": (
                    "valid native samples / expected native samples, capped at 1.0"
                ),
                "future_mapping": {
                    "local_weight": future_mapping.get("local_weight"),
                    "global_fusion_weight": future_mapping.get("global_fusion_weight"),
                    "status": future_mapping.get("status"),
                    "note": (
                        "Declared semantics reserved for the future auxiliary mapping stage. "
                        "Stage 1 does not execute this mapping and Stage 2 must not use these "
                        "target-only fields as future-known features."
                    ),
                },
            }
        )
    return entries


def build_target_metadata(
    frame: pd.DataFrame,
    quality: Mapping,
    config: Mapping,
    *,
    source_metadata: Sequence[Mapping] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON-serialisable Stage 1 target metadata document.

    ``frame`` is the target frame from :func:`build_target_power_frame` and
    ``quality`` is the audit mapping returned with it. ``source_metadata`` is an
    optional pre-computed list of raw-file metadata entries (the prepare script
    computes them once); when omitted, the raw files are hashed with the streaming
    :func:`src.utils.sha256_file` helper (raw files are never loaded whole).
    """
    _validate_config(config)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ContractError("metadata requires a non-empty target frame.")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None:
        raise ContractError("Target frame must be indexed by a timezone-naive DatetimeIndex.")
    if frame.index.name != TIMESTAMP_COLUMN:
        raise ContractError(f"Target frame index must be named {TIMESTAMP_COLUMN!r}.")
    missing_columns = [column for column in CORE_TARGET_COLUMNS[1:] if column not in frame.columns]
    if missing_columns:
        raise ContractError(f"Target frame is missing core columns: {missing_columns}.")
    if not isinstance(quality, Mapping):
        raise ContractError(
            "quality must be the audit mapping returned by build_target_power_frame."
        )

    target = target_spec_from_config(config)
    label_type = classify_target(target)
    require_main_model_label(label_type, dataset_id=str(_cfg(config, "dataset_id")))
    dataset_id = str(_cfg(config, "dataset_id"))
    site_id = str(_cfg(config, "site_id"))
    domain_id = str(_cfg(config, "domain_id"))

    if source_metadata is None:
        sources = [
            _source_file_metadata(_cfg(config, "sources.power.path"), role="power_label_source"),
            _source_file_metadata(
                _cfg(config, "sources.weather.path"), role="target_domain_weather_source"
            ),
        ]
    else:
        sources = [dict(entry) for entry in source_metadata]

    grid = quality.get("grid", {})
    validity = quality.get("validity", {})
    coverage = quality.get("coverage", {})
    raw = quality.get("raw", {})
    anchor = quality.get("anchor", {})

    metadata: dict[str, Any] = {
        "schema_version": "stage1_target_metadata_v1",
        "stage": "stage1_target_power",
        "generated_at_utc": utc_timestamp(),
        # Flat convenience keys (also available in the structured sections below).
        "dataset_id": dataset_id,
        "site_id": site_id,
        "domain_id": domain_id,
        "label_type": label_type.value,
        "label_unit": target.unit,
        "timezone": str(_cfg(config, "timezone")),
        "timezone_verified": False,
        "anchor_timestamp": anchor.get("anchor_timestamp"),
        "frequency": _frequency(config),
        "synthetic": False,
        "research_result": False,
        "dataset": {
            "dataset_id": dataset_id,
            "site_id": site_id,
            "domain_id": domain_id,
            "role": str(_cfg(config, "role")),
            "frequency": _frequency(config),
            "points_per_day": int(_cfg(config, "points_per_day")),
            "synthetic": False,
            "research_result": False,
            "stage1_target_domain_data_complete": True,
        },
        "label": {
            "label_type": label_type.value,
            "label_origin": target.label_origin.value,
            "label_name": str(_cfg(config, "label.name")),
            "unit": target.unit,
            "kind": target.kind.value,
            "scope": target.scope.value,
            "derived_columns": _components(config),
            "sum_rule": (
                "facility_power_kw = sum of the five component 15-minute interval means "
                "(min_count=5, so a missing component makes the label NaN instead of summing "
                "fewer components)"
            ),
            "measurement_boundary": target.measurement_boundary,
            "sampling_semantics": target.sampling_semantics,
            "aggregation_rule": str(_cfg(config, "label.aggregation_rule")),
            "composition_warning": str(_cfg(config, "label.composition_warning")),
            "target_spec": target.to_dict(),
            "honesty_note": (
                "This label is a component-sum derived facility power value "
                "(facility_derived_kw). It is not an independent facility total-meter "
                "measurement and must never be reported as facility_measured_kw."
            ),
        },
        "timezone_metadata": {
            "declared": str(_cfg(config, "timezone")),
            "verified": False,
            "timezone_verified": False,
            "assumption": "raw naive timestamps are treated as naive UTC",
            "local_time_operations_blocked": True,
            "note": str(_cfg(config, "timezone_note")),
            "anchor_based_trace_days_require_verified_timezone": False,
            "dst_rules_applied": False,
        },
        "time": {
            "anchor_timestamp": anchor.get("anchor_timestamp"),
            "anchor_basis": anchor.get("anchor_basis"),
            "configured_window_start": anchor.get("configured_window_start"),
            "first_timestamp_with_all_components_finite": raw.get(
                "first_timestamp_with_all_components_finite"
            ),
            "first_prepared_interval": grid.get("start"),
            "last_prepared_interval": grid.get("end"),
            "grid_slots": grid.get("slots"),
            "interval_label": "start; [timestamp, timestamp + 15 minutes)",
            "anchor_matches_first_complete_slot": anchor.get("anchor_matches_first_complete_slot"),
        },
        "aggregation": {
            "frequency": _frequency(config),
            "points_per_day": int(_cfg(config, "points_per_day")),
            "interval_step_seconds": grid.get("interval_step_seconds"),
            "rule": (
                "native ~60 s component samples are averaged per left-closed right-open "
                "15-minute interval; the interval start is the label timestamp"
            ),
            "expected_native_samples_per_interval": _expected_samples(config),
            "coverage_ratio_definition": str(_cfg(config, "coverage.coverage_ratio_rule")),
            "coverage_validity_threshold": _coverage_threshold(config),
            "validity_rule": str(_cfg(config, "coverage.validity_rule")),
            "coverage_ratio_capped_at_one": True,
            "gap_compression": False,
            "backward_fill_used": False,
            "imputation_used": False,
            "label_timestamp_is_interval_start": True,
        },
        "data_selection": {
            "raw_power_start_note": str(_cfg(config, "raw_power_start_note")),
            "rows_loaded": raw.get("rows_loaded"),
            "rows_excluded_before_anchor": raw.get("rows_excluded_before_anchor"),
            "rows_excluded_before_anchor_with_all_components_nan": raw.get(
                "rows_excluded_before_anchor_with_all_components_nan"
            ),
            "rows_excluded_at_or_after_window_end": raw.get(
                "rows_excluded_at_or_after_window_end"
            ),
            "raw_power_first_timestamp_loaded": raw.get("first_timestamp_loaded"),
            "raw_power_first_finite_component_timestamp": raw.get(
                "first_timestamp_with_all_components_finite"
            ),
            "rationale": (
                "The pre-2016-06-12 power era has 0% finite component values (all five channels "
                "NaN), so it cannot produce a label. The prepared label frame therefore starts "
                "at the first 15-minute interval with a complete component observation; every "
                "grid position inside the selected range is kept, so the time axis is not "
                "compressed."
            ),
        },
        "quality": {
            "valid_slots": validity.get("valid_slots"),
            "invalid_slots": validity.get("invalid_slots"),
            "valid_fraction": validity.get("valid_fraction"),
            "missing_interval_count": grid.get("slots_without_any_native_sample"),
            "slots_with_at_least_one_native_sample": grid.get(
                "slots_with_at_least_one_native_sample"
            ),
            "coverage_ratio_mean": coverage.get("mean"),
            "coverage_ratio_share_full": coverage.get("share_full_coverage"),
            "native_sampling_audit": quality.get("native_sampling"),
            "component_audit": quality.get("components"),
            "power_kw_audit": quality.get("power_kw"),
            "quality_warnings": list(quality.get("quality_warnings", [])),
        },
        "features": {
            "role": str(_cfg(config, "sources.weather.field_role")),
            "registry": _feature_registry_entries(config),
            "availability": str(_cfg(config, "sources.weather.availability")),
            "future_mapping": dict(_cfg(config, "sources.weather.future_mapping")),
            "warning": (
                "Weather features are target-domain historical observations registered for the "
                "future auxiliary path only. Stage 1 does not use them; they are not known-future "
                "features and must not be used as such in Stage 2."
            ),
        },
        "source_files": sources,
        "artifacts": {
            "target_power": {
                "file": str(_cfg(config, "outputs.target_power_file")),
                "role": "stage1_target_label",
                "columns": list(CORE_TARGET_COLUMNS),
                "format": "wide parquet, one row per 15-minute interval start",
            },
            "target_components": {
                "file": str(_cfg(config, "outputs.target_components_file")),
                "role": "label_composition_only",
                "columns": [TIMESTAMP_COLUMN, *_components(config)],
                "label_composition_only": True,
                "warning": (
                    "The five component series are the composition of the facility label. They "
                    "are provided for audit/analysis only and must NOT be used as future-known "
                    "features when forecasting the label they reconstruct."
                ),
            },
            "target_features": {
                "file": str(_cfg(config, "outputs.target_features_file")),
                "role": "target_domain_covariates_long",
                "columns": list(FEATURE_COLUMNS),
                "format": "long parquet: one row per (timestamp, feature_name)",
            },
            "target_metadata": {"file": str(_cfg(config, "outputs.metadata_file"))},
            "target_quality": {"file": str(_cfg(config, "outputs.quality_file"))},
        },
        "provenance": environment_report(),
        "config_digest": config_digest(config),
    }
    return metadata


# ---------------------------------------------------------------------------
# artefact writing
# ---------------------------------------------------------------------------


def _validate_target_frame_for_write(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ContractError("target frame must be a non-empty pandas DataFrame.")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None:
        raise ContractError("target frame must be indexed by a timezone-naive DatetimeIndex.")
    if frame.index.name != TIMESTAMP_COLUMN:
        raise ContractError(f"target frame index must be named {TIMESTAMP_COLUMN!r}.")
    if list(frame.columns) != list(CORE_TARGET_COLUMNS[1:]):
        raise ContractError(
            f"target frame must have exactly the core columns {list(CORE_TARGET_COLUMNS[1:])}, "
            f"got {list(frame.columns)}."
        )
    if not frame.index.is_monotonic_increasing:
        raise ContractError("target frame index must be strictly increasing.")


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> Path:
    """Write ``frame`` to ``destination`` via a temporary file then atomic replace."""
    temporary = destination.with_name(destination.name + ".tmp.parquet")
    try:
        write_table(frame, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _artifact_entry(
    path: Path, *, role: str, rows: int, columns: Sequence[str], **extra: Any
) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "rows": int(rows),
        "columns": [str(column) for column in columns],
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        **extra,
    }


def write_target_artifacts(
    frame: pd.DataFrame,
    components: pd.DataFrame,
    features: pd.DataFrame,
    metadata: Mapping,
    quality: Mapping,
    config: Mapping,
    *,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write the Stage 1 target artefacts under ``data/processed`` (never under raw).

    Writes:

    * ``processed_target_power.parquet`` — core five columns only;
    * ``target_components.parquet`` — five component interval means + timestamp,
      flagged as label composition only in ``target_metadata.json``;
    * ``processed_target_features.parquet`` — LONG weather features
      (``timestamp, domain_id, feature_name, feature_value, unit, observed_mask,
      coverage_ratio``), role ``target_only``;
    * ``target_metadata.json`` and ``target_quality.json``.

    Re-running regenerates these *declared* Stage 1 artefact names atomically; no
    other file is touched and nothing is ever written below ``data/raw/``.
    """
    _validate_config(config)
    if not isinstance(metadata, Mapping) or not isinstance(quality, Mapping):
        raise TypeError("metadata and quality must be mappings.")
    _validate_target_frame_for_write(frame)

    declared_components = _components(config)
    if not isinstance(components, pd.DataFrame) or components.empty:
        raise ContractError("components frame must be a non-empty pandas DataFrame.")
    if list(components.columns) != declared_components:
        raise ContractError(
            f"components frame must have exactly the declared components {declared_components}, "
            f"got {list(components.columns)}."
        )
    if len(components) != len(frame):
        raise ContractError("components frame and target frame must have one row per interval.")
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ContractError("features frame must be a non-empty pandas DataFrame.")
    if list(features.columns) != list(FEATURE_COLUMNS):
        raise ContractError(
            f"features frame must have exactly the long-format columns {list(FEATURE_COLUMNS)}, "
            f"got {list(features.columns)}."
        )

    outputs = _cfg(config, "outputs")
    destination_dir = ensure_dir(
        assert_not_raw_path(
            resolve_path(out_dir if out_dir is not None else outputs["out_dir"])
        )
    )

    target_table = frame.reset_index()[list(CORE_TARGET_COLUMNS)]
    component_table = components.reset_index()[[TIMESTAMP_COLUMN, *declared_components]]

    target_path = _write_parquet_atomic(
        target_table, destination_dir / str(outputs["target_power_file"])
    )
    component_path = _write_parquet_atomic(
        component_table, destination_dir / str(outputs["target_components_file"])
    )
    feature_path = _write_parquet_atomic(
        features.reset_index(drop=True), destination_dir / str(outputs["target_features_file"])
    )

    stamp = utc_timestamp()
    artifacts: dict[str, Any] = {
        "target_power": _artifact_entry(
            target_path,
            role="stage1_target_label",
            rows=len(target_table),
            columns=list(CORE_TARGET_COLUMNS),
            label_type=classify_target(target_spec_from_config(config)).value,
            unit="kW",
        ),
        "target_components": _artifact_entry(
            component_path,
            role="label_composition_only",
            rows=len(component_table),
            columns=[TIMESTAMP_COLUMN, *declared_components],
            label_composition_only=True,
            warning=(
                "Components are the composition of facility_power_kw and must NOT be used as "
                "future-known features."
            ),
        ),
        "target_features": _artifact_entry(
            feature_path,
            role="target_domain_covariates_long",
            rows=len(features),
            columns=list(FEATURE_COLUMNS),
            format="long",
        ),
    }

    quality_document = jsonable(dict(quality))
    quality_document["stage"] = str(quality.get("stage", "stage1_target_power"))
    quality_document["written_at_utc"] = stamp
    quality_document["artifacts"] = {
        "target_power": artifacts["target_power"],
        "target_components": artifacts["target_components"],
        "target_features": artifacts["target_features"],
    }
    quality_path = destination_dir / str(outputs["quality_file"])
    write_json(quality_path, quality_document)

    metadata_document = jsonable(dict(metadata))
    metadata_document["written_at_utc"] = stamp
    artifacts["target_quality"] = _artifact_entry(
        quality_path, role="quality_report", rows=0, columns=[]
    )
    artifacts["target_metadata"] = {
        "role": "metadata",
        "path": str(destination_dir / str(outputs["metadata_file"])),
        "columns": [],
        "rows": 0,
    }
    metadata_document["artifacts"] = artifacts
    metadata_path = destination_dir / str(outputs["metadata_file"])
    write_json(metadata_path, metadata_document)

    artifacts["target_metadata"] = _artifact_entry(
        metadata_path, role="metadata", rows=0, columns=[]
    )
    return {
        "paths": {
            "target_power": str(target_path),
            "target_components": str(component_path),
            "target_features": str(feature_path),
            "target_metadata": str(metadata_path),
            "target_quality": str(quality_path),
        },
        "artifacts": artifacts,
        "written_at_utc": stamp,
    }


__all__ = [
    "CORE_TARGET_COLUMNS",
    "FEATURE_COLUMNS",
    "TIMESTAMP_COLUMN",
    "build_target_feature_frame",
    "build_target_metadata",
    "build_target_power_frame",
    "build_target_power_frame_with_components",
    "dataset_spec_from_config",
    "load_raw_components",
    "target_spec_from_config",
    "write_target_artifacts",
]
