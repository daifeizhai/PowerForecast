"""Quality inspection, safe causal filling, and versioned processed output.

Responsibilities (task book §5.1, §5.3):

* :func:`quality_report` — read-only description of the frame: completeness,
  duplicates, grid gaps, per-column NaN fraction, out-of-physical-range counts,
  complete trace days and target coverage.
* :func:`validate_frame` — structural gate that raises on invalid input.
  Physical-range excursions are reported as diagnostics (not silently clipped)
  and a negative target is counted, never clamped.
* :func:`fill_causal` — **only** forward-fills covariates, only from the past,
  at most ``fill_limit`` consecutive steps, and **never** touches the target.
  Every filled cell is recorded in the returned audit mapping.
* :func:`write_processed` — writes a fresh versioned parquet file plus a JSON
  sidecar under ``data/processed/``.  It can never write below ``data/raw/``.

Nothing in this module writes to ``data/raw/``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.adapters import mapping_table
from src.data.contracts import (
    TARGET_COLUMN,
    ContractError,
    DatasetSpec,
    PhysicalUnitError,
    TargetKind,
    is_power_unit,
    to_kw,
)
from src.data.loaders import assert_not_raw_path, write_table
from src.utils import (
    ensure_dir,
    environment_report,
    file_metadata,
    resolve_path,
    utc_timestamp,
    write_json,
)

POINTS_PER_DAY: int = 96


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _require_frame(frame: Any) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(frame).__name__}.")
    return frame


def _require_spec(spec: Any) -> DatasetSpec:
    if not isinstance(spec, DatasetSpec):
        raise ContractError("Expected a DatasetSpec.")
    return spec


def _frequency_delta(frequency: str) -> pd.Timedelta:
    try:
        offset = pd.tseries.frequencies.to_offset(frequency)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported frequency {frequency!r}: {exc}") from exc
    delta = pd.Timedelta(offset)
    if delta <= pd.Timedelta(0):
        raise ValueError(f"frequency must be positive, got {frequency!r}.")
    return delta


def _iso(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return pd.Timestamp(value).isoformat()


def _numeric_array(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.to_numpy(dtype=float, na_value=np.nan)


def _anchor_reference(frame: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.Timestamp, bool, str]:
    """Return (reference timestamp, matches first row, source of reference)."""
    index = frame.index
    first = index[0]
    anchor = spec.trace_anchor_timestamp
    if anchor is None:
        return first, True, "first_timestamp_no_declared_anchor"
    try:
        anchor_ts = pd.Timestamp(anchor)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"trace_anchor_timestamp={anchor!r} is invalid: {exc}") from exc
    if anchor_ts.tzinfo is None:
        if index.tz is not None:
            anchor_ts = anchor_ts.tz_localize(index.tz)
        else:
            anchor_ts = anchor_ts.tz_localize("UTC")
    else:
        anchor_ts = anchor_ts.tz_convert(index.tz) if index.tz is not None else anchor_ts.tz_convert("UTC")
    return anchor_ts, bool(first == anchor_ts), "declared_trace_anchor"


# ---------------------------------------------------------------------------
# quality report
# ---------------------------------------------------------------------------


def quality_report(frame: pd.DataFrame, spec: DatasetSpec, *, frequency: str = "15min") -> dict:
    """Read-only quality/completeness summary of one adapted frame.

    The returned mapping is JSON-serialisable and is stored in every processed
    sidecar.  It does not modify the frame and it never fills a missing value.
    """
    frame = _require_frame(frame)
    spec = _require_spec(spec)
    if len(frame) == 0:
        raise ContractError("quality_report requires a non-empty frame.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ContractError("quality_report requires a DatetimeIndex (see adapt_frame).")
    delta = _frequency_delta(frequency)
    index = frame.index

    # --- grid completeness ------------------------------------------------------------
    span = index.max() - index.min()
    expected_interval_count = int(span / delta) + 1
    unique_count = int(index.unique().size)
    duplicate_timestamps = int(len(index) - unique_count)
    missing_intervals = max(expected_interval_count - unique_count, 0)
    rows_before_anchor = 0
    off_grid_rows = 0
    trace_day_count = 0
    complete_trace_days = 0
    complete_trace_days_with_finite_target: int | None = None

    if index.tz is None:
        raise ContractError("quality_report requires a timezone-aware UTC DatetimeIndex.")
    reference, anchor_matches, reference_source = _anchor_reference(frame, spec)
    freq_ns = int(delta.value)
    deltas = index.asi8 - int(reference.value)
    positions = np.floor_divide(deltas, freq_ns)
    remainders = np.mod(deltas, freq_ns)
    off_grid_rows = int(np.count_nonzero(remainders != 0))
    rows_before_anchor = int(np.count_nonzero(positions < 0))
    in_grid = (positions >= 0) & (remainders == 0)
    day_ids = np.where(in_grid, positions // POINTS_PER_DAY, -1)
    valid_days = day_ids[day_ids >= 0]
    if valid_days.size:
        trace_day_count = int(valid_days.max()) + 1

    present_by_day: dict[int, set[int]] = {}
    for position, day_id in zip(positions[in_grid].tolist(), day_ids[in_grid].tolist()):
        present_by_day.setdefault(int(day_id), set()).add(int(position) % POINTS_PER_DAY)
    complete_days = sorted(
        day for day, slots in present_by_day.items() if len(slots) == POINTS_PER_DAY
    )
    complete_trace_days = len(complete_days)

    target_present = spec.target is not None
    if target_present and TARGET_COLUMN in frame.columns:
        target_values = _numeric_array(frame[TARGET_COLUMN])
        finite_target = np.isfinite(target_values)
        target_coverage_count = int(finite_target.sum())
        target_coverage_fraction = float(finite_target.mean())
        complete_with_target = 0
        for day in complete_days:
            day_mask = day_ids == day
            if int(day_mask.sum()) == POINTS_PER_DAY and bool(
                np.isfinite(target_values[day_mask]).all()
            ):
                complete_with_target += 1
        complete_trace_days_with_finite_target = complete_with_target
        target_negative_count = int(np.count_nonzero(np.isfinite(target_values) & (target_values < 0)))
        target_nan_count = int(np.count_nonzero(~np.isfinite(target_values)))
    else:
        target_coverage_count = None
        target_coverage_fraction = None
        target_negative_count = None
        target_nan_count = None
        complete_trace_days_with_finite_target = None

    # --- per-column NaN and declared physical ranges ----------------------------------
    nan_counts: dict[str, int] = {}
    nan_fraction: dict[str, float] = {}
    for column in frame.columns:
        values = _numeric_array(frame[column])
        count = int(len(values) - np.count_nonzero(np.isfinite(values)))
        nan_counts[str(column)] = count
        nan_fraction[str(column)] = float(count / len(values)) if len(values) else 0.0

    out_of_range_counts: dict[str, int] = {}
    for canonical_name, feature in spec.features.items():
        if canonical_name not in frame.columns or feature.physical_range is None:
            continue
        values = _numeric_array(frame[canonical_name])
        low, high = float(feature.physical_range[0]), float(feature.physical_range[1])
        finite = np.isfinite(values)
        out_of_range_counts[canonical_name] = int(
            np.count_nonzero(finite & ((values < low) | (values > high)))
        )

    present_features = [name for name in spec.feature_names if name in frame.columns]
    missing_declared_features = [name for name in spec.feature_names if name not in frame.columns]
    unexpected_columns = [
        str(column) for column in frame.columns if column not in spec.features and column != TARGET_COLUMN
    ]

    return {
        "dataset_id": spec.dataset_id,
        "synthetic": bool(spec.synthetic),
        "frequency": frequency,
        "rows": int(len(frame)),
        "start": _iso(index.min()),
        "end": _iso(index.max()),
        "index_timezone": str(index.tz),
        "timezone": spec.timezone if spec.timezone is not None else str(index.tz),
        "timezone_verified": bool(spec.timezone_verified and str(index.tz) == "UTC"),
        "trace_grid_reference": _iso(reference),
        "trace_grid_reference_source": reference_source,
        "anchor_matches_first_timestamp": bool(anchor_matches),
        "duplicate_timestamps": duplicate_timestamps,
        "expected_interval_count": expected_interval_count,
        "missing_intervals": missing_intervals,
        "off_grid_rows": off_grid_rows,
        "rows_before_anchor": rows_before_anchor,
        "trace_day_count": trace_day_count,
        "complete_trace_days": complete_trace_days,
        "complete_trace_days_with_finite_target": complete_trace_days_with_finite_target,
        "declared_features": present_features,
        "missing_declared_features": missing_declared_features,
        "unexpected_columns": unexpected_columns,
        "nan_counts": nan_counts,
        "nan_fraction": nan_fraction,
        "out_of_range_counts": out_of_range_counts,
        "target_present": bool(target_present and TARGET_COLUMN in frame.columns),
        "target_coverage_count": target_coverage_count,
        "target_coverage_fraction": target_coverage_fraction,
        "target_nan_count": target_nan_count,
        "target_negative_count": target_negative_count,
    }


# ---------------------------------------------------------------------------
# structural validation
# ---------------------------------------------------------------------------


def validate_frame(frame: pd.DataFrame, spec: DatasetSpec) -> dict:
    """Structural gate for an adapted frame.

    Raises :class:`ContractError` / :class:`PhysicalUnitError` for structurally
    invalid input (wrong index type/timezone, unsorted/duplicated timestamps,
    missing declared fields, non-numeric values, non-finite values, a kW target
    where the spec declares none or a non-power target).

    Physical-range excursions and negative targets are **diagnostics only**: they
    are counted in the returned mapping and are never silently clipped.
    """
    frame = _require_frame(frame)
    spec = _require_spec(spec)
    errors: list[str] = []

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ContractError("Frame index must be a DatetimeIndex named 'timestamp'.")
    index = frame.index
    if index.tz is None:
        errors.append(
            "Frame index is timezone-naive; adapted frames must carry a timezone-aware UTC index."
        )
    elif str(index.tz) != "UTC":
        errors.append(f"Frame index timezone is {index.tz}; expected UTC.")
    if index.name != "timestamp":
        errors.append(f"Frame index must be named 'timestamp', got {index.name!r}.")
    duplicate_timestamps = int(index.duplicated().sum())
    if duplicate_timestamps:
        errors.append(f"Frame has {duplicate_timestamps} duplicate timestamps.")
    if not index.is_monotonic_increasing:
        errors.append("Frame index is not sorted in increasing time order.")

    if spec.target is None:
        if TARGET_COLUMN in frame.columns:
            raise PhysicalUnitError(
                f"Unlabelled dataset {spec.dataset_id!r} must not carry the power target column "
                f"{TARGET_COLUMN!r}; unlabelled pretraining data has no kW label."
            )
        target_present = False
    else:
        if spec.target.kind is not TargetKind.ACTIVE_POWER:
            raise PhysicalUnitError(
                f"Dataset {spec.dataset_id!r} declares target kind {spec.target.kind.value!r}; "
                f"the supervised kW frame may only carry an active_power target, not {TARGET_COLUMN!r}."
            )
        target_present = TARGET_COLUMN in frame.columns
        if not target_present:
            errors.append(
                f"Declared target is missing: frame must contain {TARGET_COLUMN!r} in kW."
            )

    declared = list(spec.feature_names) + ([TARGET_COLUMN] if target_present else [])
    missing_declared_features = [name for name in spec.feature_names if name not in frame.columns]
    if missing_declared_features:
        errors.append(f"Declared features are missing from the frame: {missing_declared_features}.")
    unexpected_columns = [
        str(column) for column in frame.columns if column not in spec.features and column != TARGET_COLUMN
    ]

    non_numeric_columns = [
        name for name in declared if name in frame.columns and not pd.api.types.is_numeric_dtype(frame[name])
    ]
    if non_numeric_columns:
        errors.append(f"Declared columns are not numeric: {non_numeric_columns}.")

    infinite_counts: dict[str, int] = {}
    physical_range_violations: dict[str, int] = {}
    for name in declared:
        if name not in frame.columns or not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        values = _numeric_array(frame[name])
        infinite = int(np.count_nonzero(np.isinf(values)))
        if infinite:
            infinite_counts[name] = infinite
            errors.append(
                f"Column {name!r} contains {infinite} infinite value(s); adapters must replace "
                "±inf with NaN, never keep it as an observation."
            )
    for canonical_name, feature in spec.features.items():
        if canonical_name not in frame.columns or feature.physical_range is None:
            continue
        if not pd.api.types.is_numeric_dtype(frame[canonical_name]):
            continue
        values = _numeric_array(frame[canonical_name])
        low, high = float(feature.physical_range[0]), float(feature.physical_range[1])
        finite = np.isfinite(values)
        violations = int(np.count_nonzero(finite & ((values < low) | (values > high))))
        if violations:
            physical_range_violations[canonical_name] = violations

    target_nan_count: int | None = None
    target_negative_count: int | None = None
    if target_present and TARGET_COLUMN in frame.columns and pd.api.types.is_numeric_dtype(frame[TARGET_COLUMN]):
        target_values = _numeric_array(frame[TARGET_COLUMN])
        target_nan_count = int(np.count_nonzero(~np.isfinite(target_values)))
        target_negative_count = int(
            np.count_nonzero(np.isfinite(target_values) & (target_values < 0))
        )

    if errors:
        raise ContractError("Frame violates the data contract: " + " ".join(errors))

    return {
        "dataset_id": spec.dataset_id,
        "valid": True,
        "rows": int(len(frame)),
        "timezone": str(index.tz),
        "timezone_verified": bool(spec.timezone_verified and str(index.tz) == "UTC"),
        "duplicate_timestamps": duplicate_timestamps,
        "declared_features": list(spec.feature_names),
        "missing_declared_features": missing_declared_features,
        "unexpected_columns": unexpected_columns,
        "non_numeric_columns": non_numeric_columns,
        "infinite_counts": infinite_counts,
        "physical_range_violations": physical_range_violations,
        "target_present": target_present,
        "target_nan_count": target_nan_count,
        "target_negative_count": target_negative_count,
        "physical_range_violations_are_diagnostics_only": True,
        "negative_target_is_diagnostic_not_clipped": True,
    }


# ---------------------------------------------------------------------------
# causal filling
# ---------------------------------------------------------------------------


def fill_causal(
    frame: pd.DataFrame, spec: DatasetSpec, *, fill_limit: int = 4
) -> tuple[pd.DataFrame, dict]:
    """Forward-fill covariates only, bounded by ``fill_limit``, never the target.

    The fill is strictly causal (past -> present), never backward, and the target
    column is left byte-identically untouched.  The audit mapping records every
    column that was filled and the effective number of filled cells.
    """
    frame = _require_frame(frame)
    spec = _require_spec(spec)
    if isinstance(fill_limit, bool) or not isinstance(fill_limit, int) or fill_limit < 1:
        raise ValueError(f"fill_limit must be a positive integer, got {fill_limit!r}.")

    result = frame.copy()
    covariate_columns = [
        name for name in spec.feature_names if name in result.columns and name != TARGET_COLUMN
    ]
    filled_counts: dict[str, int] = {}
    for column in covariate_columns:
        before = int(result[column].isna().sum())
        if before:
            result[column] = result[column].ffill(limit=fill_limit)
        after = int(result[column].isna().sum())
        filled_counts[column] = before - after

    target_unchanged = True
    if TARGET_COLUMN in frame.columns:
        target_unchanged = bool(
            np.array_equal(
                frame[TARGET_COLUMN].to_numpy(dtype=float, na_value=np.nan),
                result[TARGET_COLUMN].to_numpy(dtype=float, na_value=np.nan),
                equal_nan=True,
            )
        )
    audit = {
        "policy": "causal_ffill_covariates_only",
        "fill_limit": int(fill_limit),
        "fill_direction": "forward_only",
        "backward_fill_used": False,
        "covariate_columns": covariate_columns,
        "filled_counts": filled_counts,
        "total_filled_cells": int(sum(filled_counts.values())),
        "target_column": TARGET_COLUMN if TARGET_COLUMN in frame.columns else None,
        "target_filled": 0,
        "target_values_modified": not target_unchanged,
        "notes": (
            "Only covariates are forward-filled, only from known past values, at most "
            "fill_limit consecutive missing steps. The target is never filled and no future "
            "value is used."
        ),
    }
    if not target_unchanged:
        raise ContractError("fill_causal modified the target values; this must never happen.")
    return result, audit


# ---------------------------------------------------------------------------
# versioned processed output
# ---------------------------------------------------------------------------


def _sanitize_tag(tag: str) -> str:
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("tag must be a nonempty string.")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if any(character not in allowed for character in tag):
        raise ValueError(
            f"tag={tag!r} contains characters outside [A-Za-z0-9_.-]; it is used as a file prefix."
        )
    return tag


def _unit_conversion_notes(spec: DatasetSpec) -> dict:
    converted: dict[str, dict[str, Any]] = {}
    unchanged: dict[str, str] = {}
    for canonical_name, feature in spec.features.items():
        if is_power_unit(feature.unit):
            converted[canonical_name] = {
                "declared_unit": feature.unit,
                "stored_unit": "kW",
                "multiply_by": float(to_kw(np.array([1.0]), feature.unit)[0]),
            }
        else:
            unchanged[canonical_name] = feature.unit
    target_note: dict[str, Any] | None = None
    if spec.target is not None:
        target_note = {
            "column": TARGET_COLUMN,
            "declared_unit": spec.target.unit,
            "stored_unit": "kW",
            "multiply_by": float(to_kw(np.array([1.0]), spec.target.unit)[0]),
            "target_kind": spec.target.kind.value,
            "target_scope": spec.target.scope.value,
            "label_origin": spec.target.label_origin.value,
            "synthetic": bool(spec.target.synthetic),
        }
    return {
        "target": target_note,
        "power_features_converted_to_kw": converted,
        "features_left_in_declared_unit": unchanged,
        "notes": (
            "All stored power values are kW. Utilisation/ratio/percent columns are never "
            "converted to kW; that would require an audited capacity calibration."
        ),
    }


def write_processed(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    *,
    out_dir: str = "data/processed",
    tag: str = "synthetic",
    source_files: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    """Write a new versioned parquet file plus a JSON metadata sidecar.

    The destination is always a *new* file; an existing processed file is never
    overwritten.  Writing below ``data/raw/`` is refused.  The returned mapping
    (also written to ``<file>.metadata.json``) records the dataset spec, source
    file hashes, processing code version, field mapping, unit conversions, the
    sampling grid, timezone/availability assumptions, quality mask statistics and
    the synthetic flag.
    """
    frame = _require_frame(frame)
    spec = _require_spec(spec)
    tag = _sanitize_tag(tag)
    if extra is not None and not isinstance(extra, Mapping):
        raise TypeError("extra must be a mapping or None.")

    validation = validate_frame(frame, spec)
    frequency = spec.native_frequency or "15min"
    quality = quality_report(frame, spec, frequency=frequency)

    directory = ensure_dir(assert_not_raw_path(resolve_path(out_dir)))
    stamp = utc_timestamp()
    # utc_timestamp() is RFC3339 and contains colons, which are illegal in Windows
    # file names; the versioned file name uses the colon-free form (the sidecar
    # keeps the exact RFC3339 timestamp under written_at_utc).
    file_stamp = stamp.replace(":", "")
    base = f"{tag}_{spec.dataset_id}_{file_stamp}"
    destination = directory / f"{base}.parquet"
    sidecar_path = directory / f"{base}.metadata.json"
    counter = 1
    while destination.exists() or sidecar_path.exists():
        base = f"{tag}_{spec.dataset_id}_{file_stamp}_{counter:02d}"
        destination = directory / f"{base}.parquet"
        sidecar_path = directory / f"{base}.metadata.json"
        counter += 1
        if counter > 1000:
            raise ContractError(
                f"Could not find a fresh processed filename for tag={tag!r} in {directory}."
            )

    destination = write_table(frame, destination)
    written_meta = file_metadata(destination)

    source_metadata: list[dict[str, Any]] | None = None
    generated = True
    if source_files:
        generated = False
        source_metadata = [file_metadata(path) for path in source_files]

    environment = environment_report()
    covariate_missing_counts = {
        name: quality["nan_counts"][name]
        for name in quality["declared_features"]
        if name in quality["nan_counts"]
    }
    payload: dict[str, Any] = {
        "dataset_id": spec.dataset_id,
        "site_id": spec.site_id,
        "domain_id": spec.domain_id,
        "tag": tag,
        "file": str(destination),
        "sidecar": str(sidecar_path),
        "bytes": int(written_meta["bytes"]),
        "sha256": written_meta["sha256"],
        "written_at_utc": stamp,
        "synthetic": bool(spec.synthetic),
        "spec": spec.to_dict(),
        "source_files": source_metadata,
        "provenance": {
            "generated": generated,
            "generated_at_utc": stamp,
            "git_head": environment.get("git_head"),
            "git_branch": environment.get("git_branch"),
            "git_base_main": environment.get("git_base_main"),
            "git_dirty_tracked": environment.get("git_dirty_tracked"),
            "python": environment.get("python"),
            "dependencies": environment.get("dependencies"),
            "generator": "src.data.preprocessing.write_processed",
        },
        "processing": {
            "module": "src.data.preprocessing",
            "code_version": {"git_head": environment.get("git_head")},
            "config": dict(extra or {}),
        },
        "field_mapping": mapping_table(spec),
        "unit_conversion": _unit_conversion_notes(spec),
        "sampling": {
            "native_frequency": spec.native_frequency,
            "frequency": frequency,
            "points_per_day": POINTS_PER_DAY if frequency in {"15min", "15T"} else None,
            "trace_anchor_timestamp": spec.trace_anchor_timestamp,
            "interval_semantics": spec.interval_semantics,
            "rows": quality["rows"],
            "start": quality["start"],
            "end": quality["end"],
            "grid": "timezone-aware UTC DatetimeIndex named 'timestamp'",
        },
        "timezone": {
            "declared": spec.timezone,
            "verified": bool(spec.timezone_verified),
            "index_timezone": quality["index_timezone"],
            "anchor_matches_first_timestamp": quality["anchor_matches_first_timestamp"],
        },
        "availability": {
            "semantics": spec.availability_semantics,
            "target_available_at_lag_seconds": float(
                spec.target.available_at_lag.total_seconds()
            )
            if spec.target is not None
            else None,
            "feature_available_at_lag_seconds": {
                name: float(feature.availability_lag.total_seconds())
                for name, feature in spec.features.items()
            },
            "assumption_notes": (
                "Availability lags are declared assumptions, not verified measurement latency."
            ),
        },
        "quality": quality,
        "quality_mask_statistics": {
            "rows": quality["rows"],
            "target_present": quality["target_present"],
            "target_missing_count": quality["target_nan_count"],
            "target_coverage_count": quality["target_coverage_count"],
            "covariate_missing_counts": covariate_missing_counts,
            "complete_trace_days": quality["complete_trace_days"],
            "complete_trace_days_with_finite_target": quality[
                "complete_trace_days_with_finite_target"
            ],
            "duplicate_timestamps": quality["duplicate_timestamps"],
            "missing_intervals": quality["missing_intervals"],
        },
        "validation": validation,
    }
    write_json(sidecar_path, payload)
    return payload


__all__ = [
    "fill_causal",
    "quality_report",
    "validate_frame",
    "write_processed",
]
