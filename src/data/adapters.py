"""Adapt a raw source into the canonical supervised kW frame.

The adapter is driven exclusively by the declared :class:`DatasetSpec`; it never
guesses meaning from column position or from a bare column name:

* only declared fields (the features and the declared target) are selected and
  renamed to their canonical names;
* power-unit features and an ``active_power`` target are converted to kW through
  :func:`src.data.contracts.to_kw`; utilisation / ratio / percent columns are
  **never** converted to kW (a use that would imply a power conversion raises
  :class:`PhysicalUnitError`);
* naive timestamps are localised to the dataset's **verified** timezone with
  ``ambiguous="raise"`` and ``nonexistent="raise"``, then converted to a
  timezone-aware UTC ``DatetimeIndex`` named ``"timestamp"``;
* ``±inf`` becomes ``NaN`` (never a valid observation) and exact duplicate
  timestamps are collapsed deterministically by their arithmetic mean while the
  count is reported; no other row is dropped.

Raw target column name
----------------------
``TargetSpec`` intentionally carries label semantics only.  To know which raw
column holds the label, the adapter reads the optional
``DatasetSpec.provenance["target_raw_column"]`` hint and otherwise assumes the
canonical ``TARGET_COLUMN`` name.  A ``power_utilization`` target is refused by
this adapter because it does not build a kW frame at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import (
    TARGET_COLUMN,
    ContractError,
    DatasetSpec,
    PhysicalUnitError,
    TargetKind,
    is_power_unit,
    to_kw,
)

#: Optional provenance hint naming the raw column that stores the supervised label.
TARGET_RAW_COLUMN_KEY: str = "target_raw_column"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _target_raw_name(spec: DatasetSpec) -> str:
    raw_name = spec.provenance.get(TARGET_RAW_COLUMN_KEY, TARGET_COLUMN)
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ContractError(
            f"{TARGET_RAW_COLUMN_KEY} must be a nonempty string, got {raw_name!r}."
        )
    return raw_name


def _looks_like_kw_column(canonical_name: str) -> bool:
    """Guard against a column explicitly named as kW while declaring another unit."""
    return canonical_name.lower() == TARGET_COLUMN.lower() or canonical_name.lower().endswith("_kw")


def _extract_timestamps(
    raw: pd.DataFrame, spec: DatasetSpec, timestamp_column: str | None
) -> pd.DatetimeIndex:
    if timestamp_column is not None:
        if timestamp_column not in raw.columns:
            raise ContractError(
                f"timestamp_column={timestamp_column!r} is not present in the raw frame; "
                f"available columns: {list(raw.columns)}."
            )
        values: Any = raw[timestamp_column]
    elif isinstance(raw.index, pd.DatetimeIndex):
        values = raw.index
    elif "timestamp" in raw.columns:
        values = raw["timestamp"]
    else:
        raise ContractError(
            "adapt_frame needs timestamps: pass timestamp_column=..., provide a DatetimeIndex, "
            "or include a 'timestamp' column."
        )
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Cannot parse the source timestamps: {exc}") from exc
    if index.hasnans:
        raise ContractError("Source timestamps contain missing values (NaT).")
    if index.tz is None:
        timezone = spec.require_verified_timezone("localizing naive source timestamps")
        index = index.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    index = index.tz_convert("UTC").as_unit("ns")
    return index


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def mapping_table(spec: DatasetSpec) -> dict:
    """``raw_name -> canonical/unit/statistic/available_at`` for audit sidecars."""
    if not isinstance(spec, DatasetSpec):
        raise ContractError("mapping_table expects a DatasetSpec.")
    table: dict[str, dict[str, Any]] = {}
    for canonical_name, feature in spec.features.items():
        table[feature.raw_name] = {
            "canonical_name": canonical_name,
            "unit": feature.unit,
            "statistic": feature.statistic.value,
            "available_at_lag_seconds": float(feature.availability_lag.total_seconds()),
            "available_at_rule": "a value observed at t is usable from t + available_at_lag_seconds",
            "future_known": bool(feature.future_known),
            "physical_range": list(feature.physical_range) if feature.physical_range else None,
            "source": feature.source,
            "is_target": False,
        }
    if spec.target is not None:
        raw_name = _target_raw_name(spec)
        if raw_name in table:
            raise ContractError(
                f"The declared target raw column {raw_name!r} collides with a feature raw name; "
                "each raw column must have exactly one semantic role."
            )
        table[raw_name] = {
            "canonical_name": TARGET_COLUMN,
            "unit": spec.target.unit,
            "statistic": "label",
            "available_at_lag_seconds": float(spec.target.available_at_lag.total_seconds()),
            "available_at_rule": "a value observed at t is usable from t + available_at_lag_seconds",
            "future_known": False,
            "physical_range": None,
            "source": spec.dataset_id,
            "is_target": True,
            "target_kind": spec.target.kind.value,
            "target_scope": spec.target.scope.value,
            "label_origin": spec.target.label_origin.value,
            "measurement_boundary": spec.target.measurement_boundary,
            "sampling_semantics": spec.target.sampling_semantics,
        }
    return table


def adapt_frame(
    raw: pd.DataFrame,
    spec: DatasetSpec,
    *,
    timestamp_column: str | None = None,
) -> pd.DataFrame:
    """Convert one raw table into the canonical supervised frame.

    Returns a frame indexed by a timezone-aware UTC ``DatetimeIndex`` named
    ``"timestamp"`` whose columns are exactly the declared feature canonical
    names plus (for a labelled dataset) ``TARGET_COLUMN`` in kW.  Adaptation
    diagnostics are attached as ``result.attrs["adaptation"]``.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"adapt_frame expects a pandas DataFrame, got {type(raw).__name__}.")
    if not isinstance(spec, DatasetSpec):
        raise ContractError("adapt_frame expects a DatasetSpec.")
    if raw.columns.duplicated().any():
        duplicated = sorted(set(raw.columns[raw.columns.duplicated()].tolist()))
        raise ContractError(f"Raw frame has duplicate column labels: {duplicated}.")

    if spec.target is not None and spec.target.kind is not TargetKind.ACTIVE_POWER:
        raise PhysicalUnitError(
            f"adapt_frame builds the supervised kW frame and refuses target kind "
            f"{spec.target.kind.value!r}: a utilisation/ratio target is not power and cannot be "
            "converted to kW without an audited capacity calibration. Register it as a separate "
            "proxy task instead."
        )

    if spec.target is not None:
        if TARGET_COLUMN in spec.features:
            raise ContractError(
                f"{TARGET_COLUMN!r} must not be declared as a covariate FeatureSpec: the label is "
                "not an input feature and must never be stored in the covariate history."
            )
        target_raw = _target_raw_name(spec)
    else:
        target_raw = None
        for canonical_name in spec.features:
            if _looks_like_kw_column(canonical_name):
                raise PhysicalUnitError(
                    f"Unlabelled dataset {spec.dataset_id!r} declares power-looking feature "
                    f"{canonical_name!r}; unlabelled pretraining sources must not carry a power "
                    "column. Remove it from the feature registry instead of renaming it."
                )

    # --- declared-field selection -----------------------------------------------------
    declared: list[tuple[str, str, str, bool]] = []  # (raw, canonical, unit, is_target)
    for canonical_name, feature in spec.features.items():
        if _looks_like_kw_column(canonical_name) and not is_power_unit(feature.unit):
            raise PhysicalUnitError(
                f"Feature {canonical_name!r} is named like a power column but declares unit "
                f"{feature.unit!r}; it cannot be converted to kW."
            )
        declared.append((feature.raw_name, canonical_name, feature.unit, False))
    if target_raw is not None:
        declared.append((target_raw, TARGET_COLUMN, spec.target.unit, True))  # type: ignore[union-attr]

    raw_names = [entry[0] for entry in declared]
    if len(set(raw_names)) != len(raw_names):
        raise ContractError(f"Declared raw column names are not unique: {raw_names}.")
    missing = [name for name in raw_names if name not in raw.columns]
    if missing:
        raise ContractError(
            f"Declared raw columns are missing from {spec.dataset_id!r}: {missing}. "
            f"Available columns: {list(raw.columns)}."
        )

    index = _extract_timestamps(raw, spec, timestamp_column)

    # --- numeric coercion, unit conversion, inf -> NaN --------------------------------
    columns: dict[str, np.ndarray] = {}
    inf_replaced: dict[str, int] = {}
    power_converted: dict[str, dict[str, Any]] = {}
    for raw_name, canonical_name, unit, is_target in declared:
        source_values = pd.to_numeric(raw[raw_name], errors="coerce")
        array = source_values.to_numpy(dtype=float, na_value=np.nan)
        infinite = int(np.isinf(array).sum())
        if infinite:
            array = np.where(np.isinf(array), np.nan, array)
            inf_replaced[canonical_name] = infinite
        if is_power_unit(unit):
            array = to_kw(array, unit)
            power_converted[canonical_name] = {
                "declared_unit": unit,
                "stored_unit": "kW",
                "factor": float(to_kw(np.array([1.0]), unit)[0]),
                "is_target": is_target,
            }
        columns[canonical_name] = array

    result = pd.DataFrame(columns, index=index)

    # --- ordering, duplicate collapse --------------------------------------------------
    duplicate_timestamps = int(index.duplicated().sum())
    result = result.sort_index(kind="mergesort")
    if duplicate_timestamps:
        # Deterministic collapse of exactly repeated timestamps (pandas mean ignores NaN).
        result = result.groupby(level=0).mean().sort_index()
    if len(result) == 0:
        raise ContractError(f"adapt_frame produced no rows for dataset {spec.dataset_id!r}.")
    if not result.index.is_monotonic_increasing or result.index.has_duplicates:
        raise ContractError("adapt_frame failed to produce a strictly increasing timestamp index.")
    if result.index.tz is None or str(result.index.tz) != "UTC":
        raise ContractError("adapt_frame must produce a timezone-aware UTC index.")
    result.index = result.index.rename("timestamp")

    result.attrs["adaptation"] = {
        "dataset_id": spec.dataset_id,
        "rows_input": int(len(raw)),
        "rows_output": int(len(result)),
        "rows_dropped": int(len(raw) - len(result) - duplicate_timestamps),
        "duplicate_timestamps_collapsed": duplicate_timestamps,
        "duplicate_collapse_policy": "arithmetic_mean_of_exactly_repeated_timestamps",
        "infinite_values_replaced_with_nan": inf_replaced,
        "power_columns_converted_to_kw": power_converted,
        "target_raw_column": target_raw,
        "target_declared_unit": spec.target.unit if spec.target is not None else None,
        "timezone": str(result.index.tz),
        "timezone_verified": bool(spec.timezone_verified),
    }
    return result


__all__ = ["TARGET_RAW_COLUMN_KEY", "adapt_frame", "mapping_table"]
