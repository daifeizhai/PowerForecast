"""Typed data contracts shared by the unlabelled and labelled data flows.

Design rules (task book §4, §5):

* :class:`PretrainBatch` carries **no power label at all**; unlabelled
  pretraining is structurally incapable of leaking kW into a supervised fit.
* :class:`ForecastBatch` carries ``targets_kw`` only when the declared target is
  a legal ``active_power`` target whose unit is kW.  Utilisation / ratio targets
  and node-level power cannot be smuggled into it.
* Timestamps and availability are validated at construction time, so a batch
  that would let information produced *after* the forecast origin enter the
  model can simply not be built.
* ``F=0`` (no covariates) and ``K=0`` (no known-future features) are legal: the
  power-only main path must always remain usable.

Internal time convention: all assembled arrays use timezone-naive
``datetime64[ns]`` values in **UTC**.  Conversion to local time happens at the
edges (data adapters, reporting), never inside the contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class ContractError(ValueError):
    """A data object violates the declared data contract."""


class PhysicalUnitError(ContractError):
    """A unit conversion would silently change physical meaning."""


class TargetEligibilityError(ContractError):
    """A target cannot be used for the requested (research) purpose."""


class LeakageError(ContractError):
    """Information produced after the forecast origin tried to enter a batch."""


class InsufficientDataError(ContractError):
    """The requested window cannot be built from the available history/labels."""


# ---------------------------------------------------------------------------
# enumerations
# ---------------------------------------------------------------------------


class DatasetRole(str, Enum):
    UNLABELED_PRETRAIN = "unlabeled_pretrain"
    TARGET_SUPERVISED = "target_supervised"
    FINAL_HOLDOUT = "final_holdout"


class AuditStatus(str, Enum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"


class TargetKind(str, Enum):
    ACTIVE_POWER = "active_power"
    POWER_UTILIZATION = "power_utilization"


class TargetScope(str, Enum):
    FACILITY = "facility"
    POWER_DOMAIN = "power_domain"
    RACK = "rack"
    NODE = "node"
    SERVER = "server"
    IT = "it"


class LabelOrigin(str, Enum):
    MEASURED = "measured"
    COMPONENT_SUM = "component_sum"
    ENERGY_CONVERTED = "energy_converted"
    ESTIMATED = "estimated"
    POWER_UTILIZATION = "power_utilization"
    SYNTHETIC_SIMULATED = "synthetic_simulated"


class IntervalStatistic(str, Enum):
    INSTANTANEOUS = "instantaneous"
    INTERVAL_MEAN = "interval_mean"
    INTERVAL_INTEGRAL = "interval_integral"
    EVENT_STATE = "event_state"
    LIFECYCLE_SUMMARY = "lifecycle_summary"


REAL_MEASUREMENT_ORIGINS = frozenset(
    {LabelOrigin.MEASURED, LabelOrigin.COMPONENT_SUM, LabelOrigin.ENERGY_CONVERTED}
)

#: Power units that can be converted to kW without extra assumptions.
POWER_TO_KW: dict[str, float] = {"w": 1e-3, "kw": 1.0, "mw": 1e3, "gw": 1e6}

#: Units that describe a ratio / utilisation, never a power.
UNITLESS_UNITS: frozenset[str] = frozenset(
    {"ratio", "fraction", "percent", "%", "pct", "dimensionless", "utilization", "utilisation"}
)

MISSING_POLICIES: frozenset[str] = frozenset({"mask", "reject"})

#: Canonical column holding the labelled power target in a supervised frame.
#: Never reuse this name for utilisation/ratio data.
TARGET_COLUMN: str = "active_power_kw"


def normalize_unit(unit: str) -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise PhysicalUnitError(f"Unit must be a nonempty string, got {unit!r}.")
    return unit.strip()


def is_power_unit(unit: str) -> bool:
    return normalize_unit(unit).lower() in POWER_TO_KW


def is_unitless(unit: str) -> bool:
    return normalize_unit(unit).lower() in UNITLESS_UNITS


def to_kw(values: Any, unit: str) -> np.ndarray:
    """Convert power values to kW.  Unitless data can never be converted."""
    key = normalize_unit(unit).lower()
    factor = POWER_TO_KW.get(key)
    if factor is None:
        raise PhysicalUnitError(
            f"Cannot convert unit {unit!r} to kW: only {sorted(POWER_TO_KW)} are power units. "
            "Utilisation/ratio data is not power and needs an external capacity calibration."
        )
    return np.asarray(values, dtype=float) * factor


def from_kw(values_kw: Any, unit: str) -> np.ndarray:
    """Convert kW back to the declared unit (used only for reporting)."""
    key = normalize_unit(unit).lower()
    factor = POWER_TO_KW.get(key)
    if factor is None:
        raise PhysicalUnitError(f"Cannot express kW in unit {unit!r}.")
    return np.asarray(values_kw, dtype=float) / factor


# ---------------------------------------------------------------------------
# feature / target / dataset specifications
# ---------------------------------------------------------------------------


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string, got {value!r}.")
    return value


@dataclass(frozen=True)
class FeatureSpec:
    """Declared physical/semantic meaning of one input field.

    ``availability_lag`` is the reporting latency: a value whose observation
    interval ends at ``t`` becomes usable at ``t + availability_lag``.  It must
    be declared explicitly; unknown latency is recorded as an assumption in
    ``notes`` rather than silently treated as zero.
    """

    raw_name: str
    canonical_name: str
    unit: str
    statistic: IntervalStatistic = IntervalStatistic.INTERVAL_MEAN
    physical_range: tuple[float, float] | None = None
    native_frequency: str | None = None
    availability_lag: pd.Timedelta = pd.Timedelta(0)
    future_known: bool = False
    missing_policy: str = "mask"
    source: str = ""
    audit_status: AuditStatus = AuditStatus.UNVERIFIED
    notes: str = ""

    def __post_init__(self) -> None:
        _require_text("raw_name", self.raw_name)
        _require_text("canonical_name", self.canonical_name)
        _require_text("unit", self.unit)
        if self.physical_range is not None:
            low, high = (float(self.physical_range[0]), float(self.physical_range[1]))
            if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
                raise ContractError(
                    f"physical_range must be finite and increasing for {self.canonical_name}."
                )
        if not isinstance(self.availability_lag, pd.Timedelta):
            raise ContractError("availability_lag must be a pandas Timedelta.")
        if self.availability_lag < pd.Timedelta(0):
            raise ContractError("availability_lag cannot be negative.")
        if self.missing_policy not in MISSING_POLICIES:
            raise ContractError(f"missing_policy must be one of {sorted(MISSING_POLICIES)}.")
        if self.statistic is IntervalStatistic.LIFECYCLE_SUMMARY and self.future_known:
            raise ContractError(
                "A post-hoc lifecycle summary can never be a known-future feature."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_name": self.raw_name,
            "canonical_name": self.canonical_name,
            "unit": self.unit,
            "statistic": self.statistic.value,
            "physical_range": list(self.physical_range) if self.physical_range else None,
            "native_frequency": self.native_frequency,
            "availability_lag_seconds": float(self.availability_lag.total_seconds()),
            "future_known": self.future_known,
            "missing_policy": self.missing_policy,
            "source": self.source,
            "audit_status": self.audit_status.value,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TargetSpec:
    """Declared label semantics: what is measured, where, in which unit, when."""

    kind: TargetKind
    scope: TargetScope
    unit: str
    label_origin: LabelOrigin
    measurement_boundary: str
    sampling_semantics: str
    available_at_lag: pd.Timedelta = pd.Timedelta(0)
    synthetic: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        _require_text("measurement_boundary", self.measurement_boundary)
        _require_text("sampling_semantics", self.sampling_semantics)
        _require_text("unit", self.unit)
        if not isinstance(self.available_at_lag, pd.Timedelta):
            raise ContractError("available_at_lag must be a pandas Timedelta.")
        if self.available_at_lag < pd.Timedelta(0):
            raise ContractError("available_at_lag cannot be negative.")
        if self.synthetic and self.label_origin is not LabelOrigin.SYNTHETIC_SIMULATED:
            raise TargetEligibilityError(
                "synthetic=True requires label_origin=synthetic_simulated; synthetic values "
                "must never be declared as measured."
            )
        if self.label_origin is LabelOrigin.SYNTHETIC_SIMULATED and not self.synthetic:
            raise TargetEligibilityError(
                "label_origin=synthetic_simulated requires synthetic=True."
            )
        if self.kind is TargetKind.ACTIVE_POWER:
            if not is_power_unit(self.unit):
                raise PhysicalUnitError(
                    f"active_power target must use a power unit, got {self.unit!r}."
                )
            if self.label_origin is LabelOrigin.POWER_UTILIZATION:
                raise TargetEligibilityError(
                    "active_power target cannot have label_origin=power_utilization."
                )
        else:  # POWER_UTILIZATION: a proxy target with no physical power unit
            if is_power_unit(self.unit):
                raise PhysicalUnitError(
                    "power_utilization target cannot declare a power unit; it is not kW."
                )
            if not is_unitless(self.unit):
                raise PhysicalUnitError(
                    f"power_utilization target must use a dimensionless unit, got {self.unit!r}."
                )
            if self.label_origin is not LabelOrigin.POWER_UTILIZATION:
                raise TargetEligibilityError(
                    "power_utilization target requires label_origin=power_utilization."
                )

    @property
    def is_facility_active_power(self) -> bool:
        return self.kind is TargetKind.ACTIVE_POWER and self.scope is TargetScope.FACILITY

    @property
    def requires_kw_conversion(self) -> bool:
        return normalize_unit(self.unit).lower() != "kw"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "scope": self.scope.value,
            "unit": self.unit,
            "label_origin": self.label_origin.value,
            "measurement_boundary": self.measurement_boundary,
            "sampling_semantics": self.sampling_semantics,
            "available_at_lag_seconds": float(self.available_at_lag.total_seconds()),
            "synthetic": self.synthetic,
            "notes": self.notes,
        }


def require_facility_active_power_kw(
    target: TargetSpec | None, *, allow_synthetic: bool = False
) -> None:
    """Guard the formal facility-kW main task.

    Node/server power, utilisation, and synthetic labels are rejected here
    instead of being silently promoted to a facility-level result.
    """
    if target is None:
        raise TargetEligibilityError("No target label is declared for a facility-kW task.")
    if target.kind is not TargetKind.ACTIVE_POWER:
        raise TargetEligibilityError(
            f"Target kind {target.kind.value!r} is not active_power; utilisation data cannot be "
            "evaluated as facility kW."
        )
    if target.scope is not TargetScope.FACILITY:
        raise TargetEligibilityError(
            f"Target scope {target.scope.value!r} is not facility; node/server power belongs to an "
            "explicitly named auxiliary task, not the facility main task."
        )
    if not is_power_unit(target.unit):
        raise PhysicalUnitError(f"Facility active power requires a power unit, got {target.unit!r}.")
    if target.synthetic and not allow_synthetic:
        raise TargetEligibilityError(
            "synthetic_simulated labels cannot produce a formal facility-kW research result."
        )


@dataclass(frozen=True)
class DatasetSpec:
    """Semantic description of one dataset/domain, independent of storage."""

    dataset_id: str
    site_id: str
    domain_id: str
    role: DatasetRole
    native_frequency: str | None = None
    trace_anchor_timestamp: str | None = None
    timezone: str | None = None
    timezone_verified: bool = False
    interval_semantics: str = ""
    availability_semantics: str = ""
    features: Mapping[str, FeatureSpec] = field(default_factory=dict)
    target: TargetSpec | None = None
    source_version: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    audit_status: AuditStatus = AuditStatus.UNVERIFIED

    def __post_init__(self) -> None:
        _require_text("dataset_id", self.dataset_id)
        _require_text("site_id", self.site_id)
        _require_text("domain_id", self.domain_id)
        if self.role is DatasetRole.UNLABELED_PRETRAIN and self.target is not None:
            raise ContractError(
                "unlabeled_pretrain datasets must declare target=None: an unlabelled source may "
                "not carry a power label."
            )
        if self.role is DatasetRole.TARGET_SUPERVISED and self.target is None:
            raise ContractError(
                "target_supervised datasets must declare a TargetSpec describing the label."
            )
        if self.role is DatasetRole.TARGET_SUPERVISED:
            for name, value in (
                ("timezone", self.timezone),
                ("trace_anchor_timestamp", self.trace_anchor_timestamp),
                ("interval_semantics", self.interval_semantics),
                ("availability_semantics", self.availability_semantics),
            ):
                _require_text(f"target_supervised.{name}", value)
        seen_raw: set[str] = set()
        for key, spec in self.features.items():
            if not isinstance(spec, FeatureSpec):
                raise ContractError(f"features[{key!r}] must be a FeatureSpec.")
            if key != spec.canonical_name:
                raise ContractError(
                    f"features key {key!r} must equal FeatureSpec.canonical_name "
                    f"{spec.canonical_name!r}; canonical names are the only join key."
                )
            if spec.raw_name in seen_raw:
                raise ContractError(f"duplicate raw_name {spec.raw_name!r} in one dataset.")
            seen_raw.add(spec.raw_name)

    @property
    def synthetic(self) -> bool:
        declared = bool(self.provenance.get("synthetic", False))
        label = bool(self.target.synthetic) if self.target is not None else False
        return declared or label

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.features.keys())

    def feature_specs(self) -> tuple[FeatureSpec, ...]:
        return tuple(self.features.values())

    def require_verified_timezone(self, purpose: str = "this operation") -> str:
        if not self.timezone_verified or not self.timezone:
            raise TargetEligibilityError(
                f"{purpose} needs a verified timezone for dataset {self.dataset_id!r}; "
                f"timezone={self.timezone!r}, timezone_verified={self.timezone_verified}. "
                "Keep the uncertainty explicit instead of assuming UTC."
            )
        return self.timezone

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "site_id": self.site_id,
            "domain_id": self.domain_id,
            "role": self.role.value,
            "native_frequency": self.native_frequency,
            "trace_anchor_timestamp": self.trace_anchor_timestamp,
            "timezone": self.timezone,
            "timezone_verified": self.timezone_verified,
            "interval_semantics": self.interval_semantics,
            "availability_semantics": self.availability_semantics,
            "features": {name: spec.to_dict() for name, spec in self.features.items()},
            "target": self.target.to_dict() if self.target is not None else None,
            "source_version": self.source_version,
            "provenance": dict(self.provenance),
            "audit_status": self.audit_status.value,
            "synthetic": self.synthetic,
        }


# ---------------------------------------------------------------------------
# array helpers
# ---------------------------------------------------------------------------

FORBIDDEN_LABEL_ATTRIBUTES = ("targets_kw", "targets", "future_kw", "power_labels")


def as_datetime64(values: Any, *, name: str = "timestamps") -> np.ndarray:
    """Coerce timestamps to UTC-naive ``datetime64[ns]``.

    Accepts ``datetime64`` arrays, ``DatetimeIndex`` (tz-aware or naive),
    sequences of ``Timestamp``/``datetime``/ISO strings, and object arrays that
    mix timezone-aware and naive values (naive values are interpreted as UTC,
    which is this project's internal convention).
    """
    if isinstance(values, np.ndarray) and values.dtype.kind == "M":
        result = values.astype("datetime64[ns]")
        if result.size and np.isnat(result).any():
            raise ContractError(f"{name} contains missing timestamps (NaT).")
        return result
    try:
        index = pd.DatetimeIndex(values)
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        result = index.to_numpy(dtype="datetime64[ns]")
    except (TypeError, ValueError):
        array = np.asarray(values)
        try:
            flat = pd.to_datetime(array.reshape(-1), utc=True, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{name} cannot be interpreted as timestamps: {exc}") from exc
        result = (
            flat.tz_localize(None).to_numpy(dtype="datetime64[ns]").reshape(array.shape)
        )
    if result.size and np.isnat(result).any():
        raise ContractError(f"{name} contains missing timestamps (NaT).")
    return result


def _float_array(name: str, values: Any, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim:
        raise ContractError(f"{name} must be {ndim}-D, got shape {array.shape}.")
    return array


def _bool_array(name: str, values: Any, ndim: int) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != bool:
        raise ContractError(f"{name} must be boolean, got dtype {array.dtype}.")
    if array.ndim != ndim:
        raise ContractError(f"{name} must be {ndim}-D, got shape {array.shape}.")
    return array


def _time_array(name: str, values: Any, ndim: int) -> np.ndarray:
    array = as_datetime64(values, name=name)
    if array.ndim != ndim:
        raise ContractError(f"{name} must be {ndim}-D, got shape {array.shape}.")
    return array


def _require_shape(name: str, array: np.ndarray, expected: tuple[int, ...]) -> None:
    if array.shape != expected:
        raise ContractError(f"{name} must have shape {expected}, got {array.shape}.")


def _require_finite(name: str, array: np.ndarray, mask: np.ndarray) -> None:
    if bool(mask.any()) and not bool(np.isfinite(array[mask]).all()):
        raise ContractError(f"{name} contains non-finite values where the mask says they exist.")


def _require_strictly_increasing(name: str, stamps: np.ndarray) -> None:
    if stamps.shape[-1] > 1 and not bool(np.all(np.diff(stamps.astype("int64"), axis=-1) > 0)):
        raise ContractError(f"{name} must be strictly increasing along the time axis.")


def _names_tuple(name: str, values: Any, expected: int) -> tuple[str, ...]:
    names = tuple(values)
    if len(names) != expected:
        raise ContractError(f"{name} declares {len(names)} names but {expected} are needed.")
    if any(not isinstance(item, str) or not item for item in names):
        raise ContractError(f"{name} must contain nonempty strings.")
    if len(set(names)) != len(names):
        raise ContractError(f"{name} contains duplicate names.")
    return names


# ---------------------------------------------------------------------------
# unlabelled batch
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PretrainBatch:
    """Unlabelled operational-metric batch: no kW and no label placeholder."""

    features: np.ndarray  # [B, L, F]
    observed_mask: np.ndarray  # [B, L, F]
    timestamps: np.ndarray  # [B, L]
    available_at: np.ndarray  # [B, L, F]
    feature_names: tuple[str, ...]
    domain_ids: tuple[str, ...]
    sampling_metadata: Mapping[str, Any] = field(default_factory=dict)
    dataset_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _float_array("features", self.features, 3))
        object.__setattr__(self, "observed_mask", _bool_array("observed_mask", self.observed_mask, 3))
        object.__setattr__(self, "timestamps", _time_array("timestamps", self.timestamps, 2))
        object.__setattr__(self, "available_at", _time_array("available_at", self.available_at, 3))
        batch, length, width = self.features.shape
        if batch < 1 or length < 1 or width < 1:
            raise ContractError(f"PretrainBatch needs B>=1, L>=1, F>=1, got {self.features.shape}.")
        _require_shape("observed_mask", self.observed_mask, (batch, length, width))
        _require_shape("timestamps", self.timestamps, (batch, length))
        _require_shape("available_at", self.available_at, (batch, length, width))
        object.__setattr__(
            self, "feature_names", _names_tuple("feature_names", self.feature_names, width)
        )
        domains = tuple(self.domain_ids)
        if len(domains) != batch:
            raise ContractError(f"domain_ids needs {batch} entries, got {len(domains)}.")
        if any(not isinstance(item, str) or not item for item in domains):
            raise ContractError("domain_ids must contain nonempty strings.")
        object.__setattr__(self, "domain_ids", domains)
        _require_strictly_increasing("timestamps", self.timestamps)
        _require_finite("features", self.features, self.observed_mask)
        if bool((self.available_at < self.timestamps[:, :, None]).any()):
            raise ContractError(
                "available_at cannot be earlier than the observation timestamp it describes."
            )
        for attribute in FORBIDDEN_LABEL_ATTRIBUTES:
            if hasattr(self, attribute):
                raise ContractError(
                    f"PretrainBatch must not expose {attribute!r}; unlabelled data carries no kW."
                )
        if not isinstance(self.sampling_metadata, Mapping):
            raise ContractError("sampling_metadata must be a mapping.")

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_timesteps(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[2])

    def slice(self, indices: Any) -> "PretrainBatch":
        return PretrainBatch(
            features=self.features[indices],
            observed_mask=self.observed_mask[indices],
            timestamps=self.timestamps[indices],
            available_at=self.available_at[indices],
            feature_names=self.feature_names,
            domain_ids=tuple(np.asarray(self.domain_ids)[indices].tolist()),
            sampling_metadata=self.sampling_metadata,
            dataset_id=self.dataset_id,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "feature_names": list(self.feature_names),
            "domain_ids": list(self.domain_ids),
            "shape": [self.batch_size, self.n_timesteps, self.n_features],
            "sampling_metadata": dict(self.sampling_metadata),
            "contains_power_labels": False,
        }


def require_pretrain_batch(batch: Any) -> PretrainBatch:
    """Reject supervised batches at the unlabelled entry point."""
    if not isinstance(batch, PretrainBatch):
        raise ContractError(
            "Unlabelled pretraining accepts PretrainBatch only; "
            f"got {type(batch).__name__}. Supervised fit must not be called with unlabelled data, "
            "and unlabelled pretraining must never see labels."
        )
    return batch


# ---------------------------------------------------------------------------
# labelled forecast batches
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ForecastInputs:
    """Everything a forecaster may see at prediction time (no labels).

    ``past_power_kw`` is the main-path history.  Covariates (``F``) and
    known-future features (``K``) may both be empty.  All availability times are
    guaranteed to be ``<=`` the corresponding origin: the constructor raises
    :class:`LeakageError` otherwise.
    """

    past_power_kw: np.ndarray  # [B, L]
    past_power_valid: np.ndarray  # [B, L]
    past_timestamps: np.ndarray  # [B, L]
    origin_timestamps: np.ndarray  # [B]
    target_timestamps: np.ndarray  # [B, H]
    target_spec: TargetSpec
    domain_ids: tuple[str, ...]
    dataset_id: str = ""
    past_covariates: np.ndarray | None = None  # [B, L, F] or None when F=0
    past_covariate_mask: np.ndarray | None = None
    past_covariate_available_at: np.ndarray | None = None
    covariate_names: tuple[str, ...] = ()
    future_known: np.ndarray | None = None  # [B, H, K] or None when K=0
    future_known_mask: np.ndarray | None = None
    future_known_available_at: np.ndarray | None = None
    future_known_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.target_spec, TargetSpec):
            raise ContractError("target_spec must be a TargetSpec.")
        if self.target_spec.kind is not TargetKind.ACTIVE_POWER:
            raise PhysicalUnitError(
                "ForecastInputs.past_power_kw requires an active_power target; "
                f"got {self.target_spec.kind.value!r}."
            )
        if not is_power_unit(self.target_spec.unit):
            raise PhysicalUnitError(
                f"ForecastInputs.past_power_kw requires a power unit, got {self.target_spec.unit!r}."
            )
        object.__setattr__(self, "past_power_kw", _float_array("past_power_kw", self.past_power_kw, 2))
        object.__setattr__(
            self, "past_power_valid", _bool_array("past_power_valid", self.past_power_valid, 2)
        )
        object.__setattr__(
            self, "past_timestamps", _time_array("past_timestamps", self.past_timestamps, 2)
        )
        object.__setattr__(
            self, "origin_timestamps", _time_array("origin_timestamps", self.origin_timestamps, 1)
        )
        object.__setattr__(
            self, "target_timestamps", _time_array("target_timestamps", self.target_timestamps, 2)
        )
        batch, horizon = self.target_timestamps.shape
        length = self.past_power_kw.shape[1]
        _require_shape("past_power_valid", self.past_power_valid, (batch, length))
        _require_shape("past_timestamps", self.past_timestamps, (batch, length))
        _require_shape("origin_timestamps", self.origin_timestamps, (batch,))
        if batch < 1 or length < 1 or horizon < 1:
            raise ContractError("ForecastInputs needs B>=1, L>=1, H>=1.")
        domains = tuple(self.domain_ids)
        if len(domains) != batch:
            raise ContractError(f"domain_ids needs {batch} entries, got {len(domains)}.")
        object.__setattr__(self, "domain_ids", domains)
        if not bool((self.target_timestamps[:, 0] == self.origin_timestamps).all()):
            raise ContractError(
                "origin_timestamps must equal the first target timestamp: the origin is the start "
                "of the first predicted 15-minute interval and the information cutoff."
            )
        _require_strictly_increasing("past_timestamps", self.past_timestamps)
        _require_strictly_increasing("target_timestamps", self.target_timestamps)
        _require_finite("past_power_kw", self.past_power_kw, self.past_power_valid)
        if bool((self.past_timestamps >= self.origin_timestamps[:, None]).any()):
            raise LeakageError(
                "Every context timestamp must lie before its forecast origin; otherwise the "
                "current (unfinished) interval would be treated as observed history."
            )
        self._validate_covariates(batch, length, horizon)
        self._validate_future_known(batch, horizon)
        if not isinstance(self.metadata, Mapping):
            raise ContractError("metadata must be a mapping.")

    def _validate_covariates(self, batch: int, length: int, horizon: int) -> None:
        covariates = self.past_covariates
        names = tuple(self.covariate_names)
        if covariates is None:
            if names:
                raise ContractError("covariate_names were given without past_covariates.")
            if self.past_covariate_mask is not None or self.past_covariate_available_at is not None:
                raise ContractError("covariate mask/availability given without past_covariates.")
            return
        object.__setattr__(
            self, "past_covariates", _float_array("past_covariates", covariates, 3)
        )
        width = self.past_covariates.shape[2]
        _require_shape("past_covariates", self.past_covariates, (batch, length, width))
        mask = self.past_covariate_mask
        if mask is None:
            mask = np.ones_like(self.past_covariates, dtype=bool)
        object.__setattr__(
            self, "past_covariate_mask", _bool_array("past_covariate_mask", mask, 3)
        )
        _require_shape("past_covariate_mask", self.past_covariate_mask, (batch, length, width))
        available = self.past_covariate_available_at
        if available is None:
            available = np.broadcast_to(
                self.past_timestamps[:, :, None], (batch, length, width)
            ).copy()
        object.__setattr__(
            self, "past_covariate_available_at", _time_array("past_covariate_available_at", available, 3)
        )
        _require_shape(
            "past_covariate_available_at", self.past_covariate_available_at, (batch, length, width)
        )
        object.__setattr__(self, "covariate_names", _names_tuple("covariate_names", names, width))
        _require_finite("past_covariates", self.past_covariates, self.past_covariate_mask)
        # Availability is judged on the observations that are actually declared
        # usable (mask=True).  Entries masked out are not inputs at all.
        usable = self.past_covariate_mask
        origin_grid = np.broadcast_to(self.origin_timestamps[:, None, None], usable.shape)
        if bool(np.any(usable & (self.past_covariate_available_at > origin_grid))):
            raise LeakageError(
                "A past covariate would not yet be available at its forecast origin "
                "(available_at > origin). Audit its measurement latency or mask it out."
            )
        stamp_grid = np.broadcast_to(self.past_timestamps[:, :, None], usable.shape)
        if bool(np.any(usable & (self.past_covariate_available_at < stamp_grid))):
            raise ContractError("available_at cannot be earlier than its observation timestamp.")

    def _validate_future_known(self, batch: int, horizon: int) -> None:
        known = self.future_known
        names = tuple(self.future_known_names)
        if known is None:
            if names:
                raise ContractError("future_known_names were given without future_known.")
            if self.future_known_mask is not None or self.future_known_available_at is not None:
                raise ContractError("future-known mask/availability given without future_known.")
            return
        object.__setattr__(self, "future_known", _float_array("future_known", known, 3))
        width = self.future_known.shape[2]
        _require_shape("future_known", self.future_known, (batch, horizon, width))
        mask = self.future_known_mask
        if mask is None:
            mask = np.ones_like(self.future_known, dtype=bool)
        object.__setattr__(self, "future_known_mask", _bool_array("future_known_mask", mask, 3))
        _require_shape("future_known_mask", self.future_known_mask, (batch, horizon, width))
        available = self.future_known_available_at
        if available is None:
            available = np.broadcast_to(
                self.origin_timestamps[:, None, None], (batch, horizon, width)
            ).copy()
        object.__setattr__(
            self, "future_known_available_at", _time_array("future_known_available_at", available, 3)
        )
        _require_shape(
            "future_known_available_at", self.future_known_available_at, (batch, horizon, width)
        )
        object.__setattr__(
            self, "future_known_names", _names_tuple("future_known_names", names, width)
        )
        _require_finite("future_known", self.future_known, self.future_known_mask)
        usable = self.future_known_mask
        origin_grid = np.broadcast_to(self.origin_timestamps[:, None, None], usable.shape)
        if bool(np.any(usable & (self.future_known_available_at > origin_grid))):
            raise LeakageError(
                "A 'known future' feature is not actually available at the origin "
                "(available_at > origin); only audited calendar/time information is known future."
            )

    @property
    def batch_size(self) -> int:
        return int(self.past_power_kw.shape[0])

    @property
    def context_length(self) -> int:
        return int(self.past_power_kw.shape[1])

    @property
    def horizon(self) -> int:
        return int(self.target_timestamps.shape[1])

    @property
    def n_covariates(self) -> int:
        return 0 if self.past_covariates is None else int(self.past_covariates.shape[2])

    @property
    def n_future_known(self) -> int:
        return 0 if self.future_known is None else int(self.future_known.shape[2])

    def slice(self, indices: Any) -> "ForecastInputs":
        def take(value: np.ndarray | None) -> np.ndarray | None:
            return None if value is None else value[indices]

        return ForecastInputs(
            past_power_kw=self.past_power_kw[indices],
            past_power_valid=self.past_power_valid[indices],
            past_timestamps=self.past_timestamps[indices],
            origin_timestamps=self.origin_timestamps[indices],
            target_timestamps=self.target_timestamps[indices],
            target_spec=self.target_spec,
            domain_ids=tuple(np.asarray(self.domain_ids)[indices].tolist()),
            dataset_id=self.dataset_id,
            past_covariates=take(self.past_covariates),
            past_covariate_mask=take(self.past_covariate_mask),
            past_covariate_available_at=take(self.past_covariate_available_at),
            covariate_names=self.covariate_names,
            future_known=take(self.future_known),
            future_known_mask=take(self.future_known_mask),
            future_known_available_at=take(self.future_known_available_at),
            future_known_names=self.future_known_names,
            metadata=self.metadata,
        )

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "domain_ids": list(self.domain_ids),
            "shape": {
                "batch": self.batch_size,
                "context_length": self.context_length,
                "horizon": self.horizon,
                "n_covariates": self.n_covariates,
                "n_future_known": self.n_future_known,
            },
            "covariate_names": list(self.covariate_names),
            "future_known_names": list(self.future_known_names),
            "target": self.target_spec.to_dict(),
            "extra": dict(self.metadata),
        }


@dataclass(frozen=True, eq=False)
class ForecastBatch:
    """Supervised labelled batch: inputs plus verified kW targets."""

    inputs: ForecastInputs
    targets_kw: np.ndarray  # [B, H]
    target_valid_mask: np.ndarray  # [B, H]
    target_available_at: np.ndarray | None = None  # [B, H]

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, ForecastInputs):
            raise ContractError("ForecastBatch.inputs must be a ForecastInputs instance.")
        target = self.inputs.target_spec
        if target.kind is not TargetKind.ACTIVE_POWER:
            raise TargetEligibilityError(
                f"targets_kw only accepts active_power labels, got {target.kind.value!r}."
            )
        if normalize_unit(target.unit).lower() != "kw":
            raise PhysicalUnitError(
                f"targets_kw requires the declared target unit to be kW, got {target.unit!r}. "
                "Convert W/MW with to_kw() before building the batch."
            )
        if target.scope is TargetScope.NODE and target.kind is TargetKind.ACTIVE_POWER:
            # Legal only as an explicitly named auxiliary task; the CLI guards the
            # facility main task, here we only refuse silent promotion.
            pass
        object.__setattr__(self, "targets_kw", _float_array("targets_kw", self.targets_kw, 2))
        object.__setattr__(
            self, "target_valid_mask", _bool_array("target_valid_mask", self.target_valid_mask, 2)
        )
        expected = (self.inputs.batch_size, self.inputs.horizon)
        _require_shape("targets_kw", self.targets_kw, expected)
        _require_shape("target_valid_mask", self.target_valid_mask, expected)
        _require_finite("targets_kw", self.targets_kw, self.target_valid_mask)
        if self.target_available_at is not None:
            object.__setattr__(
                self,
                "target_available_at",
                _time_array("target_available_at", self.target_available_at, 2),
            )
            _require_shape("target_available_at", self.target_available_at, expected)

    @property
    def batch_size(self) -> int:
        return self.inputs.batch_size

    @property
    def horizon(self) -> int:
        return self.inputs.horizon

    def slice(self, indices: Any) -> "ForecastBatch":
        return ForecastBatch(
            inputs=self.inputs.slice(indices),
            targets_kw=self.targets_kw[indices],
            target_valid_mask=self.target_valid_mask[indices],
            target_available_at=None
            if self.target_available_at is None
            else self.target_available_at[indices],
        )

    def metadata_dict(self) -> dict[str, Any]:
        return {
            **self.inputs.metadata_dict(),
            "labelled": True,
            "valid_target_fraction": float(self.target_valid_mask.mean()),
        }
