"""Configurable trace-day protocol, trace grid, and leakage-safe window feasibility.

Time semantics implemented here (task book §5.2, §6):

* Every dataset gets **one explicit trace anchor** (``DatasetSpec.trace_anchor_timestamp``).
  Trace days are counted from that anchor on the absolute slot grid: ``points_per_day``
  consecutive slots of ``frequency`` form one trace day, and the forecast origin of trace
  day ``k`` is the first slot of day ``k+1``.  A domain whose first record is late keeps the
  same day numbers as every other domain of the dataset; missing rows never renumber days
  and the first observation is never promoted to day 1.
* Protocols are configuration data.  History length, horizons, development/test origins and
  per-stage training cutoffs all come from YAML; no dataset-specific day numbers or history
  lengths are coded here.  The only built-in protocol assumptions are the ones the task book
  fixes for *every* protocol: 15-minute grid labels describe interval starts, one trace day
  is 24 hours of slots, and the anchor must be dataset-explicit (never per-domain renumbering).
* :func:`training_window_audit` answers the quantitative question "how many supervised samples
  can legally be built before a training cutoff?"  A sample is legal only when its whole
  context *and* its whole label window end at or before the cutoff.  Requiring a long context
  is not the same as having enough supervised samples: both numbers are reported separately.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from src.data.contracts import ContractError, InsufficientDataError, as_datetime64
from src.utils import load_config

#: The four run phases of the experiment lifecycle (task book §8.1).
PROTOCOL_PHASES: tuple[str, ...] = ("development", "finalize", "final-test", "smoke")

#: Only a dataset-explicit anchor is supported: per-domain renumbering is forbidden.
SUPPORTED_ANCHOR_POLICIES: tuple[str, ...] = ("dataset_explicit",)
#: Only explicitly listed origins are supported in this skeleton.
SUPPORTED_ORIGIN_MODES: tuple[str, ...] = ("explicit",)
#: Strict availability: a value is usable only once it is actually available at the origin.
SUPPORTED_AVAILABILITY_POLICIES: tuple[str, ...] = ("strict",)
SUPPORTED_PROTOCOL_TYPES: tuple[str, ...] = (
    "configurable",
    "fixed_24h",
    "fixed_7d",
    "demo",
    # Stage-2 main-model protocol (task book V2.0 §5): anchored D14/D22 day numbers
    # with per-horizon origin lists from the experiment config.  A configuration
    # instance of the same generic engine, not a new hard-coded execution path.
    "fixed_d14_d22",
)

TRACE_DAY = pd.Timedelta(24 * 60 * 60 * 1_000_000_000)


# ---------------------------------------------------------------------------
# small validation helpers
# ---------------------------------------------------------------------------


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}.")
    return int(value)


def frequency_step(frequency: str) -> pd.Timedelta:
    """Parse a pandas frequency alias into a positive fixed-width slot width."""
    if not isinstance(frequency, str) or not frequency.strip():
        raise ValueError(f"frequency must be a nonempty string, got {frequency!r}.")
    try:
        offset = to_offset(frequency)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frequency {frequency!r} is not a pandas offset alias.") from exc
    nanos = getattr(offset, "nanos", None)
    if nanos is None:
        raise ValueError(
            f"frequency {frequency!r} is not a fixed-width slot width; a trace grid needs a "
            "fixed number of nanoseconds per slot."
        )
    nanos = int(nanos)
    if nanos <= 0:
        raise ValueError(f"frequency must be positive, got {frequency!r}.")
    return pd.Timedelta(nanos)


def _component_frequency(frequency: Any, points_per_day: Any) -> pd.Timedelta:
    """Validate that ``points_per_day`` slots of ``frequency`` tile one 24h trace day."""
    step = frequency_step(frequency)
    points = _positive_int(points_per_day, "points_per_day")
    if step * points != TRACE_DAY:
        raise ValueError(
            f"{points} slots of {frequency!r} span {step * points}, not one 24h trace day. "
            "A trace day is defined as a fixed number of consecutive slots from the anchor, "
            "so frequency and points_per_day must tile exactly 24 hours."
        )
    return step


def _parse_timestamp(value: Any, *, name: str = "timestamp") -> pd.Timestamp:
    """UTC-naive ``Timestamp``; aware inputs are converted to UTC, never to local time."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot be interpreted as a timestamp: {value!r}.") from exc
    if pd.isna(stamp):
        raise ValueError(f"{name} is missing (NaT); an explicit timestamp is required.")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def _positive_step_from(values: Any) -> pd.Timedelta:
    step = frequency_step(str(values))
    return step


def _origin_tuple(values: Any, name: str) -> tuple[int, ...]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of trace day numbers.") from exc
    days = tuple(_positive_int(item, name) for item in items)
    if len(set(days)) != len(days):
        raise ValueError(f"{name} must not contain duplicate trace day numbers: {days}.")
    if list(days) != sorted(days):
        raise ValueError(f"{name} must be strictly increasing, got {days}.")
    return days


def _resolve_horizon_in(horizons: Mapping[str, int], horizon: Any) -> str:
    """Resolve a horizon name or point count against a declared mapping."""
    if isinstance(horizon, (int, np.integer)) and not isinstance(horizon, bool):
        points = int(horizon)
        for name, count in horizons.items():
            if int(count) == points:
                return name
    else:
        text = str(horizon).strip()
        if text in horizons:
            return text
        try:
            points = int(text)
        except (TypeError, ValueError):
            points = -1
        for name, count in horizons.items():
            if int(count) == points:
                return name
    raise ValueError(
        f"Unknown horizon {horizon!r}; this protocol declares {dict(horizons)}. "
        "Unknown horizons are rejected instead of falling back to a hidden default."
    )


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Protocol:
    """A fully resolved, configuration-driven time protocol.

    ``default_horizon`` names the horizon used when the caller does not pass one; it must be
    one of ``horizons`` (or the mapping must contain exactly one horizon).  All day numbers
    are trace day numbers counted from the dataset anchor, never calendar dates.
    """

    protocol_id: str
    frequency: str
    points_per_day: int
    context_points: int
    horizons: Mapping[str, int]
    trace_anchor_policy: str
    origin_mode: str
    development_origins: tuple[int, ...]
    test_origins: tuple[int, ...]
    train_cutoffs: Mapping[str, int | None]
    availability_policy: str
    protocol_type: str
    overlap_warning: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)
    default_horizon: str | None = None
    smoke_origins: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, str) or not self.protocol_id.strip():
            raise ValueError("protocol_id must be a nonempty string.")
        _component_frequency(self.frequency, self.points_per_day)
        object.__setattr__(self, "frequency", str(self.frequency))
        object.__setattr__(self, "points_per_day", _positive_int(self.points_per_day, "points_per_day"))
        object.__setattr__(self, "context_points", _positive_int(self.context_points, "context_points"))

        horizons = {
            str(name): _positive_int(count, f"horizons[{name!r}]")
            for name, count in dict(self.horizons).items()
        }
        if not horizons:
            raise ValueError("horizons must declare at least one horizon.")
        if len(set(horizons.values())) != len(horizons):
            raise ValueError(
                f"horizon point counts must be unique so a count maps to exactly one name: {horizons}."
            )
        object.__setattr__(self, "horizons", horizons)

        for attribute, allowed in (
            ("trace_anchor_policy", SUPPORTED_ANCHOR_POLICIES),
            ("origin_mode", SUPPORTED_ORIGIN_MODES),
            ("availability_policy", SUPPORTED_AVAILABILITY_POLICIES),
            ("protocol_type", SUPPORTED_PROTOCOL_TYPES),
        ):
            value = getattr(self, attribute)
            if value not in allowed:
                raise ValueError(
                    f"{attribute}={value!r} is not supported in this skeleton; "
                    f"choose one of {allowed}. Unsupported modes fail loudly instead of "
                    "silently falling back to a different time semantics."
                )

        object.__setattr__(
            self, "development_origins", _origin_tuple(self.development_origins, "development_origins")
        )
        object.__setattr__(self, "test_origins", _origin_tuple(self.test_origins, "test_origins"))
        object.__setattr__(self, "smoke_origins", _origin_tuple(self.smoke_origins, "smoke_origins"))

        cutoffs: dict[str, int | None] = {}
        for key, value in dict(self.train_cutoffs).items():
            if value is None:
                cutoffs[str(key)] = None
            else:
                cutoffs[str(key)] = _positive_int(value, f"train_cutoffs[{key!r}]")
        object.__setattr__(self, "train_cutoffs", cutoffs)

        default_horizon = self.default_horizon
        if default_horizon is not None:
            default_horizon = str(default_horizon)
            if default_horizon not in horizons:
                raise ValueError(
                    f"default_horizon={default_horizon!r} is not one of the declared horizons "
                    f"{sorted(horizons)}."
                )
        object.__setattr__(self, "default_horizon", default_horizon)
        object.__setattr__(self, "overlap_warning", bool(self.overlap_warning))
        if not isinstance(self.raw, Mapping):
            raise ValueError("raw must be a mapping (the original configuration).")
        object.__setattr__(self, "raw", copy.deepcopy(dict(self.raw)))

    # -- horizon handling ---------------------------------------------------

    @property
    def step(self) -> pd.Timedelta:
        return frequency_step(self.frequency)

    def horizon_name(self, horizon: Any) -> str:
        """``"24h"`` or ``"96"`` -> canonical horizon name; unknown input raises."""
        try:
            return _resolve_horizon_in(self.horizons, horizon)
        except ValueError as exc:
            raise ValueError(
                f"Horizon {horizon!r} is not declared by protocol {self.protocol_id!r}; "
                f"declared horizons are {sorted(self.horizons)}."
            ) from exc

    def horizon_points(self, horizon: Any | None) -> int:
        """Label length in points; ``None`` uses the protocol's declared default horizon."""
        if horizon is None:
            if self.default_horizon is None:
                if len(self.horizons) == 1:
                    return int(next(iter(self.horizons.values())))
                raise ValueError(
                    f"Protocol {self.protocol_id!r} declares {sorted(self.horizons)} but no "
                    "default_horizon; pass an explicit horizon."
                )
            horizon = self.default_horizon
        return int(self.horizons[self.horizon_name(horizon)])

    # -- origins ------------------------------------------------------------

    def origins_for(self, phase: str) -> tuple[int, ...]:
        """Explicit trace-day origins configured for one run phase."""
        if phase not in PROTOCOL_PHASES:
            raise ValueError(f"Unknown phase {phase!r}; expected one of {PROTOCOL_PHASES}.")
        if phase == "development":
            return self.development_origins
        if phase == "finalize":
            return ()
        if phase == "final-test":
            return self.test_origins
        # smoke: an explicitly declared demo list wins, otherwise the test origins are reused.
        return self.smoke_origins or self.test_origins

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "frequency": self.frequency,
            "points_per_day": self.points_per_day,
            "context_points": self.context_points,
            "horizons": {name: int(count) for name, count in self.horizons.items()},
            "trace_anchor_policy": self.trace_anchor_policy,
            "origin_mode": self.origin_mode,
            "development_origins": [int(day) for day in self.development_origins],
            "test_origins": [int(day) for day in self.test_origins],
            "smoke_origins": [int(day) for day in self.smoke_origins],
            "train_cutoffs": {
                name: (None if value is None else int(value))
                for name, value in self.train_cutoffs.items()
            },
            "availability_policy": self.availability_policy,
            "protocol_type": self.protocol_type,
            "default_horizon": self.default_horizon,
            "overlap_warning": bool(self.overlap_warning),
            "raw": copy.deepcopy(dict(self.raw)),
        }

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        horizon: Any | None = None,
        context_points: int | None = None,
        origins: Sequence[int] | None = None,
    ) -> "Protocol":
        """Build a protocol from a YAML-style mapping.

        ``horizon`` selects (and becomes the default for) one declared horizon; an undeclared
        horizon raises instead of being invented.  ``context_points`` overrides the configured
        history length.  ``origins`` is an explicit origin-day override applied to the
        development, test and smoke lists (callers take responsibility for the override); it
        never invents origins from the data.
        """
        if not isinstance(config, Mapping):
            raise ValueError("protocol configuration must be a mapping.")
        mapping: dict[str, Any] = dict(config)
        required = (
            "protocol_id",
            "frequency",
            "points_per_day",
            "context_points",
            "horizons",
            "trace_anchor_policy",
            "origin_mode",
            "development_origins",
            "test_origins",
            "train_cutoffs",
            "availability_policy",
            "protocol_type",
        )
        missing = [key for key in required if key not in mapping]
        if missing:
            raise ValueError(f"protocol configuration is missing required keys: {missing}.")

        horizons = {
            str(name): _positive_int(count, f"horizons[{name!r}]")
            for name, count in dict(mapping["horizons"]).items()
        }
        if not horizons:
            raise ValueError("protocol configuration must declare at least one horizon.")

        default_horizon = mapping.get("default_horizon")
        if horizon is not None:
            default_horizon = _resolve_horizon_in(horizons, horizon)
        elif default_horizon is not None:
            _resolve_horizon_in(horizons, default_horizon)
        elif len(horizons) == 1:
            default_horizon = next(iter(horizons))
        else:
            raise ValueError(
                "protocol configuration declares several horizons but no default_horizon; "
                "the configured default must be explicit."
            )

        development_origins = mapping["development_origins"]
        test_origins = mapping["test_origins"]
        smoke_origins = mapping.get("smoke_origins", ())
        if origins is not None:
            chosen = _origin_tuple(origins, "origins")
            development_origins = chosen
            test_origins = chosen
            smoke_origins = chosen

        return cls(
            protocol_id=mapping["protocol_id"],
            frequency=mapping["frequency"],
            points_per_day=mapping["points_per_day"],
            context_points=(
                mapping["context_points"] if context_points is None else context_points
            ),
            horizons=horizons,
            trace_anchor_policy=mapping["trace_anchor_policy"],
            origin_mode=mapping["origin_mode"],
            development_origins=development_origins,
            test_origins=test_origins,
            train_cutoffs=mapping["train_cutoffs"],
            availability_policy=mapping["availability_policy"],
            protocol_type=mapping["protocol_type"],
            overlap_warning=bool(mapping.get("overlap_warning", False)),
            raw=copy.deepcopy(mapping),
            default_horizon=(None if default_horizon is None else str(default_horizon)),
            smoke_origins=smoke_origins,
        )


def load_protocol(
    path: Any,
    *,
    horizon: Any | None = None,
    context_points: int | None = None,
    origins: Sequence[int] | None = None,
) -> Protocol:
    """Load a protocol presets file (relative paths resolve against the project root)."""
    return Protocol.from_config(
        load_config(path), horizon=horizon, context_points=context_points, origins=origins
    )


# ---------------------------------------------------------------------------
# trace grid
# ---------------------------------------------------------------------------


def _slot_positions(
    anchor: pd.Timestamp,
    timestamps: np.ndarray,
    step: pd.Timedelta,
    *,
    name: str,
) -> np.ndarray:
    """Absolute slot numbers of ``timestamps`` on the anchor grid (validated, never renumbered)."""
    values = np.asarray(timestamps, dtype="datetime64[ns]")
    diff = values.astype("int64") - anchor.value
    if bool((diff < 0).any()):
        first = values[np.argmin(diff)]
        raise ValueError(
            f"{name} contains {first} which precedes the trace anchor {anchor}. "
            "Timestamps before the anchor cannot be assigned a trace day."
        )
    remainder = diff % step.value
    if bool((remainder != 0).any()):
        offender = values[int(np.argmax(remainder != 0))]
        raise ValueError(
            f"{name} contains {offender} which is not aligned to anchor + k*{step} "
            f"(anchor={anchor}). Resample onto the protocol grid instead of shifting days."
        )
    return diff // step.value


@dataclass(frozen=True)
class TraceGrid:
    """The complete expected slot grid of one dataset, anchored once and shared by all domains.

    ``timestamps`` starts at ``anchor`` and is contiguous, so it is the *expected* grid, not
    the set of observed rows: interior or leading missing observations never renumber days.
    """

    anchor: pd.Timestamp
    timestamps: pd.DatetimeIndex
    points_per_day: int
    frequency: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor", _parse_timestamp(self.anchor, name="anchor"))
        object.__setattr__(self, "points_per_day", _positive_int(self.points_per_day, "points_per_day"))
        step = _component_frequency(self.frequency, self.points_per_day)
        object.__setattr__(self, "frequency", str(self.frequency))
        try:
            index = pd.DatetimeIndex(self.timestamps)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamps must be interpretable as a DatetimeIndex.") from exc
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        if len(index) == 0:
            raise ValueError("TraceGrid needs at least one slot timestamp.")
        if not index.is_unique:
            raise ValueError("TraceGrid timestamps must be unique.")
        if not index.is_monotonic_increasing:
            raise ValueError("TraceGrid timestamps must be strictly increasing.")
        if index[0] != self.anchor:
            raise ValueError(
                f"TraceGrid timestamps must start at the anchor {self.anchor}, got {index[0]}. "
                "A grid that starts at the first observation would silently renumber days."
            )
        if len(index) > 1 and bool((np.diff(index.asi8) != step.value).any()):
            raise ValueError(
                "TraceGrid timestamps must be contiguous on the protocol frequency; "
                "a sparse grid would make slot arithmetic ambiguous."
            )
        object.__setattr__(self, "timestamps", index)

    # -- geometry -----------------------------------------------------------

    @property
    def step(self) -> pd.Timedelta:
        return frequency_step(self.frequency)

    @property
    def n_slots(self) -> int:
        return int(len(self.timestamps))

    @property
    def n_complete_trace_days(self) -> int:
        return int(self.n_slots // self.points_per_day)

    def day_number(self, timestamp: Any) -> int:
        """1-based trace day of ``timestamp``, computed from the anchor only."""
        stamp = _parse_timestamp(timestamp, name="timestamp")
        step = self.step
        diff = stamp.value - self.anchor.value
        if diff < 0:
            raise ValueError(f"{stamp} precedes the trace anchor {self.anchor}.")
        if diff % step.value != 0:
            raise ValueError(
                f"{stamp} is not on the trace grid (anchor={self.anchor}, step={step}). "
                "Calendar-day numbering is never substituted for trace-day numbering."
            )
        slot = diff // step.value
        return int(slot // self.points_per_day) + 1

    def day_start(self, day: int) -> pd.Timestamp:
        """First slot of trace day ``day``."""
        day = _positive_int(day, "day")
        slot = (day - 1) * self.points_per_day
        if slot >= self.n_slots:
            raise ValueError(
                f"Trace day {day} starts at slot {slot}, outside the grid of {self.n_slots} slots."
            )
        return self.timestamps[slot]

    def origin_of_day(self, day: int) -> pd.Timestamp:
        """First slot of trace day ``day+1``: the information cutoff / forecast origin of ``day``.

        Computed arithmetically from the anchor, so it can point past the observed grid; callers
        planning windows must check coverage instead of silently shifting the origin.
        """
        day = _positive_int(day, "day")
        return pd.Timestamp(
            self.anchor.value + day * self.points_per_day * self.step.value
        )

    def origin_index(self, day: int) -> int:
        """Grid slot index of the origin of trace day ``day`` (may exceed the grid end)."""
        day = _positive_int(day, "day")
        return int(day * self.points_per_day)

    def contains(self, timestamps: Any) -> np.ndarray:
        """Boolean mask: is each timestamp one of the expected grid slots?"""
        values = as_datetime64(timestamps, name="timestamps")
        diff = values.astype("int64") - self.anchor.value
        step_value = self.step.value
        on_grid = (diff >= 0) & (diff % step_value == 0)
        within = diff < self.n_slots * step_value
        return np.asarray(on_grid & within, dtype=bool)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor.isoformat(),
            "frequency": self.frequency,
            "frequency_seconds": float(self.step.total_seconds()),
            "points_per_day": self.points_per_day,
            "n_slots": self.n_slots,
            "n_complete_trace_days": self.n_complete_trace_days,
            "first_timestamp": self.timestamps[0].isoformat(),
            "last_timestamp": self.timestamps[-1].isoformat(),
        }


def build_trace_grid(
    *,
    anchor: Any,
    timestamps: Any | None = None,
    points_per_day: int = 96,
    frequency: str = "15min",
    n_days: int | None = None,
) -> TraceGrid:
    """Build the expected slot grid from the dataset anchor.

    With ``timestamps`` the grid spans from the anchor through the last slot needed to cover
    them (it is never renumbered to start at the first observation).  Without timestamps,
    ``n_days`` complete trace days are built.
    """
    anchor_stamp = _parse_timestamp(anchor, name="anchor")
    step = _component_frequency(frequency, points_per_day)
    points = _positive_int(points_per_day, "points_per_day")
    if timestamps is None:
        if n_days is None:
            raise ValueError("build_trace_grid needs either timestamps or n_days.")
        days = _positive_int(n_days, "n_days")
        n_slots = days * points
    else:
        values = as_datetime64(timestamps, name="timestamps")
        if values.size == 0:
            raise ValueError("build_trace_grid cannot infer a grid from zero timestamps.")
        slots = _slot_positions(anchor_stamp, np.unique(values), step, name="timestamps")
        n_slots = int(slots.max()) + 1
    index = pd.DatetimeIndex(
        anchor_stamp.value + np.arange(n_slots, dtype="int64") * step.value
    )
    return TraceGrid(
        anchor=anchor_stamp,
        timestamps=index,
        points_per_day=points,
        frequency=str(frequency),
    )


def grid_for_frame(
    spec: Any,
    frame: pd.DataFrame,
    *,
    frequency: str = "15min",
    points_per_day: int = 96,
) -> TraceGrid:
    """Trace grid for a frame using the dataset's single explicit anchor."""
    anchor = getattr(spec, "trace_anchor_timestamp", None)
    if not anchor:
        raise ValueError(
            f"Dataset {getattr(spec, 'dataset_id', '<unknown>')!r} does not declare "
            "trace_anchor_timestamp. Refusing to renumber the first observation as trace day 1; "
            "declare the dataset's common anchor explicitly."
        )
    if not isinstance(frame, pd.DataFrame):
        raise ContractError("frame must be a pandas DataFrame indexed by timestamp.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ContractError("frame must be indexed by a DatetimeIndex of slot timestamps.")
    timestamps = as_datetime64(frame.index, name="frame.index")
    return build_trace_grid(
        anchor=anchor,
        timestamps=timestamps,
        points_per_day=points_per_day,
        frequency=frequency,
    )


# ---------------------------------------------------------------------------
# supervised-window feasibility under a training cutoff
# ---------------------------------------------------------------------------


def training_window_audit(
    *,
    train_cutoff_day: int,
    context_points: int,
    horizon_points: int,
    points_per_day: int = 96,
    stride_points: int | None = None,
) -> dict[str, Any]:
    """Count supervised windows that are fully legal at ``train_cutoff_day``.

    A supervised sample is legal only if its whole context and its whole label window end at
    or before the cutoff, i.e. for an origin at absolute slot ``o``:

    * ``o - context_points >= 0`` (complete context exists), and
    * ``o + horizon_points - 1 <= cutoff_day * points_per_day - 1`` (labels end by the cutoff).

    ``stride_points=None`` means one candidate origin per trace day (day boundaries); an
    explicit stride spaces candidates from the earliest legal origin.  The result is the
    structured ``insufficient_training_windows`` report required by the task book, and it is
    deliberately independent of any model: a long input history does not imply that a
    supervised model can be fitted.
    """
    cutoff_day = _positive_int(train_cutoff_day, "train_cutoff_day")
    context = _positive_int(context_points, "context_points")
    horizon = _positive_int(horizon_points, "horizon_points")
    points = _positive_int(points_per_day, "points_per_day")
    stride = None if stride_points is None else _positive_int(stride_points, "stride_points")

    cutoff_last_slot = cutoff_day * points - 1
    earliest_origin = context
    latest_origin = cutoff_last_slot - horizon + 1

    first_origin_slot: int | None = None
    last_origin_slot: int | None = None
    feasible = 0
    if latest_origin >= earliest_origin:
        if stride is None:
            first_day = (earliest_origin + points - 1) // points
            last_day = latest_origin // points
            feasible = max(0, last_day - first_day + 1)
            if feasible:
                first_origin_slot = first_day * points
                last_origin_slot = last_day * points
        else:
            feasible = (latest_origin - earliest_origin) // stride + 1
            first_origin_slot = earliest_origin
            last_origin_slot = earliest_origin + (feasible - 1) * stride

    reason: str | None = None
    if feasible == 0:
        reason = (
            f"No legal supervised window ends at or before the training cutoff of trace day "
            f"{cutoff_day}: a sample needs {context} context slots ending before its origin and "
            f"a {horizon}-point label window ending at or before slot {cutoff_last_slot} (the last "
            f"slot of trace day {cutoff_day}). The earliest origin with a complete context is slot "
            f"{earliest_origin}; the latest origin whose labels still fit is slot {latest_origin} "
            f"(shortfall {earliest_origin - latest_origin} slots). A longer input history must not "
            "be mistaken for enough supervised training data."
        )

    return {
        "feasible_windows": int(feasible),
        "insufficient_training_windows": bool(feasible == 0),
        "reason": reason,
        "train_cutoff_day": int(cutoff_day),
        "context_points": int(context),
        "horizon_points": int(horizon),
        "points_per_day": int(points),
        "stride_points": None if stride is None else int(stride),
        "train_cutoff_last_slot": int(cutoff_last_slot),
        "earliest_legal_origin_slot": int(earliest_origin),
        "latest_legal_origin_slot": int(latest_origin),
        "first_feasible_origin_slot": first_origin_slot,
        "last_feasible_origin_slot": last_origin_slot,
        "required_span_slots": int(context + horizon),
    }


def require_training_windows(
    *,
    train_cutoff_day: int,
    context_points: int,
    horizon_points: int,
    points_per_day: int = 96,
    stride_points: int | None = None,
) -> dict[str, Any]:
    """Like :func:`training_window_audit`, but raise when no legal window exists."""
    audit = training_window_audit(
        train_cutoff_day=train_cutoff_day,
        context_points=context_points,
        horizon_points=horizon_points,
        points_per_day=points_per_day,
        stride_points=stride_points,
    )
    if audit["feasible_windows"] == 0:
        raise InsufficientDataError(
            f"Insufficient supervised training windows (train_cutoff_day={train_cutoff_day}, "
            f"context_points={context_points}, horizon_points={horizon_points}, "
            f"points_per_day={points_per_day}, stride_points={stride_points}). "
            f"{audit['reason']} Use a no-fit baseline, a shorter context/horizon, or a later "
            "cutoff instead of silently shortening labels or borrowing future labels."
        )
    return audit


__all__ = [
    "PROTOCOL_PHASES",
    "Protocol",
    "TraceGrid",
    "build_trace_grid",
    "frequency_step",
    "grid_for_frame",
    "load_protocol",
    "require_training_windows",
    "training_window_audit",
]
