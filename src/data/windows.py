"""Leakage-safe forecast and pretraining window assembly.

The time protocol lives in :mod:`src.data.splitting`; this module turns a resolved
:class:`~src.data.splitting.OriginPlan` into contract-validated batches:

* ``ForecastInputs`` / ``ForecastBatch`` carry history power, past covariates, the audited
  calendar known-future block, and labels for supervised work.
* ``PretrainBatch`` carries operational metrics only: no kW, no label placeholder, native
  sampling frequency preserved.

Assembly rules enforced here (task book §4.2, §5.2, §6.5):

* The context is exactly ``context_points`` slots immediately before the origin
  (``context[-1] == origin - one slot``).
* Every covariate value is usable only when ``observation_end + availability_lag <= origin``;
  otherwise it is masked out **and its value is not delivered** (NaN), never presented as if
  it had been observed.  Missing frame values stay NaN with ``mask=False``; they are never
  zero-filled.
* Calendar features are the only default known-future block; they are generated from the
  target timestamps and carry ``available_at == origin``.
* Missing frame rows inside a window are not repaired: the corresponding positions become
  NaN / invalid so a no-fit baseline can reject the window explicitly.  Windows are never
  shifted, shortened or interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.contracts import (
    TARGET_COLUMN,
    ContractError,
    DatasetRole,
    ForecastBatch,
    ForecastInputs,
    InsufficientDataError,
    PretrainBatch,
    TargetEligibilityError,
    to_kw,
)
from src.data.features import calendar_feature_matrix
from src.data.splitting import Protocol, TraceGrid, frequency_step

__all__ = [
    "OriginPlan",
    "audit_split",
    "build_forecast_batch",
    "build_forecast_inputs",
    "build_pretrain_batch",
    "plan_origins",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ContractError(f"{name} must be a positive integer, got {value!r}.")
    if int(value) < 1:
        raise ContractError(f"{name} must be >= 1, got {value!r}.")
    return int(value)


def _naive_timestamp(value: Any, *, name: str = "timestamp") -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} cannot be interpreted as a timestamp: {value!r}.") from exc
    if pd.isna(stamp):
        raise ContractError(f"{name} is missing (NaT).")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def _frame_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """UTC-naive, sorted, unique DatetimeIndex of a frame."""
    if not isinstance(frame, pd.DataFrame):
        raise ContractError("frame must be a pandas DataFrame.")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ContractError("frame must be indexed by a DatetimeIndex of slot timestamps.")
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    if len(index) == 0:
        raise InsufficientDataError("frame has no rows.")
    if not index.is_unique:
        raise ContractError(
            "frame index must be unique: duplicated timestamps would make one slot ambiguous."
        )
    if not index.is_monotonic_increasing:
        raise ContractError("frame index must be sorted in increasing time order.")
    return index


def _column_values(frame: pd.DataFrame, column: str, *, what: str = "column") -> np.ndarray:
    if column not in frame.columns:
        raise ContractError(f"frame is missing required {what} {column!r}.")
    return frame[column].to_numpy(dtype=float, na_value=np.nan)


def _resolve_feature_column(feature_spec: Any, columns: pd.Index) -> str:
    for candidate in (feature_spec.canonical_name, feature_spec.raw_name):
        if candidate in columns:
            return candidate
    raise ContractError(
        f"Declared feature {feature_spec.canonical_name!r} (raw_name={feature_spec.raw_name!r}) "
        f"is absent from the frame columns {sorted(map(str, columns))}. "
        "A declared covariate must exist; it is never silently dropped or zero-filled."
    )


def _state(plan: "OriginPlan") -> dict[str, Any]:
    return {
        "phase": plan.phase,
        "protocol_id": plan.protocol_id,
        "horizon": plan.horizon,
        "horizon_points": int(plan.horizon_points),
        "context_points": int(plan.context_points),
        "points_per_day": int(plan.points_per_day),
        "frequency": plan.frequency,
        "days": [int(day) for day in plan.days],
    }


# ---------------------------------------------------------------------------
# origin plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OriginPlan:
    """Resolved, coverage-checked forecast origins for one phase/horizon.

    ``dropped`` records every configured origin that could not be used, with the reason; a
    dropped origin is never replaced by a shifted day number and the horizon is never
    shortened.  ``overlap`` is True when the label windows of two origins share timestamps,
    in which case the audit states that the samples are not independent.
    """

    phase: str
    protocol_id: str
    horizon: str
    horizon_points: int
    days: tuple[int, ...]
    origins: pd.DatetimeIndex
    anchor: pd.Timestamp
    context_points: int
    label_start: pd.Timestamp
    label_end: pd.Timestamp
    required_history_points: int
    dropped: tuple[dict, ...] = ()
    overlap: bool = False
    frequency: str = "15min"
    points_per_day: int = 96

    def __post_init__(self) -> None:
        for attribute in ("phase", "protocol_id", "horizon", "frequency"):
            value = getattr(self, attribute)
            if not isinstance(value, str) or not value:
                raise ContractError(f"{attribute} must be a nonempty string, got {value!r}.")
        object.__setattr__(self, "horizon_points", _positive_int(self.horizon_points, "horizon_points"))
        object.__setattr__(self, "context_points", _positive_int(self.context_points, "context_points"))
        object.__setattr__(
            self,
            "required_history_points",
            _positive_int(self.required_history_points, "required_history_points"),
        )
        object.__setattr__(self, "points_per_day", _positive_int(self.points_per_day, "points_per_day"))
        object.__setattr__(self, "days", tuple(int(day) for day in self.days))
        origins = pd.DatetimeIndex(self.origins)
        if origins.tz is not None:
            origins = origins.tz_convert("UTC").tz_localize(None)
        if len(origins) != len(self.days):
            raise ContractError(
                f"OriginPlan has {len(self.days)} trace days but {len(origins)} origins."
            )
        if len(origins) == 0:
            raise ContractError("OriginPlan must contain at least one feasible origin.")
        if not origins.is_unique or not origins.is_monotonic_increasing:
            raise ContractError("OriginPlan origins must be strictly increasing and unique.")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "anchor", _naive_timestamp(self.anchor, name="anchor"))
        object.__setattr__(self, "label_start", _naive_timestamp(self.label_start, name="label_start"))
        object.__setattr__(self, "label_end", _naive_timestamp(self.label_end, name="label_end"))
        object.__setattr__(self, "dropped", tuple(dict(entry) for entry in self.dropped))
        object.__setattr__(self, "overlap", bool(self.overlap))

    @property
    def step(self) -> pd.Timedelta:
        return frequency_step(self.frequency)

    @property
    def n_origins(self) -> int:
        return int(len(self.origins))

    def audit(self) -> dict[str, Any]:
        """JSON-serialisable split audit payload (``results/<run>/split_audit.json``)."""
        step = self.step
        step_value = step.value
        points = int(self.points_per_day)
        anchor_ns = self.anchor.value
        records: list[dict[str, Any]] = []
        slot_windows: list[np.ndarray] = []
        covered: set[int] = set()
        for day, origin in zip(self.days, self.origins):
            origin_ts = pd.Timestamp(origin)
            offset = origin_ts.value - anchor_ns
            if offset % step_value != 0:
                raise ContractError(f"origin {origin_ts} is not on the trace grid.")
            origin_slot = int(offset // step_value)
            last_context = origin_ts - step
            context_start = origin_ts - self.context_points * step
            label_end = origin_ts + (self.horizon_points - 1) * step
            label_days = sorted(
                {int((origin_slot + lead) // points) + 1 for lead in range(self.horizon_points)}
            )
            covered.update(label_days)
            slots = origin_slot + np.arange(self.horizon_points, dtype="int64")
            slot_windows.append(slots)
            records.append(
                {
                    "day": int(day),
                    "origin": origin_ts.isoformat(),
                    "origin_slot": origin_slot,
                    "context_start": context_start.isoformat(),
                    "last_context": last_context.isoformat(),
                    "required_history_points": int(self.context_points),
                    "label_start": origin_ts.isoformat(),
                    "label_end": label_end.isoformat(),
                    "horizon_points": int(self.horizon_points),
                    "label_days": label_days,
                }
            )

        overlap_pairs: list[dict[str, Any]] = []
        for first, second in zip(records, records[1:]):
            previous_end = pd.Timestamp(first["label_end"])
            current_start = pd.Timestamp(second["origin"])
            overlap_points = int((previous_end.value - current_start.value) // step_value + 1)
            if overlap_points > 0:
                overlap_pairs.append(
                    {
                        "day_a": int(first["day"]),
                        "day_b": int(second["day"]),
                        "overlapping_points": overlap_points,
                        "overlap_fraction_of_window": round(
                            overlap_points / float(self.horizon_points), 6
                        ),
                    }
                )

        n_label_points = int(len(self.days)) * int(self.horizon_points)
        unique_slots = int(np.unique(np.concatenate(slot_windows)).size)
        if overlap_pairs or n_label_points != unique_slots:
            overlap_note = (
                f"{len(self.days)} label windows share timestamps: {n_label_points} predicted "
                f"points cover only {unique_slots} unique slots. Errors from these windows are "
                "not independent observations and must not be aggregated as if they were "
                "independent samples."
            )
        else:
            overlap_note = None

        payload: dict[str, Any] = {
            **_state(self),
            "anchor": self.anchor.isoformat(),
            "n_origins": int(len(self.days)),
            "origins": records,
            "first_origin": records[0]["origin"],
            "last_origin": records[-1]["origin"],
            "label_start": self.label_start.isoformat(),
            "label_end": self.label_end.isoformat(),
            "covered_label_days": sorted(covered),
            "n_label_points": n_label_points,
            "n_unique_label_timestamps": unique_slots,
            "overlap": bool(self.overlap or bool(overlap_pairs) or n_label_points != unique_slots),
            "overlap_pairs": overlap_pairs,
            "overlap_note": overlap_note,
            "statistically_independent_samples": not bool(
                self.overlap or overlap_pairs or n_label_points != unique_slots
            ),
            "dropped": [dict(entry) for entry in self.dropped],
        }
        return payload


def plan_origins(
    spec: Any,
    grid: TraceGrid,
    protocol: Protocol,
    *,
    phase: str,
    horizon: Any | None = None,
    frame: pd.DataFrame | None = None,
) -> OriginPlan:
    """Resolve configured trace-day origins to concrete, coverage-checked origins.

    Geometry is checked against the trace grid (which spans anchor → last needed slot).  When
    ``frame`` is supplied, an origin is additionally dropped if its required context or label
    window falls outside the frame's outer time coverage (a leading/trailing gap).  Interior
    missing rows are *not* a reason to drop an origin: those positions become NaN / invalid in
    the batch so a baseline rejects the window explicitly.
    """
    if not isinstance(grid, TraceGrid):
        raise ContractError("grid must be a TraceGrid built from the dataset anchor.")
    if not isinstance(protocol, Protocol):
        raise ContractError("protocol must be a Protocol instance.")
    if protocol.points_per_day != grid.points_per_day:
        raise ValueError(
            f"protocol/grid mismatch: protocol points_per_day={protocol.points_per_day} but grid "
            f"declares {grid.points_per_day}."
        )
    if frequency_step(protocol.frequency) != grid.step:
        raise ValueError(
            f"protocol/grid mismatch: protocol frequency={protocol.frequency!r} but grid "
            f"declares {grid.frequency!r}."
        )
    anchor = _naive_timestamp(getattr(spec, "trace_anchor_timestamp", None), name="trace anchor")
    if anchor != grid.anchor:
        raise ValueError(
            f"Dataset anchor {anchor} does not match the trace grid anchor {grid.anchor}; "
            "the trace grid must be built from the dataset's own explicit anchor."
        )

    if horizon is None:
        if protocol.default_horizon is None:
            raise ValueError(
                f"Protocol {protocol.protocol_id!r} declares no default_horizon; pass a horizon."
            )
        horizon_name = protocol.horizon_name(protocol.default_horizon)
    else:
        horizon_name = protocol.horizon_name(horizon)
    horizon_points = protocol.horizon_points(horizon_name)
    context_points = protocol.context_points

    configured_days = protocol.origins_for(phase)
    if not configured_days:
        raise ValueError(
            f"Protocol {protocol.protocol_id!r} declares no origins for phase {phase!r}. "
            "Origins are configuration data: they are never inferred from the data or from the "
            "test range."
        )

    frame_first: pd.Timestamp | None = None
    frame_last: pd.Timestamp | None = None
    if frame is not None:
        index = _frame_index(frame)
        if not bool(grid.contains(index).all()):
            off_grid = index[~grid.contains(index)]
            raise ValueError(
                f"frame timestamps are not covered by the trace grid; e.g. {off_grid[0]} is not "
                f"an expected slot of anchor + k*{grid.frequency} within "
                f"[{grid.timestamps[0]}, {grid.timestamps[-1]}]. Build the grid from this "
                "frame/dataset coverage; frames are never renumbered onto a shorter grid."
            )
        frame_first, frame_last = index[0], index[-1]

    step = grid.step
    kept_days: list[int] = []
    kept_origins: list[pd.Timestamp] = []
    dropped: list[dict[str, Any]] = []
    last_slot = grid.n_slots - 1
    for day in configured_days:
        origin = grid.origin_of_day(day)
        origin_slot = grid.origin_index(day)
        context_start = origin - context_points * step
        label_end = origin + (horizon_points - 1) * step
        reason: str | None = None
        detail: dict[str, Any] = {
            "origin_slot": int(origin_slot),
            "context_points": int(context_points),
            "horizon_points": int(horizon_points),
            "context_start_slot": int(origin_slot - context_points),
            "label_end_slot": int(origin_slot + horizon_points - 1),
        }
        if origin_slot < 0:
            reason = f"origin slot {origin_slot} precedes the trace anchor."
        elif context_points > origin_slot:
            reason = (
                f"trace day {int(day)} needs {context_points} context points but its origin is at "
                f"slot {origin_slot}; the grid starts at slot 0, so only {origin_slot} context "
                "points exist."
            )
        elif origin_slot + horizon_points - 1 > last_slot:
            reason = (
                f"label window of {horizon_points} points would end at slot "
                f"{origin_slot + horizon_points - 1}, beyond the last covered slot {last_slot}."
            )
        elif frame_first is not None and context_start < frame_first:
            reason = (
                f"required context starts at {context_start.isoformat()} but the frame starts at "
                f"{frame_first.isoformat()}."
            )
        elif frame_last is not None and label_end > frame_last:
            reason = (
                f"label window ends at {label_end.isoformat()} but the frame ends at "
                f"{frame_last.isoformat()}."
            )

        if reason is not None:
            dropped.append(
                {
                    "day": int(day),
                    "origin": origin.isoformat(),
                    "reason": reason,
                    "detail": detail,
                }
            )
            continue
        kept_days.append(int(day))
        kept_origins.append(origin)

    if not kept_days:
        reasons = "; ".join(entry["reason"] for entry in dropped)
        raise InsufficientDataError(
            f"No feasible origin for protocol {protocol.protocol_id!r}, phase {phase!r}, horizon "
            f"{horizon_name!r} ({horizon_points} points) with context_points={context_points}. "
            f"Configured trace days {list(configured_days)} were all dropped: {reasons}. "
            "Origins are never silently shifted and the horizon is never shortened."
        )

    origins_index = pd.DatetimeIndex(kept_origins)
    label_starts = origins_index
    label_ends = origins_index + (horizon_points - 1) * step
    overlap = bool(
        len(origins_index) > 1
        and bool((origins_index[1:] < label_ends[:-1]).any())
    )

    return OriginPlan(
        phase=phase,
        protocol_id=protocol.protocol_id,
        horizon=horizon_name,
        horizon_points=horizon_points,
        days=tuple(kept_days),
        origins=origins_index,
        anchor=grid.anchor,
        context_points=context_points,
        label_start=label_starts[0],
        label_end=label_ends[-1],
        required_history_points=context_points,
        dropped=tuple(dropped),
        overlap=overlap,
        frequency=protocol.frequency,
        points_per_day=protocol.points_per_day,
    )


# ---------------------------------------------------------------------------
# forecast inputs / batch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Geometry:
    origin_ns: np.ndarray  # [B]
    context_ns: np.ndarray  # [B, L]
    label_ns: np.ndarray  # [B, H]
    step: pd.Timedelta


@dataclass(frozen=True)
class _Gathered:
    geometry: _Geometry
    frame: pd.DataFrame  # reindexed to the union of needed slots (NaN where absent)
    context_pos: np.ndarray  # [B, L]
    label_pos: np.ndarray  # [B, H]


def _geometry(plan: OriginPlan) -> _Geometry:
    origins = pd.DatetimeIndex(plan.origins)
    if len(origins) == 0:
        raise InsufficientDataError("OriginPlan contains no origins.")
    if len(origins) != len(plan.days):
        raise ContractError("OriginPlan days and origins have different lengths.")
    step = plan.step
    step_value = step.value
    origin_ns = origins.asi8.astype("int64")
    context_points = int(plan.context_points)
    horizon_points = int(plan.horizon_points)
    context_ns = origin_ns[:, None] - (
        np.arange(context_points, 0, -1, dtype="int64")[None, :] * step_value
    )
    label_ns = origin_ns[:, None] + (
        np.arange(horizon_points, dtype="int64")[None, :] * step_value
    )
    anchor_ns = _naive_timestamp(plan.anchor, name="anchor").value
    for name, block in (("context", context_ns), ("label", label_ns)):
        diff = block - anchor_ns
        if bool((diff < 0).any()):
            raise InsufficientDataError(
                f"Planned {name} window precedes the trace anchor; the plan is inconsistent."
            )
        if bool((diff % step_value != 0).any()):
            raise ContractError(
                f"Planned {name} window is not aligned to the trace grid frequency."
            )
    return _Geometry(origin_ns=origin_ns, context_ns=context_ns, label_ns=label_ns, step=step)


def _gather(frame: pd.DataFrame, plan: OriginPlan) -> _Gathered:
    geometry = _geometry(plan)
    index = _frame_index(frame)
    union_ns = np.unique(
        np.concatenate([geometry.context_ns.ravel(), geometry.label_ns.ravel()])
    )
    union = pd.DatetimeIndex(union_ns)
    sub = frame.set_axis(index, axis=0).reindex(union)
    context_pos = union.get_indexer(pd.DatetimeIndex(geometry.context_ns.ravel())).reshape(
        geometry.context_ns.shape
    )
    label_pos = union.get_indexer(pd.DatetimeIndex(geometry.label_ns.ravel())).reshape(
        geometry.label_ns.shape
    )
    return _Gathered(
        geometry=geometry, frame=sub, context_pos=context_pos, label_pos=label_pos
    )


def _build_inputs(
    spec: Any,
    frame: pd.DataFrame,
    plan: OriginPlan,
    *,
    include_calendar: bool,
) -> tuple[ForecastInputs, _Gathered]:
    target = spec.target
    if target is None:
        raise ContractError(
            f"Dataset {spec.dataset_id!r} declares no target; supervised forecast windows need "
            "an active_power label. Unlabelled sources build PretrainBatch instead."
        )
    gathered = _gather(frame, plan)
    geometry = gathered.geometry
    batch, context_points = geometry.context_ns.shape
    horizon_points = geometry.label_ns.shape[1]
    origin_ns = geometry.origin_ns
    step_ns = geometry.step.value

    power = to_kw(_column_values(gathered.frame, TARGET_COLUMN, what="target column"), target.unit)
    past_power_kw = power[gathered.context_pos]
    past_power_valid = np.isfinite(past_power_kw)

    feature_names = tuple(sorted(spec.features))
    past_covariates: np.ndarray | None = None
    past_covariate_mask: np.ndarray | None = None
    past_covariate_available_at: np.ndarray | None = None
    covariate_names: tuple[str, ...] = ()
    if feature_names:
        width = len(feature_names)
        past_covariates = np.full((batch, context_points, width), np.nan, dtype=float)
        past_covariate_mask = np.zeros((batch, context_points, width), dtype=bool)
        available_ns = np.empty((batch, context_points, width), dtype="int64")
        for column_index, name in enumerate(feature_names):
            feature_spec = spec.features[name]
            column = _resolve_feature_column(feature_spec, gathered.frame.columns)
            values = _column_values(gathered.frame, column)[gathered.context_pos]
            lag_ns = int(feature_spec.availability_lag.value)
            # The slot is labelled by its interval start, so the observation interval ends one
            # frequency step later; the declared reporting lag is added on top of that.
            feature_available = geometry.context_ns + step_ns + lag_ns
            usable = np.isfinite(values) & (feature_available <= origin_ns[:, None])
            past_covariates[:, :, column_index] = np.where(usable, values, np.nan)
            past_covariate_mask[:, :, column_index] = usable
            available_ns[:, :, column_index] = feature_available
        past_covariate_available_at = available_ns.astype("datetime64[ns]")
        covariate_names = feature_names

    if include_calendar:
        calendar_matrix, calendar_names = calendar_feature_matrix(
            pd.DatetimeIndex(geometry.label_ns.ravel()), points_per_day=plan.points_per_day
        )
        future_known = calendar_matrix.reshape(
            batch, horizon_points, len(calendar_names)
        ).astype(float)
        future_known_mask = np.ones(future_known.shape, dtype=bool)
        future_known_available_at = np.broadcast_to(
            origin_ns[:, None, None], future_known.shape
        ).copy().astype("datetime64[ns]")
        future_known_names: tuple[str, ...] = calendar_names
    else:
        future_known = None
        future_known_mask = None
        future_known_available_at = None
        future_known_names = ()

    declared_future_known = tuple(
        sorted(name for name, feature_spec in spec.features.items() if feature_spec.future_known)
    )
    metadata = {
        **_state(plan),
        "include_calendar": bool(include_calendar),
        "future_known_source": "calendar" if include_calendar else "none",
        "declared_but_not_delivered_future_known_features": list(declared_future_known),
        "availability_policy": "strict",
        "target_unit": target.unit,
    }

    inputs = ForecastInputs(
        past_power_kw=past_power_kw,
        past_power_valid=past_power_valid,
        past_timestamps=geometry.context_ns.astype("datetime64[ns]"),
        origin_timestamps=origin_ns.astype("datetime64[ns]"),
        target_timestamps=geometry.label_ns.astype("datetime64[ns]"),
        target_spec=target,
        domain_ids=tuple(spec.domain_id for _ in range(batch)),
        dataset_id=spec.dataset_id,
        past_covariates=past_covariates,
        past_covariate_mask=past_covariate_mask,
        past_covariate_available_at=past_covariate_available_at,
        covariate_names=covariate_names,
        future_known=future_known,
        future_known_mask=future_known_mask,
        future_known_available_at=future_known_available_at,
        future_known_names=future_known_names,
        metadata=metadata,
    )
    return inputs, gathered


def build_forecast_inputs(
    spec: Any,
    frame: pd.DataFrame,
    plan: OriginPlan,
    *,
    include_calendar: bool = True,
) -> ForecastInputs:
    """Prediction-time inputs (no labels) for every origin in ``plan``."""
    inputs, _ = _build_inputs(spec, frame, plan, include_calendar=include_calendar)
    return inputs


def build_forecast_batch(
    spec: Any,
    frame: pd.DataFrame,
    plan: OriginPlan,
    *,
    include_calendar: bool = True,
) -> ForecastBatch:
    """Supervised batch: inputs plus the next ``horizon_points`` labelled kW values."""
    inputs, gathered = _build_inputs(spec, frame, plan, include_calendar=include_calendar)
    target = spec.target
    geometry = gathered.geometry
    labels = to_kw(
        _column_values(gathered.frame, TARGET_COLUMN, what="target column"), target.unit
    )[gathered.label_pos]
    valid = np.isfinite(labels)
    # Labels are interval means over their slot: available once the interval ends, plus the
    # declared label reporting lag.  Downstream cutoff checks use these times, not row stamps.
    target_available_at = (
        geometry.label_ns + geometry.step.value + int(target.available_at_lag.value)
    ).astype("datetime64[ns]")
    return ForecastBatch(
        inputs=inputs,
        targets_kw=labels,
        target_valid_mask=valid,
        target_available_at=target_available_at,
    )


# ---------------------------------------------------------------------------
# unlabelled pretraining window
# ---------------------------------------------------------------------------


def _infer_time_delta(index: pd.DatetimeIndex, declared_frequency: str | None) -> tuple[int, bool]:
    """Native slot width in ns plus whether the series is regularly sampled."""
    if len(index) > 1:
        diffs = np.diff(index.asi8)
        regular = bool((diffs == diffs[0]).all())
        if regular:
            return int(diffs[0]), True
        return int(np.median(diffs)), False
    if not declared_frequency:
        raise ValueError(
            "A single-row pretraining frame has no observable time delta; declare "
            "DatasetSpec.native_frequency explicitly."
        )
    return int(frequency_step(declared_frequency).value), True


def build_pretrain_batch(
    spec: Any,
    frame: pd.DataFrame,
    *,
    context_points: int,
    stride_points: int | None = None,
    domain_id: str | None = None,
    max_windows: int | None = None,
) -> PretrainBatch:
    """Unlabelled operational-metric windows at the source's native frequency.

    No power label is produced or accepted.  Windows are ``context_points`` consecutive native
    observations; missing values stay NaN with ``observed_mask=False``.
    """
    if getattr(spec, "role", None) is DatasetRole.FINAL_HOLDOUT:
        raise TargetEligibilityError(
            f"Dataset {getattr(spec, 'dataset_id', '<unknown>')!r} is a final holdout; it must "
            "not be used as an unlabelled pretraining corpus."
        )
    context = _positive_int(context_points, "context_points")
    index = _frame_index(frame)
    n_rows = int(len(index))
    stride = context if stride_points is None else _positive_int(stride_points, "stride_points")
    starts = np.arange(0, n_rows - context + 1, stride, dtype="int64")
    if max_windows is not None:
        limit = _positive_int(max_windows, "max_windows")
        starts = starts[:limit]
    if starts.size == 0:
        raise InsufficientDataError(
            f"Pretraining window of {context} points cannot fit in a frame of {n_rows} rows "
            f"(stride_points={stride_points})."
        )

    feature_names = tuple(sorted(spec.features))
    if not feature_names:
        raise ContractError(
            "build_pretrain_batch needs at least one declared feature (F>=1); an empty feature "
            "set produces no pretraining signal."
        )

    frame_naive = frame.set_axis(index, axis=0)
    batch = int(starts.size)
    width = len(feature_names)
    positions = starts[:, None] + np.arange(context, dtype="int64")[None, :]
    timestamps_ns = index.asi8[positions]
    declared = getattr(spec, "native_frequency", None)
    delta_ns, regular = _infer_time_delta(index, declared)
    if delta_ns <= 0:
        raise ContractError("native time delta must be positive.")
    if declared and regular and frequency_step(declared).value != delta_ns:
        raise ContractError(
            f"Declared native_frequency={declared!r} ({frequency_step(declared)}) conflicts with "
            f"the observed regular sampling of {pd.Timedelta(delta_ns)}. Fix the dataset spec or "
            "resample explicitly; native pretraining sampling is never silently relabelled."
        )

    features = np.full((batch, context, width), np.nan, dtype=float)
    observed_mask = np.zeros((batch, context, width), dtype=bool)
    available_ns = np.empty((batch, context, width), dtype="int64")
    for column_index, name in enumerate(feature_names):
        feature_spec = spec.features[name]
        column = _resolve_feature_column(feature_spec, frame_naive.columns)
        values = _column_values(frame_naive, column)[positions]
        lag_ns = int(feature_spec.availability_lag.value)
        usable = np.isfinite(values)
        features[:, :, column_index] = np.where(usable, values, np.nan)
        observed_mask[:, :, column_index] = usable
        available_ns[:, :, column_index] = timestamps_ns + delta_ns + lag_ns

    native_frequency = declared if declared else str(pd.Timedelta(delta_ns))
    sampling_metadata: Mapping[str, Any] = {
        "native_frequency": native_frequency,
        "time_delta_seconds": float(delta_ns / 1_000_000_000),
        "regular_frequency": bool(regular),
        "context_points": int(context),
        "stride_points": int(stride),
        "window_mode": "sliding",
        "declared_native_frequency": declared,
        "calendar_features": False,
        "contains_power_labels": False,
    }
    resolved_domain = domain_id if domain_id else spec.domain_id
    return PretrainBatch(
        features=features,
        observed_mask=observed_mask,
        timestamps=timestamps_ns.astype("datetime64[ns]"),
        available_at=available_ns.astype("datetime64[ns]"),
        feature_names=feature_names,
        domain_ids=tuple(resolved_domain for _ in range(batch)),
        sampling_metadata=sampling_metadata,
        dataset_id=spec.dataset_id,
    )


# ---------------------------------------------------------------------------
# split audit for result archiving
# ---------------------------------------------------------------------------


def audit_split(spec: Any, frame: pd.DataFrame, plan: OriginPlan) -> dict[str, Any]:
    """Split audit payload: protocol geometry, per-origin times, and per-window availability."""
    payload = plan.audit()
    index = _frame_index(frame)
    geometry = _geometry(plan)
    step_value = geometry.step.value
    anchor_ns = _naive_timestamp(plan.anchor, name="anchor").value
    context_points = int(plan.context_points)
    horizon_points = int(plan.horizon_points)
    frame_naive = frame.set_axis(index, axis=0)
    target_column = TARGET_COLUMN if TARGET_COLUMN in frame_naive.columns else None

    offset = index.asi8 - anchor_ns
    on_grid = (offset >= 0) & (offset % step_value == 0)
    n_usable = 0
    for record in payload["origins"]:
        origin_ts = pd.Timestamp(record["origin"])
        context_ns = origin_ts.value - (
            np.arange(context_points, 0, -1, dtype="int64") * step_value
        )
        label_ns = origin_ts.value + (
            np.arange(horizon_points, dtype="int64") * step_value
        )
        context_present = index.get_indexer(pd.DatetimeIndex(context_ns)) >= 0
        label_present = index.get_indexer(pd.DatetimeIndex(label_ns)) >= 0
        if target_column is not None:
            label_values = frame_naive.reindex(pd.DatetimeIndex(label_ns))[target_column]
            target_present = label_values.notna().to_numpy()
        else:
            target_present = np.zeros(horizon_points, dtype=bool)
        missing_context = int((~context_present).sum())
        missing_labels = int((~label_present).sum())
        missing_targets = int((~target_present).sum())
        context_complete = missing_context == 0
        label_complete = missing_labels == 0
        usable = bool(context_complete and label_complete and missing_targets == 0)
        n_usable += int(usable)
        record.update(
            {
                "missing_context_points": missing_context,
                "missing_label_points": missing_labels,
                "target_missing_points": missing_targets,
                "context_complete": context_complete,
                "label_complete": label_complete,
                "usable_supervised_window": usable,
            }
        )

    payload["anchor_policy"] = "dataset_explicit"
    payload["dataset"] = {
        "dataset_id": getattr(spec, "dataset_id", None),
        "site_id": getattr(spec, "site_id", None),
        "domain_id": getattr(spec, "domain_id", None),
        "role": getattr(getattr(spec, "role", None), "value", None),
        "native_frequency": getattr(spec, "native_frequency", None),
        "trace_anchor_timestamp": getattr(spec, "trace_anchor_timestamp", None),
        "timezone": getattr(spec, "timezone", None),
        "timezone_verified": bool(getattr(spec, "timezone_verified", False)),
        "synthetic": bool(getattr(spec, "synthetic", False)),
        "audit_status": getattr(getattr(spec, "audit_status", None), "value", None),
        "source_version": getattr(spec, "source_version", None),
    }
    target = getattr(spec, "target", None)
    payload["target"] = target.to_dict() if target is not None else None
    payload["frame"] = {
        "rows": int(len(index)),
        "first_timestamp": index[0].isoformat(),
        "last_timestamp": index[-1].isoformat(),
        "index_unique": bool(index.is_unique),
        "index_monotonic": bool(index.is_monotonic_increasing),
        "on_trace_grid": bool(on_grid.all()),
        "off_grid_rows": int((~on_grid).sum()),
        "rows_before_anchor": int((offset < 0).sum()),
        "columns": [str(column) for column in frame.columns],
    }
    payload["n_usable_windows"] = int(n_usable)
    payload["all_windows_usable"] = bool(n_usable == len(payload["origins"]))
    return payload
