"""Deterministic synthetic facility data for engineering validation only.

What this module produces
-------------------------
``build_synthetic_dataset`` returns a :class:`SyntheticDataset` with

* one **supervised** target frame (role ``target_supervised``) carrying the
  canonical kW target column ``active_power_kw`` plus, when ``covariates=True``,
  the operational covariates declared in its ``DatasetSpec``; and
* exactly **two heterogeneous unlabelled** sources (role
  ``unlabeled_pretrain``, ``target=None``) that share the same UTC trace anchor
  but have different feature sets, different widths ``F`` and their own
  availability semantics.  Neither source contains a power column at all.

Honesty rules encoded here (task book §1.4, §4, §10.1):

* labels are ``label_origin=synthetic_simulated`` with ``synthetic=True``;
  they are never presented as a real measurement;
* the supervised frame and the unlabelled frames are separate objects with
  separate specs, so an unlabelled source can never be fed to a supervised fit;
* ``noise=0.0`` gives an exactly reproducible profile.  It is deterministic but
  *not* a flat series: the slot profile repeats per day type (weekday vs
  weekend) and carries a slow trend, so "same slot yesterday" is deliberately
  not an identity on this data;
* ``missing_days=(k,)`` leaves a contiguous NaN block in every column of trace
  day ``k``.  The NaNs are never filled here: missing values must stay missing
  so that masks / rejection paths can be tested.

Time convention: a timezone-aware UTC ``DatetimeIndex`` named ``"timestamp"``
on an exact 15-minute grid starting at ``anchor``.  No local-time or DST
arithmetic is performed: trace days are 96 absolute 15-minute slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.contracts import (
    TARGET_COLUMN,
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

#: Default canonical dataset id (also used by configs/datasets/synthetic.yaml).
SYNTHETIC_DATASET_ID: str = "synthetic_facility"
SYNTHETIC_SITE_ID: str = "synthetic_site"
SYNTHETIC_DOMAIN_ID: str = "synthetic_facility"

#: The two unlabelled source ids; source A has F=2, source B has F=3.
SYNTHETIC_SOURCE_IDS: tuple[str, ...] = (
    "synthetic_ops_cpu_mem",
    "synthetic_ops_gpu_tasks",
)

FREQUENCY: str = "15min"
POINTS_PER_DAY: int = 96
DEFAULT_DAYS: int = 31
DEFAULT_SEED: int = 7
DEFAULT_ANCHOR: str = "2024-01-01T00:00:00Z"
DEFAULT_NOISE: float = 1.0

#: Contiguous NaN block inserted per missing trace day: 8 slots starting at 10:00.
MISSING_BLOCK_START_SLOT: int = 40
MISSING_BLOCK_LENGTH: int = 8

#: Declared physical ranges.  Power values are kW; utilisation values are ratios
#: in [0, 1] and are therefore never converted to kW.
TARGET_RANGE_KW: tuple[float, float] = (200.0, 1500.0)

SOURCE_VERSION: str = "synthetic-v1"


# ---------------------------------------------------------------------------
# datacontainers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticSource:
    """One unlabelled synthetic source: a spec (target=None) and its frame."""

    spec: DatasetSpec
    frame: pd.DataFrame

    @property
    def dataset_id(self) -> str:
        return self.spec.dataset_id

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.frame.columns)


@dataclass(frozen=True)
class SyntheticDataset:
    """Supervised synthetic target plus its heterogeneous unlabelled sources."""

    spec: DatasetSpec
    frame: pd.DataFrame
    unlabeled_sources: tuple[SyntheticSource, ...]

    @property
    def dataset_id(self) -> str:
        return self.spec.dataset_id

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Canonical covariate names (``F=0`` -> empty tuple)."""
        return self.spec.feature_names

    @property
    def target(self) -> pd.Series:
        return self.frame[TARGET_COLUMN]

    def feature_frame(self) -> pd.DataFrame:
        """Covariate-only view (``F=0`` returns an empty frame on the same index)."""
        return self.frame.loc[:, list(self.feature_names)]


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------


def _feature(
    *,
    name: str,
    unit: str,
    physical_range: tuple[float, float],
    source: str,
    statistic: IntervalStatistic = IntervalStatistic.INTERVAL_MEAN,
    availability_lag: pd.Timedelta = pd.Timedelta(0),
    notes: str = "",
) -> FeatureSpec:
    return FeatureSpec(
        raw_name=name,
        canonical_name=name,
        unit=unit,
        statistic=statistic,
        physical_range=physical_range,
        native_frequency=FREQUENCY,
        availability_lag=availability_lag,
        future_known=False,
        missing_policy="mask",
        source=source,
        audit_status=AuditStatus.VERIFIED,
        notes=notes,
    )


def _target_feature_specs() -> dict[str, FeatureSpec]:
    """Covariates present in the supervised target frame (insertion order kept)."""
    notes_common = "synthetic operative covariate; simulated, engineering-validation only"
    return {
        "cpu_util": _feature(
            name="cpu_util",
            unit="ratio",
            physical_range=(0.0, 1.0),
            source=SYNTHETIC_DATASET_ID,
            notes=notes_common,
        ),
        "gpu_util": _feature(
            name="gpu_util",
            unit="ratio",
            physical_range=(0.0, 1.0),
            source=SYNTHETIC_DATASET_ID,
            notes=notes_common,
        ),
        "mem_util": _feature(
            name="mem_util",
            unit="ratio",
            physical_range=(0.0, 1.0),
            source=SYNTHETIC_DATASET_ID,
            notes=notes_common,
        ),
        "task_count": _feature(
            name="task_count",
            unit="count",
            physical_range=(0.0, 20000.0),
            source=SYNTHETIC_DATASET_ID,
            notes=notes_common,
        ),
        "cooling_kw": _feature(
            name="cooling_kw",
            unit="kW",
            physical_range=(0.0, 600.0),
            source=SYNTHETIC_DATASET_ID,
            notes=f"{notes_common}; power value stored in kW",
        ),
        "it_power_kw": _feature(
            name="it_power_kw",
            unit="kW",
            physical_range=(0.0, 1500.0),
            source=SYNTHETIC_DATASET_ID,
            notes=f"{notes_common}; power value stored in kW",
        ),
        "temperature_c": _feature(
            name="temperature_c",
            unit="degC",
            physical_range=(-20.0, 60.0),
            source=SYNTHETIC_DATASET_ID,
            notes=notes_common,
        ),
        "extra_metric": _feature(
            name="extra_metric",
            unit="ratio",
            physical_range=(0.0, 1.0),
            source=SYNTHETIC_DATASET_ID,
            notes=(
                "target-domain-only reserved field. It is kept with its value and mask for a "
                "future local branch (target_only_local_weight=1.0) and is never assigned a "
                "global fusion weight; no fusion algorithm is implemented in this round."
            ),
        ),
    }


def _source_a_features() -> dict[str, FeatureSpec]:
    common = "unlabelled synthetic operational metric; no power label, no kW"
    return {
        "cpu_util": _feature(
            name="cpu_util",
            unit="ratio",
            physical_range=(0.0, 1.0),
            source=SYNTHETIC_SOURCE_IDS[0],
            statistic=IntervalStatistic.INSTANTANEOUS,
            notes=f"{common}; instantaneous observation available at the sample timestamp",
        ),
        "mem_util": _feature(
            name="mem_util",
            unit="ratio",
            physical_range=(0.0, 1.0),
            source=SYNTHETIC_SOURCE_IDS[0],
            statistic=IntervalStatistic.INSTANTANEOUS,
            notes=f"{common}; instantaneous observation available at the sample timestamp",
        ),
    }


def _source_b_features() -> dict[str, FeatureSpec]:
    common = "unlabelled synthetic operational metric; no power label, no kW"
    # Construct with an explicit unit to avoid numpy 2.5's deprecated generic
    # timedelta unit path.
    lag = pd.Timedelta(15, "min")
    return {
        "gpu_util": _feature(
            name="gpu_util",
            unit="ratio",
            physical_range=(0.0, 1.0),
            source=SYNTHETIC_SOURCE_IDS[1],
            availability_lag=lag,
            notes=f"{common}; 15-minute interval mean, reported 15 minutes after interval end",
        ),
        "task_count": _feature(
            name="task_count",
            unit="count",
            physical_range=(0.0, 20000.0),
            source=SYNTHETIC_SOURCE_IDS[1],
            availability_lag=lag,
            notes=f"{common}; 15-minute interval mean task count",
        ),
        "queue_depth": _feature(
            name="queue_depth",
            unit="count",
            physical_range=(0.0, 2048.0),
            source=SYNTHETIC_SOURCE_IDS[1],
            availability_lag=lag,
            notes=f"{common}; source-only field, makes source B wider (F=3 vs F=2)",
        ),
    }


def _target_spec(*, anchor_text: str) -> TargetSpec:
    return TargetSpec(
        kind=TargetKind.ACTIVE_POWER,
        scope=TargetScope.FACILITY,
        unit="kW",
        label_origin=LabelOrigin.SYNTHETIC_SIMULATED,
        measurement_boundary=(
            "simulated facility aggregate active power; no physical meter or measurement "
            "boundary exists"
        ),
        sampling_semantics=(
            "synthetic mean active power over each 15-minute interval starting at its UTC "
            "timestamp"
        ),
        available_at_lag=pd.Timedelta(0),
        synthetic=True,
        notes=(
            "engineering-validation label only: never a real measured facility-kW result and "
            "never eligible for a research leaderboard"
        ),
    )


# ---------------------------------------------------------------------------
# generation helpers
# ---------------------------------------------------------------------------


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def _validate_days(days: Any) -> int:
    days = _require_int("days", days)
    if days < DEFAULT_DAYS:
        raise ValueError(
            f"days={days} is too short: at least {DEFAULT_DAYS} complete consecutive trace days "
            "are required so the configured demo origins (up to D30) and their D23-D31 label "
            "windows exist; no origin is silently dropped or shortened."
        )
    return days


def _validate_noise(noise: Any) -> float:
    try:
        value = float(noise)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"noise must be a nonnegative number, got {noise!r}.") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"noise must be a finite nonnegative number, got {noise!r}.")
    return value


def _validate_anchor(anchor: Any) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(anchor)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"anchor={anchor!r} is not a valid timestamp.") from exc
    if pd.isna(stamp):
        raise ContractError(f"anchor={anchor!r} is NaT.")
    if stamp.tzinfo is None:
        # Synthetic data is defined in UTC; a naive anchor is interpreted as UTC.
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    if stamp.minute % 15 != 0 or stamp.second != 0 or stamp.microsecond != 0 or stamp.nanosecond != 0:
        raise ContractError(
            f"anchor={anchor!r} is not on a 15-minute UTC boundary; the synthetic trace grid "
            "must start exactly on a slot boundary."
        )
    return stamp


def _validate_missing_days(missing_days: Any, days: int) -> tuple[int, ...]:
    if missing_days is None:
        return ()
    try:
        values = tuple(missing_days)
    except TypeError as exc:
        raise ValueError(f"missing_days must be a sequence of trace-day indices: {missing_days!r}.") from exc
    result: list[int] = []
    for value in values:
        day = _require_int("missing_days entry", value)
        if day < 0 or day >= days:
            raise ValueError(
                f"missing_days entry {day} is outside [0, {days - 1}] (0-based trace days)."
            )
        if day not in result:
            result.append(day)
    return tuple(result)


def _anchor_text(stamp: pd.Timestamp) -> str:
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _rng(seed: int, channel: int) -> np.random.Generator:
    """Independent deterministic stream per noise channel."""
    return np.random.default_rng([int(seed), int(channel)])


def _noise(seed: int, channel: int, size: int, scale: float, amount: float) -> np.ndarray:
    if amount == 0.0 or scale == 0.0:
        return np.zeros(size, dtype=float)
    return _rng(seed, channel).normal(0.0, scale, size) * amount


def _apply_missing_block(frame: pd.DataFrame, missing_days: tuple[int, ...]) -> None:
    """Contiguous NaN block in every column of selected trace days (in place)."""
    for day in missing_days:
        start = day * POINTS_PER_DAY + MISSING_BLOCK_START_SLOT
        stop = start + MISSING_BLOCK_LENGTH
        frame.iloc[start:stop, :] = np.nan


def _clip(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.clip(values, float(bounds[0]), float(bounds[1]))


# ---------------------------------------------------------------------------
# public builder
# ---------------------------------------------------------------------------


def build_synthetic_dataset(
    *,
    days: int = DEFAULT_DAYS,
    seed: int = DEFAULT_SEED,
    anchor: str = DEFAULT_ANCHOR,
    covariates: bool = True,
    noise: float = DEFAULT_NOISE,
    missing_days: tuple[int, ...] = (),
) -> SyntheticDataset:
    """Build the deterministic supervised synthetic dataset and its 2 sources.

    Parameters
    ----------
    days:
        Number of complete consecutive 96-slot (15-minute) trace days; ``>= 31``.
    seed:
        Seed of every random stream.  Identical arguments always yield identical
        frames and specs.
    anchor:
        Inclusive start of the UTC trace grid, on a 15-minute boundary.  The
        same anchor is declared for the target and for both unlabelled sources.
    covariates:
        ``True`` -> the target frame also carries the declared covariates.
        ``False`` -> only ``TARGET_COLUMN`` is present and ``F=0`` (callers can
        build covariate-free batches).
    noise:
        Multiplier of every noise component.  ``0.0`` disables noise completely
        and yields an exactly reproducible noiseless profile.
    missing_days:
        0-based trace-day indices that receive a contiguous NaN block in every
        column.  Those NaNs are never filled.

    Returns
    -------
    SyntheticDataset
        ``.spec`` (role ``target_supervised``), ``.frame`` (UTC index named
        ``"timestamp"``, containing ``TARGET_COLUMN`` and the covariate
        canonical columns) and ``.unlabeled_sources`` (exactly two
        ``unlabeled_pretrain`` sources with ``target=None``).
    """
    days = _validate_days(days)
    seed = _require_int("seed", seed)
    if seed < 0:
        raise ValueError(f"seed must be nonnegative, got {seed}.")
    covariates = bool(covariates)
    noise = _validate_noise(noise)
    anchor_ts = _validate_anchor(anchor)
    anchor_text = _anchor_text(anchor_ts)
    missing = _validate_missing_days(missing_days, days)

    periods = days * POINTS_PER_DAY
    index = pd.date_range(start=anchor_ts, periods=periods, freq=FREQUENCY, name="timestamp")
    if index.tz is None or str(index.tz) != "UTC":
        raise ContractError("synthetic trace grid must be a timezone-aware UTC index.")

    slot = (index.hour.to_numpy() * 60 + index.minute.to_numpy()) // 15
    trace_day = np.repeat(np.arange(days, dtype=np.int64), POINTS_PER_DAY)
    weekend = index.dayofweek.to_numpy() >= 5
    weekend_scale = np.where(weekend, 0.82, 1.0)

    # --- deterministic daily profile -------------------------------------------------
    morning = np.exp(-0.5 * ((slot - 40) / 12.0) ** 2)
    evening = np.exp(-0.5 * ((slot - 82) / 10.0) ** 2)
    shape = 0.12 + 0.65 * morning + 0.45 * evening
    trend = 9.0 * np.sin(2.0 * np.pi * trace_day / 9.0) + 1.8 * trace_day
    it_power = (760.0 + 340.0 * shape) * weekend_scale + trend

    temperature = (
        20.0
        + 5.5 * np.sin(2.0 * np.pi * (slot - 44) / POINTS_PER_DAY)
        + 1.5 * np.sin(2.0 * np.pi * trace_day / 7.0)
    )
    cooling = 38.0 + 9.5 * np.maximum(temperature - 16.0, 0.0) + 0.05 * it_power
    aux = 32.0 + 6.0 * np.sin(2.0 * np.pi * slot / POINTS_PER_DAY + 0.3)
    target = it_power + cooling + aux

    cpu_util = 0.22 + 0.55 * shape + 0.02 * np.sin(2.0 * np.pi * trace_day / 5.0)
    mem_util = 0.48 + 0.10 * np.sin(2.0 * np.pi * trace_day / 11.0) + 0.04 * shape
    gpu_util = 0.30 + 0.45 * np.exp(-0.5 * ((slot - 60) / 16.0) ** 2) + 0.06 * np.sin(
        2.0 * np.pi * trace_day / 5.0
    )
    task_count = 3800.0 + 2600.0 * shape + 500.0 * np.sin(2.0 * np.pi * trace_day / 6.0)
    queue_depth = 45.0 + 90.0 * gpu_util + 25.0 * np.sin(2.0 * np.pi * slot / POINTS_PER_DAY)
    extra_metric = (
        0.38
        + 0.14 * np.sin(2.0 * np.pi * (slot - 20) / POINTS_PER_DAY)
        + 0.03 * np.sin(2.0 * np.pi * trace_day / 8.0)
    )

    # --- independent deterministic noise streams -------------------------------------
    target = target + _noise(seed, 1, periods, 6.0, noise)
    it_power = it_power + _noise(seed, 2, periods, 4.0, noise)
    cooling = cooling + _noise(seed, 3, periods, 2.0, noise)
    temperature = temperature + _noise(seed, 4, periods, 0.15, noise)
    cpu_util = cpu_util + _noise(seed, 5, periods, 0.008, noise)
    mem_util = mem_util + _noise(seed, 6, periods, 0.006, noise)
    gpu_util = gpu_util + _noise(seed, 7, periods, 0.010, noise)
    task_count = task_count + _noise(seed, 8, periods, 45.0, noise)
    queue_depth = queue_depth + _noise(seed, 9, periods, 4.0, noise)
    extra_metric = extra_metric + _noise(seed, 10, periods, 0.004, noise)

    # --- physical plausibility --------------------------------------------------------
    target = _clip(target, TARGET_RANGE_KW)
    it_power = _clip(it_power, (0.0, 1500.0))
    cooling = _clip(cooling, (0.0, 600.0))
    temperature = _clip(temperature, (-20.0, 60.0))
    cpu_util = _clip(cpu_util, (0.0, 1.0))
    mem_util = _clip(mem_util, (0.0, 1.0))
    gpu_util = _clip(gpu_util, (0.0, 1.0))
    task_count = _clip(np.rint(task_count), (0.0, 20000.0))
    queue_depth = _clip(np.rint(queue_depth), (0.0, 2048.0))
    extra_metric = _clip(extra_metric, (0.0, 1.0))

    provenance: dict[str, Any] = {
        "synthetic": True,
        "generator": "src.data.synthetic.build_synthetic_dataset",
        "seed": seed,
        "days": days,
        "noise": noise,
        "anchor_timestamp": anchor_text,
        "frequency": FREQUENCY,
        "points_per_day": POINTS_PER_DAY,
        "missing_days": list(missing),
        "covariates": covariates,
    }

    if covariates:
        target_frame = pd.DataFrame(
            {
                TARGET_COLUMN: target,
                "cpu_util": cpu_util,
                "gpu_util": gpu_util,
                "mem_util": mem_util,
                "task_count": task_count,
                "cooling_kw": cooling,
                "it_power_kw": it_power,
                "temperature_c": temperature,
                "extra_metric": extra_metric,
            },
            index=index,
        )
        target_spec = DatasetSpec(
            dataset_id=SYNTHETIC_DATASET_ID,
            site_id=SYNTHETIC_SITE_ID,
            domain_id=SYNTHETIC_DOMAIN_ID,
            role=DatasetRole.TARGET_SUPERVISED,
            native_frequency=FREQUENCY,
            trace_anchor_timestamp=anchor_text,
            timezone="UTC",
            timezone_verified=True,
            interval_semantics=(
                "each row is the mean of the 96-slot UTC trace grid; slot k of a trace day is "
                "the 15-minute interval starting at anchor + (day*96 + k)*15min"
            ),
            availability_semantics=(
                "synthetic values are simulated as available at their own interval start "
                "(lag=0); the label is never a real observation and must not be read as one"
            ),
            features=_target_feature_specs(),
            target=_target_spec(anchor_text=anchor_text),
            source_version=SOURCE_VERSION,
            provenance=provenance,
            audit_status=AuditStatus.VERIFIED,
        )
    else:
        target_frame = pd.DataFrame({TARGET_COLUMN: target}, index=index)
        target_spec = DatasetSpec(
            dataset_id=SYNTHETIC_DATASET_ID,
            site_id=SYNTHETIC_SITE_ID,
            domain_id=SYNTHETIC_DOMAIN_ID,
            role=DatasetRole.TARGET_SUPERVISED,
            native_frequency=FREQUENCY,
            trace_anchor_timestamp=anchor_text,
            timezone="UTC",
            timezone_verified=True,
            interval_semantics=(
                "each row is the mean of the 96-slot UTC trace grid; slot k of a trace day is "
                "the 15-minute interval starting at anchor + (day*96 + k)*15min"
            ),
            availability_semantics=(
                "synthetic values are simulated as available at their own interval start "
                "(lag=0); the label is never a real observation and must not be read as one"
            ),
            features={},
            target=_target_spec(anchor_text=anchor_text),
            source_version=SOURCE_VERSION,
            provenance={**provenance, "covariates": False, "F": 0},
            audit_status=AuditStatus.VERIFIED,
        )

    _apply_missing_block(target_frame, missing)

    source_a_frame = pd.DataFrame(
        {"cpu_util": cpu_util, "mem_util": mem_util},
        index=index,
    )
    source_b_frame = pd.DataFrame(
        {"gpu_util": gpu_util, "task_count": task_count, "queue_depth": queue_depth},
        index=index,
    )
    _apply_missing_block(source_a_frame, missing)
    _apply_missing_block(source_b_frame, missing)

    common_provenance = {
        "synthetic": True,
        "generator": "src.data.synthetic.build_synthetic_dataset",
        "seed": seed,
        "days": days,
        "noise": noise,
        "anchor_timestamp": anchor_text,
        "frequency": FREQUENCY,
        "missing_days": list(missing),
    }
    source_a_spec = DatasetSpec(
        dataset_id=SYNTHETIC_SOURCE_IDS[0],
        site_id=SYNTHETIC_SITE_ID,
        domain_id=SYNTHETIC_SOURCE_IDS[0],
        role=DatasetRole.UNLABELED_PRETRAIN,
        native_frequency=FREQUENCY,
        trace_anchor_timestamp=anchor_text,
        timezone="UTC",
        timezone_verified=True,
        interval_semantics=(
            "instantaneous operational-metric samples on the shared 15-minute UTC trace grid"
        ),
        availability_semantics="available at the sample timestamp (availability_lag=0)",
        features=_source_a_features(),
        target=None,
        source_version=SOURCE_VERSION,
        provenance={**common_provenance, "F": 2, "feature_set": "cpu_mem"},
        audit_status=AuditStatus.VERIFIED,
    )
    source_b_spec = DatasetSpec(
        dataset_id=SYNTHETIC_SOURCE_IDS[1],
        site_id=SYNTHETIC_SITE_ID,
        domain_id=SYNTHETIC_SOURCE_IDS[1],
        role=DatasetRole.UNLABELED_PRETRAIN,
        native_frequency=FREQUENCY,
        trace_anchor_timestamp=anchor_text,
        timezone="UTC",
        timezone_verified=True,
        interval_semantics=(
            "15-minute interval means on the shared 15-minute UTC trace grid"
        ),
        availability_semantics=(
            "reported 15 minutes after the interval ends (availability_lag=15min)"
        ),
        features=_source_b_features(),
        target=None,
        source_version=SOURCE_VERSION,
        provenance={**common_provenance, "F": 3, "feature_set": "gpu_tasks"},
        audit_status=AuditStatus.VERIFIED,
    )

    sources = (
        SyntheticSource(spec=source_a_spec, frame=source_a_frame),
        SyntheticSource(spec=source_b_spec, frame=source_b_frame),
    )
    # Coherence guards: labels only in the supervised frame, never in a source.
    if TARGET_COLUMN in source_a_frame.columns or TARGET_COLUMN in source_b_frame.columns:
        raise ContractError("unlabelled synthetic sources must never contain the power target.")
    return SyntheticDataset(spec=target_spec, frame=target_frame, unlabeled_sources=sources)


def unlabeled_frames(dataset: SyntheticDataset) -> tuple[tuple[str, pd.DataFrame], ...]:
    """``(dataset_id, frame)`` pairs for the unlabelled sources, in declaration order."""
    if not isinstance(dataset, SyntheticDataset):
        raise ContractError("unlabeled_frames expects a SyntheticDataset.")
    return tuple((source.dataset_id, source.frame) for source in dataset.unlabeled_sources)


def build_synthetic_dataset_from_config(config: Mapping[str, Any]) -> SyntheticDataset:
    """Build from the ``configs/datasets/synthetic.yaml`` mapping.

    The YAML names the anchor ``anchor_timestamp`` while the builder keyword is
    ``anchor``; this helper is the single place performing that mapping so every
    script consumes the identical semantics.
    """
    if not isinstance(config, Mapping):
        raise ContractError("synthetic dataset config must be a mapping.")
    dataset_id = config.get("dataset_id", SYNTHETIC_DATASET_ID)
    if dataset_id != SYNTHETIC_DATASET_ID:
        raise ContractError(
            f"config declares dataset_id={dataset_id!r} but this generator only builds "
            f"{SYNTHETIC_DATASET_ID!r}; add a real adapter instead of renaming a synthetic set."
        )
    frequency = config.get("frequency", FREQUENCY)
    if frequency != FREQUENCY:
        raise ContractError(
            f"config frequency={frequency!r} is unsupported: the synthetic generator builds an "
            f"exact {FREQUENCY} UTC grid only."
        )
    points_per_day = int(config.get("points_per_day", POINTS_PER_DAY))
    if points_per_day != POINTS_PER_DAY:
        raise ContractError(
            f"config points_per_day={points_per_day} is unsupported: expected {POINTS_PER_DAY}."
        )
    return build_synthetic_dataset(
        days=int(config.get("days", DEFAULT_DAYS)),
        seed=int(config.get("seed", DEFAULT_SEED)),
        anchor=str(config.get("anchor_timestamp", DEFAULT_ANCHOR)),
        covariates=bool(config.get("covariates", True)),
        noise=float(config.get("noise", DEFAULT_NOISE)),
        missing_days=tuple(config.get("missing_days") or ()),
    )


__all__ = [
    "DEFAULT_ANCHOR",
    "DEFAULT_DAYS",
    "DEFAULT_NOISE",
    "DEFAULT_SEED",
    "FREQUENCY",
    "POINTS_PER_DAY",
    "SYNTHETIC_DATASET_ID",
    "SYNTHETIC_DOMAIN_ID",
    "SYNTHETIC_SITE_ID",
    "SYNTHETIC_SOURCE_IDS",
    "SyntheticDataset",
    "SyntheticSource",
    "build_synthetic_dataset",
    "build_synthetic_dataset_from_config",
    "unlabeled_frames",
]
