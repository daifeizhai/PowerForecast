"""Causal, origin-safe feature construction.

Only information observable at (or before) the forecast origin may enter a
model:

* **Calendar features** are derived from the timestamps themselves and are
  therefore legitimately known in advance (the only default known-future input).
* **Rolling statistics** use the *past* side of every position only; no centred
  window and no future sample can enter a feature.
* There is deliberately no API that takes a whole series and returns a global
  aggregate: such statistics would silently describe the evaluation period.

Local per-window standardisation is exposed as an explicitly documented research
helper whose statistics come from the window's own past values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import LeakageError, as_datetime64

#: Columns produced by :func:`calendar_features`; the only default known-future set.
CALENDAR_FEATURE_NAMES: tuple[str, ...] = (
    "slot_sin",
    "slot_cos",
    "dow_sin",
    "dow_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
    "hour",
)

MINUTES_PER_DAY = 24 * 60


def _utc_index(timestamps: Any) -> pd.DatetimeIndex:
    """UTC-naive DatetimeIndex; trace arithmetic is absolute, not local."""
    return pd.DatetimeIndex(as_datetime64(timestamps))


def slot_of_day(timestamps: Any, *, points_per_day: int = 96) -> np.ndarray:
    """0-based slot number inside the UTC trace day (0..95 with 15-minute data).

    Trace days are 96 consecutive slots from the common anchor, so the slot is
    computed on the absolute UTC grid rather than on local clock days.
    """
    if points_per_day < 1 or MINUTES_PER_DAY % points_per_day != 0:
        raise ValueError(f"points_per_day={points_per_day} does not divide a day into equal slots.")
    index = _utc_index(timestamps)
    minutes = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    slot_width = MINUTES_PER_DAY // points_per_day
    return (minutes // slot_width).astype(np.int64)


def calendar_features(
    timestamps: Any,
    *,
    points_per_day: int = 96,
) -> pd.DataFrame:
    """Known-future calendar frame indexed by UTC timestamp.

    Built from the timestamp axis only; never from measured values, so the same
    features are available for the whole horizon at forecast time.
    """
    index = _utc_index(timestamps)
    if len(index) == 0:
        raise ValueError("calendar_features needs at least one timestamp.")
    slots = slot_of_day(index, points_per_day=points_per_day)
    minutes = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    weekday = index.dayofweek.to_numpy().astype(float)
    day = index.dayofyear.to_numpy().astype(float)
    hour_of_day = minutes / MINUTES_PER_DAY
    return pd.DataFrame(
        {
            "slot_sin": np.sin(2.0 * np.pi * slots / points_per_day),
            "slot_cos": np.cos(2.0 * np.pi * slots / points_per_day),
            "dow_sin": np.sin(2.0 * np.pi * weekday / 7.0),
            "dow_cos": np.cos(2.0 * np.pi * weekday / 7.0),
            "day_sin": np.sin(2.0 * np.pi * day / 365.25),
            "day_cos": np.cos(2.0 * np.pi * day / 365.25),
            "is_weekend": (weekday >= 5).astype(float),
            "hour": hour_of_day,
        },
        index=index,
    ).astype(float)


def calendar_feature_matrix(
    timestamps: Any,
    *,
    points_per_day: int = 96,
    columns: tuple[str, ...] = CALENDAR_FEATURE_NAMES,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """``[N, K]`` calendar matrix plus the matching column names."""
    frame = calendar_features(timestamps, points_per_day=points_per_day)
    missing = [name for name in columns if name not in frame]
    if missing:
        raise ValueError(f"Unknown calendar feature columns: {missing}")
    return frame.loc[:, list(columns)].to_numpy(dtype=float), tuple(columns)


def rolling_past_statistics(
    values: Any,
    *,
    window: int,
    statistic: str = "mean",
    min_periods: int | None = None,
) -> np.ndarray:
    """Rolling statistic that only ever looks backwards.

    Position ``t`` uses ``values[t-window+1 : t+1]``.  ``shift`` and centred
    windows are deliberately not offered, so future samples cannot leak in.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2):
        raise ValueError("values must be 1-D or 2-D.")
    if window < 1:
        raise ValueError("window must be positive.")
    minimum = window if min_periods is None else int(min_periods)
    if not 1 <= minimum <= window:
        raise ValueError("min_periods must satisfy 1 <= min_periods <= window.")
    if statistic not in {"mean", "std", "min", "max", "median"}:
        raise ValueError(f"Unsupported statistic {statistic!r}.")
    if array.ndim == 1:
        rolled = pd.Series(array).rolling(window=window, min_periods=minimum)
        return getattr(rolled, statistic)().to_numpy(dtype=float)
    rolled = pd.DataFrame(array).rolling(window=window, min_periods=minimum)
    return getattr(rolled, statistic)().to_numpy(dtype=float)


def local_standardization(
    values_kw: Any,
    *,
    axis: int = -1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-window z-score using that window's own values only.

    Explicitly a research helper: each window is normalised independently, so it
    cannot import statistics from the future of another window.  Diagnostics
    must be saved by any run that uses it.
    """
    array = np.asarray(values_kw, dtype=float)
    mean = np.nanmean(array, axis=axis, keepdims=True)
    std = np.nanstd(array, axis=axis, keepdims=True)
    usable = np.isfinite(std) & (std > 1e-12)
    safe_std = np.where(usable, std, 1.0)
    normalized = (array - mean) / safe_std
    return normalized, {
        "method": "per_window_zscore",
        "axis": int(axis),
        "constant_windows": int(np.sum(~usable)),
        "uses_future_values": False,
    }


def require_past_only_mask(n_points: int, origin_index: int) -> np.ndarray:
    """Boolean mask selecting positions strictly before ``origin_index``."""
    if origin_index < 0 or origin_index > n_points:
        raise LeakageError(
            f"origin_index={origin_index} is outside a series of length {n_points}."
        )
    mask = np.zeros(int(n_points), dtype=bool)
    mask[:origin_index] = True
    return mask
