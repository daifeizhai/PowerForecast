"""Chronological boundaries and rolling windows with strict label separation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastInputs:
    history: np.ndarray  # [sample, context, feature], target is feature zero
    future_known: np.ndarray  # [sample, horizon, calendar_feature]
    origins: pd.DatetimeIndex  # first interval to predict
    target_times: np.ndarray
    domain_ids: tuple[str, ...] | None = None  # optional side metadata, never a numeric feature

    def __len__(self) -> int:
        return len(self.history)

    def take(self, indices: slice | np.ndarray) -> "ForecastInputs":
        domains = None if self.domain_ids is None else tuple(np.asarray(self.domain_ids)[indices].tolist())
        return ForecastInputs(self.history[indices], self.future_known[indices], self.origins[indices], self.target_times[indices], domains)


@dataclass(frozen=True)
class WindowBatch:
    inputs: ForecastInputs
    targets: np.ndarray  # [sample, horizon], never supplied to predict()
    skipped_windows: int = 0

    def __len__(self) -> int:
        return len(self.targets)


def concatenate_domain_batches(batches: dict[str, WindowBatch]) -> WindowBatch:
    """Combine independent series windows without treating coincident times as duplicates."""
    if not batches or any(not len(b) for b in batches.values()):
        raise ValueError("Domain batches must be nonempty.")
    values = list(batches.values())
    arrays = [np.concatenate([getattr(b.inputs, key) for b in values])
              for key in ("history", "future_known", "target_times")]
    labels = np.concatenate([b.targets for b in values])
    for array in [*arrays, labels]:
        array.setflags(write=False)
    origins = values[0].inputs.origins.append([b.inputs.origins for b in values[1:]])
    domains = tuple(domain for domain, batch in batches.items() for _ in range(len(batch)))
    inputs = ForecastInputs(arrays[0], arrays[1], origins, arrays[2], domains)
    return WindowBatch(inputs, labels, sum(b.skipped_windows for b in values))


def chronological_splits(length: int, train_fraction: float, validation_fraction: float) -> dict[str, tuple[int, int]]:
    if length <= 0 or not (0 < train_fraction < 1 and 0 < validation_fraction < 1 - train_fraction):
        raise ValueError("Split fractions must be positive and leave a nonempty test split.")
    train_end = int(length * train_fraction)
    validation_end = int(length * (train_fraction + validation_fraction))
    return {"train": (0, train_end), "validation": (train_end, validation_end), "test": (validation_end, length)}


def complete_days(frame: pd.DataFrame, timezone: str, target: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Trim grid/target coverage edges; keep all interior missing intervals."""
    local = frame.index.tz_convert(timezone)
    counts = pd.Series(1, index=local).resample("D").sum()
    full = counts[counts == 96]
    if full.empty:
        raise ValueError("No complete 96-point days in the requested range.")
    grid_start, grid_stop = full.index[0], full.index[-1] + pd.DateOffset(days=1)
    grid_rows = int(((local >= grid_start) & (local < grid_stop)).sum())
    if target is not None:
        available = pd.Series(np.isfinite(frame[target].to_numpy()), index=local).resample("D").sum()
        full = full.loc[available.reindex(full.index).to_numpy() == 96]
        if full.empty:
            raise ValueError("No complete 96-point target days; check component coverage and missing labels.")
    start, stop = full.index[0], full.index[-1] + pd.DateOffset(days=1)
    keep = (local >= start) & (local < stop)
    result = frame.loc[keep]
    if len(result) % 96 or not (counts.loc[start:full.index[-1]] == 96).all():
        raise ValueError("Local days must have 96 intervals; verify timezone/DST before this protocol.")
    return result, {"dropped_edge_rows": len(frame)-len(result), "grid_edge_rows_dropped": len(frame)-grid_rows,
                    "target_edge_rows_dropped": grid_rows-len(result), "target_coverage_checked": target,
                    "days": len(result)//96,
                    "start": str(start), "end_exclusive": str(stop)}


def adaptive_day_splits(days: int, train_fraction: float = .6, validation_fraction: float = .2,
                        minimum_validation_days: int = 8, minimum_test_days: int = 9) -> dict[str, tuple[int, int]]:
    """Day-aligned 60/20/20, reserving validation/test minima for short series."""
    chronological_splits(days, train_fraction, validation_fraction)
    train = int(days * train_fraction)
    validation = max(minimum_validation_days, int(days * validation_fraction))
    test = max(minimum_test_days, days - train - validation)
    train = days - validation - test
    if train < 14:
        raise ValueError("Insufficient complete days: need >=14 train, >=8 validation and >=9 test days.")
    return {"train": (0, train*96), "validation": (train*96, (train+validation)*96),
            "test": ((train+validation)*96, days*96)}


def build_windows(history: pd.DataFrame, known: pd.DataFrame, target: pd.Series, bounds: tuple[int, int], context: int, horizon: int, stride: int) -> WindowBatch:
    if min(context, horizon, stride) <= 0:
        raise ValueError("Context, horizon, and stride must be positive.")
    if not history.index.equals(known.index) or not history.index.equals(target.index):
        raise ValueError("History, calendar, and labels must have identical timestamps.")
    start, end = bounds
    if not 0 <= start < end <= len(history):
        raise ValueError(f"Invalid split bounds: {bounds}")
    hv = history.to_numpy(dtype=np.float32)
    kv = known.to_numpy(dtype=np.float32)
    yv = target.to_numpy(dtype=np.float32)
    origins, contexts, calendars, labels, times = [], [], [], [], []
    skipped = 0
    for origin in range(max(start, context), end - horizon + 1, stride):
        past, future, y = hv[origin-context:origin], kv[origin:origin+horizon], yv[origin:origin+horizon]
        if not (np.isfinite(past).all() and np.isfinite(future).all() and np.isfinite(y).all()):
            skipped += 1
            continue
        contexts.append(past)
        calendars.append(future)
        labels.append(y)
        origins.append(history.index[origin])
        times.append(history.index[origin:origin+horizon].to_numpy())
    if not labels:
        raise ValueError(f"No valid windows for bounds={bounds}, context={context}, horizon={horizon}; skipped={skipped}. Check data length and missing values.")
    inputs = ForecastInputs(np.stack(contexts), np.stack(calendars), pd.DatetimeIndex(origins), np.stack(times))
    return WindowBatch(inputs, np.stack(labels), skipped)
