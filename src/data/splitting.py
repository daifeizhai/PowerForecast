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

    def __len__(self) -> int:
        return len(self.history)

    def take(self, indices: slice | np.ndarray) -> "ForecastInputs":
        return ForecastInputs(self.history[indices], self.future_known[indices], self.origins[indices], self.target_times[indices])


@dataclass(frozen=True)
class WindowBatch:
    inputs: ForecastInputs
    targets: np.ndarray  # [sample, horizon], never supplied to predict()
    skipped_windows: int = 0

    def __len__(self) -> int:
        return len(self.targets)


def chronological_splits(length: int, train_fraction: float, validation_fraction: float) -> dict[str, tuple[int, int]]:
    if length <= 0 or not (0 < train_fraction < 1 and 0 < validation_fraction < 1 - train_fraction):
        raise ValueError("Split fractions must be positive and leave a nonempty test split.")
    train_end = int(length * train_fraction)
    validation_end = int(length * (train_fraction + validation_fraction))
    return {"train": (0, train_end), "validation": (train_end, validation_end), "test": (validation_end, length)}


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
