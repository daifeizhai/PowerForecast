"""Causal cleaning: preserve missing labels and never backfill history."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_frame(frame: pd.DataFrame, target: str, config: dict) -> pd.DataFrame:
    if target not in frame:
        raise ValueError(f"Target column not found: {target}")
    result = frame.replace([np.inf, -np.inf], np.nan).copy()
    for column, bounds in config.get("bounds", {}).items():
        if column not in result:
            continue
        low, high = bounds
        invalid = pd.Series(False, index=result.index)
        if low is not None:
            invalid |= result[column] < low
        if high is not None:
            invalid |= result[column] > high
        result.loc[invalid, column] = np.nan
    limit = int(config.get("ffill_limit", 4))
    if limit < 0:
        raise ValueError("ffill_limit must be nonnegative.")
    covariates = [column for column in result if column != target]
    if covariates and limit:
        result[covariates] = result[covariates].ffill(limit=limit)
    return result


def quality_report(frame: pd.DataFrame, frequency: str = "15min") -> dict:
    if frame.empty:
        raise ValueError("Dataset is empty.")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq=frequency)
    return {
        "rows": len(frame),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "timezone": str(frame.index.tz),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "missing_intervals": len(expected.difference(frame.index)),
        "missing_fraction": {column: float(frame[column].isna().mean()) for column in frame},
    }
