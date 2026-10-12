"""History-only physical inputs and genuinely known future calendar inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calendar_features(index: pd.DatetimeIndex, timezone: str = "UTC") -> pd.DataFrame:
    local = index.tz_convert(timezone) if index.tz is not None else index.tz_localize(timezone)
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    weekday = local.dayofweek.to_numpy()
    day = local.dayofyear.to_numpy()
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "week_sin": np.sin(2 * np.pi * (weekday + hour / 24) / 7),
        "week_cos": np.cos(2 * np.pi * (weekday + hour / 24) / 7),
        "year_sin": np.sin(2 * np.pi * day / 365.25),
        "year_cos": np.cos(2 * np.pi * day / 365.25),
        "is_weekend": (weekday >= 5).astype(float),
    }, index=index, dtype=np.float32)


def make_features(frame: pd.DataFrame, target: str, covariates: list[str], timezone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(dict.fromkeys([target, *covariates]))
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Missing history features: {sorted(missing)}")
    known = calendar_features(frame.index, timezone)
    # Observed weather and IT loads are never taken from the forecast interval.
    return pd.concat([frame[columns].astype(np.float32), known], axis=1), known
