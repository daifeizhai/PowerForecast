"""Quarter-hour bins are labeled by the start of the interval."""

from __future__ import annotations

import pandas as pd


def resample_frame(frame: pd.DataFrame, frequency: str = "15min") -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_monotonic_increasing:
        raise ValueError("Resampling requires a sorted DatetimeIndex.")
    return frame.resample(frequency, closed="left", label="left").mean()
