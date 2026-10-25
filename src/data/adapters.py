"""Canonical field names, explicit units, and UTC timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def adapt_frame(frame: pd.DataFrame, source: dict, timezone: str) -> pd.DataFrame:
    identity_columns = source.get("domain_columns", ["dataset_id", "site_id", "domain_id", "power_domain_id", "rack_id", "node_id", "cell", "pdu"])
    for identity in identity_columns:
        if identity in frame and frame[identity].nunique(dropna=False) > 1:
            raise ValueError(f"Multiple values of {identity}; partition by physical domain before timestamp aggregation.")
    mapping = source["columns"]  # canonical name -> raw field name
    missing = set(mapping.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing configured columns: {sorted(missing)}")
    timestamp = source.get("timestamp")
    if timestamp:
        if timestamp not in frame.columns:
            raise ValueError(f"Missing timestamp column: {timestamp}")
        times = pd.DatetimeIndex(pd.to_datetime(frame[timestamp], errors="raise"))
    else:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Set source.timestamp or provide a DatetimeIndex.")
        times = frame.index
    if times.hasnans:
        raise ValueError("Timestamps contain missing values.")
    if times.tz is None:
        # Fail on ambiguous local timestamps instead of silently guessing DST.
        times = times.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    # Normalize parquet microseconds to nanoseconds before joins. Pandas 2.x
    # union of differently ranged, regular microsecond indexes can lose bins.
    times = times.tz_convert("UTC").as_unit("ns")
    result = frame[list(mapping.values())].copy()
    result.columns = list(mapping)
    result.index = times.rename("timestamp")
    for column in result:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] *= float(source.get("scales", {}).get(column, 1.0))
        result[column] += float(source.get("offsets", {}).get(column, 0.0))
    result = result.replace([np.inf, -np.inf], np.nan).sort_index()
    if result.index.has_duplicates:
        result = result.groupby(level=0).mean()
    for boundary, operator in [("minimum_timestamp", "lower"), ("maximum_timestamp", "upper")]:
        if source.get(boundary):
            threshold = pd.Timestamp(source[boundary])
            threshold = threshold.tz_localize("UTC") if threshold.tz is None else threshold.tz_convert("UTC")
            result = result.loc[result.index >= threshold] if operator == "lower" else result.loc[result.index <= threshold]
    if result.empty:
        raise ValueError("No observations remain after timestamp filtering.")
    return result
