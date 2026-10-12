"""Deterministic demo data used solely to verify the experiment plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(days: int = 100, seed: int = 42) -> pd.DataFrame:
    if days < 60:
        raise ValueError("Use at least 60 demo days to support the 7-day split.")
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=96*days, freq="15min", tz="UTC", name="timestamp")
    t = np.arange(len(index))
    temperature = 15 + 8*np.sin(2*np.pi*t/96) + 2*np.sin(2*np.pi*t/672)
    it = 800 + 100*np.sin(2*np.pi*t/96) + 40*(index.dayofweek < 5) + rng.normal(0, 5, len(t))
    cooling = 20 + np.maximum(temperature-10, 0)*2
    return pd.DataFrame({
        "facility_power_kw": it+cooling+30,
        "it_power_kw": it,
        "cooling_kw": cooling,
        "hvac_kw": np.full(len(t), 10),
        "pump_kw": np.full(len(t), 15),
        "temperature_c": temperature,
        "humidity_pct": 50-15*np.sin(2*np.pi*t/96),
    }, index=index)
