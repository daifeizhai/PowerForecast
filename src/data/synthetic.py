"""Deterministic demo data used solely to verify the experiment plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(days: int = 100, seed: int = 42) -> pd.DataFrame:
    if days < 31:
        raise ValueError("Use at least 31 demo days for the fixed SAPF development/test schedule.")
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


def make_demo_domains(days: int = 31, seed: int = 42):
    """Synthetic power-utilization domains, never research performance evidence."""
    from src.data.domains import DomainMetadata
    from src.data.google_panel import DomainPanel
    if days < 31:
        raise ValueError("Use at least 31 days for the SAPF 24h/3d schedule.")
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=days*96, freq="15min", tz="UTC", name="timestamp")
    points = np.arange(len(index))
    series, metadata, coverage = {}, {}, {}
    for cell in range(3):
        for number in range(4):
            domain = f"demo_cell{cell}_pdu{number}"
            level = .35+.1*cell+.03*number
            values = level+.03*np.sin(2*np.pi*points/96+.1*cell)+.01*np.cos(2*np.pi*points/672)
            values += .003*rng.normal(size=len(points))
            series[domain] = pd.DataFrame({"measured_power_util": values}, index=index)
            metadata[domain] = DomainMetadata(dataset_id="synthetic_demo", site_id=f"demo_cell{cell}", domain_id=domain,
                power_domain_id=f"pdu{number}", target_kind="power_utilization", target_scope="power_domain",
                target_unit="utilization_fraction", label_origin="assumed_scenario", timezone="UTC", timezone_verified=False,
                raw_frequency="15min", sampling_semantics="sample_mean",
                interval_definition="start; [timestamp, timestamp + 15 minutes)", coverage="Synthetic complete grid.",
                source_version=f"synthetic seed {seed}", feature_availability={"target": "history_only", "trace_calendar": "future_known"})
            coverage[domain] = pd.Series(1, index=index)
    return DomainPanel(series, metadata, coverage, {}, index[0], {d: {} for d in series})
