"""Causal load signatures computed from one observed history window."""
from __future__ import annotations

import numpy as np

SIGNATURE_NAMES = (
    "log_level", "coefficient_of_variation", "daily_correlation", "weekly_correlation",
    "relative_trend_per_day", "relative_mean_ramp", "relative_ramp_std",
    "relative_daily_amplitude",
)


def load_signature(history: np.ndarray, points_per_day: int = 96) -> np.ndarray:
    values = np.asarray(history, dtype=float).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Load signatures need a complete finite observed history.")
    level = float(np.mean(values))
    denominator = max(abs(level), 1e-8)
    def correlation(lag):
        if len(values) <= lag + 2:
            return 0.0
        a, b = values[:-lag], values[lag:]
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else 0.0
    time = np.arange(len(values), dtype=float)
    time -= time.mean()
    slope = np.dot(time, values-values.mean())/np.dot(time, time)
    ramps = np.diff(values)
    whole_days = len(values)//points_per_day
    profile = values[-whole_days*points_per_day:].reshape(whole_days, points_per_day).mean(0) if whole_days else values
    return np.array([
        np.log(denominator), np.std(values)/denominator, correlation(points_per_day), correlation(points_per_day*7),
        slope*points_per_day/denominator, np.mean(np.abs(ramps))/denominator, np.std(ramps)/denominator,
        np.std(profile)/denominator,
    ], dtype=float)


def signature_records(inputs) -> list[dict]:
    domains = inputs.domain_ids or ("unspecified",)*len(inputs)
    return [
        {"domain_id": domain, "origin": str(origin), "mean_level": float(np.mean(history[:, 0])),
         "history_std": float(np.std(history[:, 0])), **dict(zip(SIGNATURE_NAMES, load_signature(history[:, 0], inputs.points_per_day)))}
        for domain, origin, history in zip(domains, inputs.origins, inputs.history)
    ]
