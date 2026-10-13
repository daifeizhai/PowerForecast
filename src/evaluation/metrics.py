"""Percentage errors are reported in percent, with explicit zero policies."""

from __future__ import annotations

import numpy as np


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray, mape_threshold: float = 1e-8) -> dict:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metrics require nonempty actual and predicted arrays with identical shapes.")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Metrics require finite labels and predictions.")
    error = actual - predicted
    absolute = np.abs(error)
    denominator = float(np.abs(actual).sum())
    if not np.isfinite(mape_threshold) or mape_threshold < 0:
        raise ValueError("MAPE threshold must be finite and nonnegative.")
    nonzero = np.abs(actual) > mape_threshold
    return {
        "WAPE": float(100 * absolute.sum() / denominator) if denominator > 1e-8 else None,
        "MAPE": float(100 * np.mean(absolute[nonzero] / np.abs(actual[nonzero]))) if nonzero.any() else None,
        "MAE": float(absolute.mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "n": int(actual.size),
        "mape_n": int(nonzero.sum()),
        "mape_excluded": int(actual.size - nonzero.sum()),
    }
