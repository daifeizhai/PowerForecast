"""Forecast error metrics with an explicit, reproducible MAPE zero policy.

Definitions (all errors are in the declared label unit, here kW)
-----------------------------------------------------------------
* ``MAPE`` -- mean of ``|y - y_hat| / |y|`` over the points kept by the
  near-zero policy, reported in **percent**.  The policy is scale-free::

      threshold_kW = mape_relative_threshold * mean(|y|)

  A point contributes iff ``|y| > threshold_kW``.  With the default
  ``mape_relative_threshold=0.0`` only exact-zero truth points are excluded.
  Near-zero truth is therefore **never silently replaced by 1** and the
  threshold actually used is returned as ``mape_threshold`` together with
  ``mape_n`` (kept points) and ``mape_excluded`` (excluded points).
  When no point survives the policy, ``MAPE`` is ``None`` and
  ``mape_defined is False``.
* ``WAPE`` -- ``100 * sum(|y - y_hat|) / sum(|y|)`` in **percent**, computed
  over *all* points regardless of the MAPE policy.  It is undefined
  (``None``, ``wape_defined is False``) when the denominator is effectively
  zero (``sum(|y|) <= WAPE_DENOMINATOR_EPS``).
* ``MAE`` / ``RMSE`` -- in kW, over all points.

``n`` is the total number of scored points.  Shapes must match; empty or
non-finite arrays are rejected instead of silently producing NaN metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["WAPE_DENOMINATOR_EPS", "forecast_metrics"]

#: ``sum(|y|)`` at or below this value means WAPE has no meaningful denominator.
WAPE_DENOMINATOR_EPS = 1e-12

_METRIC_KEYS: tuple[str, ...] = (
    "MAPE",
    "WAPE",
    "MAE",
    "RMSE",
    "n",
    "mape_threshold",
    "mape_n",
    "mape_excluded",
    "wape_defined",
    "mape_defined",
)


def _as_float_arrays(actual: Any, predicted: Any) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            "actual and predicted must have identical shapes; got "
            f"{actual_array.shape} vs {predicted_array.shape}."
        )
    if actual_array.size == 0:
        raise ValueError("forecast_metrics requires at least one point.")
    if not bool(np.isfinite(actual_array).all()):
        raise ValueError("actual contains non-finite values; metrics cannot be defined.")
    if not bool(np.isfinite(predicted_array).all()):
        raise ValueError("predicted contains non-finite values; metrics cannot be defined.")
    return actual_array, predicted_array


def forecast_metrics(
    actual: Any,
    predicted: Any,
    *,
    mape_relative_threshold: float = 0.0,
) -> dict[str, Any]:
    """Return MAPE/WAPE/MAE/RMSE plus the complete near-zero policy record.

    Parameters
    ----------
    actual, predicted:
        Same-shape arrays (any dimension; flattened internally), in the same
        unit (kW for this project).
    mape_relative_threshold:
        Nonnegative, scale-free relative threshold.  ``threshold_kW =
        mape_relative_threshold * mean(|actual|)``.  ``0.0`` keeps every point
        with a strictly positive truth; a small positive value (e.g. ``0.01``)
        excludes truth points whose magnitude is negligible relative to the
        series level.

    Returns
    -------
    dict with exactly the keys ``MAPE``, ``WAPE``, ``MAE``, ``RMSE``, ``n``,
    ``mape_threshold``, ``mape_n``, ``mape_excluded``, ``wape_defined``,
    ``mape_defined``.  ``MAPE``/``WAPE`` are percentages, not fractions.
    """
    actual_array, predicted_array = _as_float_arrays(actual, predicted)
    relative = float(mape_relative_threshold)
    if not np.isfinite(relative) or relative < 0.0:
        raise ValueError(
            f"mape_relative_threshold must be finite and nonnegative, got {mape_relative_threshold!r}."
        )

    truth = actual_array.reshape(-1)
    forecast = predicted_array.reshape(-1)
    abs_truth = np.abs(truth)
    abs_error = np.abs(truth - forecast)
    squared_error = (truth - forecast) ** 2

    n = int(truth.size)
    mean_abs_truth = float(abs_truth.mean())
    mape_threshold = float(relative * mean_abs_truth) if relative > 0.0 else 0.0
    included = abs_truth > mape_threshold
    mape_n = int(np.count_nonzero(included))
    mape_excluded = int(n - mape_n)

    if mape_n:
        with np.errstate(divide="ignore", invalid="ignore"):
            mape: float | None = float(100.0 * np.mean(abs_error[included] / abs_truth[included]))
    else:
        mape = None

    wape_denominator = float(abs_truth.sum())
    wape_defined = bool(wape_denominator > WAPE_DENOMINATOR_EPS)
    wape = float(100.0 * abs_error.sum() / wape_denominator) if wape_defined else None

    metrics: dict[str, Any] = {
        "MAPE": mape,
        "WAPE": wape,
        "MAE": float(abs_error.mean()),
        "RMSE": float(np.sqrt(squared_error.mean())),
        "n": n,
        "mape_threshold": mape_threshold,
        "mape_n": mape_n,
        "mape_excluded": mape_excluded,
        "wape_defined": wape_defined,
        "mape_defined": mape is not None,
    }
    # Guard against accidental key drift in downstream writers.
    if tuple(metrics) != _METRIC_KEYS:  # pragma: no cover - developer invariant
        raise AssertionError(f"forecast_metrics keys drifted: {tuple(metrics)} != {_METRIC_KEYS}")
    return metrics


def undefined_metrics(*, mape_threshold: float = 0.0) -> dict[str, Any]:
    """Metrics record for an empty/censored slice (no scorable point).

    Kept internal-ish but documented: the per-origin / per-lead tables need one
    consistent column set even when a row has no valid label.
    """
    return {
        "MAPE": None,
        "WAPE": None,
        "MAE": None,
        "RMSE": None,
        "n": 0,
        "mape_threshold": float(mape_threshold),
        "mape_n": 0,
        "mape_excluded": 0,
        "wape_defined": False,
        "mape_defined": False,
    }
