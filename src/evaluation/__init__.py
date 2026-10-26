"""Evaluation layer: metrics, multi-level scoring, and run artifacts.

Public API
----------
* :func:`forecast_metrics` -- MAPE/WAPE/MAE/RMSE with an explicit near-zero
  MAPE policy and definedness flags.
* :func:`score_batch` / :class:`MetricsReport` -- per-origin, per-lead,
  per-forecast-day, daily-mean and overlap-aware scoring of a labelled batch.
* :class:`RunDirectory` / :func:`predictions_frame` -- immutable run directory,
  exact 16-column prediction table, manifest/frozen/metrics writers.
"""

from __future__ import annotations

from src.evaluation.artifacts import (
    PREDICTION_COLUMNS,
    RunDirectory,
    predictions_frame,
)
from src.evaluation.backtest import MetricsReport, score_batch
from src.evaluation.metrics import WAPE_DENOMINATOR_EPS, forecast_metrics

__all__ = [
    "PREDICTION_COLUMNS",
    "MetricsReport",
    "RunDirectory",
    "WAPE_DENOMINATOR_EPS",
    "forecast_metrics",
    "predictions_frame",
    "score_batch",
]
