"""Multi-level scoring of labelled forecast batches.

Reporting levels (task book §9):

``aggregate``
    All valid points in the batch, one MAPE/WAPE/MAE/RMSE record.
``per_origin``
    One row per forecast origin with the origin timestamp and all metrics.
``per_lead``
    One row per lead step ``h = 1..H`` (error vs forecast horizon).
``per_forecast_day``
    Only when ``H % 96 == 0``: one row per 96-point forecast day, so a 672-point
    (7d) forecast yields seven rows.  Each day is scored separately.
``daily_mean``
    When ``H % 96 == 0``: metrics of the **per-day mean power** of each forecast
    curve, kind ``daily_mean_power``.  When ``H < 96`` (4h/8h) this is the mean
    of the forecast window and is labelled ``window_mean_power`` -- it is
    **never** called a daily mean.  Any other horizon shape yields kind
    ``"none"``.
``overlap``
    Non-null only for ``H % 96 == 0`` when two origins' label windows share
    ground-truth timestamps (e.g. 672-point forecasts from consecutive days).
    The note states that error points are **not independent**; no independent
    weekly sample claim is derivable from such a batch.

Only rows whose ``target_valid_mask`` is True are scored.  Missing labels are
never imputed; a level with no valid point gets an explicit all-``None`` record
with ``n=0`` instead of being silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import ContractError, ForecastBatch
from src.evaluation.metrics import forecast_metrics, undefined_metrics
from src.utils import jsonable

__all__ = ["MetricsReport", "POINTS_PER_DAY", "score_batch"]

#: Formal 15-minute output grid: 96 points per trace day.
POINTS_PER_DAY = 96

_METRIC_COLUMNS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class MetricsReport:
    """Structured multi-level metrics for one scored forecast batch."""

    aggregate: dict[str, Any]
    per_origin: pd.DataFrame
    per_lead: pd.DataFrame
    per_forecast_day: pd.DataFrame
    daily_mean: dict[str, Any] | None
    daily_mean_kind: str
    overlap: dict[str, Any] | None
    labels: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe payload for ``metrics.json`` (all levels included)."""
        return {
            "aggregate": jsonable(self.aggregate),
            "per_origin": _frame_records(self.per_origin),
            "per_lead": _frame_records(self.per_lead),
            "per_forecast_day": _frame_records(self.per_forecast_day),
            "daily_mean": None if self.daily_mean is None else jsonable(self.daily_mean),
            "daily_mean_kind": self.daily_mean_kind,
            "overlap": None if self.overlap is None else jsonable(self.overlap),
            "labels": jsonable(self.labels),
            "horizon": int(len(self.per_lead)),
            "n_origins": int(len(self.per_origin)),
        }


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(key): jsonable(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _flatten_arrays(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _safe_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    mape_relative_threshold: float,
) -> dict[str, Any]:
    """Metrics over the valid subset, or an explicit undefined record."""
    actual_flat = _flatten_arrays(actual)
    predicted_flat = _flatten_arrays(predicted)
    valid_flat = np.asarray(valid, dtype=bool).reshape(-1)
    if not (actual_flat.size == predicted_flat.size == valid_flat.size):
        raise ContractError(
            "Score slices must have equal sizes; got "
            f"actual={actual_flat.size}, predicted={predicted_flat.size}, mask={valid_flat.size}."
        )
    if not bool(valid_flat.any()):
        return undefined_metrics()
    return forecast_metrics(
        actual_flat[valid_flat],
        predicted_flat[valid_flat],
        mape_relative_threshold=mape_relative_threshold,
    )


def _validate_predictions(batch: ForecastBatch, predictions: np.ndarray) -> np.ndarray:
    if not isinstance(batch, ForecastBatch):
        raise ContractError("score_batch expects a ForecastBatch (labelled data).")
    array = np.asarray(predictions, dtype=float)
    expected = batch.targets_kw.shape
    if array.shape != expected:
        raise ValueError(
            f"predictions must have shape {expected} [B, H]; got {array.shape}."
        )
    if not bool(np.isfinite(array).all()):
        raise ValueError(
            "predictions contain non-finite values (NaN/inf); refusing to score a diverged "
            "model.  Clipping must be an explicit, recorded step before scoring."
        )
    return array


def _daily_window_mean_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    valid: np.ndarray,
    mape_relative_threshold: float,
) -> tuple[dict[str, Any] | None, str]:
    """Per-day mean power (H multiple of 96) or forecast-window mean (H < 96)."""
    batch_size, horizon = targets.shape
    actual_means: list[float] = []
    predicted_means: list[float] = []
    if horizon % POINTS_PER_DAY == 0 and horizon > 0:
        days = horizon // POINTS_PER_DAY
        for sample in range(batch_size):
            for day in range(days):
                window = slice(day * POINTS_PER_DAY, (day + 1) * POINTS_PER_DAY)
                mask = valid[sample, window]
                if bool(mask.any()):
                    actual_means.append(float(targets[sample, window][mask].mean()))
                    predicted_means.append(float(predictions[sample, window][mask].mean()))
        kind = "daily_mean_power"
    elif horizon < POINTS_PER_DAY:
        for sample in range(batch_size):
            mask = valid[sample]
            if bool(mask.any()):
                actual_means.append(float(targets[sample][mask].mean()))
                predicted_means.append(float(predictions[sample][mask].mean()))
        kind = "window_mean_power"
    else:
        return None, "none"
    if not actual_means:
        return None, kind
    return (
        forecast_metrics(
            np.asarray(actual_means),
            np.asarray(predicted_means),
            mape_relative_threshold=mape_relative_threshold,
        ),
        kind,
    )


def _detect_overlap(batch: ForecastBatch) -> dict[str, Any] | None:
    """Flag shared ground-truth timestamps between origins (H multiple of 96)."""
    if batch.horizon % POINTS_PER_DAY != 0:
        return None
    stamps = np.asarray(batch.inputs.target_timestamps).reshape(-1)
    unique = np.unique(stamps)
    if int(unique.size) >= int(stamps.size):
        return None
    return {
        "overlapping": True,
        "note": "origins share ground truth; error points are NOT independent",
        "unique_target_timestamps": int(unique.size),
    }


def score_batch(
    batch: ForecastBatch,
    predictions: np.ndarray,
    *,
    mape_relative_threshold: float = 0.0,
) -> MetricsReport:
    """Score ``predictions`` against a labelled :class:`ForecastBatch`.

    ``predictions`` must be ``[B, H]`` in kW and match the batch horizon.  The
    returned :class:`MetricsReport` keeps every declared reporting level so a
    saved run can be re-checked level by level.
    """
    predictions = _validate_predictions(batch, predictions)
    targets = np.asarray(batch.targets_kw, dtype=float)
    valid = np.asarray(batch.target_valid_mask, dtype=bool)
    batch_size, horizon = targets.shape

    aggregate = _safe_metrics(targets, predictions, valid, mape_relative_threshold)

    origin_rows: list[dict[str, Any]] = []
    origins = np.asarray(batch.inputs.origin_timestamps)
    for sample in range(batch_size):
        row = {"origin_timestamp": pd.Timestamp(origins[sample])}
        row.update(_safe_metrics(targets[sample], predictions[sample], valid[sample], mape_relative_threshold))
        origin_rows.append(row)
    per_origin = pd.DataFrame(origin_rows, columns=("origin_timestamp", *_METRIC_COLUMNS))

    lead_rows: list[dict[str, Any]] = []
    for lead in range(horizon):
        row: dict[str, Any] = {"lead": lead + 1}
        row.update(
            _safe_metrics(
                targets[:, lead],
                predictions[:, lead],
                valid[:, lead],
                mape_relative_threshold,
            )
        )
        lead_rows.append(row)
    per_lead = pd.DataFrame(lead_rows, columns=("lead", *_METRIC_COLUMNS))

    day_rows: list[dict[str, Any]] = []
    if horizon % POINTS_PER_DAY == 0 and horizon > 0:
        for day in range(horizon // POINTS_PER_DAY):
            window = slice(day * POINTS_PER_DAY, (day + 1) * POINTS_PER_DAY)
            row = {"forecast_day": day + 1}
            row.update(
                _safe_metrics(
                    targets[:, window],
                    predictions[:, window],
                    valid[:, window],
                    mape_relative_threshold,
                )
            )
            day_rows.append(row)
        per_forecast_day = pd.DataFrame(day_rows, columns=("forecast_day", *_METRIC_COLUMNS))
    else:
        per_forecast_day = pd.DataFrame(columns=("forecast_day", *_METRIC_COLUMNS))

    daily_mean, daily_mean_kind = _daily_window_mean_metrics(
        targets, predictions, valid, mape_relative_threshold
    )

    target = batch.inputs.target_spec
    labels = {
        "unit": "kW",
        "scope": target.scope.value,
        "label_origin": target.label_origin.value,
        "synthetic": bool(target.synthetic),
        "mape_threshold": float(aggregate["mape_threshold"]),
        "mape_relative_threshold": float(mape_relative_threshold),
        "mape_policy": (
            "MAPE keeps points with |y| > mape_relative_threshold * mean(|y|); "
            "mape_threshold is the absolute kW threshold actually used"
        ),
    }

    return MetricsReport(
        aggregate=aggregate,
        per_origin=per_origin,
        per_lead=per_lead,
        per_forecast_day=per_forecast_day,
        daily_mean=daily_mean,
        daily_mean_kind=daily_mean_kind,
        overlap=_detect_overlap(batch),
        labels=labels,
    )
