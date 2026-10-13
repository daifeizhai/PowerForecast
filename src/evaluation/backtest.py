from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from src.data.splitting import WindowBatch
from src.evaluation.metrics import forecast_metrics
from src.models.base import BaseForecaster
from src.training.predictor import predict


@dataclass
class BacktestResult:
    metrics: dict
    predictions: np.ndarray
    per_lead: pd.DataFrame
    per_origin: pd.DataFrame
    daily_metrics: dict
    daily_per_origin: pd.DataFrame
    per_day: pd.DataFrame
    nonoverlap_metrics: dict | None
    inference_seconds: float = 0.


def backtest(model: BaseForecaster, batch: WindowBatch, clip_min: float | None = None, batch_size: int = 32,
             mape_threshold: float = 1e-8, anchor: pd.Timestamp | None = None) -> BacktestResult:
    if batch_size <= 0:
        raise ValueError("Prediction batch size must be positive.")
    # Fixed parameters, updated observed history at each origin.
    started = perf_counter()
    predictions = np.concatenate([
        predict(model, batch.inputs.take(slice(start, start+batch_size)), clip_min)
        for start in range(0, len(batch), batch_size)
    ])
    return score_predictions(batch, predictions, mape_threshold, anchor, perf_counter()-started)


def score_predictions(batch, predictions, mape_threshold=1e-8, anchor=None, inference_seconds=0.):
    metrics = lambda actual, predicted: forecast_metrics(actual, predicted, mape_threshold)
    origin_rows = [{"origin": str(origin), **metrics(y, p)} for origin, y, p in
                   zip(batch.inputs.origins, batch.targets, predictions)]
    horizon = batch.targets.shape[1]
    if horizon % 96:
        daily_truth, daily_pred = batch.targets.mean(1, keepdims=True), predictions.mean(1, keepdims=True)
    else:
        daily_truth = batch.targets.reshape(len(batch), -1, 96).mean(-1)
        daily_pred = predictions.reshape(len(batch), -1, 96).mean(-1)
    daily_rows = [{"origin": str(origin), **metrics(y, p)} for origin, y, p in
                  zip(batch.inputs.origins, daily_truth, daily_pred)]
    summary = metrics(batch.targets, predictions)
    for name in ("MAPE", "WAPE", "RMSE"):
        values = [row[name] for row in origin_rows if row[name] is not None]
        summary.update({f"origin_{name}_{stat}": float(function(values)) if values else None
                        for stat, function in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max))})
    lead_rows = [{"lead": i+1, **metrics(batch.targets[:, i], predictions[:, i])} for i in range(horizon)]
    day_rows = [{"forecast_day": i+1, **metrics(batch.targets[:, i*96:(i+1)*96], predictions[:, i*96:(i+1)*96])}
                for i in range(horizon//96)]
    nonoverlap = None
    if horizon == 672:
        anchor = anchor if anchor is not None else batch.inputs.origins[0]
        elapsed_days = (batch.inputs.origins.as_unit("ns").asi8-pd.Timestamp(anchor).as_unit("ns").value)/(86400*1e9)
        keep = np.asarray(elapsed_days % 7 == 0)
        if keep.any():
            nonoverlap = {"origins": [str(o) for o in batch.inputs.origins[keep]],
                          **metrics(batch.targets[keep], predictions[keep]),
                          "daily_mean": metrics(daily_truth[keep], daily_pred[keep])}
    return BacktestResult(summary, predictions, pd.DataFrame(lead_rows), pd.DataFrame(origin_rows),
                          metrics(daily_truth, daily_pred), pd.DataFrame(daily_rows), pd.DataFrame(day_rows),
                          nonoverlap, inference_seconds)


def prediction_table(batch: WindowBatch, predictions: np.ndarray) -> pd.DataFrame:
    horizon = batch.targets.shape[1]
    return pd.DataFrame({
        "origin": np.repeat(batch.inputs.origins.to_numpy(), horizon),
        "timestamp": batch.inputs.target_times.reshape(-1),
        "lead": np.tile(np.arange(1, horizon+1), len(batch)),
        "actual": batch.targets.reshape(-1),
        "predicted": predictions.reshape(-1),
    })
