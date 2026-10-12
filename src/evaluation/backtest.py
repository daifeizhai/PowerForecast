from __future__ import annotations

from dataclasses import dataclass

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


def backtest(model: BaseForecaster, batch: WindowBatch, clip_min: float | None = None, batch_size: int = 32) -> BacktestResult:
    if batch_size <= 0:
        raise ValueError("Prediction batch size must be positive.")
    # Fixed parameters, updated observed history at each origin.
    predictions = np.concatenate([
        predict(model, batch.inputs.take(slice(start, start+batch_size)), clip_min)
        for start in range(0, len(batch), batch_size)
    ])
    lead_metrics = [{"lead": lead+1, **forecast_metrics(batch.targets[:, lead], predictions[:, lead])} for lead in range(batch.targets.shape[1])]
    return BacktestResult(forecast_metrics(batch.targets, predictions), predictions, pd.DataFrame(lead_metrics))


def prediction_table(batch: WindowBatch, predictions: np.ndarray) -> pd.DataFrame:
    horizon = batch.targets.shape[1]
    return pd.DataFrame({
        "origin": np.repeat(batch.inputs.origins.to_numpy(), horizon),
        "timestamp": batch.inputs.target_times.reshape(-1),
        "lead": np.tile(np.arange(1, horizon+1), len(batch)),
        "actual": batch.targets.reshape(-1),
        "predicted": predictions.reshape(-1),
    })
