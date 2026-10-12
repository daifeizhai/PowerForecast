from __future__ import annotations

import numpy as np

from src.data.splitting import ForecastInputs, WindowBatch
from src.models.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    name = "naive"

    def fit(self, batch: WindowBatch) -> "NaiveForecaster":
        return self

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        return np.repeat(inputs.history[:, -1, 0:1], inputs.future_known.shape[1], axis=1)


class SeasonalNaiveForecaster(BaseForecaster):
    name = "seasonal_naive"

    def __init__(self, period: int = 96):
        if period <= 0:
            raise ValueError("Seasonal period must be positive.")
        self.period = period

    def fit(self, batch: WindowBatch) -> "SeasonalNaiveForecaster":
        if batch.inputs.history.shape[1] < self.period:
            raise ValueError(f"seasonal_naive requires context >= {self.period}.")
        return self

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        if inputs.history.shape[1] < self.period:
            raise ValueError(f"seasonal_naive requires context >= {self.period}.")
        horizon = inputs.future_known.shape[1]
        return inputs.history[:, -self.period:, 0][:, np.arange(horizon) % self.period]
