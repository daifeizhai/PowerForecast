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


class ThreeDayMeanForecaster(BaseForecaster):
    name = "three_day_mean"

    def fit(self, batch):
        if batch.inputs.history.shape[1] < batch.inputs.points_per_day*3:
            raise ValueError("three_day_mean requires 3 complete observed days.")
        return self

    def predict(self, inputs):
        period = inputs.points_per_day
        shape = inputs.history[:, -3*period:, 0].reshape(len(inputs), 3, period).mean(1)
        return shape[:, np.arange(inputs.future_known.shape[1]) % period]


class SeasonalARForecaster(BaseForecaster):
    """Fixed-parameter seasonal AR with recursive generated-value rollout."""
    name = "seasonal_ar"

    def fit(self, batch):
        from sklearn.linear_model import Ridge
        self.lags = (1, 96, 192, 672)
        contexts, labels = [], []
        # Each day used once; all lagged features are observed within the training sample.
        for past, target in zip(batch.inputs.history[:, :, 0], batch.targets[:, :96]):
            series = np.concatenate([past, target])
            idx = len(past)+np.arange(96)
            contexts.append(np.stack([series[idx-lag] for lag in self.lags], axis=-1))
            labels.append(target)
        self.estimator = Ridge(alpha=1.).fit(np.concatenate(contexts), np.concatenate(labels))
        return self

    def predict(self, inputs):
        horizon = inputs.future_known.shape[1]
        generated = np.empty((len(inputs), inputs.history.shape[1]+horizon), dtype=float)
        generated[:, :inputs.history.shape[1]] = inputs.history[:, :, 0]
        for i in range(inputs.history.shape[1], generated.shape[1]):
            generated[:, i] = self.estimator.predict(generated[:, [i-lag for lag in self.lags]])
        return generated[:, -horizon:]
