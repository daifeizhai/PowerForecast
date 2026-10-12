"""Fit SARIMA parameters once, then filter each observed rolling context."""

from __future__ import annotations

import numpy as np

from src.data.splitting import ForecastInputs, WindowBatch
from src.models.base import BaseForecaster


class SarimaForecaster(BaseForecaster):
    name = "sarima"

    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 0, 0, 96), maxiter: int = 50, fit_points: int = 2048):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:
            raise ImportError("sarima requires statsmodels. Install requirements-optional.txt or pip install statsmodels.") from exc
        self.sarimax = SARIMAX
        self.order, self.seasonal_order = tuple(order), tuple(seasonal_order)
        self.maxiter, self.fit_points = maxiter, fit_points
        self.params = None

    def _spec(self, series):
        return self.sarimax(series, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)

    def fit(self, batch: WindowBatch) -> "SarimaForecaster":
        # The latest training window is a contiguous observed series; never
        # concatenate overlapping windows or include validation/test labels.
        series = np.concatenate([batch.inputs.history[-1, :, 0], batch.targets[-1]])[-self.fit_points:]
        self.params = self._spec(series).fit(disp=False, maxiter=self.maxiter).params
        return self

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Fit sarima before predicting.")
        horizon = inputs.future_known.shape[1]
        return np.stack([self._spec(history[:, 0]).filter(self.params).forecast(horizon) for history in inputs.history])
