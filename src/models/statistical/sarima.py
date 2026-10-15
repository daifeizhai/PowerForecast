"""Fit SARIMA parameters once, then filter each observed rolling context."""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from src.data.splitting import ForecastInputs, WindowBatch
from src.models.base import BaseForecaster


class SarimaForecaster(BaseForecaster):
    name = "sarima"

    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 0, 0, 96), maxiter: int = 50, fit_points: int = 2048,
                 estimation: str = "statespace", center: bool = False):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:
            raise ImportError("sarima requires statsmodels. Install requirements-optional.txt or pip install statsmodels.") from exc
        self.sarimax = SARIMAX
        self.order, self.seasonal_order = tuple(order), tuple(seasonal_order)
        self.maxiter, self.fit_points = maxiter, fit_points
        if estimation not in {"statespace", "innovations_mle"}:
            raise ValueError("SARIMA estimation must be statespace or innovations_mle.")
        self.estimation, self.center = estimation, bool(center)
        self.params = None

    def _spec(self, series):
        stationary = getattr(self, "estimation", "statespace") == "innovations_mle"
        return self.sarimax(series, order=self.order, seasonal_order=self.seasonal_order,
                           enforce_stationarity=stationary, enforce_invertibility=stationary)

    def fit(self, batch: WindowBatch) -> "SarimaForecaster":
        # The latest training window is a contiguous observed series; never
        # concatenate overlapping windows or include validation/test labels.
        series = np.concatenate([batch.inputs.history[-1, :, 0], batch.targets[-1]])[-self.fit_points:]
        return self.fit_series(series)

    def fit_series(self, series) -> "SarimaForecaster":
        """Fit one contiguous training trace, optionally with missing observations."""
        series = np.asarray(series, dtype=float)[-self.fit_points:]
        self.level = float(np.mean(series)) if self.center else 0.
        series = series-self.level
        # Point forecasts need the terminal state, not covariance estimates.
        with threadpool_limits(limits=1, user_api="blas"):
            if self.estimation == "innovations_mle":
                from statsmodels.tsa.arima.estimators.innovations import innovations_mle
                result, details = innovations_mle(series, order=self.order, seasonal_order=self.seasonal_order,
                    demean=False, enforce_invertibility=True,
                    minimize_kwargs={"method": "L-BFGS-B", "options": {"maxiter": self.maxiter}})
                self.params = result.params
                self.optimization = dict(converged=bool(details.minimize_results.success),
                    iterations=int(details.minimize_results.nit), message=str(details.minimize_results.message))
            else:
                result = self._spec(series).fit(disp=False, maxiter=self.maxiter,
                                                low_memory=True, cov_type="none")
                self.params = result.params
                self.optimization = result.mle_retvals
        self._plans = {}
        return self

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Fit sarima before predicting.")
        horizon = inputs.future_known.shape[1]
        with threadpool_limits(limits=1, user_api="blas"):
            from scipy.sparse import csr_matrix
            level = getattr(self, "level", 0.)
            history = np.asarray(inputs.history[:, :, 0], dtype=float)-level
            if not np.isfinite(history).all():
                raise ValueError("SARIMA forecast histories must be finite.")
            length = history.shape[1]
            if not hasattr(self, "_plans"):
                self._plans = {}
            if length not in self._plans:
                # For equal-length, fully observed contexts, Kalman gains depend
                # only on frozen parameters, not on the observed load values.
                spec = self._spec(np.zeros(length))
                result = spec.filter(self.params, conserve_memory=0, cov_type="none")
                filtered = result.filter_results
                self._plans[length] = (csr_matrix(spec.ssm.transition[:, :, 0]),
                    spec.ssm.design[0, :, 0].copy(), filtered.kalman_gain[:, 0, :].copy(),
                    filtered.initial_state.copy(), spec.ssm.state_intercept[:, 0].copy(),
                    float(spec.ssm.obs_intercept[0, 0]))
            transition, design, gain, initial, intercept, offset = self._plans[length]
            states = np.repeat(initial[:, None], len(history), axis=1)
            for i in range(length):
                residual = history[:, i] - (design @ states + offset)
                states = transition @ states + intercept[:, None] + gain[:, i, None]*residual[None, :]
            prediction = np.empty((len(history), horizon))
            for i in range(horizon):
                prediction[:, i] = design @ states + offset + level
                states = transition @ states + intercept[:, None]
            return prediction
