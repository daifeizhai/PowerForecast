from __future__ import annotations

import numpy as np
from joblib import parallel_backend
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.splitting import ForecastInputs, WindowBatch
from src.models.base import BaseForecaster, tabular_inputs


class TabularForecaster(BaseForecaster):
    def __init__(self, name: str, params: dict, seed: int = 42):
        self.name = name
        params = dict(params)
        self.feature_mode = params.pop("feature_mode", "observed_history")
        self.output_jobs = int(params.pop("output_jobs", 4 if name in {"lightgbm", "xgboost"} else 1))
        if name == "ridge":
            self.estimator = make_pipeline(StandardScaler(), Ridge(**params))
        elif name == "extra_trees":
            self.estimator = ExtraTreesRegressor(random_state=seed, **params)
        elif name == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError("lightgbm requires pip install lightgbm (or requirements-optional.txt).") from exc
            self.estimator = MultiOutputRegressor(LGBMRegressor(random_state=seed, **params), n_jobs=self.output_jobs)
        elif name == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError("xgboost requires pip install xgboost (or requirements-optional.txt).") from exc
            self.estimator = MultiOutputRegressor(XGBRegressor(random_state=seed, **params), n_jobs=self.output_jobs)
        else:
            raise ValueError(f"Unsupported tabular model: {name}")
        self.fitted = False

    def fit(self, batch: WindowBatch) -> "TabularForecaster":
        with parallel_backend("threading", n_jobs=self.output_jobs):
            self.estimator.fit(self._features(batch.inputs), batch.targets)
        self.fitted = True
        return self

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(f"Fit {self.name} before predicting.")
        with parallel_backend("threading", n_jobs=getattr(self, "output_jobs", 1)):
            prediction = self.estimator.predict(self._features(inputs))
        return np.asarray(prediction).reshape(len(inputs), inputs.future_known.shape[1])

    def _features(self, inputs):
        mode = getattr(self, "feature_mode", "flatten")
        if mode == "flatten":
            return tabular_inputs(inputs)
        if mode != "observed_history":
            raise ValueError(f"Unknown tabular feature mode: {mode}")
        known_count = inputs.future_known.shape[2]
        observed_count = inputs.history.shape[2]-known_count
        if observed_count < 1:
            raise ValueError("Expected observed features before appended calendar features.")
        # Retain every observed lag. Direct lead-specific models need only the
        # origin/day calendar phases, not thousands of repeated calendar columns.
        return np.concatenate([inputs.history[:, :, :observed_count].reshape(len(inputs), -1),
            inputs.history[:, -1, observed_count:], inputs.future_known[:, ::96].reshape(len(inputs), -1),
            inputs.future_known[:, -1]], axis=1)
