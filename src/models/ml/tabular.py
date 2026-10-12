from __future__ import annotations

import numpy as np
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
        if name == "ridge":
            self.estimator = make_pipeline(StandardScaler(), Ridge(**params))
        elif name == "extra_trees":
            self.estimator = ExtraTreesRegressor(random_state=seed, **params)
        elif name == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError("lightgbm requires pip install lightgbm (or requirements-optional.txt).") from exc
            self.estimator = MultiOutputRegressor(LGBMRegressor(random_state=seed, **params), n_jobs=1)
        elif name == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError("xgboost requires pip install xgboost (or requirements-optional.txt).") from exc
            self.estimator = MultiOutputRegressor(XGBRegressor(random_state=seed, **params), n_jobs=1)
        else:
            raise ValueError(f"Unsupported tabular model: {name}")
        self.fitted = False

    def fit(self, batch: WindowBatch) -> "TabularForecaster":
        self.estimator.fit(tabular_inputs(batch.inputs), batch.targets)
        self.fitted = True
        return self

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(f"Fit {self.name} before predicting.")
        prediction = self.estimator.predict(tabular_inputs(inputs))
        return np.asarray(prediction).reshape(len(inputs), inputs.future_known.shape[1])
