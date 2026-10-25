"""History-window normalization with forecast-scale restoration; no target fitting."""
from __future__ import annotations

import numpy as np

from src.data.splitting import ForecastInputs, WindowBatch
from src.models.base import BaseForecaster


def normalized_inputs(inputs, scale_fraction=0.01, epsilon=1e-6):
    if inputs.history.shape[2] != inputs.future_known.shape[2]+1:
        raise ValueError("Normalization currently supports target history plus calendar only.")
    past = np.asarray(inputs.history[:, :, 0], dtype=float)
    if not np.isfinite(past).all():
        raise ValueError("Normalization cannot fill missing target history.")
    location = past.mean(axis=1, keepdims=True)
    scale = np.maximum(past.std(axis=1, keepdims=True),
                       np.maximum(np.abs(location)*scale_fraction, epsilon))
    history = np.array(inputs.history, dtype=np.float32, copy=True)
    history[:, :, 0] = ((past-location)/scale).astype(np.float32)
    transformed = ForecastInputs(history, inputs.future_known, inputs.origins, inputs.target_times, inputs.domain_ids, inputs.points_per_day)
    return transformed, location, scale


class HistoryNormalizedForecaster(BaseForecaster):
    def __init__(self, base_model="ridge", base_params=None, scale_fraction=0.01, epsilon=1e-6, seed=42,
                 name="normalized_ridge"):
        if base_model not in {"ridge", "extra_trees", "dlinear"} or scale_fraction < 0 or epsilon <= 0:
            raise ValueError("Use a supported base model and positive normalization scale.")
        self.name, self.base_model, self.base_params = name, base_model, dict(base_params or {})
        self.scale_fraction, self.epsilon, self.seed = float(scale_fraction), float(epsilon), int(seed)

    def _transform(self, inputs):
        return normalized_inputs(inputs, self.scale_fraction, self.epsilon)

    def fit(self, batch):
        from src.models import create_model
        inputs, location, scale = self._transform(batch.inputs)
        labels = ((np.asarray(batch.targets, dtype=float)-location)/scale).astype(np.float32)
        self.model = create_model(self.base_model, self.base_params, self.seed).fit(WindowBatch(inputs, labels))
        return self

    def predict(self, inputs):
        if not hasattr(self, "model"):
            raise RuntimeError("Fit the normalized model before prediction.")
        transformed, location, scale = self._transform(inputs)
        return np.asarray(self.model.predict(transformed), float)*scale+location
