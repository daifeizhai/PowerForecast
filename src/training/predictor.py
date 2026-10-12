from __future__ import annotations

import numpy as np

from src.data.splitting import ForecastInputs
from src.models.base import BaseForecaster


def predict(model: BaseForecaster, inputs: ForecastInputs, clip_min: float | None = None) -> np.ndarray:
    values = np.asarray(model.predict(inputs), dtype=np.float64)
    expected = (len(inputs), inputs.future_known.shape[1])
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"{model.name} returned invalid predictions: expected finite shape {expected}, got {values.shape}.")
    return np.maximum(values, clip_min) if clip_min is not None else values
