"""All models fit on training labels and predict without access to labels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.data.splitting import ForecastInputs, WindowBatch


class BaseForecaster(ABC):
    name: str

    @abstractmethod
    def fit(self, batch: WindowBatch) -> "BaseForecaster":
        """Fit using only the supplied training split."""

    @abstractmethod
    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        """Return [sample, horizon] forecasts using history and known inputs."""

    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path).with_suffix(".joblib")
        joblib.dump(self, path)
        return path


def tabular_inputs(inputs: ForecastInputs) -> np.ndarray:
    return np.concatenate([
        inputs.history.reshape(len(inputs), -1),
        inputs.future_known.reshape(len(inputs), -1),
    ], axis=1)
