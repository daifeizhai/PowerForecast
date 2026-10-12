from __future__ import annotations

import time
from dataclasses import dataclass

from src.data.splitting import WindowBatch
from src.models import create_model
from src.models.base import BaseForecaster


@dataclass
class TrainingResult:
    model: BaseForecaster
    seconds: float
    train_windows: int


def train_model(name: str, batch: WindowBatch, params: dict | None = None, seed: int = 42) -> TrainingResult:
    if not len(batch):
        raise ValueError("Training requires at least one valid window.")
    model = create_model(name, params, seed)
    start = time.perf_counter()
    model.fit(batch)
    return TrainingResult(model, time.perf_counter() - start, len(batch))
