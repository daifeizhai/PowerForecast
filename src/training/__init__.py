"""Training layer: phase policy, pre-test freezing/reload, and prediction runner."""

from __future__ import annotations

from src.training.lifecycle import (
    PHASES,
    FrozenModel,
    FrozenRunError,
    freeze_model,
    load_frozen,
    phase_allows_fit,
    phase_allows_test_scoring,
    require_frozen_record,
)
from src.training.runner import predict_batch

__all__ = [
    "PHASES",
    "FrozenModel",
    "FrozenRunError",
    "freeze_model",
    "load_frozen",
    "phase_allows_fit",
    "phase_allows_test_scoring",
    "predict_batch",
    "require_frozen_record",
]
