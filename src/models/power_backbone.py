"""Planned main power path (A): an explicit entry point for future research.

No learned backbone is implemented in this round.  The class below is a
*registration-only* placeholder: it may be constructed (so that registry/CLI
code can reason about it) but every execution method raises
:class:`NotImplementedError`.  It never returns weights, never trains and never
produces a prediction that could be mistaken for a research result.

Future work is tracked in ``docs/TODO_RESEARCH.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.data.contracts import ForecastBatch, ForecastInputs
from src.models.interfaces import BasePowerForecaster, ModelStatus

__all__ = [
    "AbstractPowerBackbone",
    "BasePowerBackbone",
    "PlannedPowerBackbone",
    "TODO_RESEARCH_DOC",
]

TODO_RESEARCH_DOC = "docs/TODO_RESEARCH.md"


@runtime_checkable
class BasePowerBackbone(BasePowerForecaster, Protocol):
    """Structural protocol for a future trainable power backbone."""

    status: ModelStatus


class AbstractPowerBackbone(ABC):
    """Abstract base class future power backbones should inherit from."""

    name: str = "power_backbone"
    family: str = "backbone"
    status: ModelStatus = ModelStatus.PLANNED
    requires_fit: bool = True

    @abstractmethod
    def fit(self, batch: ForecastBatch) -> None:
        """Fit on a supervised, origin-safe ForecastBatch."""

    @abstractmethod
    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        """Return ``[B, H]`` absolute active power in kW."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return JSON-serialisable metadata."""


class PlannedPowerBackbone(AbstractPowerBackbone):
    """Placeholder registered as ``planned``; execution is refused explicitly."""

    def __init__(self, **params: Any) -> None:
        # Only configuration is recorded; no random or fake "trained" state.
        self.params: dict[str, Any] = dict(params)
        self.status: ModelStatus = ModelStatus.PLANNED

    @staticmethod
    def _blocked(operation: str) -> NotImplementedError:
        return NotImplementedError(
            f"power_backbone is PLANNED, not implemented: '{operation}' cannot run in this "
            "rebuild round.  The main path is currently provided by the real parameter-free "
            f"baselines.  Future backbone work is tracked in {TODO_RESEARCH_DOC}."
        )

    def fit(self, batch: ForecastBatch) -> None:
        raise self._blocked("fit")

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        raise self._blocked("predict")

    def save(self, path: str | Path) -> None:
        raise self._blocked("save")

    def describe(self) -> dict[str, Any]:
        """Honest metadata only; declaring a model planned is not an implementation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "family": self.family,
            "requires_fit": self.requires_fit,
            "params": dict(self.params),
            "description": (
                "planned learned main power path; intentionally NOT implemented in this round"
            ),
            "todo": TODO_RESEARCH_DOC,
        }
