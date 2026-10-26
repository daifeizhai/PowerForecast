"""Planned supervised target-domain power mapping (C): interface only.

The mapping is trained on target-domain history only, using representations
built from information known at/before the forecast origin.  No mapping model is
implemented in this round; every execution method raises
:class:`NotImplementedError` instead of emitting a fake auxiliary kW candidate.

Future work is tracked in ``docs/TODO_RESEARCH.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.data.contracts import ForecastBatch, ForecastInputs
from src.models.interfaces import BaseTargetMapper, ModelStatus

__all__ = ["AbstractTargetMapper", "PlannedTargetMapper", "TODO_RESEARCH_DOC"]

TODO_RESEARCH_DOC = "docs/TODO_RESEARCH.md"


class AbstractTargetMapper(ABC):
    """Abstract base class for future supervised target-domain mappings."""

    name: str = "target_mapping"
    family: str = "target_mapping"
    status: ModelStatus = ModelStatus.PLANNED
    requires_fit: bool = True

    @abstractmethod
    def fit(self, batch: ForecastBatch, historic_repr: np.ndarray) -> None:
        """Fit using origin-safe historical representations and kW labels."""

    @abstractmethod
    def predict_aux(self, inputs: ForecastInputs, historic_repr: np.ndarray) -> np.ndarray:
        """Return an auxiliary kW candidate ``[B, H]``."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return JSON-serialisable metadata."""


class PlannedTargetMapper(AbstractTargetMapper):
    """Placeholder registered as ``planned``; fit/predict_aux are refused."""

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.status: ModelStatus = ModelStatus.PLANNED

    @staticmethod
    def _blocked(operation: str) -> NotImplementedError:
        return NotImplementedError(
            f"target_mapping is PLANNED, not implemented: '{operation}' cannot run in this "
            "round.  The supervised mapping relation has not been specified or validated, so no "
            f"auxiliary kW prediction is returned.  Future work: {TODO_RESEARCH_DOC}."
        )

    def fit(self, batch: ForecastBatch, historic_repr: np.ndarray) -> None:
        raise self._blocked("fit")

    def predict_aux(self, inputs: ForecastInputs, historic_repr: np.ndarray) -> np.ndarray:
        raise self._blocked("predict_aux")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "family": self.family,
            "requires_fit": self.requires_fit,
            "params": dict(self.params),
            "description": (
                "planned supervised target-domain power mapping; intentionally NOT implemented"
            ),
            "todo": TODO_RESEARCH_DOC,
        }
