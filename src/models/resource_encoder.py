"""Planned unlabelled operational-metric encoder (B): interface only.

The encoder may *only* consume :class:`~src.data.contracts.PretrainBatch`
objects, which structurally carry no kW label.  No real pretraining is
implemented in this round; every execution method raises
:class:`NotImplementedError` instead of returning a randomly initialised
"trained" encoder.

Future work is tracked in ``docs/TODO_RESEARCH.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.data.contracts import ForecastInputs, PretrainBatch
from src.models.interfaces import BaseResourceEncoder, ModelStatus

__all__ = [
    "AbstractResourceEncoder",
    "PlannedResourceEncoder",
    "TODO_RESEARCH_DOC",
]

TODO_RESEARCH_DOC = "docs/TODO_RESEARCH.md"


class AbstractResourceEncoder(ABC):
    """Abstract base class for future unlabelled representation encoders."""

    name: str = "resource_pretrain"
    family: str = "resource_pretrain"
    status: ModelStatus = ModelStatus.PLANNED
    requires_fit: bool = True

    @abstractmethod
    def pretrain(self, batches: Iterable[PretrainBatch]) -> None:
        """Pretrain on unlabelled batches (no power label may be present)."""

    @abstractmethod
    def encode(self, inputs: "PretrainBatch | ForecastInputs") -> np.ndarray:
        """Encode observed metrics into a representation array."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return JSON-serialisable metadata."""


class PlannedResourceEncoder(AbstractResourceEncoder):
    """Placeholder registered as ``planned``; pretraining/encoding is refused."""

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.status: ModelStatus = ModelStatus.PLANNED
        self.feature_names: tuple[str, ...] = tuple(params.get("feature_names") or ())

    @staticmethod
    def _blocked(operation: str) -> NotImplementedError:
        return NotImplementedError(
            f"resource_pretrain is PLANNED, not implemented: '{operation}' cannot run in this "
            "round.  No representation encoder has been specified, trained or validated, so no "
            "random/fake 'trained' encoder is returned.  Future work: "
            f"{TODO_RESEARCH_DOC}."
        )

    def pretrain(self, batches: Iterable[PretrainBatch]) -> None:
        raise self._blocked("pretrain")

    def encode(self, inputs: "PretrainBatch | ForecastInputs") -> np.ndarray:
        raise self._blocked("encode")

    def save(self, path: str | Path) -> None:
        raise self._blocked("save")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "family": self.family,
            "requires_fit": self.requires_fit,
            "params": dict(self.params),
            "description": (
                "planned unlabelled operational-metric encoder; intentionally NOT implemented"
            ),
            "accepts_labels": False,
            "todo": TODO_RESEARCH_DOC,
        }
