"""Planned primary/auxiliary fusion (D): interface only, no algorithm chosen.

The fusion method, coefficients, dynamic gating and loss are *unresolved* by
design.  This round must never emit a fused prediction such as ``0 * aux`` or a
fixed-weight blend and call it validated fusion.  Every execution method raises
:class:`NotImplementedError`.

The global fusion weight therefore stays explicitly unresolved (``None``).
Future work is tracked in ``docs/TODO_RESEARCH.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.data.contracts import ForecastInputs
from src.models.interfaces import BaseFusion, ModelStatus

__all__ = [
    "AbstractFusion",
    "PlannedFusion",
    "TODO_RESEARCH_DOC",
    "UNRESOLVED_GLOBAL_FUSION_WEIGHT",
]

TODO_RESEARCH_DOC = "docs/TODO_RESEARCH.md"

#: No global fusion weight has been selected; it must not be invented here.
UNRESOLVED_GLOBAL_FUSION_WEIGHT: None = None


class AbstractFusion(ABC):
    """Abstract base class for a future (still unspecified) fusion layer."""

    name: str = "fusion"
    family: str = "fusion"
    status: ModelStatus = ModelStatus.PLANNED
    requires_fit: bool = True
    global_fusion_weight: None = UNRESOLVED_GLOBAL_FUSION_WEIGHT

    @abstractmethod
    def predict(
        self,
        primary_kw: np.ndarray,
        auxiliary_kw: np.ndarray | None,
        inputs: ForecastInputs,
    ) -> np.ndarray:
        """Return fused ``[B, H]`` absolute kW (to be defined by future research)."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return JSON-serialisable metadata."""


class PlannedFusion(AbstractFusion):
    """Placeholder registered as ``planned``; fusion prediction is refused."""

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.status: ModelStatus = ModelStatus.PLANNED
        self.global_fusion_weight = UNRESOLVED_GLOBAL_FUSION_WEIGHT

    @staticmethod
    def _blocked(operation: str) -> NotImplementedError:
        return NotImplementedError(
            f"fusion is PLANNED, not implemented: '{operation}' cannot run in this round.  The "
            "fusion algorithm, coefficients and gating are unresolved; returning a fixed-weight "
            "blend or 0*auxiliary would fake a validated fusion result.  The global fusion "
            f"weight remains None.  Future work: {TODO_RESEARCH_DOC}."
        )

    def predict(
        self,
        primary_kw: np.ndarray,
        auxiliary_kw: np.ndarray | None,
        inputs: ForecastInputs,
    ) -> np.ndarray:
        raise self._blocked("predict")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "family": self.family,
            "requires_fit": self.requires_fit,
            "params": dict(self.params),
            "description": (
                "planned primary/auxiliary fusion; intentionally NOT implemented and no "
                "algorithm/coefficients selected"
            ),
            "global_fusion_weight": self.global_fusion_weight,
            "todo": TODO_RESEARCH_DOC,
        }
