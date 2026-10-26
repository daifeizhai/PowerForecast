"""Frozen foundation-model adapters (A1 Chronos-2, A2 TiRex-2).

Public API
----------
* :class:`Chronos2Forecaster` -- zero-shot Chronos-2, P50 median, ``[B,H]`` kW.
* :class:`TiRex2Forecaster` -- zero-shot TiRex-2, P50 median at quantile index 4,
  documented blockwise recursion for ``H > native_horizon`` (320).
* :func:`resolve_checkpoint` / :class:`CheckpointSnapshot` -- offline-only
  checkpoint resolution with SHA256 evidence; no silent downloads.
* :func:`configure_offline_environment` -- pins Hugging-Face libraries offline.

Both adapters share the same contract: ``predict(context_kw: [B, L] float kW,
horizon: int) -> [B, H] finite float64 kW``, ``requires_fit = False`` and a
``describe()`` JSON manifest block.  Neither adapter accepts future truth, and
missing weights raise instead of creating random/partial state.
"""

from __future__ import annotations

from src.models.foundation.chronos2 import CHRONOS2_FILES, CHRONOS2_MODEL_ID, Chronos2Forecaster
from src.models.foundation.offline import (
    CheckpointSnapshot,
    configure_offline_environment,
    library_versions,
    resolve_checkpoint,
    snapshot_to_dict,
    validate_context,
    validate_prediction,
)
from src.models.foundation.tirex2 import TIRex2_FILES, TIRex2_MODEL_ID, TiRex2Forecaster

__all__ = [
    "CHRONOS2_FILES",
    "CHRONOS2_MODEL_ID",
    "TIRex2_FILES",
    "TIRex2_MODEL_ID",
    "CheckpointSnapshot",
    "Chronos2Forecaster",
    "TiRex2Forecaster",
    "configure_offline_environment",
    "library_versions",
    "resolve_checkpoint",
    "snapshot_to_dict",
    "validate_context",
    "validate_prediction",
]
