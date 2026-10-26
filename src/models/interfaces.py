"""Public model interfaces and the single source of model-status truth.

Scope of this rebuild round
---------------------------
* ``seasonal_naive`` and ``three_day_mean`` are *real*, parameter-free CPU
  baselines implemented in :mod:`src.models.baselines`.
* ``power_backbone``, ``resource_pretrain``, ``target_mapping`` and ``fusion``
  are planned research interfaces only.  They are registered with
  :attr:`ModelStatus.PLANNED`; their concrete placeholders raise
  ``NotImplementedError`` and no fake weights, fake pretraining or fake fusion
  prediction is ever produced.
* ``lightgbm`` is registered as *planned* only.  This round must never return a
  Naive prediction dressed up as LightGBM.

:data:`MODEL_REGISTRY` is the single place that decides whether a model name may
be built.  :func:`implemented_models` is the authoritative list of supported
models; :func:`planned_models` lists research interfaces that must not be
reported as supported.

The registry can be cross-checked against ``configs/models/baselines.yaml`` with
:func:`load_model_config`; a planned entry can never be validated as supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

import numpy as np

from src.data.contracts import (
    ContractError,
    ForecastBatch,
    ForecastInputs,
    PretrainBatch,
)

__all__ = [
    "MODEL_REGISTRY",
    "BaseFusion",
    "BasePowerForecaster",
    "BaseResourceEncoder",
    "BaseTargetMapper",
    "ModelEntry",
    "ModelStatus",
    "describe_models",
    "implemented_models",
    "load_model_config",
    "planned_models",
]


# ---------------------------------------------------------------------------
# status and public protocols
# ---------------------------------------------------------------------------


class ModelStatus(str, Enum):
    """Whether a registered model can actually be executed this round."""

    IMPLEMENTED = "implemented"
    PLANNED = "planned"
    UNAVAILABLE = "unavailable"


@runtime_checkable
class BasePowerForecaster(Protocol):
    """Main power path (A): history in kW -> absolute future kW ``[B, H]``.

    ``fit`` and ``evaluate`` may read labels; ``predict`` receives only
    :class:`~src.data.contracts.ForecastInputs`, which structurally carries no
    future labels.
    """

    name: str
    requires_fit: bool

    def fit(self, batch: ForecastBatch) -> None:
        """Supervised entry point; parameter-free models may document a no-op."""

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        """Return predicted absolute active power in kW, shape ``[B, H]``."""

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable metadata: name, status, parameters, requirements."""


@runtime_checkable
class BaseResourceEncoder(Protocol):
    """Unlabelled multi-source representation pretraining (B).  Interface only."""

    def pretrain(self, batches: Iterable[PretrainBatch]) -> None:
        """Pretrain from unlabelled batches; no kW label may ever be seen."""

    def encode(self, inputs: "PretrainBatch | ForecastInputs") -> np.ndarray:
        """Encode observed metrics into a representation array."""

    def save(self, path: str | Path) -> None:
        """Persist encoder state (a planned model must not pretend to have any)."""


@runtime_checkable
class BaseTargetMapper(Protocol):
    """Supervised target-domain power mapping (C).  Interface only."""

    def fit(self, batch: ForecastBatch, historic_repr: np.ndarray) -> None:
        """Fit the mapping using origin-safe historical representations only."""

    def predict_aux(self, inputs: ForecastInputs, historic_repr: np.ndarray) -> np.ndarray:
        """Return an auxiliary kW candidate for the forecast horizon."""


@runtime_checkable
class BaseFusion(Protocol):
    """Primary + auxiliary fusion (D).  Interface only; no algorithm is chosen."""

    def predict(
        self,
        primary_kw: np.ndarray,
        auxiliary_kw: np.ndarray | None,
        inputs: ForecastInputs,
    ) -> np.ndarray:
        """Return fused absolute kW.  Must not be a fixed/zero-weight fake."""


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelEntry:
    """Immutable description of one model name and its implementation status."""

    name: str
    status: ModelStatus
    requires_fit: bool
    family: str
    description: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def implemented(self) -> bool:
        return self.status is ModelStatus.IMPLEMENTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "requires_fit": self.requires_fit,
            "family": self.family,
            "description": self.description,
            "params": dict(self.params),
        }


#: The only registry the rest of the code base may consult.  Insertion order is
#: the reporting order; implemented entries come first.
MODEL_REGISTRY: dict[str, ModelEntry] = {
    "seasonal_naive": ModelEntry(
        name="seasonal_naive",
        status=ModelStatus.IMPLEMENTED,
        requires_fit=False,
        family="baseline",
        description="previous-day same-slot profile (96 points)",
        params={"season_length": 96},
    ),
    "three_day_mean": ModelEntry(
        name="three_day_mean",
        status=ModelStatus.IMPLEMENTED,
        requires_fit=False,
        family="baseline",
        description="slot-wise mean over the previous three trace days",
        params={"season_length": 96},
    ),
    "lightgbm": ModelEntry(
        name="lightgbm",
        status=ModelStatus.PLANNED,
        requires_fit=True,
        family="baseline",
        description="planned gradient boosting baseline; NOT implemented",
        params={"train_cutoff": None},
    ),
    "power_backbone": ModelEntry(
        name="power_backbone",
        status=ModelStatus.PLANNED,
        requires_fit=True,
        family="backbone",
        description="planned learned main power path; interface only",
        params={},
    ),
    "resource_pretrain": ModelEntry(
        name="resource_pretrain",
        status=ModelStatus.PLANNED,
        requires_fit=True,
        family="resource_pretrain",
        description="planned unlabelled operational-metric encoder; interface only",
        params={},
    ),
    "target_mapping": ModelEntry(
        name="target_mapping",
        status=ModelStatus.PLANNED,
        requires_fit=True,
        family="target_mapping",
        description="planned supervised target-domain power mapping; interface only",
        params={},
    ),
    "fusion": ModelEntry(
        name="fusion",
        status=ModelStatus.PLANNED,
        requires_fit=True,
        family="fusion",
        description="planned primary/auxiliary fusion; no algorithm selected",
        params={"global_fusion_weight": None},
    ),
}


def implemented_models() -> tuple[str, ...]:
    """Names of models that can actually run this round (authoritative list)."""
    return tuple(
        name for name, entry in MODEL_REGISTRY.items() if entry.status is ModelStatus.IMPLEMENTED
    )


def planned_models() -> tuple[str, ...]:
    """Names of research interfaces registered as planned (never supported)."""
    return tuple(
        name for name, entry in MODEL_REGISTRY.items() if entry.status is ModelStatus.PLANNED
    )


def describe_models() -> dict[str, dict[str, Any]]:
    """JSON-safe registry description keyed by model name."""
    return {name: entry.to_dict() for name, entry in MODEL_REGISTRY.items()}


# ---------------------------------------------------------------------------
# configuration cross-check
# ---------------------------------------------------------------------------


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate ``configs/models/baselines.yaml`` against the registry.

    Validation is intentionally strict about *support status*:

    * every configured model name must exist in :data:`MODEL_REGISTRY`;
    * the declared ``status`` / ``requires_fit`` must equal the registry entry;
    * ``default_model`` must be an implemented model.

    Therefore a configuration file can never silently list a planned research
    interface as if it were a supported model.
    """
    from src.utils import load_config

    source = "configs/models/baselines.yaml" if path is None else str(path)
    payload = load_config(source)
    return _validate_model_config(payload, source=source)


def _validate_model_config(payload: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ContractError(f"{source}: model configuration must be a mapping.")
    models = payload.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ContractError(f"{source}: 'models' must be a nonempty mapping of model name -> spec.")
    for name, spec in models.items():
        if not isinstance(name, str) or name not in MODEL_REGISTRY:
            raise ContractError(
                f"{source}: model {name!r} is not registered in MODEL_REGISTRY; "
                "register it with an explicit ModelStatus before configuring it."
            )
        if not isinstance(spec, Mapping):
            raise ContractError(f"{source}: model {name!r} must map to a mapping.")
        entry = MODEL_REGISTRY[name]
        declared_status = spec.get("status")
        if declared_status != entry.status.value:
            raise ContractError(
                f"{source}: model {name!r} declares status {declared_status!r} but the registry "
                f"says {entry.status.value!r}.  Planned models cannot be configured as supported."
            )
        declared_fit = spec.get("requires_fit", entry.requires_fit)
        if bool(declared_fit) != bool(entry.requires_fit):
            raise ContractError(
                f"{source}: model {name!r} declares requires_fit={declared_fit!r} but the "
                f"registry says {entry.requires_fit!r}."
            )
    default_model = payload.get("default_model")
    if not isinstance(default_model, str) or not default_model:
        raise ContractError(f"{source}: 'default_model' must be a nonempty string.")
    if default_model not in models:
        raise ContractError(
            f"{source}: default_model {default_model!r} is not present in 'models'."
        )
    default_entry = MODEL_REGISTRY[default_model]
    if default_entry.status is not ModelStatus.IMPLEMENTED:
        raise ContractError(
            f"{source}: default_model {default_model!r} must be implemented, not "
            f"{default_entry.status.value!r}."
        )
    return dict(payload)
