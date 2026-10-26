"""Model layer of the rebuilt PowerForecast skeleton.

Public API
----------
* :func:`build_forecaster` builds **implemented** models only.
* :data:`MODEL_REGISTRY`, :func:`implemented_models`, :func:`planned_models` and
  :func:`describe_models` are the single source of truth about support status.
* :class:`SeasonalNaiveForecaster` and :class:`ThreeDayMeanForecaster` are the
  only real models this round.
* :class:`BasePowerForecaster`, :class:`BaseResourceEncoder`,
  :class:`BaseTargetMapper` and :class:`BaseFusion` are the protocols future
  research modules must satisfy.

Research interfaces (``power_backbone``, ``resource_pretrain``,
``target_mapping``, ``fusion``, ``lightgbm``) are registered as ``planned`` and
must never be reported as supported or trained.
"""

from __future__ import annotations

from src.models.interfaces import (
    MODEL_REGISTRY,
    BaseFusion,
    BasePowerForecaster,
    BaseResourceEncoder,
    BaseTargetMapper,
    ModelEntry,
    ModelStatus,
    describe_models,
    implemented_models,
    load_model_config,
    planned_models,
)

# Imported after the interfaces so that circular imports stay impossible.
from src.models.baselines import (
    SeasonalNaiveForecaster,
    ThreeDayMeanForecaster,
    build_forecaster,
)

__all__ = [
    "MODEL_REGISTRY",
    "BaseFusion",
    "BasePowerForecaster",
    "BaseResourceEncoder",
    "BaseTargetMapper",
    "ModelEntry",
    "ModelStatus",
    "SeasonalNaiveForecaster",
    "ThreeDayMeanForecaster",
    "build_forecaster",
    "describe_models",
    "implemented_models",
    "load_model_config",
    "planned_models",
]
