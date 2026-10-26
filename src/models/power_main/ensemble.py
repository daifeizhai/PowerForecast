"""Model A3: the frozen 0.5/0.5 equal-weight ensemble of the two foundation models.

Scope and honesty boundary
--------------------------
This ensemble is the **main-path** (A) combination of Chronos-2 (A1) and TiRex-2
(A2).  Its ``(0.5, 0.5)`` coefficients are **internal main-path coefficients**;
they are *not* a main/auxiliary fusion weight and they say nothing about the
unimplemented research fusion path (``src/models/fusion.py``).  The weight pair is
fixed a priori: the constructor accepts **only** ``(0.5, 0.5)`` and there is no
weight-search, tuning or development-set optimisation API.

Guarantees
----------
* ``predict(context_kw, horizon)`` returns the elementwise weighted mean
  ``[B, H]`` finite float64 kW of the member forecasts;
* every member output is validated for shape ``[B, H]`` and finiteness, and any
  member failure is re-raised as :class:`EnsembleMemberError` naming the member;
* the requested horizon is always returned in full -- member forecasts are never
  truncated, padded or repeated;
* ``describe()`` records the exact fixed weights and each member's descriptor
  (checkpoint path/SHA256 included) for the run manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.models.foundation.offline import validate_context, validate_prediction

__all__ = [
    "ENSEMBLE_METHOD",
    "FIXED_WEIGHTS",
    "EnsembleMemberError",
    "FixedWeightedEnsemble",
    "build_power_ensemble",
]

#: The only supported fusion method in the Stage-2 main path.
ENSEMBLE_METHOD = "fixed_weighted_mean"

#: The frozen a-priori equal weights; no search is implemented or allowed.
FIXED_WEIGHTS: tuple[float, float] = (0.5, 0.5)

ENSEMBLE_VERSION = "1.0.0"

#: Canonical member order of model A3.
MEMBER_ORDER: tuple[str, ...] = ("chronos2", "tirex2")


class EnsembleMemberError(RuntimeError):
    """A member forecaster failed or violated the ``[B, H]`` finite-kW contract."""


def _member_label(member: Any, position: int) -> str:
    name = getattr(member, "name", None)
    return f"{position}:{name if isinstance(name, str) and name else type(member).__name__}"


@dataclass(frozen=True, eq=False)
class FixedWeightedEnsemble:
    """Fixed ``(0.5, 0.5)`` weighted mean of Chronos-2 and TiRex-2 forecasts.

    Parameters
    ----------
    members:
        Exactly two forecasters exposing ``predict(context_kw, horizon)``.
    weights:
        Must be exactly ``(0.5, 0.5)``.  Any other weights -- including negative
        weights, weights that do not sum to one, or searched/tuned weights -- raise
        :class:`ValueError`: the fixed equal-weight policy is part of the frozen
        research design, not a tunable hyper-parameter.
    """

    members: tuple[Any, ...]
    weights: tuple[float, ...]
    name: str = "power_ensemble"
    requires_fit: bool = False
    family: str = "ensemble"
    version: str = ENSEMBLE_VERSION
    method: str = ENSEMBLE_METHOD
    #: Diagnostics of the last predict() call (mutable, excluded from identity).
    last_output_diagnostics: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        members = tuple(self.members)
        object.__setattr__(self, "members", members)
        try:
            weights = tuple(float(weight) for weight in self.weights)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.name} weights must be a pair of numbers, got {self.weights!r}."
            ) from exc
        if len(members) != 2:
            raise ValueError(
                f"{self.name} needs exactly 2 member forecasters (A1 Chronos-2, A2 TiRex-2), "
                f"got {len(members)}."
            )
        if len(weights) != 2:
            raise ValueError(
                f"{self.name} accepts exactly the fixed weight pair (0.5, 0.5), got "
                f"{len(weights)} weights {weights}."
            )
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"{self.name} weights must be nonnegative, got {weights}.")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError(
                f"{self.name} weights must sum to 1.0, got {weights} (sum={sum(weights)!r})."
            )
        if weights != FIXED_WEIGHTS:
            raise ValueError(
                f"{self.name} only supports the frozen, a-priori equal weights "
                f"{FIXED_WEIGHTS}, got {weights}. This is not a tunable parameter: no weight "
                "search, development-set tuning or data-dependent reweighting is implemented."
            )
        if self.method != ENSEMBLE_METHOD:
            raise ValueError(
                f"{self.name} only supports method {ENSEMBLE_METHOD!r}, got {self.method!r}."
            )
        for position, member in enumerate(members):
            label = _member_label(member, position)
            if not callable(getattr(member, "predict", None)):
                raise ValueError(f"{self.name} member {label!r} has no callable predict().")
            if bool(getattr(member, "requires_fit", False)):
                raise ValueError(
                    f"{self.name} member {label!r} requires fit(); model A3 is zero-shot and "
                    "never fits on target-domain labels."
                )
        if members[0] is members[1]:
            raise ValueError(
                f"{self.name} members must be two distinct models (Chronos-2 and TiRex-2); "
                "the same object was passed twice."
            )
        object.__setattr__(self, "weights", weights)

    # -- inference -----------------------------------------------------------
    def predict(self, context_kw: np.ndarray, horizon: int) -> np.ndarray:
        """Fixed 0.5/0.5 weighted mean of the member forecasts, ``[B, H]`` kW."""
        context, horizon = validate_context(context_kw, horizon, model_name=self.name)
        batch, _context_length = context.shape
        weighted_sum = np.zeros((batch, horizon), dtype=np.float64)
        for position, (member, weight) in enumerate(zip(self.members, self.weights)):
            label = _member_label(member, position)
            try:
                raw = member.predict(context, horizon)
            except Exception as exc:  # noqa: BLE001 - re-raised with member identity
                raise EnsembleMemberError(
                    f"{self.name}: member {label!r} failed during predict: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            try:
                forecast = validate_prediction(
                    f"{self.name} member {label!r}",
                    raw,
                    batch_size=batch,
                    horizon=horizon,
                )
            except RuntimeError as exc:
                raise EnsembleMemberError(
                    f"{self.name}: member {label!r} returned an invalid forecast. {exc}"
                ) from exc
            weighted_sum += weight * forecast
        result = validate_prediction(self.name, weighted_sum, batch_size=batch, horizon=horizon)
        self.last_output_diagnostics.clear()
        self.last_output_diagnostics.update(
            {
                "model": self.name,
                "shape": [int(batch), int(horizon)],
                "weights": [float(weight) for weight in self.weights],
                "members": [
                    _member_label(member, position)
                    for position, member in enumerate(self.members)
                ],
                "finite": True,
            }
        )
        return result

    # -- metadata ------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """JSON-safe description with the exact fixed weights and member summaries."""
        member_descriptions: list[dict[str, Any]] = []
        for position, member in enumerate(self.members):
            label = _member_label(member, position)
            describe = getattr(member, "describe", None)
            if callable(describe):
                try:
                    payload = describe()
                except Exception:  # pragma: no cover - description must not break a run
                    payload = {"name": getattr(member, "name", type(member).__name__)}
            else:  # pragma: no cover - adapters always expose describe()
                payload = {"name": getattr(member, "name", type(member).__name__)}
            member_descriptions.append(
                {
                    "position": int(position),
                    "label": label,
                    "weight": float(self.weights[position]),
                    "description": payload,
                }
            )
        return {
            "name": self.name,
            "family": self.family,
            "version": self.version,
            "requires_fit": self.requires_fit,
            "method": self.method,
            "weights": [float(weight) for weight in self.weights],
            "member_order": [str(member.get("label", "")) for member in member_descriptions],
            "member_weights": {
                str(member.get("label", "")): float(self.weights[position])
                for position, member in enumerate(member_descriptions)
            },
            "members": member_descriptions,
            "weight_scope": (
                "INTERNAL main-path coefficients between the two foundation forecasters "
                "(A1 Chronos-2 + A2 TiRex-2). They are NOT a main/auxiliary (primary + auxiliary) "
                "fusion weight; the research fusion path remains unimplemented."
            ),
            "fixed_weight_policy": {
                "weights": [float(weight) for weight in FIXED_WEIGHTS],
                "search_allowed": False,
                "tuning_api": "none",
                "source": "frozen a-priori equal-weight design (task book §4/§6), not fitted",
            },
            "output": "[B, H] finite float64 absolute active power in kW",
            "last_output_diagnostics": dict(getattr(self, "last_output_diagnostics", {})),
        }


def build_power_ensemble(
    forecasters: Mapping[str, object],
    *,
    method: str = ENSEMBLE_METHOD,
) -> FixedWeightedEnsemble:
    """Build model A3 from the ``chronos2`` / ``tirex2`` forecasters.

    Only the fixed ``fixed_weighted_mean`` method with weights ``(0.5, 0.5)`` is
    supported.  Unknown methods, missing/extra member keys and substituted models
    raise :class:`ValueError`.
    """
    if not isinstance(forecasters, Mapping):
        raise ValueError(
            f"build_power_ensemble needs a mapping of member name -> forecaster, got "
            f"{type(forecasters).__name__}."
        )
    if method != ENSEMBLE_METHOD:
        raise ValueError(
            f"build_power_ensemble only supports {ENSEMBLE_METHOD!r} with fixed weights "
            f"{FIXED_WEIGHTS}, got method={method!r}. No other fusion/weighting method is "
            "implemented in the Stage-2 main path."
        )
    missing = [key for key in MEMBER_ORDER if key not in forecasters]
    extra = [key for key in forecasters if key not in MEMBER_ORDER]
    if missing or extra:
        raise ValueError(
            f"build_power_ensemble needs exactly the members {list(MEMBER_ORDER)}; "
            f"missing={missing}, unexpected={extra}."
        )
    members = tuple(forecasters[key] for key in MEMBER_ORDER)
    for key, member in zip(MEMBER_ORDER, members):
        declared = getattr(member, "name", None)
        if declared != key:
            raise ValueError(
                f"build_power_ensemble member {key!r} must be a forecaster named {key!r}, got "
                f"name={declared!r}; impossible to substitute another model's output."
            )
    return FixedWeightedEnsemble(members=members, weights=FIXED_WEIGHTS)
