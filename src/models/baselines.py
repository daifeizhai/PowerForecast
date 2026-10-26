"""The two real, parameter-free CPU baselines of this rebuild round.

Both models predict **future absolute active power in kW** (never a change
rate), are deterministic, use only numpy, and never touch labels:

``seasonal_naive``
    Take the last ``season_length`` (default 96 = one trace day) valid context
    points as the daily profile and repeat it to cover ``H`` points.

``three_day_mean``
    Average the **same 15-minute slot across the three most recent trace days**
    (the last ``3 * season_length`` valid context points reshaped to
    ``[3, 96]``) into a 96-point profile, then repeat it to cover ``H`` points.
    This is deliberately *not* the constant mean of the last 288 values.

Missing power is never treated as 0 kW: when the required trailing context is
shorter than needed or not fully valid, the window is rejected with
:class:`~src.data.contracts.InsufficientDataError` and a per-sample reason.

``fit`` is a documented no-op returning ``self`` because both baselines have
``requires_fit = False``; skipping ``fit`` is legal.  Small JSON state files
(type / version / params, never a pickle) support save/load and the freeze
lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data.contracts import (
    ContractError,
    ForecastBatch,
    ForecastInputs,
    InsufficientDataError,
)
from src.models.interfaces import MODEL_REGISTRY, ModelStatus
from src.utils import read_json, resolve_path, write_json

__all__ = ["SeasonalNaiveForecaster", "ThreeDayMeanForecaster", "build_forecaster"]

#: Formal output grid: 96 fifteen-minute slots per trace day.
POINTS_PER_DAY = 96

#: Version of the baseline algorithms and of their JSON state files.
BASELINE_VERSION = "1.0.0"


class _ProfileForecaster:
    """Shared profile -> tile/truncate logic and output validation."""

    name: str = ""
    family: str = "baseline"
    status = ModelStatus.IMPLEMENTED
    requires_fit: bool = False
    version: str = BASELINE_VERSION
    #: Number of complete trace days required at the end of the context.
    season_days: int = 1

    def __init__(self, season_length: int = POINTS_PER_DAY) -> None:
        if isinstance(season_length, bool) or not isinstance(season_length, (int, np.integer)):
            raise ContractError(
                f"{type(self).__name__}.season_length must be an integer, got {season_length!r}."
            )
        season_length = int(season_length)
        if season_length < 1:
            raise ContractError("season_length must be positive.")
        self._season_length = season_length
        self.params: dict[str, Any] = {"season_length": season_length}
        self.last_output_diagnostics: dict[str, Any] = {}

    # -- metadata -----------------------------------------------------------
    @property
    def season_length(self) -> int:
        return self._season_length

    @property
    def required_context_points(self) -> int:
        return self._season_length * self.season_days

    def describe(self) -> dict[str, Any]:
        entry = MODEL_REGISTRY[self.name]
        return {
            "name": self.name,
            "status": self.status.value,
            "family": self.family,
            "version": self.version,
            "requires_fit": self.requires_fit,
            "params": dict(self.params),
            "description": entry.description,
            "required_context_points": self.required_context_points,
            "returns": "[B, H] absolute active power in kW",
            "skipping_fit_is_legal": True,
        }

    def state_dict(self) -> dict[str, Any]:
        """Recoverable configuration state (no pickle, no fake weights)."""
        return {
            "model_type": self.name,
            "name": self.name,
            "version": self.version,
            "requires_fit": self.requires_fit,
            "params": dict(self.params),
            "status": self.status.value,
            "profile_rule": self._profile_rule(),
        }

    def _profile_rule(self) -> str:
        return "last_complete_trace_day_same_slot_profile"

    # -- API compatibility ---------------------------------------------------
    def fit(self, batch: ForecastBatch) -> "_ProfileForecaster":
        """Documented no-op: both baselines are parameter-free.

        ``requires_fit=False`` means the caller may skip this method entirely.
        It only verifies that a genuine labelled batch was supplied; it learns
        nothing and never reads ``targets_kw``.
        """
        if not isinstance(batch, ForecastBatch):
            raise ContractError(
                f"{self.name}.fit expects a ForecastBatch; got {type(batch).__name__}. "
                "Unlabelled data must go through pretrain(), never through supervised fit."
            )
        return self

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        if not isinstance(inputs, ForecastInputs):
            raise ContractError(
                f"{self.name}.predict expects ForecastInputs; got {type(inputs).__name__}."
            )
        self._require_context(inputs)
        batch_size = inputs.batch_size
        horizon = inputs.horizon
        output = np.empty((batch_size, horizon), dtype=np.float64)
        for sample in range(batch_size):
            output[sample] = self._tile(self._profile(inputs, sample), horizon)
        self._validate_output(output, batch_size=batch_size, horizon=horizon)
        return output

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        destination = resolve_path(path)
        if destination.is_dir():
            destination = destination / "state.json"
        write_json(destination, self.state_dict())
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "_ProfileForecaster":
        source = resolve_path(path)
        if source.is_dir():
            source = source / "state.json"
        if not source.is_file():
            raise ContractError(f"Model state file not found: {source}")
        payload = read_json(source)
        if not isinstance(payload, Mapping):
            raise ContractError(f"Model state file {source} does not contain a JSON object.")
        model_type = payload.get("model_type") or payload.get("name")
        constructor = _STATE_CONSTRUCTORS.get(model_type)  # type: ignore[arg-type]
        if constructor is None:
            raise ContractError(
                f"Model state file {source} declares unsupported model_type {model_type!r}; "
                f"implemented: {sorted(_STATE_CONSTRUCTORS)}."
            )
        if getattr(cls, "name", "") and model_type != cls.name:
            raise ContractError(
                f"{cls.__name__}.load cannot load state for model_type {model_type!r}."
            )
        recorded_version = payload.get("version", BASELINE_VERSION)
        if recorded_version != BASELINE_VERSION:
            raise ContractError(
                f"Incompatible baseline state version {recorded_version!r}; "
                f"this code implements {BASELINE_VERSION!r}.  Refusing to guess."
            )
        if bool(payload.get("requires_fit", False)) is not False:
            raise ContractError(
                "Baseline state declares requires_fit=True, which contradicts a parameter-free "
                "baseline; the state file is inconsistent."
            )
        params = payload.get("params") or {}
        if not isinstance(params, Mapping):
            raise ContractError("Baseline state 'params' must be a mapping.")
        try:
            return constructor(**dict(params))
        except (TypeError, ContractError) as exc:
            raise ContractError(f"Cannot rebuild baseline from state {source}: {exc}") from exc

    # -- hooks implemented by concrete baselines -----------------------------
    def _profile(self, inputs: ForecastInputs, sample: int) -> np.ndarray:
        raise NotImplementedError

    def _require_context(self, inputs: ForecastInputs) -> None:
        required = self.required_context_points
        length = inputs.context_length
        if length < required:
            details = "; ".join(
                f"sample {sample}: context_length={length} < required {required}"
                for sample in range(inputs.batch_size)
            )
            raise InsufficientDataError(
                f"{self.name} cannot forecast ({details}).  It needs at least {required} "
                "context points so that the last complete trace day(s) can be read without "
                "borrowing future values or treating missing points as 0 kW."
            )
        invalid = ~inputs.past_power_valid[:, -required:]
        counts = invalid.sum(axis=1)
        if bool(counts.any()):
            details = "; ".join(
                f"sample {int(sample)}: {int(counts[sample])}/{required} required trailing "
                "context positions are invalid/missing"
                for sample in np.nonzero(counts)[0]
            )
            raise InsufficientDataError(
                f"{self.name} cannot forecast from incomplete history ({details}).  Missing "
                "power is never filled in or treated as 0 kW; this window is rejected."
            )

    @staticmethod
    def _tile(profile: np.ndarray, horizon: int) -> np.ndarray:
        profile = np.asarray(profile, dtype=np.float64)
        if profile.size < 1:
            raise ContractError("A daily profile cannot be empty.")
        repeats = int(np.ceil(horizon / profile.size))
        return np.tile(profile, repeats)[:horizon]

    def _validate_output(self, output: np.ndarray, *, batch_size: int, horizon: int) -> None:
        expected = (batch_size, horizon)
        if output.shape != expected:
            raise ContractError(
                f"{self.name} produced shape {output.shape}, expected {expected} [B, H]."
            )
        if not bool(np.isfinite(output).all()):
            raise ContractError(f"{self.name} produced non-finite kW predictions.")
        negatives = int(np.count_nonzero(output < 0.0))
        self.last_output_diagnostics = {
            "model": self.name,
            "shape": [int(batch_size), int(horizon)],
            "finite": True,
            "negative_predictions": negatives,
            "clipped": False,
            "negative_policy": "recorded and passed through; optional clipping is the runner's job",
        }


class SeasonalNaiveForecaster(_ProfileForecaster):
    """Previous-day same-slot profile repeated to the forecast horizon."""

    name = "seasonal_naive"
    season_days = 1

    def _profile(self, inputs: ForecastInputs, sample: int) -> np.ndarray:
        window = inputs.past_power_kw[sample, -self._season_length :]
        return np.asarray(window, dtype=np.float64)


class ThreeDayMeanForecaster(_ProfileForecaster):
    """Slot-wise mean over the previous three trace days (not a constant mean)."""

    name = "three_day_mean"
    season_days = 3

    def _profile(self, inputs: ForecastInputs, sample: int) -> np.ndarray:
        window = np.asarray(
            inputs.past_power_kw[sample, -self.required_context_points :], dtype=np.float64
        )
        recent_days = window.reshape(self.season_days, self._season_length)
        return np.mean(recent_days, axis=0)

    def _profile_rule(self) -> str:
        return "slotwise_mean_of_last_three_complete_trace_days"


_STATE_CONSTRUCTORS: dict[str, type[_ProfileForecaster]] = {
    SeasonalNaiveForecaster.name: SeasonalNaiveForecaster,
    ThreeDayMeanForecaster.name: ThreeDayMeanForecaster,
}

_IMPLEMENTED_CONSTRUCTORS: dict[str, type[_ProfileForecaster]] = dict(_STATE_CONSTRUCTORS)


def build_forecaster(name: str, **params: Any) -> _ProfileForecaster:
    """Build an **implemented** baseline by name.

    Planned research interfaces (``lightgbm``, ``power_backbone``,
    ``resource_pretrain``, ``target_mapping``, ``fusion``) are refused with an
    explicit "planned, not implemented" :class:`ContractError`; unknown names
    are refused as unknown.  A planned model is never silently replaced by a
    Naive prediction.
    """
    if not isinstance(name, str) or not name.strip():
        raise ContractError(f"Model name must be a nonempty string, got {name!r}.")
    entry = MODEL_REGISTRY.get(name)
    if entry is None:
        planned = [item for item, item_entry in MODEL_REGISTRY.items() if item_entry.status is ModelStatus.PLANNED]
        raise ContractError(
            f"Unknown model {name!r}.  Implemented models: {list(_IMPLEMENTED_CONSTRUCTORS)}; "
            f"planned research interfaces: {planned}."
        )
    if entry.status is ModelStatus.PLANNED:
        raise ContractError(
            f"Model {name!r} is planned, not implemented: {entry.description}.  "
            "It cannot be selected for a run in this round; see docs/TODO_RESEARCH.md."
        )
    if entry.status is not ModelStatus.IMPLEMENTED:
        raise ContractError(
            f"Model {name!r} is registered as {entry.status.value!r} and cannot be built."
        )
    constructor = _IMPLEMENTED_CONSTRUCTORS.get(name)
    if constructor is None:
        raise ContractError(
            f"Model {name!r} is registered as implemented but has no constructor; "
            "the registry and the implementation are inconsistent."
        )
    try:
        return constructor(**params)
    except TypeError as exc:
        raise ContractError(f"Invalid parameters for model {name!r}: {exc}") from exc
