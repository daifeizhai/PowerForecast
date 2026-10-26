"""Stage-2 main-path model registry: B0, B1, A1, A2 and the A3 ensemble.

:func:`build_power_forecasters` returns a uniform ``dict`` keyed exactly by the
five Stage-2 model names:

===========  ==========================================================
key          implementation
===========  ==========================================================
seasonal_naive  B0 ``SeasonalNaiveForecaster`` (wrapped ``predict_from_context``)
three_day_mean  B1 ``ThreeDayMeanForecaster`` (wrapped ``predict_from_context``)
chronos2        A1 :class:`~src.models.foundation.chronos2.Chronos2Forecaster`
tirex2          A2 :class:`~src.models.foundation.tirex2.TiRex2Forecaster`
power_ensemble  A3 :class:`~src.models.power_main.ensemble.FixedWeightedEnsemble`
===========  ==========================================================

Every returned object exposes ``name``, ``requires_fit = False``,
``predict(context_kw: [B, L], horizon: int) -> [B, H]`` and ``describe()``.

The builder accepts either a *foundation* configuration
(``configs/models/foundation.yaml``) or an *experiment* configuration that points
at it through ``models.config`` (as ``configs/experiments/power_main_v1.yaml``
does), and it never loads model weights: foundation checkpoint files are resolved
and SHA256-hashed eagerly (so missing weights fail fast), while the heavy
libraries and tensors are loaded lazily on the first ``predict``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.models.baselines import SeasonalNaiveForecaster, ThreeDayMeanForecaster
from src.models.foundation.chronos2 import (
    CHRONOS2_FILES,
    CHRONOS2_MODEL_ID,
    Chronos2Forecaster,
)
from src.models.foundation.offline import CheckpointSnapshot, resolve_checkpoint
from src.models.foundation.tirex2 import TIRex2_FILES, TIRex2_MODEL_ID, TiRex2Forecaster
from src.models.power_main.ensemble import (
    ENSEMBLE_METHOD,
    FIXED_WEIGHTS,
    EnsembleMemberError,
    FixedWeightedEnsemble,
    build_power_ensemble,
)
from src.utils import load_config

__all__ = [
    "ALL_MODELS",
    "BASELINE_MODELS",
    "FOUNDATION_MODELS",
    "CheckpointSnapshot",
    "Chronos2Forecaster",
    "EnsembleMemberError",
    "FixedWeightedEnsemble",
    "TiRex2Forecaster",
    "build_power_ensemble",
    "build_power_forecasters",
    "resolve_checkpoint",
]

#: Canonical reporting order; a full build returns exactly these five keys.
ALL_MODELS: tuple[str, ...] = (
    "seasonal_naive",
    "three_day_mean",
    "chronos2",
    "tirex2",
    "power_ensemble",
)
BASELINE_MODELS: tuple[str, ...] = ("seasonal_naive", "three_day_mean")
FOUNDATION_MODELS: tuple[str, ...] = ("chronos2", "tirex2")


class _BaselineContextForecaster:
    """Stage-2 wrapper exposing ``predict_from_context`` as ``predict``.

    The wrapped baseline is the unchanged skeleton implementation; the wrapper only
    adapts the input shape, so B0/B1 produce exactly the same numbers as the
    official baseline models on the same context.
    """

    family = "baseline"
    requires_fit = False

    def __init__(self, forecaster: SeasonalNaiveForecaster | ThreeDayMeanForecaster) -> None:
        self._forecaster = forecaster
        self.name = forecaster.name
        self.version = forecaster.version

    def predict(self, context_kw: Any, horizon: int) -> Any:
        """Predict ``[B, H]`` kW from a raw ``[B, L]`` finite kW context."""
        return self._forecaster.predict_from_context(context_kw, horizon)

    def describe(self) -> dict[str, Any]:
        payload = self._forecaster.describe()
        payload.update(
            {
                "stage2_adapter": "_BaselineContextForecaster",
                "stage2_input": "[B, L] finite historical kW context (no mask, no labels)",
                "stage2_entry_point": "predict_from_context",
            }
        )
        return payload


# ---------------------------------------------------------------------------
# configuration helpers
# ---------------------------------------------------------------------------


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _resolve_foundation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the foundation-model configuration (direct or via ``models.config``)."""
    if not isinstance(config, Mapping):
        raise ValueError(
            f"build_power_forecasters config must be a mapping, got {type(config).__name__}."
        )
    models_section = config.get("models")
    if isinstance(models_section, Mapping) and "config" in models_section:
        path = models_section["config"]
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError(
                f"models.config must be a nonempty config path, got {path!r}."
            )
        try:
            payload = load_config(path)
        except OSError as exc:
            raise ValueError(
                f"Cannot load the foundation-model configuration {path!r}: {exc}"
            ) from exc
        return _as_mapping(payload, name=f"foundation config {path!r}")
    return dict(config)


def _pick(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _model_spec(foundation: Mapping[str, Any], name: str) -> dict[str, Any]:
    specs = foundation.get("models")
    if not isinstance(specs, Mapping):
        return {}
    spec = specs.get(name)
    return dict(spec) if isinstance(spec, Mapping) else {}


def _resolve_weights_cache(
    config: Mapping[str, Any],
    foundation: Mapping[str, Any],
    override: str | Path | None,
) -> str | Path:
    models_section = config.get("models") if isinstance(config.get("models"), Mapping) else {}
    defaults = foundation.get("defaults") if isinstance(foundation.get("defaults"), Mapping) else {}
    value = _pick(
        override,
        models_section.get("weights_cache"),
        foundation.get("weights_cache"),
        defaults.get("weights_cache") if isinstance(defaults, Mapping) else None,
    )
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(
            "No weights cache configured. Set models.weights_cache in the experiment config, "
            "or weights_cache in configs/models/foundation.yaml, or pass weights_cache=..."
        )
    return value


def _resolve_optional_positive_int(value: Any, *, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int,)):
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _validate_ensemble_config(foundation: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Reject any non-(0.5, 0.5) ensemble configuration (no weight search)."""
    ensemble_cfg: dict[str, Any] = {}
    foundation_ensemble = foundation.get("ensemble")
    if isinstance(foundation_ensemble, Mapping):
        ensemble_cfg.update(foundation_ensemble)
    experiment_ensemble = config.get("ensemble")
    if isinstance(experiment_ensemble, Mapping):
        ensemble_cfg.update(experiment_ensemble)
    method = ensemble_cfg.get("method", ENSEMBLE_METHOD)
    if method != ENSEMBLE_METHOD:
        raise ValueError(
            f"ensemble.method must be {ENSEMBLE_METHOD!r} with fixed weights {FIXED_WEIGHTS}, "
            f"got {method!r}. No other fusion/weighting method is implemented."
        )
    weights = (
        float(ensemble_cfg.get("chronos2_weight", FIXED_WEIGHTS[0])),
        float(ensemble_cfg.get("tirex2_weight", FIXED_WEIGHTS[1])),
    )
    if any(abs(weight - fixed) > 1e-12 for weight, fixed in zip(weights, FIXED_WEIGHTS)):
        raise ValueError(
            f"ensemble weights {weights} differ from the frozen 0.5/0.5 main-path policy "
            f"{FIXED_WEIGHTS}. Weights are not searched, tuned or data-dependent."
        )


def _selected_models(
    config: Mapping[str, Any],
    foundation: Mapping[str, Any],
    models: Sequence[str] | None,
) -> list[str]:
    experiment_models = config.get("models") if isinstance(config.get("models"), Mapping) else {}
    requested = _pick(
        list(models) if models is not None else None,
        experiment_models.get("members"),
        foundation.get("members"),
        list(ALL_MODELS),
    )
    if isinstance(requested, (str, bytes)):
        raise ValueError(
            f"models must be a sequence of model names, got a single string {requested!r}."
        )
    try:
        requested_names = list(requested)
    except TypeError as exc:
        raise ValueError(f"models must be iterable, got {requested!r}.") from exc
    unknown = [name for name in requested_names if name not in ALL_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown Stage-2 model name(s) {unknown}; supported models are {list(ALL_MODELS)}."
        )
    if not requested_names:
        raise ValueError("models must request at least one model.")
    selected = [name for name in ALL_MODELS if name in set(requested_names)]
    if "power_ensemble" in selected:
        # A3 needs both foundation members to exist.
        for name in FOUNDATION_MODELS:
            if name not in selected:
                selected.append(name)
        selected = [name for name in ALL_MODELS if name in set(selected)]
    return selected


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def build_power_forecasters(
    config: Mapping[str, Any],
    *,
    weights_cache: str | Path | None = None,
    models: Sequence[str] | None = None,
    device: str | None = None,
    dtype: str | None = None,
    jobs: int | None = None,
    batch_size: int | None = None,
    allow_download: bool | None = None,
) -> dict[str, object]:
    """Build the Stage-2 forecasters without loading any model weights.

    Parameters
    ----------
    config:
        Either a foundation configuration (``configs/models/foundation.yaml``) or an
        experiment configuration whose ``models.config`` points at one.  Explicit
        keyword arguments override configuration values.
    weights_cache:
        Local ``HF_HUB_CACHE``-layout weights directory (required unless configured).
    models:
        Optional subset of ``("seasonal_naive", "three_day_mean", "chronos2",
        "tirex2", "power_ensemble")``.  Requesting ``power_ensemble`` also builds
        its two members.  Defaults to every configured member.
    device, dtype, jobs, batch_size:
        Runtime overrides for the foundation adapters (CPU / float32 verified).
    allow_download:
        Must stay ``False`` for inference runs; weight downloads belong to a
        separate explicit preparation step.

    Returns
    -------
    dict[str, object]
        ``{name: forecaster}`` in canonical order; every object exposes ``name``,
        ``requires_fit = False``, ``predict(context_kw, horizon)`` and
        ``describe()``.  Foundation weights are loaded lazily on first ``predict``.
    """
    foundation = _resolve_foundation_config(config)
    experiment_models = config.get("models") if isinstance(config.get("models"), Mapping) else {}
    experiment_runtime = config.get("runtime") if isinstance(config.get("runtime"), Mapping) else {}

    resolved_cache = _resolve_weights_cache(config, foundation, weights_cache)
    resolved_device = str(
        _pick(device, experiment_runtime.get("device"), foundation.get("device"), "cpu")
    )
    resolved_dtype = str(_pick(dtype, foundation.get("dtype"), "float32"))
    if resolved_dtype not in {"float32", "float64"}:
        raise ValueError(
            f"dtype={resolved_dtype!r} is not supported by the Stage-2 CPU adapters; the verified "
            "dtypes are 'float32' (default) and 'float64'. TiRex-2 requires float32/float64."
        )
    resolved_jobs = _resolve_optional_positive_int(
        _pick(jobs, experiment_runtime.get("jobs"), foundation.get("jobs")), name="jobs", default=4
    )
    resolved_batch_size = _resolve_optional_positive_int(
        _pick(batch_size, experiment_runtime.get("batch_size"), foundation.get("batch_size")),
        name="batch_size",
        default=1,
    )
    resolved_allow_download = bool(
        _pick(
            allow_download,
            experiment_models.get("allow_download"),
            foundation.get("allow_download"),
            False,
        )
    )
    selected = _selected_models(config, foundation, models)
    if "power_ensemble" in selected:
        _validate_ensemble_config(foundation, config)

    chronos_spec = _model_spec(foundation, "chronos2")
    tirex_spec = _model_spec(foundation, "tirex2")
    for name, spec, expected_id, expected_files in (
        ("chronos2", chronos_spec, CHRONOS2_MODEL_ID, CHRONOS2_FILES),
        ("tirex2", tirex_spec, TIRex2_MODEL_ID, TIRex2_FILES),
    ):
        declared_id = spec.get("model_id", expected_id)
        if declared_id != expected_id:
            raise ValueError(
                f"models.{name}.model_id={declared_id!r} does not match the implemented adapter "
                f"({expected_id!r}); substituting another checkpoint is not supported."
            )
        declared_files = spec.get("checkpoint_filenames")
        if declared_files is not None and tuple(declared_files) != tuple(expected_files):
            raise ValueError(
                f"models.{name}.checkpoint_filenames={declared_files!r} does not match the "
                f"implemented adapter files {list(expected_files)}."
            )

    tirex_recursion = tirex_spec.get("recursion") if isinstance(tirex_spec.get("recursion"), Mapping) else {}
    top_recursion = foundation.get("recursion") if isinstance(foundation.get("recursion"), Mapping) else {}
    tirex_max_horizon = _pick(
        tirex_recursion.get("max_horizon"), top_recursion.get("max_horizon")
    )
    if tirex_max_horizon is not None:
        tirex_max_horizon = _resolve_optional_positive_int(
            tirex_max_horizon, name="max_horizon", default=320
        )

    built: dict[str, object] = {}
    if "seasonal_naive" in selected:
        built["seasonal_naive"] = _BaselineContextForecaster(SeasonalNaiveForecaster())
    if "three_day_mean" in selected:
        built["three_day_mean"] = _BaselineContextForecaster(ThreeDayMeanForecaster())
    if "chronos2" in selected:
        built["chronos2"] = Chronos2Forecaster(
            weights_cache=resolved_cache,
            device=resolved_device,
            dtype=resolved_dtype,
            batch_size=resolved_batch_size,
            jobs=resolved_jobs,
            allow_download=resolved_allow_download,
        )
    if "tirex2" in selected:
        built["tirex2"] = TiRex2Forecaster(
            weights_cache=resolved_cache,
            device=resolved_device,
            dtype=resolved_dtype,
            batch_size=resolved_batch_size,
            jobs=resolved_jobs,
            allow_download=resolved_allow_download,
            max_horizon=tirex_max_horizon,
        )
    if "power_ensemble" in selected:
        built["power_ensemble"] = build_power_ensemble(
            {"chronos2": built["chronos2"], "tirex2": built["tirex2"]}
        )
    return {name: built[name] for name in selected if name in built}
