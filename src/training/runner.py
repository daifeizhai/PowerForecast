"""Shared prediction runner: batched inference plus output validation.

The runner never hides a diverged model.  Shape, dtype and finiteness are
validated per chunk; optional non-negative clipping is explicit, recorded in the
returned diagnostics, and never silent.  ``0 * auxiliary`` style fusion does not
happen here either -- this module only runs a single :class:`BasePowerForecaster`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.contracts import ContractError, ForecastInputs
from src.models.interfaces import BasePowerForecaster

__all__ = ["predict_batch"]


def _validate_batch_size(batch_size: Any) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise ValueError(f"batch_size must be an integer, got {batch_size!r}.")
    size = int(batch_size)
    if size < 1:
        raise ValueError(f"batch_size must be positive, got {size}.")
    return size


def _validate_model_output(
    raw: Any,
    *,
    model_name: str,
    chunk_size: int,
    horizon: int,
) -> np.ndarray:
    array = np.asarray(raw)
    if array.dtype.kind != "f":
        raise ContractError(
            f"model {model_name!r} predict() must return a floating-point array; got dtype "
            f"{array.dtype!s}."
        )
    if array.ndim != 2:
        raise ContractError(
            f"model {model_name!r} predict() must return a 2-D [B, H] array; got "
            f"{array.ndim}-D shape {array.shape}."
        )
    expected = (chunk_size, horizon)
    if array.shape != expected:
        raise ContractError(
            f"model {model_name!r} predict() returned shape {array.shape}, expected {expected} "
            "[B, H]."
        )
    array = array.astype(np.float64, copy=False)
    if not bool(np.isfinite(array).all()):
        raise ValueError(
            f"model {model_name!r} returned non-finite kW predictions (NaN/inf).  A diverged "
            "model must be fixed or rejected, never hidden by silent clipping."
        )
    return array


def predict_batch(
    model: BasePowerForecaster,
    inputs: ForecastInputs,
    *,
    batch_size: int = 64,
    clip_min: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run ``model.predict`` in chunks and validate the assembled output.

    Parameters
    ----------
    model:
        Any implemented :class:`~src.models.interfaces.BasePowerForecaster`.
    inputs:
        Labelled-input container without future labels.
    batch_size:
        Positive chunk size used to slice ``inputs``; results are concatenated in
        order, so chunking never changes the prediction.
    clip_min:
        Optional lower bound.  When given, the number of clipped values is
        recorded in ``diagnostics["n_clipped"]`` and
        ``diagnostics["clipped"]`` is ``True``.  ``None`` means no clipping.

    Returns
    -------
    ``(predictions, diagnostics)`` where ``predictions`` is ``[B, H]`` float
    kW and diagnostics contains ``clipped``, ``clip_min``, ``n_clipped``,
    ``model``, ``requires_fit`` and (non-negative policy record)
    ``negative_predictions``.
    """
    if not isinstance(inputs, ForecastInputs):
        raise ContractError(
            f"predict_batch expects ForecastInputs; got {type(inputs).__name__}."
        )
    if not hasattr(model, "predict") or not callable(model.predict):
        raise ContractError("predict_batch expects a model with a callable predict() method.")
    chunk_size = _validate_batch_size(batch_size)

    if clip_min is not None:
        clip_min = float(clip_min)
        if not np.isfinite(clip_min):
            raise ValueError(f"clip_min must be finite when provided, got {clip_min!r}.")

    model_name = str(getattr(model, "name", type(model).__name__))
    requires_fit = bool(getattr(model, "requires_fit", False))
    total = inputs.batch_size
    horizon = inputs.horizon

    chunks: list[np.ndarray] = []
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk_inputs = inputs.slice(slice(start, stop))
        raw = model.predict(chunk_inputs)
        chunks.append(
            _validate_model_output(
                raw,
                model_name=model_name,
                chunk_size=stop - start,
                horizon=horizon,
            )
        )
    predictions = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
    if predictions.shape != (total, horizon):  # pragma: no cover - defensive
        raise ContractError(
            f"Assembled predictions have shape {predictions.shape}, expected {(total, horizon)}."
        )
    predictions = np.array(predictions, dtype=np.float64, copy=True)

    negative_predictions = int(np.count_nonzero(predictions < 0.0))
    clipped = clip_min is not None
    n_clipped = int(np.count_nonzero(predictions < clip_min)) if clipped else 0
    if clipped:
        predictions = np.maximum(predictions, clip_min)

    diagnostics: dict[str, Any] = {
        "clipped": bool(clipped),
        "clip_min": None if clip_min is None else float(clip_min),
        "n_clipped": n_clipped,
        "model": model_name,
        "requires_fit": requires_fit,
        # Recorded even when no clipping is requested: negative active power is
        # suspicious and must be visible instead of silently ignored.
        "negative_predictions": negative_predictions,
        "n_predictions": int(predictions.size),
    }
    return predictions, diagnostics
