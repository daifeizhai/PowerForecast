"""Chronos-2 adapter (model A1) for the Stage-2 historical-power main path.

The adapter is a thin, auditable wrapper around the frozen ``amazon/chronos-2``
checkpoint:

* weights are resolved **offline** by :func:`src.models.foundation.offline.resolve_checkpoint`
  (snapshot path / revision / file SHA256 are recorded) and loaded lazily on the
  first :meth:`Chronos2Forecaster.predict` call with ``local_files_only=True``;
* inference is zero-shot: one independent univariate series per batch row, no
  cross-learning between rows, no fitting and no target-domain training;
* the point forecast is the **P50 (median)** of the model's forecast
  distribution, selected with ``predict_quantiles(..., quantile_levels=[0.5])``;
* the output is ``[B, H]`` finite ``float64`` absolute active power in kW.

Batch invariance
----------------
Every row is a **separate** univariate series (``cross_learning=False``); with
the default ``batch_size=1`` the library processes rows one at a time, so a
series predicted alone and the same series inside a batch are bit-identical.
Increasing ``batch_size`` changes only grouped kernel batching (last-bit float
rounding, far below 1e-6 relative), never the information a row receives.

Missing weights, non-finite/ragged/too-short contexts and non-positive horizons
raise explicit errors; no fake or partial weights and no silent truncation of
the requested horizon are ever produced.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.models.foundation.offline import (
    CheckpointSnapshot,
    configure_offline_environment,
    library_versions,
    resolve_checkpoint,
    snapshot_to_dict,
    validate_context,
    validate_prediction,
)

__all__ = ["CHRONOS2_FILES", "CHRONOS2_MODEL_ID", "Chronos2Forecaster"]

CHRONOS2_MODEL_ID = "amazon/chronos-2"
CHRONOS2_FILES: tuple[str, ...] = ("config.json", "model.safetensors")

#: Adapter implementation version (not the model checkpoint version).
ADAPTER_VERSION = "1.0.0"

#: ``dtype`` names accepted by the constructor -> torch dtypes (mapped lazily).
_DTYPE_NAMES: tuple[str, ...] = ("float32", "float64", "float16", "bfloat16")

#: Verified upper bound of the checkpoint's context window (``chronos_config``).
_FALLBACK_MAX_CONTEXT = 8192


def _read_max_context(checkpoint: CheckpointSnapshot) -> int:
    """Read the checkpoint's context window from its own ``config.json``."""
    config_path = Path(checkpoint.snapshot_dir) / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        context_length = int(payload["chronos_config"]["context_length"])
    except (OSError, KeyError, TypeError, ValueError):  # pragma: no cover - config is verified
        return _FALLBACK_MAX_CONTEXT
    if context_length < 1:  # pragma: no cover - defensive
        return _FALLBACK_MAX_CONTEXT
    return context_length


class Chronos2Forecaster:
    """Zero-shot Chronos-2 P50 forecaster with an offline frozen checkpoint."""

    name = "chronos2"
    family = "foundation"
    model_id = CHRONOS2_MODEL_ID
    version = ADAPTER_VERSION
    requires_fit = False
    native_horizon: int | None = None  # Chronos-2 returns the full requested horizon.
    #: Smallest accepted context: one trace day (96 x 15 min).
    min_context_points = 96

    def __init__(
        self,
        *,
        weights_cache: str | Path,
        device: str = "cpu",
        dtype: str = "float32",
        batch_size: int = 1,
        jobs: int = 4,
        allow_download: bool = False,
    ) -> None:
        if not isinstance(weights_cache, (str, Path)) or not str(weights_cache).strip():
            raise ValueError("Chronos2Forecaster needs a nonempty weights_cache path.")
        if not isinstance(device, str) or not device.strip():
            raise ValueError(f"Chronos2Forecaster.device must be a nonempty string, got {device!r}.")
        if not isinstance(dtype, str) or dtype not in _DTYPE_NAMES:
            raise ValueError(
                f"Chronos2Forecaster.dtype must be one of {list(_DTYPE_NAMES)}, got {dtype!r}."
            )
        if device == "cpu" and dtype == "float16":
            raise ValueError(
                "Chronos2Forecaster.dtype='float16' is not usable on CPU; use 'float32' (verified)."
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise ValueError(f"Chronos2Forecaster.batch_size must be an integer, got {batch_size!r}.")
        if int(batch_size) < 1:
            raise ValueError("Chronos2Forecaster.batch_size must be >= 1.")
        if isinstance(jobs, bool) or not isinstance(jobs, (int, np.integer)):
            raise ValueError(f"Chronos2Forecaster.jobs must be an integer, got {jobs!r}.")
        if int(jobs) < 1:
            raise ValueError("Chronos2Forecaster.jobs must be >= 1.")

        self.device = device
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.jobs = int(jobs)
        self.allow_download = bool(allow_download)
        # Offline resolution only; no model weights are loaded here.
        self.checkpoint: CheckpointSnapshot = resolve_checkpoint(
            CHRONOS2_MODEL_ID,
            weights_cache,
            CHRONOS2_FILES,
            allow_download=self.allow_download,
        )
        self.max_context_points = _read_max_context(self.checkpoint)
        self.load_seconds: float | None = None
        self.first_inference_seconds: float | None = None
        self.last_inference_seconds: float | None = None
        self.last_output_diagnostics: dict[str, Any] = {}
        self._pipeline: Any = None
        self._torch: Any = None

    # -- loading -------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Import torch/chronos and load the frozen pipeline on first use."""
        if self._pipeline is not None:
            return
        configure_offline_environment()
        import torch
        from chronos import Chronos2Pipeline

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Chronos2Forecaster.device={self.device!r} requested but CUDA is not available. "
                "This Stage-2 run is CPU-only."
            )
        # NB: ``jobs`` is recorded as provenance only.  Overriding torch's CPU
        # thread pool here measurably slows the first CPU inference down (thread
        # pool spin-up); the verified timings use torch's default thread count.
        dtype = getattr(torch, self.dtype)
        started = time.perf_counter()
        pipeline = Chronos2Pipeline.from_pretrained(
            str(self.checkpoint.snapshot_dir),
            device_map=self.device,
            dtype=dtype,
            cache_dir=self.checkpoint.cache_dir,
            local_files_only=True,
        )
        self._pipeline = pipeline
        self._torch = torch
        self.load_seconds = time.perf_counter() - started

    # -- inference -----------------------------------------------------------
    def predict(self, context_kw: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` 15-minute absolute-power points in kW.

        Parameters
        ----------
        context_kw:
            ``[B, L]`` float kW history, one independent series per row.  ``L``
            must be at least 96 and at most the checkpoint context window.
        horizon:
            Positive number of future 15-minute steps; the full horizon is always
            returned (never truncated or filled by repetition).

        Returns
        -------
        numpy.ndarray
            ``[B, horizon]`` finite ``float64`` P50 kW predictions.
        """
        context, horizon = validate_context(
            context_kw,
            horizon,
            model_name=self.name,
            min_context_points=self.min_context_points,
            max_context_points=self.max_context_points,
        )
        batch, context_length = context.shape
        self._ensure_loaded()

        started = time.perf_counter()
        rows = [context[row].copy() for row in range(batch)]
        with self._torch.inference_mode():
            quantiles, _mean = self._pipeline.predict_quantiles(
                inputs=rows,
                prediction_length=horizon,
                quantile_levels=[0.5],
                batch_size=self.batch_size,
                context_length=int(context_length),
                cross_learning=False,
            )
        if len(quantiles) != batch:
            raise RuntimeError(
                f"Chronos2Forecaster expected {batch} forecast tensors, got {len(quantiles)}."
            )
        forecasts: list[np.ndarray] = []
        for index, tensor in enumerate(quantiles):
            values = tensor.detach().to("cpu").numpy()
            # predict_quantiles -> (n_variates, prediction_length, n_quantiles)
            if values.shape != (1, horizon, 1):
                raise RuntimeError(
                    f"Chronos2Forecaster row {index} returned shape {values.shape}, expected "
                    f"(1, {horizon}, 1); the P50 selection contract is violated."
                )
            forecasts.append(values[0, :, 0])
        output = np.stack(forecasts).astype(np.float64)
        self.last_inference_seconds = time.perf_counter() - started
        if self.first_inference_seconds is None:
            self.first_inference_seconds = self.last_inference_seconds
        result = validate_prediction(
            self.name, output, batch_size=batch, horizon=horizon
        )
        self.last_output_diagnostics = {
            "model": self.name,
            "shape": [int(batch), int(horizon)],
            "context_points": int(context_length),
            "point_policy": "quantile_0.5",
            "finite": True,
            "batch_size": int(self.batch_size),
            "seconds": round(float(self.last_inference_seconds), 3),
        }
        return result

    # -- metadata ------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """JSON-safe adapter/checkpoint description for run manifests."""
        return {
            "name": self.name,
            "family": self.family,
            "model_id": self.model_id,
            "version": self.version,
            "requires_fit": self.requires_fit,
            "native_horizon": self.native_horizon,
            "min_context_points": int(self.min_context_points),
            "max_context_points": int(self.max_context_points),
            "device": self.device,
            "dtype": self.dtype,
            "batch_size": int(self.batch_size),
            "jobs": int(self.jobs),
            "jobs_note": (
                "recorded as provenance; CPU inference uses torch's default thread pool, which "
                "is faster for the first call than forcing a smaller pool"
            ),
            "offline_only": not self.allow_download,
            "loading": "lazy on first predict(); checkpoint resolved and hashed at construction",
            "checkpoint": snapshot_to_dict(self.checkpoint),
            "point_policy": {
                "quantile": 0.5,
                "prediction_target": "median (P50) absolute active power in kW",
                "selection": "predict_quantiles(quantile_levels=[0.5]) -> quantiles[0][:, :, 0]",
                "cross_learning": False,
                "output": "[B, H] float64 kW",
            },
            "series_batching": {
                "independent_series": True,
                "one_row_per_series": True,
                "cross_learning": False,
                "batch_invariance_note": (
                    "Rows never share information (cross_learning=False). With batch_size=1 rows "
                    "are processed individually, so alone-vs-batched predictions are bit-identical."
                ),
            },
            "load_seconds": self.load_seconds,
            "load_seconds_note": (
                "time to construct the pipeline from the resolved snapshot; the first predict() "
                "additionally materialises the frozen tensors, so first_inference_seconds can be "
                "larger than later calls"
            ),
            "first_inference_seconds": self.first_inference_seconds,
            "last_inference_seconds": self.last_inference_seconds,
            "last_output_diagnostics": dict(self.last_output_diagnostics),
            "library_versions": library_versions(),
        }
