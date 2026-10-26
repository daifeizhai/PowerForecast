"""TiRex-2 adapter (model A2) for the Stage-2 historical-power main path.

The adapter wraps the frozen ``NX-AI/TiRex-2`` checkpoint resolved **offline** by
:func:`src.models.foundation.offline.resolve_checkpoint`:

* ``tirex2.load_model(snapshot_dir, device="cpu", use_flex_attention=False,
  compile=False)`` is imported lazily on first :meth:`TiRex2Forecaster.predict`;
* one independent univariate series per batch row, no covariates, no fitting;
* the point forecast is the **P50 (median)** of the checkpoint's 9 quantiles,
  i.e. index ``4`` of ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``; the
  selected level is read back from the loaded model's own ``quantiles`` buffer
  and recorded in :meth:`describe` (never assumed blindly);
* the output is ``[B, H]`` finite ``float64`` absolute active power in kW.

Long horizons (> ``native_horizon``)
------------------------------------
The checkpoint has ``future_len == 320``: the library silently truncates a
longer ``prediction_length`` to 320.  This adapter **never relies on that**: for
``H > 320`` it performs an explicit, documented blockwise rollout

    block 1 : forecast 320 steps from the real context
    hop  k  : append the model's own predicted medians to the context and
              forecast the next block from that extended context

Only the model's own previous **medians** are appended; true future values are
structurally unavailable (``predict`` receives only ``[B, L]`` context and
``H``), so they can never enter the rollout.  The last block may be shorter than
320 and is forecast at exactly its length.  The number of blocks, hops and the
context-update rule are recorded in ``describe()["recursion"]``.
"""

from __future__ import annotations

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

__all__ = ["TIRex2_FILES", "TIRex2_MODEL_ID", "TiRex2Forecaster"]

TIRex2_MODEL_ID = "NX-AI/TiRex-2"
TIRex2_FILES: tuple[str, ...] = ("model-config.yaml", "model.ckpt")

#: Adapter implementation version (not the model checkpoint version).
ADAPTER_VERSION = "1.0.0"

#: Verified quantile grid of the checkpoint; index 4 is the P50 median.
_EXPECTED_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: Fallback values if the checkpoint config cannot be read at construction.
_FALLBACK_NATIVE_HORIZON = 320
_FALLBACK_MAX_CONTEXT = 2048


def _read_checkpoint_config(checkpoint: CheckpointSnapshot) -> dict[str, Any]:
    """Read ``future_len`` / ``context_len`` / ``quantiles`` from the snapshot."""
    import yaml

    path = Path(checkpoint.snapshot_dir) / "model-config.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):  # pragma: no cover - config is verified
        return {}
    return payload if isinstance(payload, dict) else {}


class TiRex2Forecaster:
    """Zero-shot TiRex-2 P50 forecaster with documented long-horizon recursion."""

    name = "tirex2"
    family = "foundation"
    model_id = TIRex2_MODEL_ID
    version = ADAPTER_VERSION
    requires_fit = False
    #: Native forecast length (320) is only known after the checkpoint is loaded.
    native_horizon: int | None = None
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
        max_horizon: int | None = None,
    ) -> None:
        if not isinstance(weights_cache, (str, Path)) or not str(weights_cache).strip():
            raise ValueError("TiRex2Forecaster needs a nonempty weights_cache path.")
        if not isinstance(device, str) or not device.strip():
            raise ValueError(f"TiRex2Forecaster.device must be a nonempty string, got {device!r}.")
        if not isinstance(dtype, str) or dtype not in {"float32", "float64"}:
            raise ValueError(
                "TiRex2Forecaster.dtype must be 'float32' (verified) or 'float64', "
                f"got {dtype!r}."
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise ValueError(f"TiRex2Forecaster.batch_size must be an integer, got {batch_size!r}.")
        if int(batch_size) < 1:
            raise ValueError("TiRex2Forecaster.batch_size must be >= 1.")
        if isinstance(jobs, bool) or not isinstance(jobs, (int, np.integer)):
            raise ValueError(f"TiRex2Forecaster.jobs must be an integer, got {jobs!r}.")
        if int(jobs) < 1:
            raise ValueError("TiRex2Forecaster.jobs must be >= 1.")
        if max_horizon is not None:
            if isinstance(max_horizon, bool) or not isinstance(max_horizon, (int, np.integer)):
                raise ValueError(f"TiRex2Forecaster.max_horizon must be an integer, got {max_horizon!r}.")
            if int(max_horizon) < 1:
                raise ValueError("TiRex2Forecaster.max_horizon must be >= 1 (or None).")

        self.device = device
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.jobs = int(jobs)
        self.allow_download = bool(allow_download)
        self.max_horizon = None if max_horizon is None else int(max_horizon)
        # Offline resolution only; no model weights are loaded here.
        self.checkpoint: CheckpointSnapshot = resolve_checkpoint(
            TIRex2_MODEL_ID,
            weights_cache,
            TIRex2_FILES,
            allow_download=self.allow_download,
        )
        config = _read_checkpoint_config(self.checkpoint)
        self.configured_future_len = int(
            config.get("future_len") or _FALLBACK_NATIVE_HORIZON
        )
        self.max_context_points = int(config.get("context_len") or _FALLBACK_MAX_CONTEXT)
        configured_quantiles = config.get("quantiles")
        self.configured_quantiles: tuple[float, ...] = (
            tuple(float(value) for value in configured_quantiles)
            if isinstance(configured_quantiles, (list, tuple)) and configured_quantiles
            else _EXPECTED_QUANTILES
        )
        self.median_index = self._locate_median_index(self.configured_quantiles)

        self.quantile_levels: tuple[float, ...] = self.configured_quantiles
        self.median_quantile_level: float | None = None
        self.load_seconds: float | None = None
        self.first_inference_seconds: float | None = None
        self.last_inference_seconds: float | None = None
        self.last_recursion: dict[str, Any] | None = None
        self.last_output_diagnostics: dict[str, Any] = {}
        self._model: Any = None
        self._torch: Any = None

    @staticmethod
    def _locate_median_index(quantiles: tuple[float, ...]) -> int:
        """Index of the 0.5 quantile in the checkpoint's quantile grid."""
        for index, level in enumerate(quantiles):
            if abs(float(level) - 0.5) <= 1e-6:
                return index
        raise ValueError(
            "The TiRex-2 quantile grid does not contain the P50 median 0.5: "
            f"{list(quantiles)}. The adapter cannot select a median honestly."
        )

    # -- loading -------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Import torch/tirex2 and load the frozen checkpoint on first use."""
        if self._model is not None:
            return
        configure_offline_environment()
        import torch
        from tirex2 import load_model

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"TiRex2Forecaster.device={self.device!r} requested but CUDA is not available. "
                "This Stage-2 run is CPU-only."
            )
        if self.device == "cpu":
            # The checkpoint is uncompiled; force eager execution on CPU.  ``jobs`` is
            # recorded as provenance only (torch's default thread pool is faster here).
            torch.compiler.set_stance("force_eager")
        started = time.perf_counter()
        model = load_model(
            str(self.checkpoint.snapshot_dir),
            device=self.device,
            use_flex_attention=False,
            compile=False,
        )
        self._model = model
        self._torch = torch

        raw_quantiles = [round(float(level), 6) for level in model.quantiles]
        self.quantile_levels = tuple(raw_quantiles)
        self.median_index = self._locate_median_index(self.quantile_levels)
        self.median_quantile_level = float(raw_quantiles[self.median_index])
        if abs(self.median_quantile_level - 0.5) > 1e-6:
            raise RuntimeError(
                f"Loaded TiRex-2 model reports median quantile level "
                f"{self.median_quantile_level!r} at index {self.median_index}; expected 0.5. "
                "Refusing to label a non-median forecast as P50."
            )
        future_len = int(getattr(model, "future_len", 0))
        if future_len < 1:
            raise RuntimeError("TiRex-2 checkpoint does not expose a positive future_len.")
        self.native_horizon = future_len
        self.load_seconds = time.perf_counter() - started

    # -- inference -----------------------------------------------------------
    def _forecast_native(self, history: np.ndarray, length: int) -> np.ndarray:
        """One native TiRex-2 call: ``[B, L]`` context -> ``[B, length]`` medians."""
        assert self._model is not None and self._torch is not None  # nosec - ensured by caller
        from tirex2 import TimeseriesType

        batch = history.shape[0]
        series = [
            TimeseriesType(
                target=self._torch.as_tensor(row, dtype=self._torch.float32).unsqueeze(0),
                past_covariates=None,
                future_covariates=None,
            )
            for row in history
        ]
        with self._torch.inference_mode():
            outputs = self._model.forecast(
                series,
                prediction_length=int(length),
                output_type="numpy",
                batch_size=self.batch_size,
            )
        if len(outputs) != batch:
            raise RuntimeError(
                f"TiRex2Forecaster expected {batch} forecast arrays, got {len(outputs)}."
            )
        expected_shape = (1, len(self.quantile_levels), int(length))
        medians: list[np.ndarray] = []
        for index, values in enumerate(outputs):
            array = np.asarray(values)
            if array.shape != expected_shape:
                raise RuntimeError(
                    f"TiRex2Forecaster row {index} returned shape {array.shape}, expected "
                    f"{expected_shape}; the quantile/P50 contract is violated."
                )
            medians.append(array[0, self.median_index, :])
        return np.stack(medians).astype(np.float64)

    def predict(self, context_kw: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` 15-minute absolute-power points in kW.

        For ``horizon > native_horizon`` the forecast is rolled out blockwise with
        only the model's own previously predicted medians appended to the context.
        The full requested horizon is always returned.
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
        native = int(self.native_horizon)
        if self.max_horizon is not None and horizon > self.max_horizon:
            raise ValueError(
                f"{self.name}.predict horizon {horizon} exceeds the configured recursion limit "
                f"max_horizon={self.max_horizon}. Raise the limit explicitly if a longer rollout "
                "is really intended; it is never silently shortened."
            )

        started = time.perf_counter()
        if horizon <= native:
            output = self._forecast_native(context, horizon)
            recursion: dict[str, Any] = {
                "used": False,
                "block_size": int(native),
                "hops": 0,
                "blocks": 1,
                "context_update": "predicted_median_only",
            }
        else:
            pieces: list[np.ndarray] = []
            history = context.copy()
            remaining = int(horizon)
            blocks = 0
            hops = 0
            context_sizes_after_hop: list[int] = []
            while remaining > 0:
                block_length = int(min(native, remaining))
                block = self._forecast_native(history, block_length)
                pieces.append(block)
                remaining -= block_length
                blocks += 1
                if remaining > 0:
                    # Recursive context update: append ONLY the model's own medians.
                    history = np.concatenate([history, block], axis=1)
                    if history.shape[1] > self.max_context_points:
                        history = history[:, -self.max_context_points :]
                    hops += 1
                    context_sizes_after_hop.append(int(history.shape[1]))
            output = np.concatenate(pieces, axis=1)
            recursion = {
                "used": True,
                "block_size": int(native),
                "hops": int(hops),
                "blocks": int(blocks),
                "context_update": "predicted_median_only",
                "context_points_after_hop": context_sizes_after_hop,
                "truth_values_consumed": False,
            }
        self.last_inference_seconds = time.perf_counter() - started
        if self.first_inference_seconds is None:
            self.first_inference_seconds = self.last_inference_seconds
        self.last_recursion = recursion
        result = validate_prediction(self.name, output, batch_size=batch, horizon=horizon)
        self.last_output_diagnostics = {
            "model": self.name,
            "shape": [int(batch), int(horizon)],
            "context_points": int(context_length),
            "point_policy": "quantile_0.5",
            "median_index": int(self.median_index),
            "recursion_used": bool(recursion["used"]),
            "batch_size": int(self.batch_size),
            "seconds": round(float(self.last_inference_seconds), 3),
        }
        return result

    # -- metadata ------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """JSON-safe adapter/checkpoint/recursion description for run manifests."""
        last = self.last_recursion
        recursion = (
            {
                "used": bool(last["used"]),
                "block_size": int(last["block_size"]),
                "hops": int(last["hops"]),
                "blocks": int(last.get("blocks", 1)),
                "context_update": str(last["context_update"]),
            }
            if last is not None
            else {
                "used": False,
                "block_size": int(self.configured_future_len),
                "hops": 0,
                "blocks": 0,
                "context_update": "predicted_median_only",
            }
        )
        return {
            "name": self.name,
            "family": self.family,
            "model_id": self.model_id,
            "version": self.version,
            "requires_fit": self.requires_fit,
            "native_horizon": self.native_horizon,
            "configured_future_len": int(self.configured_future_len),
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
                "median_index": int(self.median_index),
                "quantile_levels": list(self.quantile_levels),
                "verified_median_quantile_level": self.median_quantile_level,
                "prediction_target": "median (P50) absolute active power in kW",
                "selection": "forecast(...)[0][median_index, :] with median_index=4",
                "output": "[B, H] float64 kW",
            },
            "recursion": recursion,
            "recursion_policy": {
                "native_horizon": self.native_horizon,
                "when_horizon_exceeds_native": "blockwise autoregressive rollout",
                "block_size": "native_horizon (320); the final block is shorter",
                "context_update": "predicted_median_only",
                "true_future_values": "never consumed: predict() only accepts [B, L] context and H",
                "max_context_points": int(self.max_context_points),
                "max_horizon": self.max_horizon,
            },
            "future_truth": {
                "accepted": False,
                "note": (
                    "predict(context_kw, horizon) has no future-target or future-covariate "
                    "argument, so true future values cannot be passed in; the recursion only "
                    "appends this adapter's own predicted medians."
                ),
            },
            "load_seconds": self.load_seconds,
            "load_seconds_note": (
                "time to construct the model from the resolved snapshot; the first predict() "
                "additionally materialises the frozen tensors, so first_inference_seconds can be "
                "larger than later calls"
            ),
            "first_inference_seconds": self.first_inference_seconds,
            "last_inference_seconds": self.last_inference_seconds,
            "last_output_diagnostics": dict(self.last_output_diagnostics),
            "library_versions": library_versions(),
        }
