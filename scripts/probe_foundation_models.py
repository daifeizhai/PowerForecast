"""Stage-2 evidence probe for the two frozen foundation models (A1 Chronos-2, A2 TiRex-2).

This is the explicit, reproducible **weight-probe / evidence** step.  It

* resolves both checkpoints from the local offline cache and prints the snapshot
  path, revision, per-file SHA256 (first 12 chars), file sizes and library versions;
* loads both models lazily and measures load / inference time on CPU;
* runs ``H = 16, 32, 96, 288`` on a small deterministic ``[2, 1344]`` kW sample
  context and verifies ``[B, H]`` shape, float64 dtype, finiteness, no NaN and a
  plausible kW magnitude;
* verifies the point policy (Chronos-2 quantile 0.5; TiRex-2 median quantile
  index 4) both from ``describe()`` and by a direct raw-quantile check;
* checks the determinism of repeated calls and the TiRex-2 long-horizon
  recursion at ``H = 672`` (block size 320, hops 2, predicted-medians-only context
  update, first block identical to a native 320-step forecast);
* verifies the fixed 0.5/0.5 ensemble against the member forecasts;
* writes the JSON evidence and exits non-zero if any check fails.

Examples::

    python -m scripts.probe_foundation_models
    python -m scripts.probe_foundation_models --horizons 16 96 288 --recursion-horizon 672
    python -m scripts.probe_foundation_models --allow-download   # weight preparation step

This script never fits or fine-tunes anything and never searches weights.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):  # allow `python scripts/probe_foundation_models.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.power_main import build_power_forecasters
from src.utils import environment_report, load_config, utc_timestamp, write_json

DEFAULT_CONFIG = "configs/models/foundation.yaml"
DEFAULT_OUTPUT = "results/analysis/stage2/model_probe.json"
DEFAULT_HORIZONS: tuple[int, ...] = (16, 32, 96, 288)
DEFAULT_RECURSION_HORIZON = 672

CONTEXT_POINTS = 1344
PROBE_BATCH = 2
#: A generous plausibility window for the synthetic probe context (~8-12 MW).
MAX_PLAUSIBLE_KW = 1e7


# ---------------------------------------------------------------------------
# deterministic sample context (never real data, never random)
# ---------------------------------------------------------------------------


def make_sample_context(points: int = CONTEXT_POINTS, batch: int = PROBE_BATCH) -> np.ndarray:
    """Deterministic, noise-free ``[B, L]`` kW context for shape/timing evidence."""
    slots = np.arange(points, dtype=np.float64)
    day = np.sin(2.0 * np.pi * slots / 96.0)
    week = np.sin(2.0 * np.pi * slots / (7.0 * 96.0))
    levels = np.array([8000.0, 12500.0][:batch], dtype=np.float64)
    rows = [level + 400.0 * day + 150.0 * week for level in levels]
    return np.stack(rows).astype(np.float64)


# ---------------------------------------------------------------------------
# evidence helpers
# ---------------------------------------------------------------------------


def _check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> bool:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def _summarise_array(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "nan_count": int(np.isnan(array).sum()),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _checkpoint_brief(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": checkpoint["model_id"],
        "snapshot_dir": checkpoint["snapshot_dir"],
        "revision": checkpoint["revision"],
        "source": checkpoint["source"],
        "cache_dir": checkpoint["cache_dir"],
        "files": {
            name: {
                "sha256": checkpoint["file_sha256"][name],
                "sha256_12": checkpoint["file_sha256"][name][:12],
                "bytes": checkpoint["file_bytes"][name],
            }
            for name in checkpoint["files"]
        },
    }


def _verify_p50_chronos(model: Any, context: np.ndarray, horizon: int = 16) -> dict[str, Any]:
    """Direct evidence that the adapter returns Chronos-2's quantile 0.5."""
    model._ensure_loaded()  # noqa: SLF001 - evidence probe inspects the loaded pipeline
    import torch

    with torch.inference_mode():
        quantiles, _mean = model._pipeline.predict_quantiles(  # noqa: SLF001
            inputs=[context[0].copy()],
            prediction_length=horizon,
            quantile_levels=[0.5],
            batch_size=1,
            context_length=int(context.shape[1]),
            cross_learning=False,
        )
    raw = quantiles[0][0, :, 0].detach().to("cpu").numpy().astype(np.float64)
    adapter = np.asarray(model.predict(context[:1], horizon), dtype=np.float64)[0]
    return {
        "raw_quantile_level": 0.5,
        "raw_shape": [int(quantiles[0].shape[0]), int(quantiles[0].shape[1])],
        "adapter_matches_quantile_0_5": bool(np.array_equal(adapter, raw)),
        "max_abs_diff": float(np.max(np.abs(adapter - raw))),
    }


def _verify_p50_tirex(model: Any, context: np.ndarray, horizon: int = 16) -> dict[str, Any]:
    """Direct evidence that the adapter returns TiRex-2 quantile index 4 (P50)."""
    model._ensure_loaded()  # noqa: SLF001 - evidence probe inspects the loaded model
    from tirex2 import TimeseriesType

    torch = model._torch  # noqa: SLF001
    series = TimeseriesType(
        target=torch.as_tensor(context[0], dtype=torch.float32).unsqueeze(0),
        past_covariates=None,
        future_covariates=None,
    )
    with torch.inference_mode():
        raw = np.asarray(
            model._model.forecast(  # noqa: SLF001
                [series], prediction_length=horizon, output_type="numpy", batch_size=1
            )[0]
        )
    median_index = int(model.median_index)
    direct_median = raw[0, median_index, :].astype(np.float64)
    adapter = np.asarray(model.predict(context[:1], horizon), dtype=np.float64)[0]
    levels = [round(float(level), 6) for level in model.quantile_levels]
    return {
        "median_index": median_index,
        "quantile_levels": levels,
        "level_at_median_index": levels[median_index],
        "raw_shape": list(raw.shape),
        "adapter_matches_median_index": bool(np.array_equal(adapter, direct_median)),
        "max_abs_diff": float(np.max(np.abs(adapter - direct_median))),
    }


# ---------------------------------------------------------------------------
# probe body
# ---------------------------------------------------------------------------


def run_probe(
    config_path: str | Path,
    *,
    weights_cache: str | Path | None,
    horizons: Sequence[int],
    recursion_horizon: int,
    output_path: str | Path,
    allow_download: bool,
) -> int:
    payload: dict[str, Any] = {
        "generated_at_utc": utc_timestamp(),
        "script": "scripts.probe_foundation_models",
        "config": str(config_path),
        "weights_cache": str(weights_cache) if weights_cache is not None else None,
        "requested_horizons": [int(horizon) for horizon in horizons],
        "recursion_horizon": int(recursion_horizon),
        "allow_download": bool(allow_download),
        "checks": [],
        "status": "failed",
    }
    checks: list[dict[str, Any]] = payload["checks"]
    try:
        config = load_config(config_path)
        print(f"config: {config_path}")
        forecasters = build_power_forecasters(
            config,
            weights_cache=weights_cache,
            models=["chronos2", "tirex2", "power_ensemble"],
            allow_download=allow_download,
        )
        chronos = forecasters["chronos2"]
        tirex = forecasters["tirex2"]
        ensemble = forecasters["power_ensemble"]
        context = make_sample_context()
        payload["sample_context"] = {
            "kind": "deterministic_synthetic_probe_context",
            "generator": "8000/12500 kW + 400 kW daily sine + 150 kW weekly sine, no noise",
            "unit": "kW",
            **_summarise_array(context),
        }
        payload["environment"] = environment_report()

        # ------------------------------------------------------------- chronos
        print("\n=== Chronos-2 (A1) ===")
        chronos_desc = chronos.describe()
        chronos_checkpoint = _checkpoint_brief(chronos_desc["checkpoint"])
        print(f"checkpoint: {chronos_checkpoint['snapshot_dir']}")
        for name, info in chronos_checkpoint["files"].items():
            print(f"  {name}: sha256={info['sha256_12']}... bytes={info['bytes']}")
        print(f"versions: {chronos_desc['library_versions']}")
        print(f"device={chronos_desc['device']} dtype={chronos_desc['dtype']} policy={chronos_desc['point_policy']['quantile']}")
        chronos_outputs: dict[int, np.ndarray] = {}
        for horizon in horizons:
            started = time.perf_counter()
            forecast = np.asarray(chronos.predict(context, horizon))
            elapsed = time.perf_counter() - started
            chronos_outputs[int(horizon)] = forecast
            summary = _summarise_array(forecast)
            print(f"  H={horizon:<4d} shape={summary['shape']} dtype={summary['dtype']} "
                  f"mean={summary['mean']:.2f} kW ({elapsed:.3f}s)")
            _check(checks, f"chronos2_shape_H{horizon}", summary["shape"] == [PROBE_BATCH, int(horizon)], summary["shape"])
            _check(checks, f"chronos2_float64_H{horizon}", summary["dtype"] == "float64", summary["dtype"])
            _check(checks, f"chronos2_finite_H{horizon}", summary["finite"] and summary["nan_count"] == 0, summary)
            _check(checks, f"chronos2_plausible_kw_H{horizon}", 0.0 < summary["mean"] < MAX_PLAUSIBLE_KW, summary["mean"])
        repeat = np.asarray(chronos.predict(context, horizons[0]))
        determinism = float(np.max(np.abs(repeat - chronos_outputs[int(horizons[0])])))
        print(f"  determinism H={horizons[0]}: max|diff|={determinism:.3e}")
        _check(checks, "chronos2_determinism", determinism <= 1e-6, determinism)
        chronos_p50 = _verify_p50_chronos(chronos, context)
        print(f"  P50 direct check: {chronos_p50}")
        _check(
            checks,
            "chronos2_p50_policy",
            chronos_desc["point_policy"]["quantile"] == 0.5
            and "quantile_levels=[0.5]" in chronos_desc["point_policy"]["selection"],
            chronos_desc["point_policy"],
        )
        _check(checks, "chronos2_p50_direct", chronos_p50["adapter_matches_quantile_0_5"], chronos_p50)
        payload["models"] = payload.get("models", {})
        payload["models"]["chronos2"] = {
            "checkpoint": chronos_checkpoint,
            "describe": chronos.describe(),
            "horizons": {
                str(horizon): {
                    **_summarise_array(output),
                    "seconds": round(
                        float(chronos.last_output_diagnostics.get("seconds", 0.0)), 3
                    ),
                }
                for horizon, output in chronos_outputs.items()
            },
            "determinism_max_abs_diff": determinism,
            "p50_direct_check": chronos_p50,
            "load_seconds": chronos.load_seconds,
            "first_inference_seconds": chronos.first_inference_seconds,
        }

        # --------------------------------------------------------------- tirex
        print("\n=== TiRex-2 (A2) ===")
        tirex_desc = tirex.describe()
        tirex_checkpoint = _checkpoint_brief(tirex_desc["checkpoint"])
        print(f"checkpoint: {tirex_checkpoint['snapshot_dir']}")
        for name, info in tirex_checkpoint["files"].items():
            print(f"  {name}: sha256={info['sha256_12']}... bytes={info['bytes']}")
        print(f"device={tirex_desc['device']} dtype={tirex_desc['dtype']} "
              f"median_index={tirex_desc['point_policy']['median_index']} native_horizon={tirex_desc['native_horizon']}")
        tirex_outputs: dict[int, np.ndarray] = {}
        for horizon in horizons:
            started = time.perf_counter()
            forecast = np.asarray(tirex.predict(context, horizon))
            elapsed = time.perf_counter() - started
            tirex_outputs[int(horizon)] = forecast
            summary = _summarise_array(forecast)
            print(f"  H={horizon:<4d} shape={summary['shape']} dtype={summary['dtype']} "
                  f"mean={summary['mean']:.2f} kW ({elapsed:.3f}s)")
            _check(checks, f"tirex2_shape_H{horizon}", summary["shape"] == [PROBE_BATCH, int(horizon)], summary["shape"])
            _check(checks, f"tirex2_float64_H{horizon}", summary["dtype"] == "float64", summary["dtype"])
            _check(checks, f"tirex2_finite_H{horizon}", summary["finite"] and summary["nan_count"] == 0, summary)
            _check(checks, f"tirex2_plausible_kw_H{horizon}", 0.0 < summary["mean"] < MAX_PLAUSIBLE_KW, summary["mean"])
        repeat = np.asarray(tirex.predict(context, horizons[0]))
        determinism = float(np.max(np.abs(repeat - tirex_outputs[int(horizons[0])])))
        print(f"  determinism H={horizons[0]}: max|diff|={determinism:.3e}")
        _check(checks, "tirex2_determinism", determinism <= 1e-6, determinism)
        tirex_p50 = _verify_p50_tirex(tirex, context)
        print(f"  P50 direct check: {tirex_p50}")
        _check(
            checks,
            "tirex2_p50_policy",
            tirex_desc["point_policy"]["median_index"] == 4
            and abs(float(tirex_desc["point_policy"]["quantile_levels"][4]) - 0.5) <= 1e-6,
            tirex_desc["point_policy"],
        )
        _check(checks, "tirex2_p50_direct", tirex_p50["adapter_matches_median_index"], tirex_p50)
        payload["models"]["tirex2"] = {
            "checkpoint": tirex_checkpoint,
            "describe": tirex.describe(),
            "horizons": {
                str(horizon): {
                    **_summarise_array(output),
                    "seconds": round(
                        float(tirex.last_output_diagnostics.get("seconds", 0.0)), 3
                    ),
                }
                for horizon, output in tirex_outputs.items()
            },
            "determinism_max_abs_diff": determinism,
            "p50_direct_check": tirex_p50,
            "load_seconds": tirex.load_seconds,
            "first_inference_seconds": tirex.first_inference_seconds,
        }

        # ----------------------------------------------------------- recursion
        if recursion_horizon and recursion_horizon > int(tirex.native_horizon or 0):
            print(f"\n=== TiRex-2 recursion H={recursion_horizon} (native={tirex.native_horizon}) ===")
            started = time.perf_counter()
            long_forecast = np.asarray(tirex.predict(context, recursion_horizon))
            elapsed = time.perf_counter() - started
            recursion = tirex.describe()["recursion"]
            summary = _summarise_array(long_forecast)
            print(f"  shape={summary['shape']} finite={summary['finite']} "
                  f"recursion={recursion} ({elapsed:.2f}s)")
            native_block = np.asarray(tirex.predict(context, int(tirex.native_horizon)))
            first_block_diff = float(
                np.max(np.abs(long_forecast[:, : int(tirex.native_horizon)] - native_block))
            )
            print(f"  first native block vs direct H={tirex.native_horizon}: max|diff|={first_block_diff:.3e}")
            _check(
                checks,
                "tirex2_recursion_shape",
                summary["shape"] == [PROBE_BATCH, int(recursion_horizon)],
                summary["shape"],
            )
            _check(checks, "tirex2_recursion_finite", summary["finite"], summary["finite"])
            _check(checks, "tirex2_recursion_used", recursion["used"] is True, recursion)
            _check(checks, "tirex2_recursion_blocks", recursion["block_size"] == 320, recursion)
            _check(
                checks,
                "tirex2_recursion_hops",
                recursion["hops"] == int(np.ceil(recursion_horizon / 320.0)) - 1,
                recursion,
            )
            _check(
                checks,
                "tirex2_recursion_context_update",
                recursion["context_update"] == "predicted_median_only",
                recursion,
            )
            _check(checks, "tirex2_recursion_first_block", first_block_diff <= 1e-6, first_block_diff)
            _check(
                checks,
                "tirex2_recursion_no_future_truth",
                "never consumed" in tirex.describe()["recursion_policy"]["true_future_values"],
                tirex.describe()["recursion_policy"],
            )
            payload["recursion_probe"] = {
                "horizon": int(recursion_horizon),
                "seconds": round(elapsed, 3),
                "recursion": recursion,
                "first_native_block_max_abs_diff": first_block_diff,
                "policy": tirex.describe()["recursion_policy"],
            }
        else:
            print("\n=== TiRex-2 recursion: skipped (H <= native_horizon) ===")
            payload["recursion_probe"] = {"skipped": True}

        # ------------------------------------------------------------- ensemble
        print("\n=== A3 fixed 0.5/0.5 ensemble ===")
        ensemble_cfg = config.get("ensemble") if isinstance(config.get("ensemble"), dict) else {}
        _check(
            checks,
            "ensemble_fixed_weights",
            abs(float(ensemble_cfg.get("chronos2_weight", 0.5)) - 0.5) <= 1e-12
            and abs(float(ensemble_cfg.get("tirex2_weight", 0.5)) - 0.5) <= 1e-12,
            ensemble_cfg,
        )
        ensemble_horizons: dict[str, Any] = {}
        for horizon in horizons:
            forecast = np.asarray(ensemble.predict(context, horizon))
            expected = 0.5 * chronos_outputs[int(horizon)] + 0.5 * tirex_outputs[int(horizon)]
            max_diff = float(np.max(np.abs(forecast - expected)))
            summary = _summarise_array(forecast)
            print(f"  H={horizon:<4d} shape={summary['shape']} mean={summary['mean']:.2f} kW "
                  f"max|ensemble-mean|={max_diff:.3e}")
            ensemble_horizons[str(horizon)] = {
                **_summarise_array(forecast),
                "max_abs_diff_vs_member_mean": max_diff,
            }
            _check(checks, f"ensemble_shape_H{horizon}", summary["shape"] == [PROBE_BATCH, int(horizon)], summary["shape"])
            _check(checks, f"ensemble_finite_H{horizon}", summary["finite"], summary["finite"])
            _check(checks, f"ensemble_equals_member_mean_H{horizon}", max_diff <= 1e-9, max_diff)
        payload["ensemble"] = {
            "weights": [0.5, 0.5],
            "method": "fixed_weighted_mean",
            "horizons": ensemble_horizons,
            "describe": ensemble.describe(),
        }

        failed = [check for check in checks if not check["ok"]]
        payload["status"] = "passed" if not failed else "failed"
        payload["checks_total"] = len(checks)
        payload["checks_failed"] = len(failed)
        destination = write_json(output_path, payload)
        print(f"\nstatus: {payload['status']} ({len(checks)} checks, {len(failed)} failed)")
        print(f"evidence: {destination}")
        if failed:
            for check in failed:
                print(f"  FAILED: {check['name']} -> {check['detail']}", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001 - the CLI converts any failure to evidence
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        try:
            write_json(output_path, payload)
        except Exception:  # pragma: no cover - best-effort evidence
            pass
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the frozen Chronos-2 / TiRex-2 checkpoints offline: checkpoint hashes, "
            "shapes, finiteness, P50 policy, timings, TiRex-2 recursion and the fixed "
            "0.5/0.5 ensemble. Exits non-zero on any failed check."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="foundation-model configuration")
    parser.add_argument("--weights-cache", default=None, help="local HF_HUB_CACHE-layout weights directory")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument(
        "--recursion-horizon",
        type=int,
        default=DEFAULT_RECURSION_HORIZON,
        help="TiRex-2 long-horizon recursion probe (0 disables it)",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSON evidence output path")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="explicit weight-preparation step: allow downloading missing checkpoint files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(horizon < 1 for horizon in args.horizons):
        print("ERROR: horizons must be positive integers.", file=sys.stderr)
        return 2
    return run_probe(
        args.config,
        weights_cache=args.weights_cache,
        horizons=[int(horizon) for horizon in args.horizons],
        recursion_horizon=int(args.recursion_horizon),
        output_path=args.output,
        allow_download=bool(args.allow_download),
    )


if __name__ == "__main__":
    raise SystemExit(main())
