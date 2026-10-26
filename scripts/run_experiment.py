"""The single official experiment entry point.

Stages (task book §8):

``development``
    Pre-checks and development diagnostics on allowed data only.  Never used to
    select or report a formal test result.
``finalize``
    Train/configure with the declared cutoff, persist model/configuration state
    and write ``frozen_before_test.json``.  Produces no test score.
``final-test``
    Requires an existing frozen run, verifies its hashes, reloads the model and
    only then predicts and scores.  Refused without frozen credentials.
``smoke``
    One command performs prepare -> freeze -> reload -> test on synthetic data
    only.  The output always carries ``synthetic=true`` and
    ``research_result=false``.

Examples (Windows PowerShell)::

    python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 24h
    python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 7d
    python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 24h \
        --protocol configs/protocols/fixed_24h.yaml
    python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 7d \
        --protocol configs/protocols/fixed_7d.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/run_experiment.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.contracts import (
    ContractError,
    DatasetSpec,
    ForecastBatch,
    InsufficientDataError,
    LeakageError,
    PhysicalUnitError,
    TargetEligibilityError,
    require_facility_active_power_kw,
)
from src.data.preprocessing import quality_report
from src.data.splitting import Protocol, grid_for_frame, training_window_audit
from src.data.synthetic import build_synthetic_dataset
from src.data.windows import OriginPlan, audit_split, build_forecast_batch, plan_origins
from src.evaluation.artifacts import RunDirectory, predictions_frame
from src.evaluation.backtest import score_batch
from src.evaluation.metrics import forecast_metrics
from src.models import (
    build_forecaster,
    describe_models,
    implemented_models,
    planned_models,
)
from src.training.lifecycle import (
    FrozenRunError,
    freeze_model,
    load_frozen,
    require_frozen_record,
)
from src.training.runner import predict_batch
from src.utils import (
    config_digest,
    environment_report,
    jsonable,
    load_config,
    resolve_path,
    seed_everything,
    sha256_file,
    sha256_text,
    utc_timestamp,
    write_json,
)

SUPPORTED_PHASES: tuple[str, ...] = ("development", "finalize", "final-test", "smoke")
HORIZON_NAMES: tuple[str, ...] = ("4h", "8h", "24h", "3d", "7d")
DEMO_COMMAND = "python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 24h"
SUPPORTED_DATASETS: frozenset[str] = frozenset({"synthetic_facility"})


class ExperimentError(RuntimeError):
    """A user-facing experiment configuration or lifecycle error."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalars are replaced."""
    merged: dict[str, Any] = {key: value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a config, following an optional ``inherit: <path>`` chain."""
    payload = load_config(path)
    parent = payload.pop("inherit", None)
    if parent:
        base = load_experiment_config(parent)
        payload = deep_merge(base, payload)
    return payload


def _resolve_seed(config: Mapping[str, Any], args: argparse.Namespace) -> int:
    raw = args.seed if args.seed is not None else config.get("seed", 7)
    seed = int(raw)
    if seed < 0:
        raise ExperimentError("Seed must be a nonnegative integer.")
    return seed


def _dataset_config(config: Mapping[str, Any]) -> dict[str, Any]:
    inline = dict(config.get("dataset") or {})
    path = config.get("dataset_config")
    file_config = dict(load_config(path)) if path else {}
    merged = deep_merge(file_config, inline)
    merged.setdefault("dataset_id", "synthetic_facility")
    return merged


def _build_dataset(config: Mapping[str, Any], *, seed: int):
    """Build the synthetic dataset, or block a real (unaudited) source."""
    settings = _dataset_config(config)
    dataset_id = str(settings.get("dataset_id", "synthetic_facility"))
    if dataset_id not in SUPPORTED_DATASETS:
        raise ExperimentError(
            f"Dataset {dataset_id!r} is only registered as a candidate/unverified source; no "
            "adapter is implemented in this skeleton and no formal facility-kW result may be "
            f"produced from it. Run the synthetic engineering demo instead: {DEMO_COMMAND}"
        )
    dataset_seed = int(settings.get("seed", seed))
    missing_days = tuple(int(day) for day in (settings.get("missing_days") or ()))
    return build_synthetic_dataset(
        days=int(settings.get("days", 31)),
        seed=dataset_seed,
        anchor=str(settings.get("anchor_timestamp", "2024-01-01T00:00:00Z")),
        covariates=bool(settings.get("covariates", True)),
        noise=float(settings.get("noise", 1.0)),
        missing_days=missing_days,
    )


def _parse_origin_days(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, str):
        if not values.strip():
            return []
        return [int(part) for part in values.replace(";", ",").split(",") if part.strip()]
    return [int(value) for value in values]


def _build_protocol(config: Mapping[str, Any], args: argparse.Namespace) -> Protocol:
    protocol_path = args.protocol or config.get("protocol_config") or "configs/protocols/default.yaml"
    protocol_config = dict(load_config(protocol_path))
    protocol_type = protocol_config.get("protocol_type") or "configurable"
    if args.demo and protocol_type == "configurable":
        protocol_type = "demo"
    protocol_config["protocol_type"] = protocol_type

    horizon = args.horizon or config.get("horizon") or protocol_config.get("default_horizon") or "24h"
    context_points = int(
        args.context_points
        if args.context_points is not None
        else config.get("context_points") or protocol_config.get("context_points") or 1344
    )

    explicit_origins = _parse_origin_days(args.origins)
    configured_demo_origins = _parse_origin_days(config.get("demo_origins"))
    configured_protocol_origins = _parse_origin_days(protocol_config.get("test_origins"))

    if protocol_type == "demo":
        # A demo protocol has no research meaning: reuse the configured demo days
        # (a fixed preset keeps its own explicit day numbers instead).
        origins = explicit_origins or configured_protocol_origins or configured_demo_origins
        if not origins:
            raise ExperimentError(
                "A demo protocol needs explicit origin days; set demo_origins in the config or "
                "pass --origins."
            )
        protocol_config["test_origins"] = origins
        protocol_config["development_origins"] = origins

    try:
        return Protocol.from_config(
            protocol_config, horizon=horizon, context_points=context_points
        )
    except ValueError as exc:
        raise ExperimentError(f"Invalid protocol configuration: {exc}") from exc


def _resolve_phase(args: argparse.Namespace, config: Mapping[str, Any]) -> str:
    phase = args.phase or config.get("phase")
    if phase is None:
        phase = "smoke" if args.demo else "development"
    if phase not in SUPPORTED_PHASES:
        raise ExperimentError(f"Unknown phase {phase!r}; expected one of {SUPPORTED_PHASES}.")
    return phase


def _research_result_allowed(
    *, spec: DatasetSpec, phase: str, policy: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    """Decide whether a run may be labelled a research result, with reasons."""
    reasons: list[str] = []
    if spec.synthetic:
        reasons.append("labels are synthetic_simulated (engineering validation only)")
    if bool(policy.get("demo_only", False)):
        reasons.append("configuration is marked demo_only")
    if phase != "final-test":
        reasons.append(f"phase {phase!r} never yields a formal test result")
    if bool(policy.get("require_real_facility_kw", True)):
        try:
            require_facility_active_power_kw(spec.target, allow_synthetic=False)
        except (TargetEligibilityError, PhysicalUnitError) as exc:
            reasons.append(f"facility-kW label not eligible: {exc}")
    return (not reasons, reasons)


# ---------------------------------------------------------------------------
# artifact payloads
# ---------------------------------------------------------------------------


def _payload_sha(payload: Any) -> str:
    import json

    return sha256_text(json.dumps(jsonable(payload), sort_keys=True, ensure_ascii=False))


def _data_audit_payload(*, spec: DatasetSpec, quality: Mapping[str, Any], phase: str) -> dict[str, Any]:
    target = spec.target
    return {
        "dataset": spec.to_dict(),
        "label": None if target is None else target.to_dict(),
        "label_honesty": {
            "synthetic": spec.synthetic,
            "label_origin": None if target is None else target.label_origin.value,
            "may_produce_formal_facility_kw": (
                None if target is None else target.is_facility_active_power
            ),
        },
        "unit_policy": {
            "output_unit": "kW",
            "conversion": "W/MW converted to kW with contracts.to_kw; unitless utilisation is never converted",
            "rounding": "none",
        },
        "time": {
            "trace_anchor_timestamp": spec.trace_anchor_timestamp,
            "timezone": spec.timezone,
            "timezone_verified": spec.timezone_verified,
            "native_frequency": spec.native_frequency,
            "interval_semantics": spec.interval_semantics,
            "availability_semantics": spec.availability_semantics,
        },
        "quality": dict(quality),
        "phase": phase,
    }


def _split_audit_payload(
    *, spec: DatasetSpec, frame: pd.DataFrame, plan: OriginPlan, run_phase: str
) -> dict[str, Any]:
    return {
        "protocol_id": plan.protocol_id,
        "horizon": plan.horizon,
        "horizon_points": plan.horizon_points,
        "context_points": plan.context_points,
        "anchor": str(plan.anchor),
        "run_phase": run_phase,
        "geometry_resolved_for_phase": plan.phase,
        "origins": [str(origin) for origin in plan.origins],
        "origin_days": list(plan.days),
        "label_start": str(plan.label_start),
        "label_end": str(plan.label_end),
        "overlap": plan.overlap,
        "dropped": list(plan.dropped),
        "plan": plan.audit(),
        "frame": audit_split(spec, frame, plan),
    }


def _verify_prediction_metrics(
    *, frame: pd.DataFrame, aggregate: Mapping[str, Any], mape_relative_threshold: float
) -> dict[str, Any]:
    """Recompute metrics from the saved prediction rows and compare."""
    valid = frame.loc[frame["valid_label"].astype(bool)]
    if valid.empty:
        return {"checked": False, "reason": "no valid prediction rows"}
    recomputed = forecast_metrics(
        valid["actual"].to_numpy(dtype=float),
        valid["predicted"].to_numpy(dtype=float),
        mape_relative_threshold=mape_relative_threshold,
    )
    differences: dict[str, float | None] = {}
    matches = True
    for key in ("MAPE", "WAPE", "MAE", "RMSE"):
        saved, again = aggregate.get(key), recomputed.get(key)
        if saved is None and again is None:
            differences[key] = None
            continue
        if saved is None or again is None:
            matches = False
            differences[key] = None
            continue
        delta = abs(float(saved) - float(again))
        differences[key] = delta
        if delta > 1e-9:
            matches = False
    return {
        "checked": True,
        "rows": int(len(valid)),
        "matches_metrics_json": matches,
        "absolute_differences": differences,
        "recomputed": recomputed,
    }


def _model_training_feasibility(protocol: Protocol, plan: OriginPlan, *, model_requires_fit: bool) -> dict:
    if not model_requires_fit:
        return {
            "requires_fit": False,
            "note": "parameter-free baseline: no supervised label window is created, so no "
            "training-window shortage can occur",
        }
    audit = training_window_audit(
        train_cutoff_day=max(plan.days) if plan.days else 0,
        context_points=plan.context_points,
        horizon_points=plan.horizon_points,
        points_per_day=protocol.points_per_day,
    )
    return {"requires_fit": True, **audit}


# ---------------------------------------------------------------------------
# main flow
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    seed = _resolve_seed(config, args)
    phase = _resolve_phase(args, config)
    policy = dict(config.get("policy") or {})
    evaluation = dict(config.get("evaluation") or {})
    mape_threshold = float(evaluation.get("mape_relative_threshold", 0.0))
    batch_size = int(evaluation.get("prediction_batch_size", 64))
    clip_min = evaluation.get("clip_min")
    clip_min = None if clip_min is None else float(clip_min)
    include_calendar = bool(evaluation.get("include_calendar", True))

    if phase not in SUPPORTED_PHASES:
        raise ExperimentError(f"Unknown phase {phase!r}.")

    seed_everything(seed)

    dataset = _build_dataset(config, seed=seed)
    spec: DatasetSpec = dataset.spec
    frame: pd.DataFrame = dataset.frame

    if spec.synthetic and phase != "smoke" and not (args.demo or bool(policy.get("demo_only", False))):
        raise ExperimentError(
            "Synthetic labels may only be used for a demo/smoke run. Pass --demo (or use "
            "configs/smoke.yaml) for engineering validation, or register a verified real "
            "facility-kW dataset before a formal run."
        )

    protocol = _build_protocol(config, args)
    horizon_requested = args.horizon or str(config.get("horizon") or "24h")
    grid = grid_for_frame(spec, frame)
    # 'finalize' must know the declared test geometry (so the freeze record covers
    # the right horizon/lead set) but never reads test labels; the geometry plan is
    # therefore resolved with the same origin set final-test will use.
    planning_phase = "final-test" if phase in ("finalize", "final-test") else phase
    plan = plan_origins(
        spec,
        grid,
        protocol,
        phase=planning_phase,
        horizon=protocol.horizon_name(horizon_requested),
        frame=frame,
    )
    horizon_name = plan.horizon

    model_name = str(args.model or config.get("model") or "seasonal_naive")
    model = build_forecaster(model_name)

    quality = quality_report(frame, spec)
    research_result, blocked_reasons = _research_result_allowed(
        spec=spec, phase=phase, policy=policy
    )

    frozen_run_path: Path | None = None
    frozen_record: dict[str, Any] = {}
    if phase == "final-test":
        frozen_run_path = _resolve_frozen_run(args)
        frozen_record = require_frozen_record(frozen_run_path)

    output_root = resolve_path(args.output or config.get("output_root") or "results/runs")
    run_dir = RunDirectory.create(output_root)
    print(
        f"[{run_dir.run_id}] phase={phase} model={model_name} horizon={horizon_name} "
        f"synthetic={spec.synthetic} research_result={research_result}"
    )

    artifact_digests: dict[str, str] = {}
    phases_executed: list[str] = []
    warnings_list: list[str] = list(blocked_reasons)
    try:
        effective_config = {
            **config,
            "phase": phase,
            "demo": bool(args.demo),
            "seed": seed,
            "horizon": horizon_name,
            "model": model_name,
            "protocol_config": args.protocol or config.get("protocol_config"),
        }
        # The frozen credential must be verifiable by a *later* run whose phase
        # necessarily differs (finalize/smoke -> final-test), so the hashed
        # snapshot excludes per-run metadata while keeping every research choice.
        config_hash_excluded = ("phase", "demo")
        hash_config = {
            key: value for key, value in effective_config.items() if key not in config_hash_excluded
        }
        config_sha = config_digest(hash_config)
        dataset_sha = _payload_sha(spec.to_dict())
        protocol_sha = _payload_sha(protocol.to_dict())

        data_audit = _data_audit_payload(spec=spec, quality=quality, phase=phase)
        run_dir.write_data_audit(data_audit)
        split_audit = _split_audit_payload(spec=spec, frame=frame, plan=plan, run_phase=phase)
        run_dir.write_split_audit(split_audit)
        if plan.dropped:
            warnings_list.extend(
                f"origin D{entry.get('day')} dropped: {entry.get('reason')}" for entry in plan.dropped
            )
        if plan.overlap:
            warnings_list.append(
                "forecast windows overlap and reuse ground truth; error points are not "
                "independent observations"
            )

        batch: ForecastBatch | None = None
        if phase == "finalize":
            run_dir.write_json(
                "batch_summary.json",
                {
                    "skipped": True,
                    "reason": (
                        "finalize freezes model/configuration state and deliberately does not "
                        "construct the labelled test batch; test labels are only read by "
                        "final-test after the frozen credential has been verified."
                    ),
                },
            )
        else:
            batch = build_forecast_batch(
                spec, frame, plan, include_calendar=include_calendar
            )
            run_dir.write_json("batch_summary.json", batch.metadata_dict())

        freeze_phase = "smoke" if phase == "smoke" else "finalize"
        frozen_payload: dict[str, Any] = {}
        test_report: dict[str, Any] = {}
        aggregate: dict[str, Any] = {}
        predictions: np.ndarray | None = None
        runner_diagnostics: dict[str, Any] = {}
        metric_check: dict[str, Any] = {}

        if phase in ("finalize", "smoke"):
            frozen_payload = freeze_model(
                model,
                run_dir=run_dir,
                config=hash_config,
                protocol=protocol.to_dict(),
                dataset=spec.to_dict(),
                phase=freeze_phase,
                seed=seed,
                horizon=horizon_name,
                extra={
                    "synthetic": spec.synthetic,
                    "research_result": research_result,
                    "horizon_points": plan.horizon_points,
                    "context_points": plan.context_points,
                    "origins": [str(origin) for origin in plan.origins],
                },
            )
            phases_executed.append(freeze_phase)
            frozen_path = run_dir.path / "frozen_before_test.json"
            artifact_digests["frozen_before_test.json"] = sha256_text(
                frozen_path.read_text(encoding="utf-8")
            )

        if phase in ("smoke", "final-test", "development"):
            if phase == "final-test":
                frozen = load_frozen(
                    frozen_run_path,
                    expected_config=hash_config,
                    expected_protocol=protocol.to_dict(),
                    expected_dataset=spec.to_dict(),
                )
                model = frozen.model
                test_report.update(
                    {
                        "frozen_run": str(frozen_run_path),
                        "frozen_before_test_sha256": sha256_file(
                            frozen_run_path / "frozen_before_test.json"
                        ),
                        "model_reloaded_from_disk": True,
                        "hashes_verified": True,
                    }
                )
            else:
                frozen = load_frozen(
                    run_dir.path,
                    expected_config=hash_config,
                    expected_protocol=protocol.to_dict(),
                    expected_dataset=spec.to_dict(),
                )
                model = frozen.model
                test_report.update(
                    {
                        "frozen_run": str(run_dir.path),
                        "frozen_before_test_sha256": sha256_file(
                            run_dir.path / "frozen_before_test.json"
                        ),
                        "model_reloaded_from_disk": True,
                        "hashes_verified": True,
                    }
                )
            phases_executed.append("final-test")
            predictions, runner_diagnostics = predict_batch(
                model, batch.inputs, batch_size=batch_size, clip_min=clip_min
            )
            report = score_batch(
                batch, predictions, mape_relative_threshold=mape_threshold
            )
            aggregate = dict(report.aggregate)
            predictions_table = predictions_frame(
                batch,
                predictions,
                run_id=run_dir.run_id,
                dataset_id=spec.dataset_id,
                model=model_name,
                protocol_id=protocol.protocol_id,
                horizon=horizon_name,
            )
            prediction_path = run_dir.write_predictions(predictions_table)
            artifact_digests[prediction_path.name] = sha256_file(prediction_path)
            metrics_json_path, metrics_csv_path = run_dir.write_metrics(report)
            for artifact in (metrics_json_path, metrics_csv_path):
                artifact_digests[artifact.name] = sha256_text(
                    artifact.read_text(encoding="utf-8")
                )
            metric_check = _verify_prediction_metrics(
                frame=predictions_table,
                aggregate=aggregate,
                mape_relative_threshold=mape_threshold,
            )
            test_report.update(
                {
                    "phase": phase,
                    "predictions_checked": True,
                    "metric_recomputation": metric_check,
                    "runner_diagnostics": runner_diagnostics,
                    "metrics": aggregate,
                    "metrics_levels": {
                        "per_origin_rows": int(len(report.per_origin)),
                        "per_lead_rows": int(len(report.per_lead)),
                        "per_forecast_day_rows": int(len(report.per_forecast_day)),
                        "daily_mean_kind": report.daily_mean_kind,
                        "overlap": report.overlap,
                        "labels": report.labels,
                    },
                }
            )
            if not metric_check.get("matches_metrics_json", False):
                raise ExperimentError(
                    "Saved predictions do not reproduce metrics.json; refusing to report a "
                    f"completed run. Detail: {metric_check}"
                )
            run_dir.write_test_report(test_report)

        manifest = {
            "run_id": run_dir.run_id,
            "experiment_id": config.get("experiment_id", "unnamed"),
            "status": "running",
            "phase": phase,
            "phases_executed": phases_executed,
            "demo": bool(args.demo),
            "synthetic": spec.synthetic,
            "research_result": research_result,
            "created_at": utc_timestamp(),
            "environment": environment_report(seed=seed),
            "config": jsonable(effective_config),
            "config_sha256": config_sha,
            "config_hash_policy": {
                "excluded_keys": list(config_hash_excluded),
                "reason": "per-run metadata is excluded so a later final-test run can verify the "
                "same frozen research configuration",
                "hashed_snapshot": jsonable(hash_config),
            },
            "protocol": protocol.to_dict(),
            "protocol_sha256": protocol_sha,
            "dataset": spec.to_dict(),
            "dataset_sha256": dataset_sha,
            "data_source": {
                "kind": "synthetic" if spec.synthetic else "unverified",
                "rows": int(len(frame)),
                "quality": quality,
                "provenance": dict(spec.provenance),
            },
            "model": model.describe(),
            "model_name": model_name,
            "model_registry": describe_models(),
            "horizon": horizon_name,
            "horizon_points": plan.horizon_points,
            "context_points": plan.context_points,
            "origins": [str(origin) for origin in plan.origins],
            "origin_days": list(plan.days),
            "anchor": str(plan.anchor),
            "label_start": str(plan.label_start),
            "label_end": str(plan.label_end),
            "overlap": plan.overlap,
            "dropped_origins": list(plan.dropped),
            "evaluation": {
                "mape_relative_threshold": mape_threshold,
                "prediction_batch_size": batch_size,
                "clip_min": clip_min,
                "include_calendar": include_calendar,
            },
            "training_feasibility": _model_training_feasibility(
                protocol, plan, model_requires_fit=bool(getattr(model, "requires_fit", False))
            ),
            "frozen": frozen_payload,
            "frozen_run": None if frozen_run_path is None else str(frozen_run_path),
            "test_report": test_report,
            "metrics": aggregate,
            "metric_recomputation": metric_check,
            "artifact_sha256": artifact_digests,
            "label_policy": {
                "unit": None if spec.target is None else spec.target.unit,
                "scope": None if spec.target is None else spec.target.scope.value,
                "label_origin": None if spec.target is None else spec.target.label_origin.value,
                "measurement_boundary": None
                if spec.target is None
                else spec.target.measurement_boundary,
            },
            "warnings": warnings_list,
        }
        run_dir.write_manifest(manifest)
        run_dir.finalize(status="completed")
        print(f"[{run_dir.run_id}] status=completed")
        if aggregate:
            _print_metrics(aggregate, spec=spec, phase=phase, research_result=research_result)
        print(f"run directory: {run_dir.path}")
        if spec.synthetic:
            print(
                "[synthetic] Engineering validation only (synthetic=true, research_result=false). "
                "This is not a real facility-kW research result."
            )
        return 0
    except (ContractError, FrozenRunError, ExperimentError, ValueError, KeyError, OSError, NotImplementedError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        try:
            run_dir.finalize(status="failed", error=message)
        except Exception:  # pragma: no cover - best effort bookkeeping
            pass
        print(f"ERROR: {message}", file=sys.stderr)
        return 1


def _resolve_frozen_run(args: argparse.Namespace) -> Path:
    candidate = args.frozen_run or args.run_dir
    if not candidate:
        raise ExperimentError(
            "phase 'final-test' requires --frozen-run (or --run-dir) pointing to an existing "
            "frozen run directory. Running final-test without frozen credentials is refused."
        )
    path = resolve_path(candidate)
    if not (path / "frozen_before_test.json").is_file():
        raise FrozenRunError(
            f"{path} has no frozen_before_test.json; final-test must load an existing frozen run."
        )
    return path


def _print_metrics(
    aggregate: Mapping[str, Any], *, spec: DatasetSpec, phase: str, research_result: bool
) -> None:
    def fmt(key: str, unit: str = "") -> str:
        value = aggregate.get(key)
        return "undefined" if value is None else f"{float(value):.4f}{unit}"

    print(
        f"metrics (n={aggregate.get('n')}): MAPE={fmt('MAPE', '%')} WAPE={fmt('WAPE', '%')} "
        f"MAE={fmt('MAE', ' kW')} RMSE={fmt('RMSE', ' kW')}"
    )
    print(f"scope={spec.target.scope.value if spec.target else None} unit=kW phase={phase} research_result={research_result}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single official experiment entry point for the rebuilt PowerForecast skeleton. "
            "Synthetic/demo runs always carry synthetic=true and research_result=false."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/base.yaml", help="experiment configuration YAML")
    parser.add_argument("--protocol", default=None, help="optional protocol YAML overriding the config")
    parser.add_argument(
        "--horizon",
        default=None,
        help=f"forecast horizon name, one of {HORIZON_NAMES} (validated against the protocol)",
    )
    parser.add_argument("--context-points", type=int, default=None, help="history length in 15-minute points")
    parser.add_argument("--origins", default=None, help="comma-separated trace-day origin numbers")
    parser.add_argument("--phase", choices=SUPPORTED_PHASES, default=None, help="run phase")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="safe alias: synthetic data only; equals 'smoke' when --phase is omitted",
    )
    parser.add_argument("--model", default=None, help="model name (see --list-models)")
    parser.add_argument("--run-dir", default=None, help="existing frozen run dir (final-test) or target run dir")
    parser.add_argument("--frozen-run", default=None, help="existing frozen run dir for final-test")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--output", default=None, help="root directory for new run directories")
    parser.add_argument("--list-models", action="store_true", help="print the model registry and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        print("implemented:")
        for name in implemented_models():
            print(f"  - {name}")
        print("planned (NOT implemented, cannot be selected):")
        for name in planned_models():
            print(f"  - {name}")
        return 0
    try:
        return run_experiment(args)
    except (ContractError, FrozenRunError, ExperimentError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
