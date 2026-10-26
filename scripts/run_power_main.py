"""Stage-2 official entry point: the historical-power main model A.

Runs the zero-shot main model on the **real** NLR ESIF derived facility power
under the pre-declared fixed D14/D22 protocol (task book §4–§7):

* main path only (history kW -> future kW); no CPU/GPU/task features are read;
* Chronos-2 and TiRex-2 are loaded from frozen local checkpoints, P50 medians;
* the frozen 0.5/0.5 equal-weight ensemble is model A3 (fixed, never searched);
* simple optional baselines B0 ``seasonal_naive`` and B1 ``three_day_mean`` run
  on exactly the same origins/labels/metric code for honest comparison;
* per-horizon origin day numbers come from the experiment config, never from
  dataset peeking.

Examples::

    python -m scripts.run_power_main --config configs/experiments/power_main_v1.yaml
    python -m scripts.run_power_main --horizons 24h 3d --splits test
    python -m scripts.run_power_main --models seasonal_naive power_ensemble
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/run_power_main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.contracts import (
    TARGET_COLUMN,
    AuditStatus,
    DatasetRole,
    DatasetSpec,
    ForecastBatch,
    LabelOrigin,
    TargetKind,
    TargetScope,
    TargetSpec,
)
from src.data.labels import LabelType, classify_target, require_main_model_label
from src.data.splitting import Protocol, grid_for_frame
from src.data.windows import build_forecast_batch, plan_origins
from src.evaluation.backtest import score_batch
from src.evaluation.metrics import forecast_metrics
from src.evaluation.power_run import (
    METRIC_KEYS,
    PREDICTION_COLUMNS,
    PowerRunDirectory,
    PowerRunError,
    aggregate_rows,
    daily_rows,
    improvement_rows,
    manifest_payload,
    power_predictions_frame,
    recompute_aggregate,
    verify_saved_predictions,
    write_json_safe,
)
from src.evaluation.visualization import generate_all_figures
from src.utils import (
    config_digest,
    environment_report,
    jsonable,
    load_config,
    read_json,
    resolve_path,
    seed_everything,
    sha256_file,
    utc_timestamp,
    write_json,
)

DEFAULT_CONFIG = "configs/experiments/power_main_v1.yaml"
HORIZON_ORDER: tuple[str, ...] = ("4h", "8h", "24h", "3d", "7d")
SPLITS: tuple[str, ...] = ("development", "test")
PLAN_PHASE: dict[str, str] = {"development": "development", "test": "final-test"}


class ExperimentError(RuntimeError):
    """A Stage-2 experiment cannot run as configured."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    payload = load_config(path)
    parent = payload.pop("inherit", None)
    if parent:
        payload = deep_merge(load_experiment_config(parent), payload)
    return payload


# ---------------------------------------------------------------------------
# target data
# ---------------------------------------------------------------------------


def load_target_experiment(
    config: Mapping[str, Any], *, target_power: str | Path | None = None
) -> tuple[DatasetSpec, pd.DataFrame, dict[str, Any], str, pd.Timestamp]:
    """Load the Stage-1 target artefact into a Stage-2 ``DatasetSpec`` + frame.

    The frame keeps only the label column (``active_power_kw``) plus masks; no
    component or operational feature ever enters the main model's inputs.
    """
    data_cfg = dict(config.get("data") or {})
    power_path = resolve_path(
        target_power or data_cfg.get("target_power") or "data/processed/processed_target_power.parquet"
    )
    if not power_path.is_file():
        raise ExperimentError(
            f"Stage-1 target artefact is missing: {power_path}. Run "
            "`python -m scripts.prepare_target_power --config configs/datasets/nlr_target.yaml` first."
        )
    frame = pd.read_parquet(power_path)
    if "timestamp" in frame.columns:
        frame = frame.set_index(pd.DatetimeIndex(frame["timestamp"]))
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ExperimentError(f"{power_path} must be indexed by a DatetimeIndex of interval starts.")
    index = frame.index.tz_convert("UTC").tz_localize(None) if frame.index.tz is not None else frame.index
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(index, name="timestamp")
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ExperimentError(f"{power_path} contains duplicate timestamps.")

    if TARGET_COLUMN not in frame.columns:
        if "power_kw" not in frame.columns:
            raise ExperimentError(f"{power_path} has neither {TARGET_COLUMN!r} nor 'power_kw'.")
        frame[TARGET_COLUMN] = frame["power_kw"]
    if "power_valid" in frame.columns:
        valid = frame["power_valid"].astype(bool).to_numpy()
    else:
        valid = np.isfinite(frame[TARGET_COLUMN].to_numpy(dtype=float))
    values = frame[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(values[valid]).all():
        raise ExperimentError("target power is non-finite where power_valid is True.")
    # The main path only needs the label column; component/operational columns must
    # never leak into the window builder or the figures.
    frame = pd.DataFrame(
        {TARGET_COLUMN: np.where(valid, values, np.nan)}, index=frame.index
    )

    metadata_path = resolve_path(data_cfg.get("target_metadata") or "data/processed/target_metadata.json")
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    dataset_id = str(metadata.get("dataset_id") or data_cfg.get("target_dataset") or "nlr_esif")
    site_id = str(metadata.get("site_id") or metadata.get("domain_id") or dataset_id)
    label_metadata = dict(metadata.get("label") or {})

    target_spec = TargetSpec(
        kind=TargetKind.ACTIVE_POWER,
        scope=TargetScope.FACILITY,
        unit="kW",
        label_origin=LabelOrigin.COMPONENT_SUM,
        measurement_boundary=str(
            label_metadata.get("measurement_boundary")
            or metadata.get("measurement_boundary")
            or "sum of NLR ESIF facility sub-meter components (derived, not an independent total meter)"
        ),
        sampling_semantics=str(
            label_metadata.get("sampling_semantics")
            or metadata.get("sampling_semantics")
            or "15-minute interval mean of ~1-minute component samples"
        ),
        synthetic=False,
    )
    declared = str(metadata.get("label_type") or "").strip()
    label_type = declared or classify_target(target_spec).value
    try:
        label_enum = LabelType(label_type)
    except ValueError as exc:
        raise ExperimentError(f"unknown label type {label_type!r} in {metadata_path}.") from exc
    require_main_model_label(label_enum, dataset_id=dataset_id)

    anchor_value = (
        metadata.get("trace_anchor_timestamp")
        or metadata.get("anchor_timestamp")
        or (metadata.get("time") or {}).get("anchor_timestamp")
    )
    if anchor_value is not None:
        anchor = pd.Timestamp(anchor_value)
        if anchor.tzinfo is not None:
            anchor = anchor.tz_convert("UTC").tz_localize(None)
    else:
        first_valid = frame[TARGET_COLUMN].first_valid_index()
        if first_valid is None:
            raise ExperimentError("the target power series has no valid observation.")
        anchor = pd.Timestamp(first_valid)

    provenance = {
        "label_type": label_type,
        "label_honesty_note": label_metadata.get("honesty_note"),
        "components_used_as_features": False,
        "target_power_file": str(power_path),
        "target_metadata_file": str(metadata_path),
    }
    quality_config = data_cfg.get("target_quality")
    if quality_config:
        provenance["quality_file"] = str(resolve_path(quality_config))

    spec = DatasetSpec(
        dataset_id=dataset_id,
        site_id=site_id,
        domain_id=str(metadata.get("domain_id") or dataset_id),
        role=DatasetRole.TARGET_SUPERVISED,
        native_frequency="15min",
        trace_anchor_timestamp=anchor.isoformat(),
        timezone=str(metadata.get("timezone") or "UTC"),
        timezone_verified=bool(metadata.get("timezone_verified", False)),
        interval_semantics=str(
            metadata.get("interval_semantics")
            or "each timestamp is the start of a 15-minute interval [t, t+15min)"
        ),
        availability_semantics=str(
            metadata.get("availability_semantics")
            or "interval mean becomes available after the interval ends"
        ),
        features={},  # main path A: history kW only
        target=target_spec,
        source_version=str(metadata.get("source_version") or "nlr_esif_stage1"),
        provenance=provenance,
        audit_status=AuditStatus.PARTIAL
        if not bool(metadata.get("timezone_verified", False))
        else AuditStatus.VERIFIED,
    )
    return spec, frame, metadata, label_type, anchor


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------


def protocol_for(config: Mapping[str, Any], horizon: str) -> Protocol:
    protocol_cfg = dict(config.get("protocol") or {})
    path = protocol_cfg.get("config") or "configs/protocols/power_main_fixed.yaml"
    payload = dict(load_config(path))
    origins = dict(protocol_cfg.get("origins") or {})
    if horizon not in origins:
        raise ExperimentError(
            f"no pre-declared origin days for horizon {horizon!r} in the experiment config."
        )
    entry = origins[horizon]
    payload["development_origins"] = [int(day) for day in entry.get("development", [])]
    payload["test_origins"] = [int(day) for day in entry.get("test", [])]
    payload["protocol_type"] = payload.get("protocol_type", "fixed_d14_d22")
    context_points = int(protocol_cfg.get("context_points") or payload.get("context_points") or 1344)
    try:
        return Protocol.from_config(payload, horizon=horizon, context_points=context_points)
    except ValueError as exc:
        raise ExperimentError(f"invalid Stage-2 protocol for {horizon!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# forecasters
# ---------------------------------------------------------------------------


def default_forecaster_factory(
    config: Mapping[str, Any], args: argparse.Namespace, model_names: Sequence[str]
) -> dict[str, Any]:
    from src.models.power_main import build_power_forecasters

    models_cfg = dict(config.get("models") or {})
    weights_cache = args.weights_cache or models_cfg.get("weights_cache")
    if not weights_cache:
        raise ExperimentError("no weights cache configured; set models.weights_cache or --weights-cache.")
    overrides: dict[str, Any] = {"weights_cache": weights_cache, "models": list(model_names)}
    if args.allow_download:
        overrides["allow_download"] = True
    runtime = dict(config.get("runtime") or {})
    overrides.setdefault("device", runtime.get("device", "cpu"))
    overrides.setdefault("jobs", int(runtime.get("jobs", 4)))
    return build_power_forecasters(config, **overrides)


def validate_forecasters(forecasters: Mapping[str, Any], selected: Sequence[str]) -> None:
    missing = [name for name in selected if name not in forecasters]
    if missing:
        raise ExperimentError(
            f"forecaster factory did not provide {missing}; available: {sorted(forecasters)}."
        )
    for name in selected:
        model = forecasters[name]
        if not callable(getattr(model, "predict", None)):
            raise ExperimentError(f"forecaster {name!r} has no callable predict().")
        if bool(getattr(model, "requires_fit", False)):
            raise ExperimentError(
                f"forecaster {name!r} requires fit(); Stage 2 is zero-shot and never fits on target data."
            )


def ensemble_weight_check(forecasters: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any] | None:
    ensemble_cfg = config.get("ensemble") or {}
    if "power_ensemble" not in forecasters:
        return None
    description = forecasters["power_ensemble"].describe() if hasattr(forecasters["power_ensemble"], "describe") else {}
    weights = description.get("weights")
    expected = [float(ensemble_cfg.get("chronos2_weight", 0.5)), float(ensemble_cfg.get("tirex2_weight", 0.5))]
    if weights is None:
        raise ExperimentError("power_ensemble must describe its fixed weights.")
    if not np.allclose(np.asarray(weights, dtype=float), np.asarray(expected, dtype=float)):
        raise ExperimentError(
            f"power_ensemble weights {weights} differ from the frozen 0.5/0.5 equal-weight policy "
            f"{expected}."
        )
    if not np.isclose(sum(float(weight) for weight in weights), 1.0):
        raise ExperimentError(f"ensemble weights must sum to 1, got {weights}.")
    return description


# ---------------------------------------------------------------------------
# one split x horizon
# ---------------------------------------------------------------------------


def _origin_metric_rows(prediction_frame: pd.DataFrame, *, mape_relative_threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for origin, group in prediction_frame.groupby("origin_time", sort=True):
        valid = group.loc[group["valid_label"].astype(bool)]
        if valid.empty:
            continue
        metrics = forecast_metrics(
            valid["actual_kw"].to_numpy(dtype=float),
            valid["prediction_kw"].to_numpy(dtype=float),
            mape_relative_threshold=mape_relative_threshold,
        )
        rows.append(
            {
                "split": str(group["split"].iloc[0]),
                "horizon": str(group["horizon"].iloc[0]),
                "model": str(group["model"].iloc[0]),
                "origin_time": pd.Timestamp(origin),
                "origin_day": int(group["origin_day"].iloc[0]),
                **{key: metrics.get(key) for key in METRIC_KEYS},
            }
        )
    return rows


def run_split_horizon(
    *,
    split: str,
    horizon: str,
    spec: DatasetSpec,
    frame: pd.DataFrame,
    grid: Any,
    protocol: Protocol,
    forecasters: Mapping[str, Any],
    model_names: Sequence[str],
    run_id: str,
    label_type: str,
    anchor: pd.Timestamp,
    mape_relative_threshold: float,
    clip_negative: bool,
    log: list[str],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    plan = plan_origins(
        spec, grid, protocol, phase=PLAN_PHASE[split], horizon=horizon, frame=frame
    )
    batch = build_forecast_batch(spec, frame, plan, include_calendar=False)
    skipped: list[dict[str, Any]] = []
    usable = np.asarray(batch.inputs.past_power_valid, dtype=bool).all(axis=1)
    if not usable.all():
        for position in np.where(~usable)[0]:
            skipped.append(
                {
                    "split": split,
                    "horizon": horizon,
                    "origin": str(plan.origins[position]),
                    "reason": "incomplete 14-day context history for a zero-shot foundation model",
                }
            )
        keep = np.where(usable)[0]
        if keep.size == 0:
            raise ExperimentError(
                f"no origin with complete context for split={split!r} horizon={horizon!r}."
            )
        plan = _plan_subset(plan, keep)
        batch = batch.slice(keep)
        log.append(f"skipped {len(skipped)} incomplete-context origins for {split}/{horizon}")
    horizon_points = batch.horizon
    context = np.asarray(batch.inputs.past_power_kw, dtype=float)
    models = [name for name in model_names if name in forecasters]
    frames: list[pd.DataFrame] = []
    origin_rows: list[dict[str, Any]] = []
    timings: dict[str, Any] = {}
    for name in models:
        model = forecasters[name]
        started = time.perf_counter()
        raw = np.asarray(model.predict(context, horizon_points), dtype=float)
        elapsed = time.perf_counter() - started
        if raw.shape != (batch.batch_size, horizon_points):
            raise ExperimentError(
                f"model {name!r} returned shape {raw.shape}, expected {(batch.batch_size, horizon_points)} "
                f"for split={split!r} horizon={horizon!r}."
            )
        if not np.isfinite(raw).all():
            raise ExperimentError(f"model {name!r} returned non-finite predictions.")
        raw_column = raw.copy()
        final = np.maximum(raw, 0.0) if clip_negative else raw
        total = int(final.size)
        timings[name] = {
            "split": split,
            "horizon": horizon,
            "seconds": round(elapsed, 3),
            "batch": int(batch.batch_size),
            "horizon_points": int(horizon_points),
            "negative_predictions": int(np.count_nonzero(raw < 0.0)),
            "negative_fraction": round(float(np.count_nonzero(raw < 0.0) / max(total, 1)), 6),
            "clipped": bool(clip_negative),
        }
        prediction_frame = power_predictions_frame(
            batch,
            final,
            run_id=run_id,
            dataset_id=spec.dataset_id,
            site_id=spec.site_id,
            label_type=label_type,
            model=name,
            split=split,
            horizon=horizon,
            anchor=anchor,
            prediction_raw=raw_column,
            clip_negative=clip_negative,
        )
        frames.append(prediction_frame)
        origin_rows.extend(_origin_metric_rows(prediction_frame, mape_relative_threshold=mape_relative_threshold))
        # Independent cross-check with the evaluation layer's own scorer.
        report = score_batch(batch, final, mape_relative_threshold=mape_relative_threshold)
        pooled = recompute_aggregate(
            prediction_frame,
            split=split,
            horizon=horizon,
            model=name,
            mape_relative_threshold=mape_relative_threshold,
        )
        for key in ("MAPE", "WAPE", "MAE", "RMSE"):
            saved, again = report.aggregate.get(key), pooled.get(key)
            if saved is None and again is None:
                continue
            if saved is None or again is None or abs(float(saved) - float(again)) > 1e-9:
                raise ExperimentError(
                    f"metric cross-check failed for {split}/{horizon}/{name}: {key} "
                    f"score_batch={saved} recomputed={again}."
                )
        log.append(
            f"{split}/{horizon}/{name}: n={pooled['n']} MAPE={pooled['MAPE']} WAPE={pooled['WAPE']} "
            f"MAE={pooled['MAE']:.4f} kW RMSE={pooled['RMSE']:.4f} kW ({elapsed:.2f}s)"
        )
    plan_payload = {
        "days": list(plan.days),
        "origins": [str(origin) for origin in plan.origins],
        "n_origins": int(len(plan.days)),
        "overlap": bool(plan.overlap),
        "label_start": str(plan.label_start),
        "label_end": str(plan.label_end),
        "dropped_by_geometry": list(plan.dropped),
    }
    return frames, origin_rows, skipped, {"plan": plan_payload, "timings": timings}


def _plan_subset(plan: Any, keep: np.ndarray) -> Any:
    """Subset an OriginPlan while preserving its audit semantics."""
    from src.data.windows import OriginPlan

    days = tuple(plan.days[index] for index in keep)
    origins = plan.origins[keep]
    return OriginPlan(
        phase=plan.phase,
        protocol_id=plan.protocol_id,
        horizon=plan.horizon,
        horizon_points=plan.horizon_points,
        days=days,
        origins=origins,
        anchor=plan.anchor,
        context_points=plan.context_points,
        label_start=min(pd.Timestamp(origin) for origin in origins),
        label_end=max(pd.Timestamp(origin) + (plan.horizon_points - 1) * plan.step for origin in origins),
        required_history_points=plan.required_history_points,
        dropped=plan.dropped,
        overlap=plan.overlap,
        frequency=plan.frequency,
        points_per_day=plan.points_per_day,
    )


# ---------------------------------------------------------------------------
# main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    args: argparse.Namespace,
    *,
    forecaster_factory: Callable[[Mapping[str, Any], argparse.Namespace, Sequence[str]], Mapping[str, Any]]
    | None = None,
) -> int:
    config = load_experiment_config(args.config)
    runtime = dict(config.get("runtime") or {})
    offline = not bool(getattr(args, "allow_download", False)) and bool(runtime.get("offline", True))
    if offline:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("MPLBACKEND", "Agg")
    seed = int(args.seed if args.seed is not None else (config.get("experiment") or {}).get("seed", 42))
    seed_everything(seed)
    evaluation_cfg = dict(config.get("evaluation") or {})
    mape_threshold = float(evaluation_cfg.get("mape_relative_threshold", 0.0))
    postprocess_cfg = dict(config.get("postprocess") or {})
    clip_negative = bool(postprocess_cfg.get("clip_negative", False))

    spec, frame, metadata, label_type, anchor = load_target_experiment(
        config, target_power=args.target_power
    )
    grid = grid_for_frame(spec, frame)

    horizons = list(args.horizons) if args.horizons else list(
        (config.get("protocol") or {}).get("horizons") or ["4h", "8h", "24h", "3d"]
    )
    unknown = [horizon for horizon in horizons if horizon not in HORIZON_ORDER]
    if unknown:
        raise ExperimentError(f"unknown horizons {unknown}; expected names from {HORIZON_ORDER}.")
    horizons = [horizon for horizon in HORIZON_ORDER if horizon in horizons]

    splits = list(args.splits) if args.splits else list(evaluation_cfg.get("split") or SPLITS)
    bad_splits = [split for split in splits if split not in SPLITS]
    if bad_splits:
        raise ExperimentError(f"unknown splits {bad_splits}; expected {SPLITS}.")
    splits = [split for split in SPLITS if split in splits]

    model_names = list(args.models) if args.models else list(
        (config.get("models") or {}).get("members")
        or ["seasonal_naive", "three_day_mean", "chronos2", "tirex2", "power_ensemble"]
    )
    factory = forecaster_factory or default_forecaster_factory
    forecasters = dict(factory(config, args, model_names))
    validate_forecasters(forecasters, model_names)
    ensemble_description = ensemble_weight_check(forecasters, config)

    output_root = args.output or (config.get("output") or {}).get("root") or "results/runs"
    run_dir = PowerRunDirectory.create(output_root)
    log: list[str] = [
        f"run_id={run_dir.run_id}",
        f"started_at={utc_timestamp()}",
        f"dataset={spec.dataset_id} site={spec.site_id} label_type={label_type} anchor={anchor}",
        f"horizons={horizons} splits={splits} models={model_names}",
        f"context_points={spec.provenance.get('context_points', 1344)} seed={seed}",
    ]
    print(f"[{run_dir.run_id}] Stage-2 power main model | {spec.dataset_id} | label={label_type}")
    all_frames: list[pd.DataFrame] = []
    all_origin_rows: list[dict[str, Any]] = []
    skipped_all: list[dict[str, Any]] = []
    timings_all: dict[str, Any] = {}
    plans: dict[str, Any] = {}
    manifest_extra_notes: list[str] = []
    started_all = time.perf_counter()
    try:
        for horizon in horizons:
            protocol = protocol_for(config, horizon)
            for split in splits:
                frames, origin_rows, skipped, info = run_split_horizon(
                    split=split,
                    horizon=horizon,
                    spec=spec,
                    frame=frame,
                    grid=grid,
                    protocol=protocol,
                    forecasters=forecasters,
                    model_names=model_names,
                    run_id=run_dir.run_id,
                    label_type=label_type,
                    anchor=anchor,
                    mape_relative_threshold=mape_threshold,
                    clip_negative=clip_negative,
                    log=log,
                )
                all_frames.extend(frames)
                all_origin_rows.extend(origin_rows)
                skipped_all.extend(skipped)
                timings_all.setdefault(horizon, {})[split] = info["timings"]
                plans.setdefault(split, {})[horizon] = info["plan"]
                overlap = info["plan"]["overlap"]
                if overlap:
                    manifest_extra_notes.append(
                        f"{split}/{horizon}: label windows overlap; overlapping truth points are not "
                        "independent statistical samples"
                    )
        predictions = pd.concat(all_frames, ignore_index=True).loc[:, list(PREDICTION_COLUMNS)]
        predictions = predictions.sort_values(["split", "horizon", "model", "origin_time", "lead_step"]).reset_index(drop=True)

        aggregate_rows_list: list[dict[str, Any]] = []
        for (split, horizon, model), _group in predictions.groupby(["split", "horizon", "model"], sort=True):
            aggregate_rows_list.append(
                recompute_aggregate(
                    predictions,
                    split=split,
                    horizon=horizon,
                    model=model,
                    mape_relative_threshold=mape_threshold,
                )
            )
        metrics_aggregate = pd.DataFrame(aggregate_rows_list)
        # add per-origin distribution statistics
        origin_table = pd.DataFrame(all_origin_rows)
        distribution_rows: list[dict[str, Any]] = []
        for row in metrics_aggregate.to_dict("records"):
            subset = origin_table.loc[
                (origin_table["split"] == row["split"])
                & (origin_table["horizon"] == row["horizon"])
                & (origin_table["model"] == row["model"])
            ]
            merged = aggregate_rows(
                report_rows=subset.to_dict("records"),
                horizon_points=int(
                    (predictions.loc[
                        (predictions["split"] == row["split"])
                        & (predictions["horizon"] == row["horizon"])
                        & (predictions["model"] == row["model"]),
                        "lead_step",
                    ].max())
                ),
                split=row["split"],
                horizon=row["horizon"],
            )
            distribution_rows.append({**row, **{k: v for k, v in merged.items() if k not in {"split", "horizon", "n_origins", "horizon_points"}}})
        metrics_aggregate = pd.DataFrame(distribution_rows)
        improvements = improvement_rows(
            metrics_aggregate,
            baselines=list(evaluation_cfg.get("improvement_baselines") or ["seasonal_naive", "three_day_mean"]),
        )
        daily = daily_rows(predictions, mape_relative_threshold=mape_threshold)

        run_dir.write_predictions(predictions)
        run_dir.write_table("metrics.csv", metrics_aggregate)
        run_dir.write_table("origin_metrics.csv", origin_table)
        run_dir.write_table("daily_metrics.csv", daily)
        run_dir.write_table("improvements.csv", improvements)
        quality_files = run_dir.copy_quality(
            resolve_path((config.get("data") or {}).get("target_quality") or "") if (config.get("data") or {}).get("target_quality") else None,
            resolve_path((config.get("data") or {}).get("source_quality") or "") if (config.get("data") or {}).get("source_quality") else None,
        )
        verification = verify_saved_predictions(
            predictions, metrics_aggregate, mape_relative_threshold=mape_threshold
        )
        if not verification["matches"]:
            raise ExperimentError(
                f"saved predictions do not reproduce metrics.csv: {verification['problems'][:3]}"
            )

        figures: list[str] = []
        if args.no_figures or not bool((config.get("output") or {}).get("figures", True)):
            manifest_extra_notes.append("figures disabled for this run")
        else:
            primary_horizon = "24h" if "24h" in horizons else horizons[0]
            produced = generate_all_figures(
                predictions=predictions,
                aggregate=metrics_aggregate,
                daily=daily,
                frame=frame.rename(columns={TARGET_COLUMN: "power_kw"}),
                figures_dir=run_dir.figures_dir,
                context_points=96,
                primary_horizon=primary_horizon,
            )
            figures = [str(path.relative_to(run_dir.path)) for path in produced]

        target_source = {
            "target_power": str(resolve_path((config.get("data") or {}).get("target_power"))),
            "target_power_sha256": sha256_file((config.get("data") or {}).get("target_power")),
            "target_metadata": str(resolve_path((config.get("data") or {}).get("target_metadata"))),
            "target_metadata_sha256": sha256_file((config.get("data") or {}).get("target_metadata")),
            "rows": int(len(frame)),
            "first_timestamp": str(frame.index.min()),
            "last_timestamp": str(frame.index.max()),
            "provenance": metadata,
        }
        protocol_payload = {
            "protocol_file": str(resolve_path((config.get("protocol") or {}).get("config"))),
            "protocol_id": protocol_for(config, horizons[0]).protocol_id,
            "context_points": int((config.get("protocol") or {}).get("context_points", 1344)),
            "horizons": {horizon: protocol_for(config, horizon).horizon_points(horizon) for horizon in horizons},
            "origin_days": {
                split: {horizon: plans[split][horizon]["days"] for horizon in horizons}
                for split in splits
            },
            "plan_details": plans,
        }
        model_descriptions = {
            name: (forecasters[name].describe() if callable(getattr(forecasters[name], "describe", None)) else {"name": name})
            for name in model_names
        }
        manifest = manifest_payload(
            config=config,
            dataset_id=spec.dataset_id,
            site_id=spec.site_id,
            label_type=label_type,
            anchor=anchor,
            target_source=target_source,
            protocol=protocol_payload,
            model_descriptions=model_descriptions,
            ensemble=ensemble_description,
            environment=environment_report(seed=seed),
            origins=plans,
            skipped=skipped_all,
            timings={"total_seconds": round(time.perf_counter() - started_all, 3), "per_call": timings_all},
            figures=figures,
            quality_files=quality_files,
            notes=manifest_extra_notes
            + [
                "main path uses only historical kW; no CPU/GPU/task features were read",
                "Chronos-2 and TiRex-2 weights are frozen (zero-shot, no target-domain training)",
                "fusion weight is the fixed internal 0.5/0.5 main-path coefficient, not a main/auxiliary weight",
                f"negative predictions recorded: {sum(item.get('negative_predictions', 0) for horizon in timings_all.values() for split in horizon.values() for item in split.values())}",
            ],
            command=" ".join([sys.executable, "-m", "scripts.run_power_main", *sys.argv[1:]]),
        )
        manifest["metric_recomputation"] = verification
        manifest["improvements_available"] = bool(len(improvements))
        run_dir.write_log(log + [f"completed_at={utc_timestamp()}"])
        run_dir.finalize(manifest=manifest, status="completed")
        _print_summary(metrics_aggregate, improvements, label_type, run_dir.path)
        return 0
    except (ExperimentError, PowerRunError, ValueError, KeyError, OSError, RuntimeError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        write_json_safe(run_dir.path / "failure.json", {"error": message, "at_utc": utc_timestamp()})
        try:
            run_dir.finalize(
                manifest={
                    "experiment": "power_main_v1",
                    "dataset": {"dataset_id": spec.dataset_id, "label_type": label_type},
                    "config": jsonable(config),
                    "config_sha256": config_digest(config),
                    "synthetic": False,
                    "research_result": False,
                },
                status="failed",
                error=message,
            )
        except Exception:  # pragma: no cover - best-effort bookkeeping
            pass
        print(f"ERROR: {message}", file=sys.stderr)
        return 1


def _print_summary(
    metrics_aggregate: pd.DataFrame, improvements: pd.DataFrame, label_type: str, run_path: Path
) -> None:
    print(f"\nlabel_type={label_type} unit=kW (real data, research_result=True)")
    columns = ["split", "horizon", "model", "MAPE", "WAPE", "MAE", "RMSE", "n"]
    view = metrics_aggregate.loc[:, [column for column in columns if column in metrics_aggregate.columns]]
    if {"split", "horizon", "model"}.issubset(view.columns):
        view = view.sort_values(["split", "horizon", "model"])
        print(view.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if not improvements.empty:
        chosen = improvements.loc[improvements["model"] == "power_ensemble"]
        if not chosen.empty:
            print("\npower_ensemble improvement vs baselines (percent):")
            print(chosen.loc[:, ["split", "horizon", "baseline", "metric", "improvement_pct"]].to_string(
                index=False, float_format=lambda value: f"{value:.2f}"
            ))
    print(f"\nrun directory: {run_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 official experiment: zero-shot historical-power main model A "
            "(Chronos-2 + TiRex-2, fixed 0.5/0.5) under the fixed D14/D22 protocol."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="experiment configuration")
    parser.add_argument("--horizons", nargs="+", default=None, choices=list(HORIZON_ORDER))
    parser.add_argument("--splits", nargs="+", default=None, choices=list(SPLITS))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output", default=None, help="root directory for new run directories")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--weights-cache", default=None, help="local frozen checkpoint cache (HF_HUB_CACHE layout)")
    parser.add_argument("--target-power", default=None, help="override the processed target power parquet")
    parser.add_argument("--offline", action="store_true", default=True, help="never download weights")
    parser.add_argument("--allow-download", action="store_true", help="allow weight download (prepare step only)")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_experiment(args)
    except (ExperimentError, PowerRunError, ValueError, KeyError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
