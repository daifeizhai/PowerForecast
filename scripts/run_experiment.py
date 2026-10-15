"""Adaptive complete-day experiments; all tasks freeze before test inference."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.data.cleaning import quality_report
from src.data.features import make_features
from src.data.loaders import read_table
from src.data.splitting import build_windows, adaptive_day_splits, complete_days
from src.data.synthetic import make_demo_data
from src.evaluation.backtest import backtest, prediction_table, score_predictions
from src.evaluation.reporting import configure_console, print_results, print_saved_results
from src.models import MODEL_NAMES, create_model
from src.training.trainer import train_model
from src.utils import PROJECT_ROOT, config_digest, load_config, resolve_path, seed_everything, write_json


def ablation_covariates(covariates, ablation):
    excluded = {"target_only": set(covariates), "no_weather": {"temperature_c", "humidity_pct"},
                "no_auxiliary": {"cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw"}}
    return [name for name in covariates if name not in excluded.get(ablation, set())]


def save_result(folder, task, name, split, batch, result, rows, seconds, unit, plots):
    model_dir = folder/task/name
    model_dir.mkdir(exist_ok=True, parents=True)
    for evaluation, metrics in (("curve", result.metrics), ("daily_mean", result.daily_metrics)):
        rows.append(dict(horizon=task, model=name, split=split, evaluation=evaluation,
                         train_seconds=seconds, inference_seconds=result.inference_seconds, **metrics))
    for label, table in (("per_lead", result.per_lead), ("per_origin", result.per_origin),
                         ("daily_per_origin", result.daily_per_origin), ("per_day", result.per_day)):
        table.to_csv(model_dir/f"{split}_{label}.csv", index=False)
    table = prediction_table(batch, result.predictions)
    table.to_parquet(model_dir/f"{split}_predictions.parquet", index=False)
    write_json(model_dir/f"{split}_metrics.json", dict(curve=result.metrics, daily_mean=result.daily_metrics,
        nonoverlap=result.nonoverlap_metrics, inference_seconds=result.inference_seconds))
    if split == "test":
        if plots:
            from src.evaluation.visualization import plot_forecast
            plot_forecast(table, model_dir/"forecast.png", f"{name}, {task}", unit)
        fmt = lambda v: "undefined" if v is None else f"{v:.4f}"
        print(f"{task} {name}: MAPE={fmt(result.metrics['MAPE'])}% RMSE={fmt(result.metrics['RMSE'])} {unit} "
              f"WAPE={fmt(result.metrics['WAPE'])}% n={result.metrics['n']}", flush=True)


def run_experiment(args):
    experiment = load_config(args.config)
    if experiment.get("protocol") == "google_hdsf_d14_d22":
        from scripts.reproduce_google_hdsf import run_reproduction
        return run_reproduction(args, experiment)
    dataset = load_config(args.dataset_config or experiment["dataset_config"])
    model_config = load_config(args.model_config or experiment["model_config"])
    names, tasks = args.models or model_config["default_models"], args.horizons or list(experiment["horizons"])
    seed = int(experiment.get("seed", 42))
    seed_everything(seed)
    if not names or len(names) != len(set(names)):
        raise ValueError("Choose distinct models.")
    for name in names:
        if name == "proposed":
            raise NotImplementedError("Implement ProposedForecaster before running the proposed model.")
        create_model(name, model_config["models"].get(name, {}), seed)
    if len(tasks) != len(set(tasks)) or any(task not in experiment["horizons"] for task in tasks):
        raise ValueError("Choose distinct configured horizons.")
    if experiment.get("frequency") != "15min" or dataset.get("frequency") != "15min":
        raise ValueError("Experiment and dataset frequencies must be 15min.")
    if args.demo:
        frame = make_demo_data(args.demo_days, seed)
        target, unit, calendar_timezone = "facility_power_kw", "kW", "UTC"
        covariates = [c for c in frame if c != target]
        data_label = "synthetic demo (not a research result)"
        provenance = dict(kind="synthetic", days=args.demo_days, seed=seed)
    else:
        path = resolve_path(args.prepared_data or dataset["processed_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Prepared data missing: {path}. Run scripts/prepare_data.py first.")
        frame = read_table(path)
        target, unit, calendar_timezone = dataset["target"], dataset.get("target_unit", "source_units"), dataset["timezone"]
        covariates, data_label = dataset.get("history_covariates", []), dataset["name"]
        stat = path.stat()
        provenance = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256(path.read_bytes()).hexdigest())
        sidecar = path.with_suffix(".metadata.json")
        if sidecar.is_file():
            provenance["preparation"] = json.loads(sidecar.read_text(encoding="utf-8"))
            if provenance["preparation"]["config_digest"] != config_digest(dataset):
                raise ValueError("Dataset configuration changed after preparation. Prepare again.")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Prepared data needs a timezone-aware DatetimeIndex.")
    if not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max(), freq="15min", name=frame.index.name)):
        raise ValueError("Prepared data needs an ordered complete 15-minute grid.")
    if args.start:
        frame = frame.loc[args.start:]
    if args.end:
        frame = frame.loc[:args.end]
    if frame.empty:
        raise ValueError("Requested date range is empty.")
    frame, day_info = complete_days(frame, calendar_timezone, target)
    covariates = ablation_covariates(experiment.get("history_covariates", covariates), args.ablation)
    history, known = make_features(frame, target, covariates, calendar_timezone)
    boundaries = adaptive_day_splits(len(frame)//96, **experiment["split"])
    context = int(experiment["context_length"])
    threshold, criterion = float(experiment.get("mape_threshold", 1e-8)), experiment.get("selection_metric", "MAPE")
    if criterion not in {"MAPE", "WAPE", "RMSE"}:
        raise ValueError("Selection metric must be MAPE, WAPE or RMSE.")
    if "hdsf" in names and float(model_config["models"]["hdsf"].get("mape_threshold", 1e-8)) != threshold:
        raise ValueError("HDSF and experiment must use the same predeclared MAPE threshold.")
    folder = resolve_path(experiment["results_dir"])/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", data=data_label, is_demo=args.demo, target=target, target_unit=unit,
        ablation=args.ablation, history_features=list(history), future_known_features=list(known), experiment=experiment,
        dataset=dataset, model_config=model_config, requested_models=names, requested_horizons=tasks,
        requested_start=args.start, requested_end=args.end, data_provenance=provenance, quality=quality_report(frame),
        complete_days=day_info, splits={name: dict(start=str(frame.index[start]), end=str(frame.index[end-1]),
            rows=end-start, days=(end-start)//96) for name, (start, end) in boundaries.items()}, versions={},
        hardware=dict(platform=platform.platform(), processor=platform.processor()),
        metrics_definition=f"MAPE/WAPE in percent; MAPE excludes |label| <= {threshold}; RMSE in {unit}; "
                           "pooled forecast instances plus per-origin statistics; daily aggregation before scoring.",
        protocol=f"Train only; validation {criterion}; optional train+validation refit; all tasks freeze before test; history rolls.",
        source_sha256={str(p.relative_to(PROJECT_ROOT)): sha256(p.read_bytes()).hexdigest()
                       for directory in ("src", "scripts", "configs") for p in (PROJECT_ROOT/directory).rglob("*")
                       if p.is_file() and p.suffix in {".py", ".yaml"}})
    for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "matplotlib", "torch", "lightgbm", "xgboost", "statsmodels",
                    "timesfm", "chronos-forecasting", "granite-tsfm", "tirex-2", "toto-2"):
        try:
            manifest["versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            manifest["versions"][package] = None
    write_json(folder/"manifest.json", manifest)
    print(f"Data: {data_label} target={target} days={len(frame)//96} rows={len(frame)}", flush=True)
    rows, best_models, prepared = [], {}, {}
    save = lambda task, name, split, batch, result, seconds, plots=False: save_result(
        folder, task, name, split, batch, result, rows, seconds, unit, plots)
    clip = lambda p: np.maximum(p, experiment["clip_min"]) if experiment.get("clip_min") is not None else p
    try:
        for task in tasks:
            horizon = int(experiment["horizons"][task])
            batches = {name: build_windows(history, known, frame[target], bounds, context, horizon,
                       int(experiment["train_stride"] if name == "train" else experiment["backtest_stride"]))
                       for name, bounds in boundaries.items() if name != "test"}
            task_dir = folder/task
            task_dir.mkdir()
            window_info = {name: dict(windows=len(batch), skipped=batch.skipped_windows,
                origins=[str(o) for o in batch.inputs.origins]) for name, batch in batches.items()}
            write_json(task_dir/"windows.json", window_info)
            trained, scores, timings = {}, {}, {}
            for name in names:
                print(f"[{task}] Fit {name}; train={len(batches['train'])}, validation={len(batches['validation'])}", flush=True)
                start = perf_counter()
                training = train_model(name, batches["train"], model_config["models"].get(name, {}), seed)
                trained[name], timings[name] = training.model, training.seconds
                if name == "hdsf":
                    calibration_started = perf_counter()
                    trained[name].calibrate(batches["validation"])
                    trained[name].configuration["calibration_seconds"] = perf_counter()-calibration_started
                    for output_name, predictions in trained[name].outputs(trained[name].validation_members).items():
                        result = score_predictions(batches["validation"], clip(predictions), threshold)
                        displayed_name = "hdsf_member__"+output_name if output_name in names and output_name != "hdsf" else output_name
                        save(task, displayed_name, "validation", batches["validation"], result,
                             training.seconds if output_name == "hdsf" else 0.)
                        if output_name == "hdsf":
                            scores[name] = trained[name].configuration["cv_origin_mean"][criterion]
                else:
                    result = backtest(trained[name], batches["validation"], experiment.get("clip_min"),
                                      int(experiment.get("prediction_batch_size", 32)), threshold)
                    scores[name] = result.metrics["origin_"+criterion+"_mean"]
                    save(task, name, "validation", batches["validation"], result, training.seconds)
                    if hasattr(trained[name], "release"):
                        trained[name].release()
            eligible = {name: score for name, score in scores.items() if score is not None}
            best_models[task] = min(eligible, key=eligible.get) if eligible else None
            if experiment.get("refit_train_validation", False):
                final = build_windows(history, known, frame[target], (0, boundaries["validation"][1]),
                                      context, horizon, int(experiment["train_stride"]))
                for name in names:
                    print(f"[{task}] Refit {name}; train+validation={len(final)}", flush=True)
                    start = perf_counter()
                    if name == "hdsf":
                        trained[name].refit(final)
                    else:
                        trained[name] = train_model(name, final, model_config["models"].get(name, {}), seed).model
                    timings[name] += perf_counter()-start
            for name in names:
                if name == "hdsf":
                    write_json(task_dir/name/"fusion.json", trained[name].configuration)
                    del trained[name].validation_members
                if experiment.get("save_models", True):
                    trained[name].save(task_dir/name/"model")
            write_json(task_dir/"selection.json", dict(criterion=f"validation origin-mean {criterion}", scores=scores,
                selected_model=best_models[task], frozen_before_test=True))
            prepared[task] = (trained, timings, window_info)
        write_json(folder/"frozen-before-test.json", dict(tasks=tasks, selected_models=best_models,
            test_start=manifest["splits"]["test"]["start"], mape_threshold=threshold,
            model_sha256={str(p.relative_to(folder)): sha256(p.read_bytes()).hexdigest() for p in folder.rglob("model.*")},
            frozen_at_utc=datetime.now(timezone.utc).isoformat()))
        for task in tasks:
            trained, timings, window_info = prepared[task]
            batch = build_windows(history, known, frame[target], boundaries["test"], context,
                                  int(experiment["horizons"][task]), int(experiment["backtest_stride"]))
            window_info["test"] = dict(windows=len(batch), skipped=batch.skipped_windows, origins=[str(o) for o in batch.inputs.origins])
            write_json(folder/task/"windows.json", window_info)
            for name in names:
                if name == "hdsf":
                    started = perf_counter()
                    outputs = trained[name].outputs(trained[name].collect(batch.inputs))
                    seconds = perf_counter()-started
                    for output_name, predictions in outputs.items():
                        result = score_predictions(batch, clip(predictions), threshold, frame.index[boundaries["test"][0]],
                                                   seconds if output_name == "hdsf" else 0.)
                        displayed_name = "hdsf_member__"+output_name if output_name in names and output_name != "hdsf" else output_name
                        save(task, displayed_name, "test", batch, result, timings[name] if output_name == "hdsf" else 0.,
                             experiment.get("save_plots", True))
                else:
                    result = backtest(trained[name], batch, experiment.get("clip_min"),
                        int(experiment.get("prediction_batch_size", 32)), threshold, frame.index[boundaries["test"][0]])
                    save(task, name, "test", batch, result, timings[name], experiment.get("save_plots", True))
                    if hasattr(trained[name], "release"):
                        trained[name].release()
            pd.DataFrame(rows).to_csv(folder/"metrics.csv", index=False)
        metrics = pd.DataFrame(rows)
        for task in tasks:
            baseline = "seasonal_naive" if task == "24h" else "three_day_mean"
            for evaluation in ("curve", "daily_mean"):
                ref = metrics[(metrics.horizon == task) & (metrics.model == baseline) &
                              (metrics.split == "test") & (metrics.evaluation == evaluation)]
                for metric in ("MAPE", "WAPE"):
                    mask = (metrics.horizon == task) & (metrics.split == "test") & (metrics.evaluation == evaluation)
                    if len(ref) and ref[metric].iloc[0] > 0:
                        metrics.loc[mask, f"{metric}_reduction_pct"] = 100*(1-metrics.loc[mask, metric]/ref[metric].iloc[0])
        metrics.to_csv(folder/"metrics.csv", index=False)
        print_results(metrics, manifest={**manifest, "selected_models": best_models},
            windows={task: prepared[task][2] for task in tasks},
            fusions={task: prepared[task][0]["hdsf"].configuration for task in tasks if "hdsf" in prepared[task][0]})
        manifest.update(status="complete", selected_models=best_models)
        write_json(folder/"manifest.json", manifest)
        print(f"Finished attempt 1. Results: {folder}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(folder/"manifest.json", manifest)
        if rows:
            pd.DataFrame(rows).to_csv(folder/"metrics.csv", index=False)
        raise
    return folder


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description="Run adaptive 24h/7d rolling load forecasts.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    for option in ("dataset-config", "model-config", "prepared-data", "start", "end"):
        parser.add_argument("--"+option)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--horizons", nargs="+")
    parser.add_argument("--demo", action="store_true", help="Synthetic pipeline verification only.")
    parser.add_argument("--demo-days", type=int, default=100)
    parser.add_argument("--ablation", choices=["full", "target_only", "no_weather", "no_auxiliary"], default="full")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--results", metavar="RUN_DIR", help="Display a saved run, or latest completed real-data run; no training.")
    args = parser.parse_args(argv)
    if args.list_models:
        print("Models: " + ", ".join(MODEL_NAMES))
        return 0
    try:
        if args.results:
            experiment = load_config(args.config)
            print_saved_results(args.results, results_dir=experiment["results_dir"])
            return 0
        run_experiment(args)
    except (OSError, ValueError, KeyError, ImportError, NotImplementedError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
