"""One-command fixed-model rolling experiments for 24h and 7d tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import sys
from pathlib import Path
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.cleaning import quality_report
from src.data.features import make_features
from src.data.loaders import read_table
from src.data.splitting import build_windows, chronological_splits
from src.data.synthetic import make_demo_data
from src.evaluation.backtest import backtest, prediction_table
from src.models import MODEL_NAMES, create_model
from src.training.trainer import train_model
from src.utils import config_digest, load_config, resolve_path, seed_everything, write_json


def ablation_covariates(covariates: list[str], ablation: str) -> list[str]:
    if ablation == "target_only":
        return []
    if ablation == "no_weather":
        return [name for name in covariates if name not in {"temperature_c", "humidity_pct"}]
    if ablation == "no_auxiliary":
        return [name for name in covariates if name not in {"cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw"}]
    return covariates


def run_experiment(args: argparse.Namespace) -> Path:
    experiment = load_config(args.config)
    dataset = load_config(args.dataset_config or experiment["dataset_config"])
    model_config = load_config(args.model_config or experiment["model_config"])
    names = args.models or model_config["default_models"]
    tasks = args.horizons or list(experiment["horizons"])
    seed = int(experiment.get("seed", 42))
    seed_everything(seed)
    if not names or len(names) != len(set(names)):
        raise ValueError("Select at least one model, without duplicates.")
    for name in names:
        if name == "proposed":
            raise NotImplementedError("Implement ProposedForecaster before running the proposed model.")
        create_model(name, model_config["models"].get(name, {}), seed)
    if any(task not in experiment["horizons"] for task in tasks):
        raise ValueError(f"Unknown horizon; configured tasks: {list(experiment['horizons'])}")
    if experiment.get("frequency") != "15min" or dataset.get("frequency") != "15min":
        raise ValueError("Both experiment and dataset frequency must be 15min.")
    if args.demo:
        frame = make_demo_data(args.demo_days, seed)
        data_label = "synthetic demo (not a research result)"
        # The demo schema matches NLR; use its own known field names even when
        # the user has a custom raw dataset configuration.
        target, unit, calendar_timezone = "facility_power_kw", "kW", "UTC"
        covariates = [column for column in frame if column != target]
        provenance = {"kind": "synthetic", "days": args.demo_days, "seed": seed}
    else:
        path = resolve_path(args.prepared_data or dataset["processed_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Prepared data missing: {path}. Run scripts/prepare_data.py first, or use --demo.")
        frame = read_table(path)
        target, unit, calendar_timezone = dataset["target"], dataset.get("target_unit", "source_units"), dataset["timezone"]
        covariates = dataset.get("history_covariates", [])
        data_label = dataset["name"]
        stat = path.stat()
        provenance = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        sidecar = path.with_suffix(".metadata.json")
        if sidecar.is_file():
            import json
            provenance["preparation"] = json.loads(sidecar.read_text(encoding="utf-8"))
            if provenance["preparation"]["config_digest"] != config_digest(dataset):
                raise ValueError("Dataset configuration changed after preparation. Prepare again or use a matching configuration.")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Prepared data must have a timezone-aware DatetimeIndex.")
    if not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max(), freq="15min", name=frame.index.name)):
        raise ValueError("Prepared data must be ordered on a complete 15-minute grid.")
    if args.start:
        frame = frame.loc[args.start:]
    if args.end:
        frame = frame.loc[:args.end]
    if frame.empty:
        raise ValueError("Requested date range is empty.")
    covariates = ablation_covariates(covariates, args.ablation)
    history, known = make_features(frame, target, covariates, calendar_timezone)
    split = experiment["split"]
    boundaries = chronological_splits(len(frame), split["train_fraction"], split["validation_fraction"])
    root = resolve_path(experiment["results_dir"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "data": data_label, "is_demo": args.demo,
        "target": target, "target_unit": unit, "ablation": args.ablation,
        "history_features": list(history), "future_known_features": list(known),
        "experiment": experiment, "dataset": dataset, "model_config": model_config,
        "requested_models": names, "requested_horizons": tasks,
        "requested_start": args.start, "requested_end": args.end,
        "data_provenance": provenance, "quality": quality_report(frame),
        "splits": {name: {"start": str(frame.index[start]), "end": str(frame.index[end-1]), "rows": end-start} for name, (start, end) in boundaries.items()},
        "versions": {package: metadata.version(package) for package in ["numpy", "pandas", "pyarrow", "scikit-learn", "matplotlib"]},
        "metrics_definition": "WAPE/MAPE in percent; MAPE excludes |label| <= 1e-8. n counts forecast instances, including overlap across origins.",
        "protocol": "Fit training only; select best by validation WAPE; freeze all models before test; update history only at rolling origins.",
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Data: {data_label} target={target} rows={len(frame)}", flush=True)
    metric_rows, best_models = [], {}
    try:
        for task in tasks:
            horizon = int(experiment["horizons"][task])
            batches = {
                name: build_windows(history, known, frame[target], bounds, int(experiment["context_length"]), horizon, int(experiment["train_stride"] if name == "train" else experiment["backtest_stride"]))
                for name, bounds in boundaries.items()
            }
            task_dir = run_dir / task
            task_dir.mkdir()
            window_info = {name: {"windows": len(batch), "skipped": batch.skipped_windows} for name, batch in batches.items()}
            write_json(task_dir / "windows.json", window_info)
            trained, validation_scores = {}, {}
            # Complete all fitting/validation before looking at test scores.
            for name in names:
                trained[name] = train_model(name, batches["train"], model_config["models"].get(name, {}), seed)
                model_dir = task_dir / name
                model_dir.mkdir()
                if experiment.get("save_models", True):
                    trained[name].model.save(model_dir / "model")
                result = backtest(trained[name].model, batches["validation"], experiment.get("clip_min"), int(experiment.get("prediction_batch_size", 32)))
                validation_scores[name] = result.metrics["WAPE"]
                metric_rows.append({"horizon": task, "model": name, "split": "validation", "train_seconds": trained[name].seconds, **result.metrics})
                result.per_lead.to_csv(model_dir / "validation_per_lead.csv", index=False)
                prediction_table(batches["validation"], result.predictions).to_parquet(model_dir / "validation_predictions.parquet", index=False)
            eligible = {name: score for name, score in validation_scores.items() if score is not None}
            best_models[task] = min(eligible, key=eligible.get) if eligible else None
            write_json(task_dir / "selection.json", {"criterion": "validation WAPE", "scores": validation_scores, "selected_model": best_models[task], "frozen_before_test": True})
            for name in names:
                result = backtest(trained[name].model, batches["test"], experiment.get("clip_min"), int(experiment.get("prediction_batch_size", 32)))
                metric_rows.append({"horizon": task, "model": name, "split": "test", "train_seconds": trained[name].seconds, **result.metrics})
                model_dir = task_dir / name
                result.per_lead.to_csv(model_dir / "test_per_lead.csv", index=False)
                table = prediction_table(batches["test"], result.predictions)
                table.to_parquet(model_dir / "test_predictions.parquet", index=False)
                write_json(model_dir / "metrics.json", {"validation": next(row for row in metric_rows if row["horizon"] == task and row["model"] == name and row["split"] == "validation"), "test": result.metrics, "train_windows": trained[name].train_windows, "train_seconds": trained[name].seconds})
                if experiment.get("save_plots", True):
                    from src.evaluation.visualization import plot_forecast
                    plot_forecast(table, model_dir / "forecast.png", f"{data_label}: {name}, {task}", unit)
                wape = "undefined" if result.metrics["WAPE"] is None else f"{result.metrics['WAPE']:.4f}%"
                print(f"{task} {name}: WAPE={wape} n={result.metrics['n']}", flush=True)
        pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
        manifest.update(status="complete", selected_models=best_models)
        write_json(run_dir / "manifest.json", manifest)
        print(f"Finished attempt 1. Results: {run_dir}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(run_dir / "manifest.json", manifest)
        if metric_rows:
            pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
        raise
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run reproducible 24h/7d rolling load forecasts.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset-config")
    parser.add_argument("--model-config")
    parser.add_argument("--prepared-data")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--horizons", nargs="+")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data to verify the pipeline.")
    parser.add_argument("--demo-days", type=int, default=100)
    parser.add_argument("--ablation", choices=["full", "target_only", "no_weather", "no_auxiliary"], default="full")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args(argv)
    if args.list_models:
        print("Models: " + ", ".join(MODEL_NAMES))
        return 0
    try:
        run_experiment(args)
    except (OSError, ValueError, KeyError, ImportError, NotImplementedError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
