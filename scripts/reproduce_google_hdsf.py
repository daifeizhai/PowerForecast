"""Run the unmodified historical HDSF engine with workspace-local artifacts.

This bridge deliberately preserves the reference's panel, masks, member inputs,
selection objective, and D14/D22 fitting. It does not use NLR window builders.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import uuid

import numpy as np
import pandas as pd

from src.evaluation.reporting import print_saved_results
from src.utils import PROJECT_ROOT, resolve_path, write_json


def load_reference(config):
    reference = resolve_path(config["reference_project"])
    if not (reference/"power_forecast/hdsf_experiment.py").is_file():
        raise FileNotFoundError(f"Historical HDSF source missing: {reference}")
    # The reference checkout is read-only, including Python bytecode caches.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference))
    from power_forecast import config as protocol
    from power_forecast.data import load_dataset
    from power_forecast.hdsf_experiment import freeze_hdsf, score_hdsf
    from power_forecast.reporting import reporting_view, series_mean_daily_metrics
    if (protocol.DEV_CUTOFF_DAY, protocol.FINAL_CUTOFF_DAY) != (14, 22):
        raise ValueError("Reference training boundaries differ from D14/D22.")
    expected = {"24h": ((14, 16, 18, 20), tuple(range(22, 31))), "7d": ((14, 15), (22, 23, 24))}
    for task, (development, test) in expected.items():
        if (protocol.TASKS[task].dev_origins, protocol.TASKS[task].test_origins) != (development, test):
            raise ValueError(f"Reference origins changed for {task}.")
    return load_dataset, freeze_hdsf, score_hdsf, reporting_view, series_mean_daily_metrics


def prepare_cache(config):
    """Link immutable checkpoints and copy optional content-addressed forecasts."""
    original, destination = resolve_path(config["weights_cache"]), resolve_path(config["cache_dir"])
    if not destination.is_relative_to(PROJECT_ROOT/"results"):
        raise ValueError("Historical HDSF writable cache must be inside project results/.")
    if not original.is_dir():
        raise FileNotFoundError(f"Offline weights cache missing: {original}")
    destination.mkdir(parents=True, exist_ok=True)
    for source in [*original.glob("models--*"), original/"hub"]:
        if not source.is_dir():
            continue
        link = destination/source.name
        if link.exists():
            if link.resolve() != source.resolve():
                raise ValueError(f"Checkpoint link points to a different directory: {link}")
        elif os.name == "nt":
            subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(source)],
                           check=True, capture_output=True)
        else:
            link.symlink_to(source, target_is_directory=True)
    copied = 0
    if config.get("reuse_prediction_cache", True):
        for source in (original/"predictions").rglob("*.npy"):
            target = destination/"predictions"/source.relative_to(original/"predictions")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied += 1
    # Fitted local models are deliberately not seeded: D14 and D22 retrain here.
    return destination, copied


def normalized_metrics(results, reporting_view, daily_metrics):
    rows = []
    for task, result in results.items():
        reported = reporting_view(result)
        for split, stage in [("validation", "development"), ("test", "test")]:
            curves = reported[stage]
            daily = daily_metrics(result[stage+"_predictions"], list(curves))
            for evaluation, scores in [("curve", curves), ("daily_mean", daily)]:
                for model, score in scores.items():
                    rows.append(dict(horizon=task, model=model, split=split, evaluation=evaluation,
                        WAPE=score["wape_pct"], MAPE=score["mape_pct"], RMSE=score["rmse"], MAE=score["mae"],
                        n=score["points"], mape_n=score["mape_points"],
                        mape_excluded=score["points"]-score["mape_points"],
                        series_count=score["series_count"], pooled_WAPE=score["pooled_wape_pct"],
                        wape_aggregation="series_mean"))
    metrics = pd.DataFrame(rows)
    for task in results:
        baseline = "daily_naive" if task == "24h" else "mean3"
        for evaluation in ("curve", "daily_mean"):
            mask = (metrics.horizon == task) & (metrics.evaluation == evaluation) & (metrics.split == "test")
            reference = metrics.loc[mask & (metrics.model == baseline)].iloc[0]
            for metric in ("WAPE", "MAPE"):
                metrics.loc[mask, metric+"_reduction_pct"] = 100*(1-metrics.loc[mask, metric]/reference[metric])
    return metrics


def run_reproduction(args, config):
    unsupported = (args.demo or args.start or args.end or args.dataset_config or args.model_config
                   or args.prepared_data or args.ablation != "full" or args.models)
    if unsupported:
        raise ValueError("D14/D22 uses the raw panel and prescribed member pool; only --horizons and --results overrides apply.")
    tasks = args.horizons or list(config["horizons"])
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks)-{"24h", "7d"}:
        raise ValueError("Choose distinct 24h/7d horizons.")
    if config.get("device", "cpu") != "cpu":
        raise ValueError("Historical HDSF reproduction requires CPU.")
    runtime = resolve_path(config.get("python_executable", sys.executable))
    if runtime != Path(sys.executable).resolve():
        if not runtime.is_file():
            raise FileNotFoundError(f"Historical Python interpreter missing: {runtime}")
        print(f"Historical runtime: {runtime}", flush=True)
        script = PROJECT_ROOT/"scripts/run_experiment.py"
        arguments = [str(script), "--config", str(resolve_path(args.config))]
        if args.horizons:
            arguments.extend(["--horizons", *args.horizons])
        # Append (never prepend) project packages for Parquet export. Historical
        # NumPy/model dependencies keep precedence; no environment is modified.
        extra = resolve_path(config.get("supplemental_site_packages", ".venv/Lib/site-packages"))
        bootstrap = (f"import sys, runpy; sys.path.append({str(extra)!r}); "
                     f"sys.argv={arguments!r}; runpy.run_path({str(script)!r}, run_name='__main__')")
        command = [str(runtime), "-B", "-c", bootstrap]
        outcome = subprocess.run(command, cwd=PROJECT_ROOT)
        if outcome.returncode:
            raise ValueError(f"Historical HDSF runtime exited with status {outcome.returncode}.")
        return None
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", MPLBACKEND="Agg")
    raw = resolve_path(config["raw_data"])
    if not raw.is_dir():
        raise FileNotFoundError(f"Raw Google directory missing: {raw}")
    files = sorted(raw.glob("*.csv.gz"))
    source_hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in files}
    load_data, freeze, score, reporting_view, daily_metrics = load_reference(config)
    data = load_data(raw)
    if (data.n_series, data.n_days) != (57, 31):
        raise ValueError(f"D14/D22 expects 57 domains x 31 days, got {data.n_series} x {data.n_days}.")
    folder = resolve_path(config["results_dir"])/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_google_"+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", protocol=config["protocol"], data="Google PowerData 2019 (57 domains)",
        target="measured_power_util", target_unit="utilization_fraction", is_demo=False,
        requested_horizons=tasks, requested_models=["hdsf"], python_executable=sys.executable,
        baseline_models={"24h": "daily_naive", "7d": "mean3"}, experiment={**config, "mape_threshold": 1e-8,
        "selection_metric": "WAPE"}, quality=data.quality, source_sha256=source_hashes,
        method_names={"daily_naive": "前一天同刻", "mean3": "前三天同刻均值"},
        metrics_aggregation_description="WAPE: compute per power domain then average 57 domains equally; MAPE/RMSE: valid forecast instances. Daily means use matching valid slots before scoring.",
        limitations=["Historical method design referenced test results; this is a reproduction, not a new untouched holdout.",
            "7d development windows overlap, as prescribed by the historical protocol.",
            "The historical D1 residual baseline uses the whole permitted training period mean; preserved for exact reproduction.",
            "Member histories and joint-domain inference retain the historical settings; they differ from NLR's single-target 14-day preset."],
        bridge_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(folder/"manifest.json", manifest)
    started = perf_counter()
    try:
        cache, copied = prepare_cache(config)
        manifest["prediction_cache"] = dict(path=str(cache), seeded_files=copied,
            policy="Only input/checkpoint/version/config matching cache entries can be reused; D14/D22 local models retrain in this cache.")
        write_json(folder/"manifest.json", manifest)
        print(f"Data: Google PowerData2019 series={data.n_series} days={data.n_days}; D14/D22", flush=True)
        print(f"Quality: valid={int(data.target_valid.sum())}/{data.target_valid.size}; cache seeded={copied}", flush=True)
        data.meta.to_csv(folder/"series.csv", index=False)
        np.savez_compressed(folder/"prepared-panel.npz", target=data.target, production=data.production,
            target_valid=data.target_valid, times=data.times, production_causal_median=data.production_causal_median)
        collector, development, configurations, reference_folder = freeze(data, tasks, cache_dir=cache,
            cache_predictions=True, batch_size=int(config["batch_size"]), jobs=int(config["jobs"]),
            device="cpu", audit_nested=bool(config.get("audit_nested", False)))
        shutil.copytree(reference_folder, folder, dirs_exist_ok=True)
        import joblib
        model_dir = folder/"models"
        model_dir.mkdir()
        for name, model in [("development-D14", collector.development_local), ("final-D22", collector.frozen_local)]:
            if model is not None:
                expected = 14 if name == "development-D14" else 22
                if model.cutoff != expected:
                    raise ValueError(f"Wrong local training cutoff: {name} {model.cutoff}")
                joblib.dump(model, model_dir/(name+".joblib"))
        frozen_path = folder/"frozen-before-test.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        model_hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in model_dir.glob("*.joblib")}
        frozen.update(local_model_sha256=model_hashes, raw_source_sha256=source_hashes,
            frozen_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(frozen_path, frozen)
        frozen_hash = sha256(frozen_path.read_bytes()).hexdigest()
        results = score(data, development, collector.collect(development=False), configurations, folder=folder)
        metrics = normalized_metrics(results, reporting_view, daily_metrics)
        metrics.to_csv(folder/"metrics.csv", index=False)
        for task, result in results.items():
            task_dir = folder/task
            for split, stage in [("validation", "development"), ("test", "test")]:
                result[stage+"_predictions"].to_parquet(task_dir/(split+"_panel.parquet"), index=False)
            write_json(task_dir/"windows.json", {"validation": dict(windows=len(result["dev_origins"]), skipped=0),
                "test": dict(windows=len(result["test_origins"]), skipped=0), "series_count": 57})
        if sha256(frozen_path.read_bytes()).hexdigest() != frozen_hash:
            raise ValueError("Frozen configuration changed during test inference.")
        if {p.name: sha256(p.read_bytes()).hexdigest() for p in files} != source_hashes:
            raise ValueError("Raw Google source files changed during reproduction.")
        if {p.name: sha256(p.read_bytes()).hexdigest() for p in model_dir.glob("*.joblib")} != model_hashes:
            raise ValueError("Frozen local models changed during test inference.")
        manifest.update(status="complete", elapsed_seconds=perf_counter()-started,
            reference_run=str(reference_folder), frozen_sha256=frozen_hash,
            selected_models={}, prescribed_model="hdsf",
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder/"manifest.json", manifest)
        print_saved_results(folder)
        print(f"Finished attempt 1. Results: {folder}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(folder/"manifest.json", manifest)
        raise
    return folder
