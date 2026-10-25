"""LODO orchestration dispatched by run_experiment; shared models, freezing and artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import platform
import subprocess
import uuid

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.cross_domain import fit_batch, validate_protocol
from src.data.google_panel import domain_windows, load_google_domains
from src.evaluation.artifacts import add_baseline_reductions, save_result
from src.evaluation.backtest import backtest
from src.evaluation.cross_domain import (audit_cross_domain_run, history_profile, plot_transfer_matrix,
                                         profile_transfer_relationship, summarize_transfer)
from src.evaluation.reporting import METHOD_NAMES, print_cross_domain_results
from src.training.lifecycle import freeze_for_test
from src.training.predictor import predict
from src.training.trainer import train_model
from src.utils import PROJECT_ROOT, config_digest, load_config, resolve_path, seed_everything, write_json


def _model_strategies(names, config, sources, target):
    for name in names:
        if name in {"seasonal_naive", "three_day_mean"}:
            yield name, name, "target_history", [target]
            continue
        for strategy in config["strategies"]:
            if strategy == "single_source":
                for source in config["single_source_domains"]:
                    yield f"{name}_single__{source}", name, strategy, [source]
            elif strategy == "multi_source":
                yield f"{name}_multi", name, strategy, sources
            elif strategy == "target_supervised_reference":
                yield f"{name}_target_reference", name, strategy, [target]
            else:
                raise ValueError(f"Unknown source strategy: {strategy}")


def run_cross_domain(args, config):
    """Fit all domains/tasks first; only then construct and evaluate target test batches."""
    unsupported = [name for name in ("demo", "start", "end", "prepared_data") if getattr(args, name, None)]
    if unsupported or getattr(args, "ablation", "full") != "full":
        raise ValueError(f"Cross-domain uses its explicit data/window protocol; unsupported overrides: {unsupported}.")
    tasks = getattr(args, "horizons", None) or list(config["horizons"])
    names = getattr(args, "models", None) or config["default_models"]
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks)-set(config["horizons"]):
        raise ValueError("Choose distinct configured horizons.")
    if not names or len(set(names)) != len(names) or set(names)-{"ridge", "seasonal_naive", "three_day_mean"}:
        raise ValueError("This validated cross-domain pilot supports ridge, seasonal_naive and three_day_mean.")
    if config.get("frequency") != "15min" or not config.get("save_models"):
        raise ValueError("The cross-domain pilot requires 15min frequency and saved frozen models.")
    dataset = load_config(getattr(args, "dataset_config", None) or config["dataset_config"])
    model_config = load_config(getattr(args, "model_config", None) or config["model_config"])
    seed = int(config["seed"])
    seed_everything(seed)
    panel = load_google_domains(dataset)
    plans = validate_protocol(config, panel, tasks)
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_cross_domain_google_"+uuid.uuid4().hex[:6]
    folder = resolve_path(config["results_dir"]) / identity
    folder.mkdir(parents=True, exist_ok=False)
    context, threshold = int(config["context_length"]), float(config["mape_threshold"])
    unit = next(iter(panel.metadata.values())).target_unit
    baselines = {task: "seasonal_naive" if task == "24h" else "three_day_mean" for task in tasks}
    manifest = dict(status="running", protocol=config["protocol"], split_protocol=config["split_protocol"],
        data="Google PowerData 2019 LODO pilot", target="measured_power_util", target_unit=unit,
        target_kind=config["target_kind"], target_scope=config["target_scope"], is_demo=False,
        requested_horizons=tasks, requested_models=names, target_domains=config["target_domains"],
        experiment=config, dataset=dataset, model_config=model_config, resolved_plans=plans,
        baseline_models=baselines, method_names={}, selection="No model/hyperparameter/source selection; predeclared parameters and source sets.",
        history_features=["measured_power_util", "day_sin", "day_cos", "week_sin", "week_cos"],
        future_known_features=["day_sin", "day_cos", "week_sin", "week_cos"],
        domain_metadata={key: value.to_dict() for key, value in panel.metadata.items()},
        data_provenance=panel.sources, data_quality=panel.audit, config_digest=config_digest(config),
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        baseline_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        versions={p: metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "scikit-learn", "matplotlib")},
        hardware=dict(platform=platform.platform(), processor=platform.processor(), blas_threads=config["blas_threads"]),
        source_sha256={str(p.relative_to(PROJECT_ROOT)): sha256(p.read_bytes()).hexdigest()
                       for directory in ("src", "scripts", "configs") for p in (PROJECT_ROOT/directory).rglob("*")
                       if p.is_file() and p.suffix in {".py", ".yaml"}},
        metrics_aggregation_description="Each target domain uses the public pooled-instance curve/daily scores. Cross-target summary is equal-domain mean and worst MAPE. Source validation is diagnostic only. Negative transfer compares fixed single-source and expanded multi-source on identical target windows.",
        limitations=["Power-domain utilization, not independent facility active-power meters or kW accuracy.",
                     "Three predeclared target domains; this is a pilot, not all 57 LODO folds.",
                     "7d windows overlap; 7d has only two final training windows per complete source and no disjoint development set before D22.",
                     "No HAST, source gating, target fine-tuning or pretrained unseen-domain claim.",
                     "Historical Google experiments have already been observed; this new identity is not a newly collected untouched dataset."])
    write_json(folder/"manifest.json", manifest)
    write_json(folder/"selection.json", dict(rule=manifest["selection"], target_selection=config["target_selection_rule"],
               source_selection="Only training/development coverage; no test-error ranking.", plans=plans,
               fixed_model_overrides=config["model_overrides"], seed=seed))
    coverage = pd.DataFrame(panel.coverage)
    coverage.to_parquet(folder/"sample-coverage.parquet")
    write_json(folder/"data-audit.json", dict(sources=panel.sources, domains=panel.audit,
               contracts=manifest["domain_metadata"], trace_anchor_utc=str(panel.anchor),
               source_documentation=dataset.get("source_documentation")))
    print(f"Data: Google LODO; domains={len(panel.series)}, targets={config['target_domains']}, history={context}", flush=True)
    rows, prepared, records, restored_checks = [], {}, [], []
    try:
        with threadpool_limits(limits=int(config["blas_threads"]), user_api="blas"):
            # The complete source/model plan for every task and target is frozen
            # before any target test batch or target performance is evaluated.
            for task in tasks:
                plan, horizon = plans[task], int(config["horizons"][task])
                for target in config["target_domains"]:
                    target_folder = folder/target
                    task_folder = target_folder/task
                    task_folder.mkdir(parents=True)
                    sources = plan["folds"][target]["source_domains"]
                    trained = {}
                    for display, algorithm, strategy, training_domains in _model_strategies(names, config, sources, target):
                        transfer = strategy in {"single_source", "multi_source"}
                        forbidden = target if transfer else None
                        params = {**model_config["models"].get(algorithm, {}),
                                  **config.get("model_overrides", {}).get(algorithm, {})}
                        extra = dict(target_domain=target, source_strategy=strategy, algorithm=algorithm,
                                     source_domains=json.dumps(training_domains if transfer else [], ensure_ascii=False))
                        label = METHOD_NAMES[algorithm]
                        suffix = {"target_history": "目标历史", "single_source": f"单源 {training_domains[0]}",
                                  "multi_source": "多源联合", "target_supervised_reference": "目标有监督参考"}[strategy]
                        manifest["method_names"][display] = f"{label}（{suffix}）"
                        seconds = 0.
                        if plan["development_origins"] and strategy != "target_history":
                            initial = fit_batch(panel, training_domains, plan["development_training_origins"], context, horizon,
                                                forbidden_domain=forbidden)
                            development_model = train_model(algorithm, initial, params, seed)
                            validation = domain_windows(panel, training_domains, plan["development_origins"], context, horizon)
                            result = backtest(development_model.model, validation, config.get("clip_min"),
                                              int(config["prediction_batch_size"]), threshold)
                            stage = "source_validation" if transfer else "validation"
                            save_result(target_folder, task, display, stage, validation, result, rows,
                                        development_model.seconds, unit, False, row_metadata=extra)
                            development_model.model.save(task_folder/display/"development")
                            seconds += development_model.seconds
                            del initial, validation, development_model
                        batch = fit_batch(panel, training_domains, plan["final_origins"], context, horizon,
                                          forbidden_domain=forbidden)
                        print(f"[{target} {task}] Fit {display}; domains={len(training_domains)}, windows={len(batch)}", flush=True)
                        fitted = train_model(algorithm, batch, params, seed)
                        directory = task_folder/display
                        directory.mkdir(exist_ok=True)
                        model_path = fitted.model.save(directory/"model")
                        record = dict(target_domain=target, horizon=task, model=display, algorithm=algorithm,
                            source_strategy=strategy, training_domains=training_domains,
                            validation_domains=training_domains if plan["development_origins"] and strategy != "target_history" else [],
                            final_cutoff_day=plan["final_cutoff_day"], final_origins=plan["final_origins"],
                            training_windows=len(batch), skipped_training_windows=batch.skipped_windows,
                            training_domain_counts={d: batch.inputs.domain_ids.count(d) for d in training_domains},
                            training_last_label=str(max(pd.Timestamp(t) for t in batch.inputs.target_times[:, -1])),
                            params=params, model_path=str(model_path.relative_to(folder)),
                            normalization="Only the supplied training batch fits StandardScaler; no local target normalization.")
                        records.append(record)
                        write_json(directory/"training.json", record)
                        trained[display] = (fitted.model, seconds+fitted.seconds, extra, model_path)
                        del batch
                    prepared[target, task] = trained
            freeze = freeze_for_test(folder, tasks=tasks, selected_models={}, mape_threshold=threshold,
                test_start={task: str(panel.anchor+pd.Timedelta(days=min(plans[task]["test_origins"]))) for task in tasks},
                domain_isolation=records, plans=plans, config_digest=config_digest(config),
                selection="Fixed preregistered models/sources; source validation diagnostic only.")
            profiles = pd.DataFrame([history_profile(panel, domain, min(min(p["test_origins"]) for p in plans.values()), context)
                                     for domain in panel.series])
            profiles.to_csv(folder/"pre-origin-profiles.csv", index=False)
            write_json(folder/"test-start.json", dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                       freeze_sha256=sha256((folder/"frozen-before-test.json").read_bytes()).hexdigest()))
            for target in config["target_domains"]:
                for task in tasks:
                    plan = plans[task]
                    batch = domain_windows(panel, [target], plan["test_origins"], context, int(config["horizons"][task]))
                    write_json(folder/target/task/"windows.json", dict(test=dict(windows=len(batch), skipped=batch.skipped_windows,
                        origins=[str(o) for o in batch.inputs.origins]), source_training=plan["final_origins"],
                        source_development=plan["development_origins"], no_validation_reason=plan["no_validation_reason"]))
                    for display, (model, seconds, extra, model_path) in prepared[target, task].items():
                        result = backtest(model, batch, config.get("clip_min"), int(config["prediction_batch_size"]), threshold,
                                          batch.inputs.origins[0])
                        save_result(folder/target, task, display, "test", batch, result, rows, seconds, unit,
                                    config.get("save_plots", True), row_metadata=extra)
                        restored = joblib.load(model_path)
                        restored_values = predict(restored, batch.inputs.take(slice(0, 1)), config.get("clip_min"))
                        difference = float(np.max(np.abs(restored_values-result.predictions[:1])))
                        restored_checks.append(dict(target_domain=target, horizon=task, model=display,
                                                    maximum_absolute_difference=difference,
                                                    passed=bool(np.allclose(restored_values, result.predictions[:1], rtol=1e-6, atol=1e-7))))
                    pd.DataFrame(rows).to_csv(folder/"metrics.csv", index=False)
        metrics = pd.concat([add_baseline_reductions(group, baselines) for _, group in
                             pd.DataFrame(rows).groupby("target_domain", sort=False)], ignore_index=True)
        metrics.to_csv(folder/"metrics.csv", index=False)
        write_json(folder/"saved-model-audit.json", restored_checks)
        comparisons, summary = summarize_transfer(metrics)
        comparisons.to_csv(folder/"negative-transfer.csv", index=False)
        summary.to_csv(folder/"domain-summary.csv", index=False)
        relationships, correlations = profile_transfer_relationship(metrics, profiles)
        relationships.to_csv(folder/"profile-transfer-analysis.csv", index=False)
        write_json(folder/"profile-correlations.json", correlations)
        for task in tasks:
            matrix = metrics[(metrics.horizon == task) & (metrics.split == "test") &
                             (metrics.evaluation == "curve")].pivot(index="target_domain", columns="model", values="MAPE")
            matrix.to_csv(folder/f"{task}_transfer_matrix.csv")
        if config.get("save_plots", True):
            plot_transfer_matrix(metrics, folder/"transfer-matrix.png")
        write_json(folder/"manifest.json", manifest)
        audit = audit_cross_domain_run(folder)
        write_json(folder/"audit.json", audit)
        if not audit["passed"]:
            raise ValueError("Cross-domain prediction/freeze/restore audit failed; inspect audit.json.")
        manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder/"manifest.json", manifest)
        print_cross_domain_results(metrics, manifest=manifest, folder=folder)
        print(f"Finished attempt 1. Results: {folder}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(folder/"manifest.json", manifest)
        if rows:
            pd.DataFrame(rows).to_csv(folder/"metrics.csv", index=False)
        raise
    return folder
