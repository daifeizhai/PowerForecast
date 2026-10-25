"""SAPF source fitting, independent-domain calibration, freeze, then target evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
from time import perf_counter
import uuid

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.cross_domain import fit_batch, validate_protocol
from src.data.google_panel import domain_windows, load_google_domains
from src.data.shape_features import SIGNATURE_NAMES, load_signature, signature_records
from src.data.splitting import WindowBatch
from src.evaluation.artifacts import add_baseline_reductions, save_result
from src.evaluation.backtest import backtest, score_predictions
from src.evaluation.cross_domain import analyze_pairwise, audit_cross_domain_run, plot_sapf_results, summarize_transfer
from src.evaluation.reporting import METHOD_NAMES, print_results
from src.models import create_model
from src.models.ml.ablation import ABLATION_MODES
from src.training.lifecycle import freeze_for_test
from src.training.predictor import predict
from src.training.trainer import train_model
from src.utils import PROJECT_ROOT, config_digest, load_config, resolve_path, seed_everything, write_json


STRATEGIES = {"naive": "target_history", "seasonal_naive": "target_history", "three_day_mean": "target_history",
              "ridge": "multi_source", "normalized_ridge": "normalized_multi_source",
              "sapf_similarity": "similarity_selected", "sapf": "reliability_selected"}
STRATEGIES.update({name: "controlled_ablation_"+mode for name, mode in ABLATION_MODES.items()})


def _train_variant(algorithm, training, params, seed, stage_models):
    if algorithm in ABLATION_MODES:
        params = {**params, "reference_model": stage_models["sapf"]}
    return train_model(algorithm, training, params, seed).model


def _calibration_domains(model):
    calibration = getattr(model, "calibration", None)
    return calibration.get("validation_domains", []) if calibration else []


def _subset(batch, domain):
    indices = np.flatnonzero(np.asarray(batch.inputs.domain_ids) == domain)
    return WindowBatch(batch.inputs.take(indices), batch.targets[indices])


def _source_profiles(batch, task, stage):
    rows = []
    for domain in sorted(set(batch.inputs.domain_ids)):
        part = _subset(batch, domain)
        last = max(range(len(part)), key=lambda i: part.inputs.origins[i])
        values = np.r_[part.inputs.history[last, :, 0], part.targets[last]][-1344:]
        rows.append(dict(domain_id=domain, role="source", stage=stage, horizon=task,
                         origin=str(pd.Timestamp(part.inputs.target_times[last, -1])+pd.Timedelta(minutes=15)),
                         mean_level=float(values.mean()), history_std=float(values.std()),
                         **dict(zip(SIGNATURE_NAMES, load_signature(values)))))
    return rows


def _variants(names, sources, pairwise):
    variants = {name: dict(algorithm=name, source_strategy=STRATEGIES.get(name, "multi_source"),
                          source_domains=[] if name in {"naive", "seasonal_naive", "three_day_mean"} else sources)
                for name in names}
    if pairwise:
        for algorithm in ("ridge", "normalized_ridge"):
            if algorithm in names:
                for source in sources:
                    name = f"single_{algorithm}__{source}"
                    variants[name] = dict(algorithm=algorithm, source_strategy="single_source" if algorithm == "ridge" else "normalized_single_source",
                                          source_domains=[source])
    return variants


def _evaluate(folder, task, split, batch, models, variants, config, rows, timings,
              profiles, pairwise, diagnostics, restored=None, restore_rows=None):
    target_profiles = signature_records(batch.inputs)
    profile_lookup = {r["domain_id"]: r for r in profiles if r["horizon"] == task and
                      r["stage"] == ("development" if split == "validation" else "final") and r["role"] == "source"}
    for name, model in models.items():
        result = backtest(model, batch, config["clip_min"], config["prediction_batch_size"], config["mape_threshold"])
        if restored is not None:
            reloaded = joblib.load(restored[name])
            values = np.concatenate([predict(reloaded, batch.inputs.take(slice(i, i+config["prediction_batch_size"])), config["clip_min"])
                                     for i in range(0, len(batch), config["prediction_batch_size"])])
            restore_rows.append(dict(horizon=task, model=name, windows=len(batch),
                maximum_absolute_difference=float(np.max(np.abs(values-result.predictions))),
                passed=bool(np.allclose(values, result.predictions, rtol=1e-6, atol=1e-7))))
        variant = variants[name]
        for domain in dict.fromkeys(batch.inputs.domain_ids):
            indices = np.flatnonzero(np.asarray(batch.inputs.domain_ids) == domain)
            part = WindowBatch(batch.inputs.take(indices), batch.targets[indices])
            scored = score_predictions(part, result.predictions[indices], config["mape_threshold"], part.inputs.origins[0])
            extra = dict(target_domain=domain, algorithm=variant["algorithm"],
                         source_strategy=variant["source_strategy"], source_domains=json.dumps(variant["source_domains"]),
                         target_role="test" if split == "test" else "validation",
                         validation_score_kind="selection_in_sample" if _calibration_domains(model) else "independent_domain",
                         target_unit=config.get("target_unit", "utilization_fraction"))
            save_result(folder/domain, task, name, split, part, scored, rows, timings[name], config.get("target_unit", "utilization_fraction"),
                        False, row_metadata=extra, quiet=True)
            if name.startswith("single_"):
                source = variant["source_domains"][0]
                source_profile = profile_lookup[source]
                for local_index, global_index in enumerate(indices):
                    target_profile = target_profiles[global_index]
                    pairwise.append(dict(horizon=task, split=split, source_domain=source, target_domain=domain,
                        origin=str(part.inputs.origins[local_index]), algorithm=variant["algorithm"],
                        MAPE=float(scored.per_origin.MAPE.iloc[local_index]),
                        **{"difference_"+key: abs(target_profile[key]-source_profile[key]) for key in SIGNATURE_NAMES}))
        if hasattr(model, "predict_with_diagnostics"):
            _, info = model.predict_with_diagnostics(batch.inputs)
            for i, (domain, origin) in enumerate(zip(batch.inputs.domain_ids, batch.inputs.origins)):
                for j, source in enumerate(model.source_domains):
                    diagnostics.append(dict(horizon=task, split=split, model=name, target_domain=domain, origin=str(origin),
                        source_domain=source, distance=float(info["distances"][i, j]), weight=float(info["weights"][i, j]),
                        transfer_gate=float(info["transfer_gate"][i]), effective_weight=float(info["weights"][i, j]*info["transfer_gate"][i])))


def run_sapf(args, config):
    if config.get("protocol") not in {"sapf_google_holdout_v1", "sapf_m100_nodes_v1"}:
        raise ValueError("Main-line experiments use SAPF; old protocols require explicit --legacy-config.")
    model_config = load_config(config["model_config"])
    names = getattr(args, "models", None) or model_config["default_models"]
    names = list(names)
    if len(names) != len(set(names)):
        raise ValueError("Choose distinct configured models and horizons.")
    if set(names) & set(ABLATION_MODES):
        # Fit/calibrate full SAPF first; interventions never select a second grid.
        names = [n for n in names if n not in ABLATION_MODES and n != "sapf"]+["sapf"]+[n for n in names if n in ABLATION_MODES]
    tasks = getattr(args, "horizons", None) or list(config["horizons"])
    if not names or len(names) != len(set(names)) or len(tasks) != len(set(tasks)) or set(tasks)-set(config["horizons"]):
        raise ValueError("Choose distinct configured models and horizons.")
    if config["mape_threshold"] < 0 or config["prediction_batch_size"] <= 0:
        raise ValueError("Use a nonnegative MAPE threshold and positive prediction batch size.")
    for name in names:
        if name in {"sapf", "sapf_similarity"} and model_config["models"][name].get("mape_threshold", 1e-8) != config["mape_threshold"]:
            raise ValueError("SAPF calibration and public evaluation must use the same MAPE threshold.")
    seed_everything(config["seed"])
    dataset = load_config(config["dataset_config"])
    demo = bool(getattr(args, "demo", False))
    if demo:
        if config["protocol"] == "sapf_m100_nodes_v1":
            raise ValueError("M100 demos must use explicit synthetic fixtures; never substitute Google utilization for kW.")
        from src.data.synthetic import make_demo_domains
        panel = make_demo_domains(getattr(args, "demo_days", 31), config["seed"])
    elif config["protocol"] == "sapf_m100_nodes_v1":
        from src.data.m100_panel import load_m100_domains
        panel = load_m100_domains(dataset)
    else:
        panel = load_google_domains(dataset)
    plan = validate_protocol(config, panel, tasks)
    roles = plan["roles"]
    sources = roles["source"]
    variants = _variants(names, sources, config["pairwise_source_diagnostics"])
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_sapf_"+uuid.uuid4().hex[:6]
    folder = resolve_path(config["results_dir"])/identifier
    folder.mkdir(parents=True, exist_ok=False)
    fixed_single = f"single_ridge__{sources[0]}"
    fixed_normalized = f"single_normalized_ridge__{sources[0]}"
    console_methods = [n for n in [*names[:2], fixed_single, "ridge", fixed_normalized, "normalized_ridge",
                                    "sapf_similarity", "sapf", *names[2:]] if n in variants]
    console_methods = list(dict.fromkeys(console_methods))
    physical = panel.metadata[sources[0]]
    m100 = config["protocol"] == "sapf_m100_nodes_v1"
    manifest = dict(status="running", protocol=config["protocol"], research_method=config.get("research_method", "SAPF mechanism study"),
        data="synthetic domains" if demo else (f"M100 ExaData: {len({m.rack_id for m in panel.metadata.values()})} racks, {len(panel.series)} nodes" if m100 else "Google PowerData 2019"), is_demo=demo,
        target=panel.target_field, target_kind=physical.target_kind, target_scope=physical.target_scope,
        target_unit=physical.target_unit, trace_anchor_utc=str(panel.anchor),
        requested_horizons=tasks, requested_models=names, experiment=config, dataset=dataset,
        model_config=model_config, domain_roles=roles, temporal_plan=plan["tasks"], console_methods=console_methods,
        method_names={fixed_single: "预定单源 Ridge", fixed_normalized: "预定单源归一化 Ridge"},
        data_provenance=panel.sources, domain_metadata={d: meta.to_dict() for d, meta in panel.metadata.items()},
        versions={p: metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "pyarrow"]},
        config_digest=config_digest(config), archive_baseline="bc6a439",
        source_sha256={str(p.relative_to(PROJECT_ROOT)): sha256(p.read_bytes()).hexdigest()
                       for directory in ("src", "scripts", "configs") for p in (PROJECT_ROOT/directory).rglob("*")
                       if p.is_file() and p.suffix in {".py", ".yaml"}},
        limitations=(["Measured node power (kW); only two racks in the same Marconi100 site, not cross-datacenter generalization.",
                      "Common-period selection used all-node observation-quality masks only; retrospective complete-coverage conditioning.",
                      "Upstream interval boundaries and raw sample counts are unavailable; preceding slots assumed available by issue time with at most one interval latency and no extra reporting delay.",
                      "One-mechanism ablations inherit full SAPF validation choices; they are conditional interventions, not independently optimized models."] if m100 else
                    ["Power-domain utilization, not facility-meter kW.", "Quality screening conditions on complete full-trace masks.",
                     "The same historical Google release has been studied before; no claim of a newly collected untouched dataset."])+
                    ["Validation domains select SAPF settings; their calibrated scores are not unbiased holdout results.",
                     "3d test windows overlap; cross-domain holdout is not independent-site generalization.", "Eagle is reserved as a final holdout and is not loaded."])
    adapted_hashes = {}
    if m100:
        for domain, frame in panel.series.items():
            path = folder/"adapted-data"/(domain+".parquet")
            path.parent.mkdir(exist_ok=True)
            frame.to_parquet(path)
            adapted_hashes[path.relative_to(folder).as_posix()] = sha256(path.read_bytes()).hexdigest()
        manifest["adapted_data_sha256"] = adapted_hashes
    write_json(folder/"manifest.json", manifest)
    write_json(folder/"predeclared-plan.json", dict(config=config, models=model_config, roles=roles,
               temporal_plan=plan["tasks"], created_at_utc=datetime.now(timezone.utc).isoformat(),
               test_metrics_not_accessed=True))
    write_json(folder/"data-audit.json", dict(sources=panel.sources, quality=panel.audit,
                                             anchor_utc=str(panel.anchor), domains=manifest["domain_metadata"]))
    print(f"配置：SAPF | 14 天历史 | {tasks} | 源域 {len(sources)} / 验证域 {len(roles['validation'])} / 测试域 {len(roles['test'])}", flush=True)
    prepared, rows, records, profiles, pairwise, diagnostics, restore_rows = {}, [], [], [], [], [], []
    try:
        with threadpool_limits(limits=config["blas_threads"], user_api="blas"):
            for task in tasks:
                spec, h = plan["tasks"][task], config["horizons"][task]
                initial = fit_batch(panel, sources, spec["development_origins"], 1344, h,
                                    forbidden_domains=roles["validation"]+roles["test"])
                validation = domain_windows(panel, roles["validation"], spec["validation_origins"], 1344, h)
                profiles.extend(_source_profiles(initial, task, "development"))
                profiles.extend(dict(**record, stage="development", horizon=task, role="validation") for record in signature_records(validation.inputs))
                models, timings = {}, {}
                for index, (name, variant) in enumerate(variants.items(), 1):
                    training = _subset(initial, variant["source_domains"][0]) if name.startswith("single_") else initial
                    algorithm = variant["algorithm"]
                    started = perf_counter()
                    model = _train_variant(algorithm, training, model_config["models"][algorithm], config["seed"], models)
                    if hasattr(model, "calibrate"):
                        model.calibrate(validation)
                        write_json(folder/task/name/"calibration.json", model.calibration)
                    directory = folder/task/name
                    directory.mkdir(parents=True, exist_ok=True)
                    model.save(directory/"development")
                    models[name], timings[name] = model, perf_counter()-started
                    if index == 1 or index == len(variants) or index % 10 == 0:
                        print(f"[{task}] 源域训练/验证准备 {index}/{len(variants)}", flush=True)
                _evaluate(folder, task, "validation", validation, models, variants, config, rows, timings,
                          profiles, pairwise, diagnostics)
                final = fit_batch(panel, sources, spec["final_origins"], 1344, h,
                                  forbidden_domains=roles["validation"]+roles["test"])
                profiles.extend(_source_profiles(final, task, "final"))
                model_paths = {}
                for name, variant in variants.items():
                    training = _subset(final, variant["source_domains"][0]) if name.startswith("single_") else final
                    started = perf_counter()
                    if hasattr(models[name], "refit"):
                        models[name].refit(training)
                    else:
                        models[name] = _train_variant(variant["algorithm"], training, model_config["models"][variant["algorithm"]], config["seed"], models)
                    timings[name] += perf_counter()-started
                    model_paths[name] = models[name].save(folder/task/name/"model")
                    record = dict(model=name, horizon=task, training_domains=variant["source_domains"],
                        calibration_domains=_calibration_domains(models[name]),
                        final_cutoff_day=spec["final_cutoff_day"], final_origins=spec["final_origins"],
                        last_training_label=str(max(pd.Timestamp(t) for t in training.inputs.target_times[:, -1])),
                        windows=len(training), model_path=str(model_paths[name].relative_to(folder)),
                        selected=getattr(models[name], "calibration", None))
                    records.append(record)
                    write_json(folder/task/name/"training.json", record)
                prepared[task] = models, timings, model_paths
                print(f"[{task}] 源域最终拟合完成；等待全任务冻结", flush=True)
            freeze_for_test(folder, tasks=tasks, training_records=records, domain_roles=roles,
                temporal_plan=plan["tasks"], config_digest=config_digest(config), mape_threshold=config["mape_threshold"],
                adapted_data_sha256=adapted_hashes,
                target_contract={k: manifest[k] for k in ("target", "target_kind", "target_scope", "target_unit")})
            write_json(folder/"test-start.json", dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                freeze_sha256=sha256((folder/"frozen-before-test.json").read_bytes()).hexdigest()))
            for task in tasks:
                batch = domain_windows(panel, roles["test"], plan["tasks"][task]["test_origins"], 1344, config["horizons"][task])
                profiles.extend(dict(**record, stage="final", horizon=task, role="test") for record in signature_records(batch.inputs))
                models, timings, model_paths = prepared[task]
                _evaluate(folder, task, "test", batch, models, variants, config, rows, timings, profiles,
                          pairwise, diagnostics, model_paths, restore_rows)
                print(f"[{task}] 测试完成：{len(batch)} 个起报，{len(models)} 个主方法/源域诊断", flush=True)
                pd.DataFrame(rows).to_csv(folder/"metrics.csv", index=False)
        metrics = pd.concat([add_baseline_reductions(group, {task: "three_day_mean" for task in tasks})
                             for _, group in pd.DataFrame(rows).groupby("target_domain", sort=False)], ignore_index=True)
        metrics.to_csv(folder/"metrics.csv", index=False)
        comparisons, summary = summarize_transfer(metrics)
        comparisons.to_csv(folder/"negative-transfer.csv", index=False)
        summary.to_csv(folder/"domain-summary.csv", index=False)
        profile_table = pd.DataFrame(profiles)
        profile_table.to_csv(folder/"domain-profiles.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(folder/"source-weights.csv", index=False)
        analyze_pairwise(pd.DataFrame(pairwise), folder)
        write_json(folder/"saved-model-audit.json", restore_rows)
        if config["save_plots"]:
            plot_sapf_results(metrics, profile_table, folder)
        audit = audit_cross_domain_run(folder)
        write_json(folder/"audit.json", audit)
        if not audit["passed"]:
            raise ValueError("SAPF freeze/role/restore/metric audit failed.")
        manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder/"manifest.json", manifest)
        print_results(metrics, manifest=manifest, detailed=getattr(args, "detailed", False))
        print(f"结果保存：{folder}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(folder/"manifest.json", manifest)
        if rows:
            pd.DataFrame(rows).to_csv(folder/"metrics.csv", index=False)
        raise
    return folder
