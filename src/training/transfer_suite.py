"""Three preregistered studies, all source groups frozen before any test outcome."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import shutil
from time import perf_counter
import uuid

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.domains import assert_data_usage
from src.data.google_panel import domain_windows
from src.data.splitting import WindowBatch
from src.data.transfer30 import load_transfer_domains
from src.evaluation.artifacts import add_baseline_reductions, save_result
from src.evaluation.backtest import backtest, score_predictions
from src.evaluation.metrics import forecast_metrics
from src.evaluation.benefit_metrics import risk_summary
from src.evaluation.cross_domain import audit_cross_domain_run, summarize_transfer
from src.evaluation.reporting import print_saved_results
from src.models import create_model
from src.models.ml.benefit import POLICY_NAMES
from src.training.benefit_learning import source_bank, train_joint_benefit
from src.training.lifecycle import freeze_for_test
from src.training.predictor import predict
from src.utils import PROJECT_ROOT, config_digest, load_config, resolve_path, write_json


def temporal_plan(config):
    plan = {}
    for task, horizon in config["horizons"].items():
        spec = config["temporal_protocol"][task]
        plan[task] = dict(**spec, development_origins=list(range(14, spec["development_cutoff_day"]-horizon//48+1)),
                         final_origins=list(range(14, spec["final_cutoff_day"]-horizon//48+1)))
    return plan


def validate_suite(panel, config):
    if config["frequency"] != "30min" or panel.points_per_day != 48 or config["context_length"] != 672 or config["horizons"] != {"24h": 48, "3d": 144}:
        raise ValueError("The transfer suite requires complete 30min 14d/24h/3d windows.")
    all_ids = []
    for group in ("m100", "ucc"):
        roles = config["domain_roles"][group]
        for role in ("source", "validation", "test"):
            if len(roles[role]) < (3 if role == "source" else 2):
                raise ValueError("Independent source/validation/test identity sets are too small.")
            all_ids.extend(roles[role])
            for domain in roles[role]:
                assert_data_usage(panel.metadata[domain], "training" if role == "source" else role)
    if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(panel.series):
        raise ValueError("Global dataset roles must be disjoint and cover the assigned panel exactly.")
    if set(all_ids) & set(config["domain_roles"]["retired_m100_previous_test"]):
        raise ValueError("Previously examined M100 test identities cannot enter any new role.")
    for task, spec in temporal_plan(config).items():
        days = config["horizons"][task]//48
        if min(spec["validation_origins"]) < spec["development_cutoff_day"] or max(spec["validation_origins"])+days > spec["final_cutoff_day"] or min(spec["test_origins"]) < spec["final_cutoff_day"] or max(spec["test_origins"])+days > min(len(f)//48 for f in panel.series.values()):
            raise ValueError("Source/validation/test chronological bounds overlap future labels.")
    training = config["benefit_training"]
    if max(training["prior_origins"])+3 > min(training["meta_cutoffs"]) or max(training["meta_cutoffs"])+training["meta_block_days"] > min(s["development_cutoff_day"] for s in config["temporal_protocol"].values()):
        raise ValueError("Past-performance priors or gain labels cross their source-only cutoff.")


def _fit_models(panel, domains, validation_domains, config, model_config, task, learner, priors, folder):
    spec = config["temporal_protocol"][task]
    horizon = config["horizons"][task]
    bank, training = source_bank(panel, domains, spec["development_cutoff_day"], horizon, model_config["sapf"], config["seed"])
    validation = domain_windows(panel, validation_domains, spec["validation_origins"], 672, horizon)
    built, timings = {}, {}
    for name in model_config["default_models"]:
        started = perf_counter()
        if name in POLICY_NAMES:
            model = create_model(name, dict(bank=bank, predictor=learner if name.startswith("sapf_gain") else None,
                priors=priors, fixed_strength=model_config["fixed_blend_strength"], policy_grid=model_config["policy_grid"]), config["seed"]).fit(training)
            if name == "sapf_gain":
                model.calibrate(validation)
            elif name == "sapf_gain_no_veto":
                model.calibration = deepcopy(built["sapf_gain"].calibration)
                model.calibration["intervention"] = "Force nonzero transfer when all sources are vetoed; other choices inherited."
        else:
            params = (dict(period=48) if name == "seasonal_naive" else {} if name == "three_day_mean" else
                      model_config["ridge"] if name == "ridge" else
                      dict(base_model="ridge", base_params=model_config["ridge"]) if name == "normalized_ridge" else model_config["sapf"])
            model = create_model(name, params, config["seed"]).fit(training)
            if hasattr(model, "calibrate"):
                model.calibrate(validation)
        built[name], timings[name] = model, perf_counter()-started
        directory = folder/task/name
        directory.mkdir(parents=True, exist_ok=True)
        model.save(directory/"development")
        if getattr(model, "calibration", None):
            write_json(directory/"calibration.json", model.calibration)
    return built, timings


def _refit_models(panel, domains, config, model_config, task, initial, learner, priors, folder):
    spec = config["temporal_protocol"][task]
    bank, training = source_bank(panel, domains, spec["final_cutoff_day"], config["horizons"][task], model_config["sapf"], config["seed"])
    models, records, paths = {}, [], {}
    for name in model_config["default_models"]:
        if name in POLICY_NAMES:
            model = create_model(name, dict(bank=bank, predictor=learner if name.startswith("sapf_gain") else None,
                priors=priors, fixed_strength=model_config["fixed_blend_strength"], policy_grid=model_config["policy_grid"]), config["seed"]).fit(training)
            model.calibration = deepcopy(initial[name].calibration)
        elif hasattr(initial[name], "refit"):
            model = initial[name].refit(training)
        else:
            params = (dict(period=48) if name == "seasonal_naive" else {} if name == "three_day_mean" else
                      model_config["ridge"] if name == "ridge" else dict(base_model="ridge", base_params=model_config["ridge"]))
            model = create_model(name, params, config["seed"]).fit(training)
        models[name] = model
        paths[name] = model.save(folder/task/name/"model")
        calibration = getattr(model, "calibration", None)
        records.append(dict(model=name, horizon=task, training_domains=[] if name in {"three_day_mean", "seasonal_naive"} else list(domains),
            calibration_domains=calibration.get("validation_domains", []) if calibration else [],
            final_cutoff_day=spec["final_cutoff_day"], final_origins=list(range(14, spec["final_cutoff_day"]-config["horizons"][task]//48+1)),
            last_training_label=str(max(pd.Timestamp(t) for t in training.inputs.target_times[:, -1])),
            windows=len(training), selected=calibration))
    return models, paths, records


def _evaluate(folder, panel, roles, config, task, split, models, timings, rows, diagnostics, restores=None):
    domains = roles["validation"] if split == "validation" else roles["test"]
    spec = config["temporal_protocol"][task]
    origins = spec["validation_origins" if split == "validation" else "test_origins"]
    batch = domain_windows(panel, domains, origins, 672, config["horizons"][task])
    for name, model in models.items():
        result = backtest(model, batch, clip_min=0., batch_size=32, mape_threshold=1e-8)
        if restores is not None:
            restored = joblib.load(folder/task/name/"model.joblib")
            values = np.concatenate([predict(restored, batch.inputs.take(slice(i, i+32)), 0.) for i in range(0, len(batch), 32)])
            restores.append(dict(horizon=task, model=name, windows=len(batch), maximum_absolute_difference=float(np.max(np.abs(values-result.predictions))),
                                 passed=bool(np.allclose(values, result.predictions, rtol=1e-6, atol=1e-7))))
        if name in POLICY_NAMES:
            parts = []
            for i in range(0, len(batch), 32):
                predicted, info = model.predict_with_diagnostics(batch.inputs.take(slice(i, i+32)))
                np.testing.assert_allclose(predicted, result.predictions[i:i+32], rtol=1e-10, atol=1e-10)
                parts.append(info)
            info = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
            for i, (domain, origin) in enumerate(zip(batch.inputs.domain_ids, batch.inputs.origins)):
                local_score = forecast_metrics(batch.targets[i], info["local_prediction"][i])["MAPE"]
                for s, expert in enumerate(model.source_domains):
                    error = np.abs(batch.targets[i].astype(float)-info["expert_predictions"][i, s])
                    expert_score = float(100*np.mean(error/np.abs(batch.targets[i].astype(float))))
                    diagnostics.append(dict(horizon=task, split=split, model=name, target_domain=domain, origin=str(origin), source_domain=expert,
                        predicted_gain_pp=float(info["predicted_gain_pp"][i, s]), negative_risk=float(info["negative_risk"][i, s]),
                        actual_gain_pp=local_score-expert_score, local_MAPE=local_score, expert_MAPE=expert_score,
                        weight=float(info["weights"][i, s]), transfer_gate=float(info["transfer_gate"][i]), admissible=bool(info["admissible"][i, s])))
        for domain in domains:
            indices = np.flatnonzero(np.asarray(batch.inputs.domain_ids) == domain)
            part = WindowBatch(batch.inputs.take(indices), batch.targets[indices])
            scored = score_predictions(part, result.predictions[indices], 1e-8, part.inputs.origins[0])
            clock = panel.audit[domain]
            extra = dict(target_domain=domain, algorithm=name, source_strategy="target_history" if name in {"three_day_mean", "seasonal_naive"} else "source_only_transfer",
                target_unit="kW", source_clock_start=clock["source_clock_start"], source_clock_timezone_verified=clock["source_clock_timezone_verified"],
                protocol_anchor=str(panel.anchor), validation_score_kind="selection_in_sample" if getattr(model, "calibration", None) else "independent_domain")
            save_result(folder/domain, task, name, split, part, scored, rows, timings.get(name, 0.), "kW", False, row_metadata=extra, quiet=True)


def run_transfer_suite(args, config):
    if getattr(args, "demo", False) or getattr(args, "models", None) or getattr(args, "horizons", None):
        raise ValueError("This preregistered joint-horizon suite runs its complete configured models/tasks; use fixture tests for demos.")
    dataset, model_config = load_config(config["dataset_config"]), load_config(config["model_config"])
    panel = load_transfer_domains(dataset, config["domain_roles"])
    validate_suite(panel, config)
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_sapf_transfer_"+uuid.uuid4().hex[:6]
    folder = resolve_path(config["results_dir"])/identifier
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", protocol=config["protocol"], data="M100/UCC measured node power", target_unit="kW", is_demo=False,
        experiment=config, dataset=dataset, model_config=model_config, studies=config["studies"], requested_horizons=list(config["horizons"]),
        timestamp_axis=config["timestamp_axis"], data_provenance=panel.sources,
        source_sha256={p.relative_to(PROJECT_ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
                       for directory in ("src", "scripts", "configs") for p in (PROJECT_ROOT/directory).rglob("*") if p.suffix in {".py", ".yaml"}})
    write_json(folder/"manifest.json", manifest)
    write_json(folder/"predeclared-plan.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(), config=config, models=model_config, forecast_test_metrics_not_accessed=True))
    children, states, state_paths, timings, validation_rows, val_diag = {}, {}, {}, {}, {}, {}
    try:
        for study in config["studies"]:
            name, group, target_group = study["name"], study["source_group"], study["test_group"]
            child = folder/"studies"/name
            child.mkdir(parents=True)
            roles = dict(source=config["domain_roles"][group]["source"], validation=config["domain_roles"][group]["validation"], test=config["domain_roles"][target_group]["test"])
            ids = sum(roles.values(), [])
            hashes = {}
            for d in ids:
                path = child/"adapted-data"/(d+".parquet")
                path.parent.mkdir(exist_ok=True)
                panel.series[d].to_parquet(path)
                hashes[path.relative_to(child).as_posix()] = sha256(path.read_bytes()).hexdigest()
            cmanifest = dict(status="running", protocol="sapf_transfer30_v1", research_method=name,
                data=f"{group.upper()} source -> {target_group.upper()} held-out nodes", target="node_power_kw", target_kind="active_power", target_scope="node", target_unit="kW",
                trace_anchor_utc=str(panel.anchor), requested_horizons=list(config["horizons"]), requested_models=model_config["default_models"],
                experiment=config, domain_roles=roles, temporal_plan=temporal_plan(config), source_group=group,
                console_methods=model_config["default_models"], domain_metadata={d: panel.metadata[d].to_dict() for d in ids},
                adapted_data_sha256=hashes, timestamp_axis=config["timestamp_axis"], target_dataset_labels_used_for_selection=group == target_group)
            children[name] = child, cmanifest
            write_json(child/"manifest.json", cmanifest)
            write_json(child/"predeclared-plan.json", dict(config=config, models=model_config, roles=roles, created_at_utc=datetime.now(timezone.utc).isoformat()))
        with threadpool_limits(config["blas_threads"], user_api="blas"):
            for group in ("m100", "ucc"):
                roles = config["domain_roles"][group]
                location = folder/"groups"/group
                print(f"[{group}] 源域内部留域收益学习；训练节点 {len(roles['source'])}，独立验证 {len(roles['validation'])}", flush=True)
                learner, priors = train_joint_benefit(panel, roles["source"], config, model_config, location/"benefit")
                (location/"joint_predictor").mkdir(parents=True)
                joblib.dump(learner, location/"joint_predictor/model.joblib")
                validation_rows[group], val_diag[group] = [], []
                for task in config["horizons"]:
                    initial, seconds = _fit_models(panel, roles["source"], roles["validation"], config, model_config, task, learner, priors[task], location)
                    _evaluate(location, panel, roles, config, task, "validation", initial, seconds, validation_rows[group], val_diag[group])
                    final, paths, records = _refit_models(panel, roles["source"], config, model_config, task, initial, learner, priors[task], location)
                    states[group, task], state_paths[group, task], timings[group, task] = final, paths, seconds
                    write_json(location/task/"training-records.json", records)
                    print(f"[{group}] {task} 最终源拟合完成，尚未访问任何测试误差", flush=True)
            for study in config["studies"]:
                name, group = study["name"], study["source_group"]
                child, cmanifest = children[name]
                source = folder/"groups"/group
                shutil.copytree(source/"benefit", child/"benefit", ignore=shutil.ignore_patterns("*.joblib"))
                records = []
                for task in config["horizons"]:
                    part = json.loads((source/task/"training-records.json").read_text(encoding="utf-8"))
                    for record in part:
                        target = child/task/record["model"]
                        target.mkdir(parents=True)
                        shutil.copyfile(state_paths[group, task][record["model"]], target/"model.joblib")
                        record["model_path"] = (target/"model.joblib").relative_to(child).as_posix()
                        records.append(record)
                freeze_for_test(child, tasks=list(config["horizons"]), training_records=records, domain_roles=cmanifest["domain_roles"],
                    temporal_plan=cmanifest["temporal_plan"], target_contract={k: cmanifest[k] for k in ("target", "target_kind", "target_scope", "target_unit")},
                    adapted_data_sha256=cmanifest["adapted_data_sha256"], config_digest=config_digest(config))
            freeze_for_test(folder, tasks=list(config["horizons"]), studies=config["studies"], global_domain_roles=config["domain_roles"],
                training_artifact_sha256={p.relative_to(folder).as_posix(): sha256(p.read_bytes()).hexdigest() for p in folder.glob("groups/*/benefit/*.csv")})
            print("三个研究的全部模型已统一冻结；现在开始独立测试。", flush=True)
            for study in config["studies"]:
                name, group = study["name"], study["source_group"]
                child, cmanifest = children[name]
                write_json(child/"test-start.json", dict(started_at_utc=datetime.now(timezone.utc).isoformat(), freeze_sha256=sha256((child/"frozen-before-test.json").read_bytes()).hexdigest()))
                rows = deepcopy(validation_rows[group])
                # Shared source-validation predictions are copied byte-for-byte to each target study.
                for d in cmanifest["domain_roles"]["validation"]:
                    shutil.copytree(folder/"groups"/group/d, child/d)
                diagnostic, restores = deepcopy(val_diag[group]), []
                for task in config["horizons"]:
                    _evaluate(child, panel, cmanifest["domain_roles"], config, task, "test", states[group, task], timings[group, task], rows, diagnostic, restores)
                metrics = pd.concat([add_baseline_reductions(g, {task: "three_day_mean" for task in config["horizons"]}) for _, g in pd.DataFrame(rows).groupby("target_domain", sort=False)], ignore_index=True)
                metrics.to_csv(child/"metrics.csv", index=False)
                diagnostic_table = pd.DataFrame(diagnostic)
                diagnostic_table.to_csv(child/"gain-diagnostics.csv", index=False)
                risk_summary(diagnostic_table).to_csv(child/"risk-summary.csv", index=False)
                pairs, summary = summarize_transfer(metrics)
                pairs.to_csv(child/"negative-transfer.csv", index=False)
                summary.to_csv(child/"domain-summary.csv", index=False)
                write_json(child/"saved-model-audit.json", restores)
                audit = audit_cross_domain_run(child)
                write_json(child/"audit.json", audit)
                if not audit["passed"]:
                    raise ValueError(f"{name}: frozen-run audit failed.")
                cmanifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
                write_json(child/"manifest.json", cmanifest)
                print(f"[{name}] 两任务全部测试与恢复核对完成", flush=True)
        from src.evaluation.transfer_audit import audit_transfer_suite
        audit = audit_transfer_suite(folder)
        write_json(folder/"audit.json", audit)
        if not audit["passed"]:
            raise ValueError("Global source/meta-label/freeze/holdout audit failed.")
        manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder/"manifest.json", manifest)
        print_saved_results(folder)
        return folder
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(folder/"manifest.json", manifest)
        raise
