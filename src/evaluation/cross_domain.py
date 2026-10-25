"""Public-score cross-domain summaries, source diagnostics and frozen-run verification."""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd

from src.evaluation.metrics import forecast_metrics


def summarize_transfer(metrics):
    test = metrics[(metrics.split == "test") & (metrics.evaluation == "curve")]
    summary = test.groupby(["horizon", "model", "source_strategy"], sort=False).agg(
        targets=("target_domain", "nunique"), mean_MAPE=("MAPE", "mean"), worst_MAPE=("MAPE", "max"),
        total_points=("n", "sum")).reset_index()
    comparisons = []
    for (target, horizon), group in test.groupby(["target_domain", "horizon"]):
        values = group.set_index("model")
        pairs = [("ridge", "normalized_ridge", "history_normalization"),
                 ("normalized_ridge", "sapf_similarity", "similarity_selection"),
                 ("ridge", "sapf", "selective_vs_raw_pool"),
                 ("normalized_ridge", "sapf", "selective_vs_normalized_pool"),
                 ("three_day_mean", "sapf", "selective_vs_local_reference")]
        from src.models.ml.ablation import ABLATION_MODES
        pairs.extend(("sapf", name, "ablation_disable_"+disabled) for name, disabled in ABLATION_MODES.items())
        for method in values.index:
            if method not in {"naive", "seasonal_naive", "three_day_mean"}:
                pairs.extend((reference, method, label) for reference, label in
                             (("three_day_mean", "vs_local_reference"), ("ridge", "vs_raw_pool"),
                              ("normalized_ridge", "vs_normalized_pool")))
        pairs.extend((name, "ridge", "adding_raw_sources") for name in values.index if name.startswith("single_ridge__"))
        pairs.extend((name, "normalized_ridge", "adding_normalized_sources")
                     for name in values.index if name.startswith("single_normalized_ridge__"))
        for baseline, method, kind in pairs:
            if baseline not in values.index or method not in values.index:
                continue
            before, after = values.loc[baseline], values.loc[method]
            if (before.n, before.mape_n) != (after.n, after.mape_n):
                raise ValueError("Transfer comparison requires identical evaluated target points.")
            delta = float(after.MAPE-before.MAPE)
            comparisons.append(dict(target_domain=target, horizon=horizon, comparison=kind,
                baseline=baseline, method=method, baseline_MAPE=float(before.MAPE), method_MAPE=float(after.MAPE),
                delta_MAPE_pp=delta, relative_change_pct=100*delta/before.MAPE if before.MAPE > 0 else None,
                negative_transfer=bool(delta > 1e-10), n=int(after.n)))
    comparisons = pd.DataFrame(comparisons)
    if not comparisons.empty:
        for label, prefix in [("vs_local_reference", "negative_vs_local"), ("vs_raw_pool", "worse_vs_raw_pool"),
                              ("vs_normalized_pool", "worse_vs_normalized_pool")]:
            subset = comparisons[comparisons.comparison == label]
            counts = subset.groupby(["horizon", "method"]).negative_transfer.agg(["sum", "count"]).reset_index()
            counts = counts.rename(columns={"method": "model", "sum": prefix+"_domains", "count": prefix+"_total"})
            counts[prefix+"_pct"] = 100*counts[prefix+"_domains"]/counts[prefix+"_total"]
            summary = summary.merge(counts, on=["horizon", "model"], how="left", validate="one_to_one")
    return comparisons, summary


def analyze_pairwise(pairwise, folder):
    pairwise.to_csv(folder/"source-target-diagnostics.csv", index=False)
    rows = []
    if not pairwise.empty:
        features = [c for c in pairwise if c.startswith("difference_")]
        for (split, task, algorithm), group in pairwise.groupby(["split", "horizon", "algorithm"]):
            for feature in features:
                valid = group[[feature, "MAPE"]].dropna()
                for method in ("pearson", "spearman"):
                    value = valid.corr(method=method).iloc[0, 1] if len(valid) >= 3 and valid[feature].std() > 0 and valid.MAPE.std() > 0 else np.nan
                    rows.append(dict(split=split, horizon=task, algorithm=algorithm, feature=feature,
                                     correlation_method=method, correlation=float(value) if np.isfinite(value) else None,
                                     pairs=len(valid), interpretation="Descriptive repeated source/target/origin pairs; no causal or independent-sample claim."))
    table = pd.DataFrame(rows)
    table.to_csv(folder/"feature-transfer-correlations.csv", index=False)
    return table


def plot_sapf_results(metrics, profiles, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test = metrics[(metrics.split == "test") & (metrics.evaluation == "curve")]
    main = test[~test.model.str.startswith("single_")]
    tasks = list(main.horizon.drop_duplicates())
    fig, axes = plt.subplots(1, len(tasks), figsize=(7*len(tasks), 4.8), squeeze=False, sharey=True)
    order = ["seasonal_naive", "three_day_mean", "ridge", "normalized_ridge", "sapf_similarity", "sapf"]
    labels = {"seasonal_naive": "Previous day", "three_day_mean": "3-day mean", "ridge": "Raw pooled",
              "normalized_ridge": "Normalized pooled", "sapf_similarity": "Similarity", "sapf": "SAPF"}
    for ax, task in zip(axes[0], tasks):
        part = main[main.horizon == task].groupby("model").MAPE.agg(["mean", "max"])
        part = part.reindex([key for key in order if key in part.index]+[key for key in part.index if key not in order])
        part.index = [labels.get(key, key) for key in part.index]
        part.plot.bar(ax=ax, color=["#277c9b", "#d36b45"], rot=28)
        ax.set_title(f"{task}: equal-domain mean and worst MAPE")
        ax.set_ylabel("MAPE (%)")
        ax.set_xlabel("")
        ax.legend(["Mean target", "Worst target"])
    fig.tight_layout()
    fig.savefig(folder/"robustness.png", dpi=160)
    plt.close(fig)
    singles = test[test.model.str.startswith("single_")].copy()
    if not singles.empty:
        singles["source"] = singles.model.str.split("__").str[-1]
        fig, axes = plt.subplots(2, len(tasks), figsize=(max(14, singles.source.nunique()*.8), 8), squeeze=False)
        maximum = float(singles.MAPE.max())
        for column, task in enumerate(tasks):
            for row, prefix in enumerate(("single_ridge__", "single_normalized_ridge__")):
                part = singles[(singles.horizon == task) & singles.model.str.startswith(prefix)]
                matrix = part.pivot(index="target_domain", columns="source", values="MAPE")
                picture = axes[row, column].imshow(matrix.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=maximum)
                axes[row, column].set_title(f"{task} / {'raw' if row == 0 else 'history normalized'} MAPE (%)")
                axes[row, column].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=65, ha="right", fontsize=8)
                axes[row, column].set_yticks(range(len(matrix)), matrix.index, fontsize=8)
                fig.colorbar(picture, ax=axes[row, column], shrink=.7)
        fig.tight_layout()
        fig.savefig(folder/"source-target-matrices.png", dpi=160)
        plt.close(fig)
    # Standardize the visualization using training source profiles only.
    if not profiles.empty:
        part = profiles[(profiles.stage == "final") & (profiles.horizon == tasks[0])].drop_duplicates("domain_id")
        keys = ["log_level", "coefficient_of_variation", "daily_correlation", "weekly_correlation",
                "relative_trend_per_day", "relative_mean_ramp"]
        source = part[part.role == "source"][keys]
        normalized = (part[keys]-source.mean())/source.std().replace(0, 1)
        fig, ax = plt.subplots(figsize=(9, max(5, len(part)*.23)))
        limit = max(1.0, float(np.abs(normalized.to_numpy()).max()))
        picture = ax.imshow(normalized.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(keys)), keys, rotation=28, ha="right", fontsize=8)
        ax.set_yticks(range(len(part)), [f"{r.role}: {r.domain_id}" for r in part.itertuples()], fontsize=7)
        ax.set_title("Observed 14-day load signatures before the first test origin")
        fig.colorbar(picture, ax=ax, label="Source-standardized feature value", shrink=.7)
        fig.tight_layout()
        fig.savefig(folder/"domain-signatures.png", dpi=160)
        plt.close(fig)


def audit_cross_domain_run(folder):
    import json
    if json.loads((folder/'manifest.json').read_text(encoding='utf-8')).get('protocol') == 'sapf_gain_v2_suite':
        from src.evaluation.gain_v2_audit import audit_gain_v2
        return audit_gain_v2(folder)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") == "sapf_transfer_suite_v1":
        from src.evaluation.transfer_audit import audit_transfer_suite
        return audit_transfer_suite(folder)
    if not manifest.get("protocol", "").startswith("sapf_"):
        raise ValueError("Use the frozen legacy audit for pre-SAPF runs.")
    freeze_path = folder/"frozen-before-test.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    start = json.loads((folder/"test-start.json").read_text(encoding="utf-8"))
    roles = manifest["domain_roles"]
    minutes = int(pd.Timedelta(manifest["experiment"].get("frequency", "15min")).total_seconds()/60)
    period = 1440//minutes
    declared = json.loads((folder/"predeclared-plan.json").read_text(encoding="utf-8"))
    plan_unchanged = roles == freeze["domain_roles"] == declared["roles"]
    disjoint = not (set(roles["source"]) & set(roles["validation"]) or set(roles["source"]) & set(roles["test"])
                    or set(roles["validation"]) & set(roles["test"]))
    frozen_first = (start["freeze_sha256"] == sha256(freeze_path.read_bytes()).hexdigest()
                    and pd.Timestamp(freeze["frozen_at_utc"]) <= pd.Timestamp(start["started_at_utc"]))
    model_hashes = bool(freeze["model_sha256"]) and all(sha256((folder/str(p).replace(chr(92), "/")).read_bytes()).hexdigest() == value
                                                  for p, value in freeze["model_sha256"].items())
    adapted_hashes = freeze.get("adapted_data_sha256", {})
    adapted_unchanged = adapted_hashes == manifest.get("adapted_data_sha256", {}) and all(
        sha256((folder/path).read_bytes()).hexdigest() == value for path, value in adapted_hashes.items())
    if manifest["protocol"] in {"sapf_m100_nodes_v1", "sapf_transfer30_v1"} and len(adapted_hashes) != sum(len(roles[r]) for r in ("source", "validation", "test")):
        adapted_unchanged = False
    physical_contract = True
    if manifest["protocol"] in {"sapf_m100_nodes_v1", "sapf_transfer30_v1"}:
        expected_contract = dict(target="node_power_kw", target_kind="active_power", target_scope="node", target_unit="kW")
        physical_contract = (all(manifest.get(k) == v for k, v in expected_contract.items())
            and freeze.get("target_contract") == expected_contract
            and all(m["target_field"] == "node_power_kw" and m["target_unit"] == "kW" and m["target_scope"] == "node"
                    for m in manifest["domain_metadata"].values()))
    records = freeze["training_records"]
    from src.models.ml.ablation import ABLATION_MODES
    lookup = {(r["horizon"], r["model"]): r for r in records}
    interventions_valid = True
    for record in records:
        if record["model"] in ABLATION_MODES:
            reference = lookup.get((record["horizon"], "sapf"))
            selected = record.get("selected") or {}
            intervention = selected.get("intervention", {})
            interventions_valid &= bool(reference and intervention.get("independently_retuned") is False
                and intervention.get("disabled") == ABLATION_MODES[record["model"]]
                and intervention.get("reference_selected") == reference["selected"]["selected"]
                and selected.get("selected") == reference["selected"]["selected"]
                and selected.get("reliability") == reference["selected"]["reliability"]
                and record["training_domains"] == reference["training_domains"]
                and record["calibration_domains"] == reference["calibration_domains"]
                and record["final_origins"] == reference["final_origins"])
    source_only = all(set(r["training_domains"]) <= set(roles["source"]) for r in records)
    calibration_only = all(set(r.get("calibration_domains", [])) <= set(roles["validation"]) for r in records)
    anchor = pd.Timestamp(manifest["trace_anchor_utc"])
    temporal = all(pd.Timestamp(r["last_training_label"]) < anchor+pd.Timedelta(days=r["final_cutoff_day"])
                   and all(day+manifest["experiment"]["horizons"][r["horizon"]]//period <= r["final_cutoff_day"] for day in r["final_origins"])
                   for r in records)
    for task, spec in manifest["temporal_plan"].items():
        temporal = temporal and (min(spec["validation_origins"]) >= spec["development_cutoff_day"]
            and max(spec["validation_origins"])+manifest["experiment"]["horizons"][task]//period <= spec["final_cutoff_day"]
            and min(spec["test_origins"]) >= spec["final_cutoff_day"])
    metrics = pd.read_csv(folder/"metrics.csv")
    keys = ["target_domain", "horizon", "model", "split", "evaluation"]
    if metrics.duplicated(keys).any():
        raise ValueError("Duplicate metric identities in saved experiment.")
    expected_keys = {(domain, record["horizon"], record["model"], split, evaluation)
                     for record in records for split, domains in (("validation", roles["validation"]), ("test", roles["test"]))
                     for domain in domains for evaluation in ("curve", "daily_mean")}
    metric_coverage = set(metrics[keys].itertuples(index=False, name=None)) == expected_keys
    checks, coordinates = [], {}
    for (domain, task, model, split), group in metrics.groupby(["target_domain", "horizon", "model", "split"]):
        table = pd.read_parquet(folder/domain/task/model/f"{split}_predictions.parquet").sort_values(["origin", "lead"]).reset_index(drop=True)
        h = manifest["experiment"]["horizons"][task]
        if (len(table) % h or table.duplicated(["origin", "lead"]).any()
                or any(part.lead.tolist() != list(range(1, h+1)) for _, part in table.groupby("origin"))):
            raise ValueError("Incomplete or duplicate saved prediction curves.")
        key = domain, task, split
        current = table[["origin", "timestamp", "lead", "actual", "domain_id"]]
        if key in coordinates and not coordinates[key].equals(current):
            raise ValueError("Compared methods do not share exactly the same target windows.")
        coordinates[key] = current
        expected_domains = roles["test"] if split == "test" else roles["validation"]
        if domain not in expected_domains or set(table.domain_id) != {domain}:
            raise ValueError("Saved evaluation has a wrong domain role.")
        origin_days = manifest["temporal_plan"][task]["test_origins" if split == "test" else "validation_origins"]
        expected_origins = [anchor+pd.Timedelta(days=day) for day in origin_days]
        if list(pd.DatetimeIndex(table.origin.drop_duplicates())) != expected_origins:
            raise ValueError("Saved evaluation does not match the fixed origin list.")
        timestamps = pd.to_datetime(table.origin, utc=True)+pd.to_timedelta((table.lead-1)*minutes, unit="min")
        if not (pd.to_datetime(table.timestamp, utc=True) == timestamps).all():
            raise ValueError("Saved target timestamps are not the complete declared 15min horizon.")
        actual = table.actual.to_numpy(np.float32).reshape(-1, h)
        if manifest["protocol"] in {"sapf_m100_nodes_v1", "sapf_transfer30_v1"}:
            adapted = pd.read_parquet(folder/"adapted-data"/(domain+".parquet"))
            expected = adapted.loc[pd.DatetimeIndex(table.timestamp), manifest["target"]].to_numpy(np.float32)
            if not np.array_equal(expected, table.actual.to_numpy(np.float32)):
                raise ValueError("Saved M100 labels do not match the adapted kW target.")
            if manifest["protocol"] == "sapf_transfer30_v1":
                source_clock = adapted.loc[pd.DatetimeIndex(table.timestamp), "source_clock_timestamp"].to_numpy()
                if not np.array_equal(source_clock, table.source_clock_timestamp.to_numpy()):
                    raise ValueError("Original source-clock labels were not preserved.")
        predicted = table.predicted.to_numpy(float).reshape(-1, h)
        for _, row in group.iterrows():
            if row.evaluation == "curve":
                computed = forecast_metrics(actual, predicted, manifest["experiment"]["mape_threshold"])
            else:
                computed = forecast_metrics(actual.reshape(len(actual), -1, period).mean(-1),
                                            predicted.reshape(len(predicted), -1, period).mean(-1),
                                            manifest["experiment"]["mape_threshold"])
            passed = all((value is None and pd.isna(row[key])) or
                         (value is not None and np.isclose(value, row[key], atol=1e-10, rtol=1e-10))
                         for key, value in computed.items())
            checks.append(dict(target_domain=domain, horizon=task, model=model, split=split,
                               evaluation=row.evaluation, passed=bool(passed)))
    restores = json.loads((folder/"saved-model-audit.json").read_text(encoding="utf-8"))
    restored = len(restores) == len(records) and bool(restores) and all(r["passed"] for r in restores)
    passed = disjoint and plan_unchanged and metric_coverage and frozen_first and model_hashes and adapted_unchanged and physical_contract and interventions_valid and source_only and calibration_only and temporal and restored and all(c["passed"] for c in checks)
    return dict(passed=bool(passed), prediction_checks=len(checks), roles_disjoint=disjoint,
                declared_plan_unchanged=bool(plan_unchanged), metric_coverage_complete=bool(metric_coverage),
                freeze_precedes_test=bool(frozen_first), model_hashes_unchanged=bool(model_hashes),
                adapted_data_hashes_unchanged=bool(adapted_unchanged), controlled_ablations_matched=bool(interventions_valid),
                physical_contract_preserved=bool(physical_contract),
                fitting_source_only=source_only, calibration_validation_only=calibration_only,
                temporal_boundaries_valid=temporal, restore_checks=len(restores), checks=checks)
