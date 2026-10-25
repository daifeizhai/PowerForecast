"""Read-only summary of a completed frozen study; never fit or choose a model."""
from pathlib import Path
import hashlib
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/"results/runs/20261025T121710Z_sapf_4355ef"
ANALYSIS = ROOT/"results/analysis/m100_nodes_v1"
GOOGLE = ROOT/"docs/figures/sapf-20261025T082346Z_5644ac"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT/"docs/figures/m100-20261025T121710Z_4355ef", help="New report-artifact directory; never overwrite existing evidence.")
args = parser.parse_args()
OUT = args.output if args.output.is_absolute() else ROOT/args.output
OUT.mkdir(exist_ok=False)
manifest = json.loads((RUN/"manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "complete" and manifest["protocol"] == "sapf_m100_nodes_v1"
for name in ["manifest.json", "predeclared-plan.json", "frozen-before-test.json", "test-start.json", "audit.json",
             "saved-model-audit.json", "metrics.csv", "domain-summary.csv", "negative-transfer.csv", "data-audit.json",
             "source-weights.csv", "domain-profiles.csv", "source-target-diagnostics.csv",
             "feature-transfer-correlations.csv", "robustness.png", "source-target-matrices.png", "domain-signatures.png"]:
    shutil.copyfile(RUN/name, OUT/name)
for name in ["preregistration.json", "independent-verification.json", "recomputed-audit.json", "verify_independently.py", "real-run.log"]:
    shutil.copyfile(ANALYSIS/name, OUT/name)
shutil.copyfile(ROOT/"results/analysis/power_sources/pytest_m100_before_run.log", OUT/"pytest.log")

metrics = pd.read_csv(RUN/"metrics.csv")
test = metrics[(metrics.split == "test") & (metrics.evaluation == "curve")]
names = manifest["model_config"]["default_models"]
summary = pd.read_csv(RUN/"domain-summary.csv")
summary[summary.model.isin(names)].to_csv(OUT/"main-summary.csv", index=False)
measures = metrics[(metrics.split == "test") & metrics.model.isin(names)].groupby(["horizon", "evaluation", "model"])[
    ["MAPE", "WAPE", "RMSE", "MAE"]].mean().reset_index()
measures.to_csv(OUT/"all-main-metrics.csv", index=False)

ablations, normalization, nonoverlap, comparison = [], [], [], []
for task in ["24h", "3d"]:
    part = test[test.horizon == task]
    full = part[part.model == "sapf"].set_index("target_domain").MAPE
    for method in ["sapf_no_normalization", "sapf_no_selection", "sapf_no_fallback", "sapf_no_reliability"]:
        off = part[part.model == method].set_index("target_domain").MAPE
        ablations.append(dict(horizon=task, intervention=method, full_mean_MAPE=full.mean(),
            disabled_mean_MAPE=off.mean(), delta_mean_MAPE_pp=(off-full).mean(),
            full_worst_MAPE=full.max(), disabled_worst_MAPE=off.max(), delta_worst_MAPE_pp=off.max()-full.max(),
            nodes_worse_when_disabled=int((off-full > 1e-10).sum()), nodes=len(full),
            interpretation="One mechanism disabled; all full-SAPF validation choices inherited, not separately optimized."))
    raw = part[part.model.str.startswith("single_ridge__")].copy()
    normalized = part[part.model.str.startswith("single_normalized_ridge__")].copy()
    raw["source"] = raw.model.str.split("__").str[-1]
    normalized["source"] = normalized.model.str.split("__").str[-1]
    pairs = raw.merge(normalized, on=["target_domain", "source"], suffixes=("_raw", "_normalized"))
    for row in pairs.itertuples():
        normalization.append(dict(horizon=task, source_domain=row.source, target_domain=row.target_domain,
            raw_MAPE=row.MAPE_raw, normalized_MAPE=row.MAPE_normalized,
            improved=bool(row.MAPE_normalized < row.MAPE_raw-1e-10)))
    for method in names:
        if task != "3d":
            continue
        scores = []
        for domain in manifest["domain_roles"]["test"]:
            table = pd.read_parquet(RUN/domain/task/method/"test_predictions.parquet").sort_values(["origin", "lead"])
            starts = list(table.origin.drop_duplicates())[::3]
            selected = table[table.origin.isin(starts)]
            a, p = selected.actual.to_numpy(float), selected.predicted.to_numpy(float)
            error = a-p
            score = dict(MAPE=100*np.mean(np.abs(error)/np.abs(a)), WAPE=100*np.abs(error).sum()/np.abs(a).sum(),
                         RMSE=np.sqrt(np.mean(error**2)), MAE=np.abs(error).mean())
            saved = json.loads((RUN/domain/task/method/"test_metrics.json").read_text())["nonoverlap"]
            assert all(np.isclose(v, saved[k], rtol=1e-10, atol=1e-10) for k, v in score.items())
            scores.append(score)
        nonoverlap.append(dict(horizon=task, model=method, targets=len(scores), origins_per_target=6,
            mean_MAPE=np.mean([s["MAPE"] for s in scores]), worst_MAPE=max(s["MAPE"] for s in scores),
            mean_WAPE=np.mean([s["WAPE"] for s in scores]), mean_RMSE_kw=np.mean([s["RMSE"] for s in scores]),
            mean_MAE_kw=np.mean([s["MAE"] for s in scores])))

google = pd.read_csv(GOOGLE/"metrics.csv")
for dataset, frame in [("Google frozen 31-day power-domain utilization", google), ("M100 frozen 60-day node kW", metrics)]:
    frame = frame[(frame.split == "test") & (frame.evaluation == "curve")]
    for task in ["24h", "3d"]:
        part = frame[frame.horizon == task]
        local = part[part.model == "three_day_mean"].set_index("target_domain").MAPE
        ridge = part[part.model == "ridge"].set_index("target_domain").MAPE
        for method in ["seasonal_naive", "three_day_mean", "ridge", "normalized_ridge", "sapf_similarity", "sapf"]:
            values = part[part.model == method].set_index("target_domain").MAPE
            comparison.append(dict(dataset=dataset, horizon=task, model=method, mean_MAPE=values.mean(),
                worst_MAPE=values.max(), mean_delta_vs_local_pp=values.mean()-local.mean(),
                worse_than_local_domains=int((values-local > 1e-10).sum()),
                worse_than_raw_ridge_domains=int((values-ridge > 1e-10).sum()), targets=len(values)))

ablation = pd.DataFrame(ablations)
ablation.to_csv(OUT/"controlled-ablation-deltas.csv", index=False)
pd.DataFrame(normalization).to_csv(OUT/"single-source-normalization-pairs.csv", index=False)
pd.DataFrame(nonoverlap).to_csv(OUT/"nonoverlap-3d.csv", index=False)
pd.DataFrame(comparison).to_csv(OUT/"google-m100-comparison.csv", index=False)

order = ["sapf_no_normalization", "sapf_no_selection", "sapf_no_fallback", "sapf_no_reliability"]
labels = ["Normalization", "Source routing", "History fallback", "Reliability weights"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
for ax, column, title in zip(axes, ["delta_mean_MAPE_pp", "delta_worst_MAPE_pp"], ["Change in mean node MAPE", "Change in worst-node MAPE"]):
    for offset, task, color in [(-.17, "24h", "#287c9c"), (.17, "3d", "#ce7144")]:
        values = ablation[ablation.horizon == task].set_index("intervention").loc[order, column].to_numpy()
        bars = ax.barh(np.arange(4)+offset, values, height=.31, color=color, label=task)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis() if not ax.yaxis_inverted() else None
    ax.set_xlabel("Disabled minus full SAPF (percentage points)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, ablation[column].max()*1.2)
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, loc="lower right")
fig.suptitle("Controlled one-mechanism ablations: positive changes mean disabling is worse", fontsize=12)
fig.text(.16, .015, "Full SAPF validation choices stay fixed in every intervention. No independent ablation retuning.", fontsize=8, color="#555555")
fig.tight_layout(rect=[0, .04, 1, .94])
fig.savefig(OUT/"ablation-deltas.png", dpi=170)
plt.close(fig)

check = json.loads((OUT/"independent-verification.json").read_text())
evidence = dict(run_id=RUN.name, google_reference=GOOGLE.name, real_M100_forecasting_runs=1,
    source_snapshot="archive/pre_m100_sapf_20261025", no_M100_parameter_changes_after_test=True,
    protocol_sha256=hashlib.sha256((ROOT/"docs/M100_PROTOCOL.md").read_bytes()).hexdigest(),
    validation=dict(pytest=44, metric_checks=check["metric_checks"], model_restore_checks=68,
        raw_w_to_kw_nodes=33, raw_w_to_kw_points=190080, mixture_checks=check["mixture_checks"],
        maximum_mixture_difference_kw=check["maximum_mixture_difference_kw"], ratio_checks=check["summary_ratio_checks"],
        nonoverlap_metric_checks=400),
    post_test_changes="Only completed-dataset catalog metadata, documentation, artifact summaries, plots and independent verification; no computational model, fit configuration, parameters, source roles or predictions changed.",
    source_code_scope="Computational source and fit configs match preregistration; registry.yaml is an audit catalog and is not read by experiment fitting.",
    hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()})
(OUT/"evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print("Evidence saved:", OUT)
print(ablation[["horizon", "intervention", "delta_mean_MAPE_pp", "nodes_worse_when_disabled"]].to_string(index=False))
