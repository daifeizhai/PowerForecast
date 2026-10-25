"""Same-target transfer comparisons and auditable metrics from saved curves."""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd

from src.data.splitting import ForecastInputs, WindowBatch
from src.evaluation.backtest import score_predictions


def summarize_transfer(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = metrics[(metrics.split == "test") & (metrics.evaluation == "curve")]
    comparisons = []
    for (target, horizon, algorithm), group in curves.groupby(["target_domain", "horizon", "algorithm"]):
        multi = group[group.source_strategy == "multi_source"]
        for _, baseline in group[group.source_strategy == "single_source"].iterrows():
            if len(multi) != 1:
                continue
            added = multi.iloc[0]
            if (baseline.n, baseline.mape_n) != (added.n, added.mape_n):
                raise ValueError("Negative transfer requires identical target evaluation coverage.")
            valid = bool(pd.notna(baseline.MAPE) and pd.notna(added.MAPE))
            delta = float(added.MAPE-baseline.MAPE) if valid else None
            comparisons.append(dict(target_domain=target, horizon=horizon, algorithm=algorithm,
                baseline=baseline.model, expanded=added.model, baseline_MAPE=baseline.MAPE,
                expanded_MAPE=added.MAPE, delta_MAPE_pp=delta,
                relative_change_pct=100*delta/baseline.MAPE if valid and baseline.MAPE > 0 else None,
                negative_transfer=delta > 1e-10 if valid else None, n=int(added.n)))
    summary = curves.groupby(["horizon", "model", "source_strategy"], sort=False).agg(
        targets=("target_domain", "nunique"), mean_MAPE=("MAPE", "mean"), worst_MAPE=("MAPE", "max"),
        total_points=("n", "sum")).reset_index()
    return pd.DataFrame(comparisons), summary


def history_profile(panel, domain: str, origin_day: int, context: int) -> dict:
    y = panel.series[domain].measured_power_util.iloc[origin_day*96-context:origin_day*96].to_numpy(float)
    finite = y[np.isfinite(y)]
    mean = float(finite.mean()) if finite.size else None
    def correlation(lag):
        a, b = y[:-lag], y[lag:]
        valid = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[valid], b[valid])[0, 1]) if valid.sum() > 2 and np.std(a[valid]) > 0 and np.std(b[valid]) > 0 else None
    differences = np.diff(y)
    differences = differences[np.isfinite(differences)]
    return dict(domain_id=domain, origin_day=origin_day, history_points=context, finite_points=len(finite), mean=mean,
                coefficient_of_variation=float(finite.std()/abs(mean)) if finite.size and mean else None,
                autocorrelation_lag1=correlation(1), daily_correlation=correlation(96), weekly_correlation=correlation(672),
                mean_absolute_ramp=float(np.mean(np.abs(differences))) if differences.size else None)


def profile_transfer_relationship(metrics, profiles):
    features = ["mean", "coefficient_of_variation", "daily_correlation", "weekly_correlation", "mean_absolute_ramp"]
    lookup = profiles.set_index("domain_id").to_dict("index")
    rows = []
    for _, row in metrics[(metrics.split == "test") & (metrics.evaluation == "curve") &
                          (metrics.source_strategy == "single_source")].iterrows():
        source = json.loads(row.source_domains)[0]
        target = row.target_domain
        record = dict(horizon=row.horizon, source_domain=source, target_domain=target, MAPE=row.MAPE)
        for key in features:
            a, b = lookup[source][key], lookup[target][key]
            record[key+"_absolute_difference"] = abs(a-b) if a is not None and b is not None else None
        rows.append(record)
    frame = pd.DataFrame(rows)
    correlations = {}
    if len(frame):
        for task, group in frame.groupby("horizon"):
            correlations[task] = {}
            for key in features:
                values = group[[key+"_absolute_difference", "MAPE"]].dropna()
                value = values.corr().iloc[0, 1] if len(values) >= 3 else np.nan
                correlations[task][key] = float(value) if np.isfinite(value) else None
    return frame, dict(pearson_correlations=correlations,
                       limitation="Descriptive correlations from a small fixed pilot; repeated targets/sources are dependent, not causal evidence or a source-selection rule.")


def plot_transfer_matrix(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = metrics[(metrics.split == "test") & (metrics.evaluation == "curve") &
                   (metrics.source_strategy == "single_source")]
    if rows.empty:
        return
    tasks = rows.horizon.unique()
    maximum = float(rows.MAPE.max())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6*len(tasks), 4.5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        table = rows[rows.horizon == task].pivot(index="target_domain", columns="model", values="MAPE")
        values = table.to_numpy(float)
        picture = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=maximum)
        ax.set_xticks(range(len(table.columns)), [v.split("__", 1)[-1] for v in table.columns], rotation=25, ha="right")
        ax.set_yticks(range(len(table)), table.index)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center")
        ax.set_title(f"{task}: target-domain curve MAPE (%)")
        ax.set_xlabel("Single training source")
        ax.set_ylabel("Held-out power domain")
        fig.colorbar(picture, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def audit_cross_domain_run(folder) -> dict:
    """Recompute metrics, compare coordinates, and verify frozen model hashes."""
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    freeze = json.loads((folder/"frozen-before-test.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(folder/"metrics.csv")
    marker = json.loads((folder/"test-start.json").read_text(encoding="utf-8"))
    freeze_precedes_test = (pd.Timestamp(freeze["frozen_at_utc"]) <= pd.Timestamp(marker["started_at_utc"])
                            and marker["freeze_sha256"] == sha256((folder/"frozen-before-test.json").read_bytes()).hexdigest())
    checks = []
    coordinates = {}
    for (domain, task, model, split), group in metrics.groupby(["target_domain", "horizon", "model", "split"]):
        table = pd.read_parquet(folder/domain/task/model/f"{split}_predictions.parquet")
        table = table.sort_values(["domain_id", "origin", "lead"]).reset_index(drop=True)
        horizon = manifest["experiment"]["horizons"][task]
        if len(table) % horizon or table.duplicated(["domain_id", "origin", "lead"]).any():
            raise ValueError("Incomplete or duplicate saved prediction curves.")
        groups = list(table.groupby(["domain_id", "origin"], sort=False))
        if any(len(part) != horizon or part.lead.tolist() != list(range(1, horizon+1)) for _, part in groups):
            raise ValueError("Saved curves must contain every forecast lead exactly once.")
        if split == "test":
            key = domain, task
            current = table[["domain_id", "origin", "timestamp", "lead", "actual"]]
            if key in coordinates and not coordinates[key].equals(current):
                raise ValueError("Compared strategies used different test targets or windows.")
            coordinates[key] = current
        origins = pd.DatetimeIndex([key[1] for key, _ in groups])
        actual = table.actual.to_numpy(np.float32).reshape(-1, horizon)
        prediction = table.predicted.to_numpy(float).reshape(-1, horizon)
        inputs = ForecastInputs(np.empty((len(groups), 1, 1)), np.empty((len(groups), horizon, 1)), origins,
                                table.timestamp.to_numpy().reshape(-1, horizon))
        result = score_predictions(WindowBatch(inputs, actual), prediction, manifest["experiment"]["mape_threshold"])
        for _, row in group.iterrows():
            recomputed = result.metrics if row.evaluation == "curve" else result.daily_metrics
            passed = all((pd.isna(row[k]) and v is None) or (v is not None and np.isclose(row[k], v, rtol=1e-10, atol=1e-10))
                         for k, v in recomputed.items() if k in row)
            checks.append(dict(target_domain=domain, horizon=task, model=model, split=split,
                               evaluation=row.evaluation, metrics_recomputed=bool(passed)))
    hashes = all(sha256((folder/p).read_bytes()).hexdigest() == digest for p, digest in freeze["model_sha256"].items())
    isolation = all(record["target_domain"] not in record["training_domains"]
                    and record["target_domain"] not in record["validation_domains"]
                    for record in freeze["domain_isolation"] if record["source_strategy"] in {"single_source", "multi_source"})
    anchor = pd.Timestamp(manifest["dataset"]["trace_start"]).tz_convert("UTC")
    temporal = all(pd.Timestamp(record["training_last_label"]) < anchor+pd.Timedelta(days=record["final_cutoff_day"])
                   and all(day+manifest["experiment"]["horizons"][record["horizon"]]//96 <= record["final_cutoff_day"]
                           for day in record["final_origins"])
                   and record["final_cutoff_day"] <= min(freeze["plans"][record["horizon"]]["test_origins"])
                   for record in freeze["domain_isolation"])
    restores = json.loads((folder/"saved-model-audit.json").read_text(encoding="utf-8"))
    return dict(passed=bool(checks) and all(c["metrics_recomputed"] for c in checks) and hashes and isolation
                       and temporal and freeze_precedes_test and len(restores) == len(freeze["domain_isolation"])
                       and all(r["passed"] for r in restores),
                prediction_checks=len(checks), model_hashes_unchanged=hashes, source_target_isolation=isolation,
                training_cutoffs_valid=temporal, freeze_precedes_test=freeze_precedes_test,
                restore_checks=len(restores), checks=checks)
