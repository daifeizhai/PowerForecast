"""Shared prediction/metric artifacts for single-domain and cross-domain runs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.backtest import prediction_table
from src.utils import write_json


def save_result(folder, task, name, split, batch, result, rows, seconds, unit, plots,
                *, row_metadata=None, quiet=False):
    model_dir = folder/task/name
    model_dir.mkdir(exist_ok=True, parents=True)
    extra = row_metadata or {}
    for evaluation, metrics in (("curve", result.metrics), ("daily_mean", result.daily_metrics)):
        rows.append(dict(horizon=task, model=name, split=split, evaluation=evaluation,
                         train_seconds=seconds, inference_seconds=result.inference_seconds, **extra, **metrics))
    for label, table in (("per_lead", result.per_lead), ("per_origin", result.per_origin),
                         ("daily_per_origin", result.daily_per_origin), ("per_day", result.per_day)):
        table = table.copy()
        if batch.inputs.domain_ids is not None and label in {"per_origin", "daily_per_origin"}:
            table["domain_id"] = batch.inputs.domain_ids
        table.to_csv(model_dir/f"{split}_{label}.csv", index=False)
    table = prediction_table(batch, result.predictions)
    if batch.inputs.domain_ids is not None:
        table["domain_id"] = np.repeat(batch.inputs.domain_ids, batch.targets.shape[1])
    if extra.get("source_clock_start"):
        start = pd.Timestamp(extra["source_clock_start"])
        anchor = pd.Timestamp(extra["protocol_anchor"])
        table["source_clock_timestamp"] = [t.isoformat() for t in start+(pd.DatetimeIndex(table.timestamp)-anchor)]
        table["source_clock_origin"] = [t.isoformat() for t in start+(pd.DatetimeIndex(table.origin)-anchor)]
        table["source_clock_timezone_verified"] = extra["source_clock_timezone_verified"]
    table.to_parquet(model_dir/f"{split}_predictions.parquet", index=False)
    write_json(model_dir/f"{split}_metrics.json", dict(curve=result.metrics, daily_mean=result.daily_metrics,
        nonoverlap=result.nonoverlap_metrics, inference_seconds=result.inference_seconds))
    if split == "test":
        if plots:
            from src.evaluation.visualization import plot_forecast
            plot_forecast(table, model_dir/"forecast.png", f"{name}, {task}", unit)
        if not quiet:
            fmt = lambda v: "undefined" if v is None else f"{v:.4f}"
            print(f"{task} {name}: MAPE={fmt(result.metrics['MAPE'])}% RMSE={fmt(result.metrics['RMSE'])} {unit} "
                  f"WAPE={fmt(result.metrics['WAPE'])}% n={result.metrics['n']}", flush=True)


def add_baseline_reductions(metrics, baselines):
    """Preserve the ordinary runner's task/evaluation-specific reduction definition."""
    metrics = metrics.copy()
    for task, baseline in baselines.items():
        for evaluation in ("curve", "daily_mean"):
            ref = metrics[(metrics.horizon == task) & (metrics.model == baseline) &
                          (metrics.split == "test") & (metrics.evaluation == evaluation)]
            for metric in ("MAPE", "WAPE"):
                mask = (metrics.horizon == task) & (metrics.split == "test") & (metrics.evaluation == evaluation)
                if len(ref) == 1 and ref[metric].iloc[0] > 0:
                    metrics.loc[mask, f"{metric}_reduction_pct"] = 100*(1-metrics.loc[mask, metric]/ref[metric].iloc[0])
    return metrics
