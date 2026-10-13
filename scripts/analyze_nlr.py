"""Raw schema/quality audit and descriptive load plots, without imputation."""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.loaders import locate_source
from src.data.splitting import complete_days, adaptive_day_splits
from src.utils import load_config, resolve_path, write_json


def analyze(config="configs/datasets/nlr.yaml", output="results/analysis/nlr"):
    config = load_config(config)
    directory = resolve_path(output)
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(resolve_path(config["processed_path"]))
    full, day_info = complete_days(frame, config["timezone"], config["target"])
    sources = {}
    for name, spec in config["sources"].items():
        path = locate_source(spec)
        schema = pq.read_schema(path)
        times = pd.to_datetime(pq.read_table(path, columns=[spec["timestamp"]]).to_pandas()[spec["timestamp"]])
        diffs = times.sort_values().diff().dt.total_seconds()
        sources[name] = {"path": str(path), "rows": len(times), "schema": str(schema),
            "sha256": sha256(path.read_bytes()).hexdigest(), "start": str(times.min()), "end": str(times.max()),
            "duplicate_timestamps": int(times.duplicated().sum()), "ordered": bool(times.is_monotonic_increasing),
            "median_interval_seconds": float(diffs[diffs > 0].median()),
            "records_before_2015": int((times < pd.Timestamp("2015-01-01")).sum())}
        kept = times[times >= pd.Timestamp("2015-01-01")]
        coverage = pd.Series(1, index=pd.DatetimeIndex(kept)).resample("15min").sum()
        sources[name]["quarter_hour_coverage"] = {
            "intervals": len(coverage), "empty_intervals": int((coverage == 0).sum()),
            "partial_intervals_1_to_14_minutes": int(((coverage > 0) & (coverage < 15)).sum()),
            "full_15_minute_intervals": int((coverage == 15).sum())}
    target = full[config["target"]]
    daily = target.resample("D").agg(["mean", "min", "max", "std", "count"])
    complete = daily["count"] == 96
    daily["complete"] = complete
    daily.to_csv(directory/"daily_quality.csv")
    groups = (complete != complete.shift()).cumsum()
    runs = daily.groupby(groups).agg(start=("count", lambda s: str(s.index[0])),
        end=("count", lambda s: str(s.index[-1])), days=("count", "size"), complete=("complete", "first"))
    runs.to_csv(directory/"continuous_runs.csv", index=False)
    corr = {str(lag): float(target.corr(target.shift(lag))) for lag in (1, 4, 96, 288, 672, 1344)}
    splits = adaptive_day_splits(len(full)//96)
    report = {"sources": sources, "complete_days": day_info, "target": config["target"],
        "target_unit": config["target_unit"], "target_kind": "derived sum of five measured component interval means",
        "timezone_assumed": config["timezone"], "timezone_verified": config["timezone_verified"],
        "missing_points": int(target.isna().sum()), "missing_fraction": float(target.isna().mean()),
        "target_complete_days": int(complete.sum()), "target_incomplete_days": int((~complete).sum()),
        "min_kw": float(target.min()), "median_kw": float(target.median()), "max_kw": float(target.max()),
        "zero_or_near_zero_points_at_1e-8": int((target.abs() <= 1e-8).sum()),
        "autocorrelation_pairwise_observed": corr,
        "split_boundaries": {name: {"start": str(full.index[start]), "end": str(full.index[stop-1]),
            "days": (stop-start)//96} for name, (start, stop) in splits.items()},
        "processing": "Physical bounds before aggregation; duplicate timestamps averaged; 15min means; "
                      "missing components invalidate target; target never filled; partial edge days trimmed; interior gaps retained."}
    report["aggregation_coverage_policy"] = ("Interval means use available finite minute observations; "
        "an interval with all five component means is scored. A target-complete day means 96 valid interval aggregates, "
        "not necessarily 1440 individually complete raw minutes. Partial raw-interval counts are disclosed separately.")
    write_json(directory/"audit.json", report)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    daily["mean"].plot(ax=axes[0], lw=.7, color="#007c91")
    axes[0].set(ylabel="Daily mean load (kW)", title="NLR: observed history, including operating-level changes")
    for name in ("validation", "test"):
        axes[0].axvline(full.index[splits[name][0]], color="#da6a00", ls="--", alpha=.8)
    (96-daily["count"]).plot(ax=axes[1], color="#b84732", lw=.6)
    axes[1].set(ylabel="Missing intervals / day", title="Missing labels are retained, never filled")
    profile = target.groupby(target.index.hour*4+target.index.minute//15).agg(["mean", "median"])
    profile.index = profile.index/4
    profile.plot(ax=axes[2])
    axes[2].set(xlabel="Hour in assumed UTC", ylabel="Load (kW)", title="Observed mean / median intraday profile")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(directory/"load_characteristics.png", dpi=160)
    plt.close(fig)
    print(f"Data audit: {directory}")
    print(f"Days={len(full)//96}, target-complete days={complete.sum()}, missing={target.isna().sum()}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets/nlr.yaml")
    parser.add_argument("--output", default="results/analysis/nlr")
    args = parser.parse_args()
    analyze(args.config, args.output)
