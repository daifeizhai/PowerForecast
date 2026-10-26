"""Render a human-readable summary of one completed Stage-2 run.

Reads only artefacts that a completed run already wrote (``manifest.json``,
``metrics.csv``, ``origin_metrics.csv``, ``daily_metrics.csv``,
``improvements.csv``) and prints/render Markdown tables.  It never recomputes or
changes evidence; the saved ``predictions.parquet`` remains the single source of
truth and the runner already verified it reproduces ``metrics.csv``.

Examples::

    python -m scripts.summarize_power_run --run results/runs/20261026T170722Z_d968c9
    python -m scripts.summarize_power_run --run <run_dir> --out results/analysis/stage2/summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__ in (None, ""):  # allow `python scripts/summarize_power_run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import read_json, resolve_path, write_json  # noqa: F401


def _md_table(frame: pd.DataFrame, *, float_format: str = "{:.4f}") -> str:
    if frame.empty:
        return "_(empty)_"
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else float_format.format(value)
            )
    header = "| " + " | ".join(str(column) for column in formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in formatted.itertuples(index=False)
    ]
    return "\n".join([header, separator, *body])


def summarize(run_dir: Path) -> str:
    manifest = read_json(run_dir / "manifest.json")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    daily = pd.read_csv(run_dir / "daily_metrics.csv")
    improvements = pd.read_csv(run_dir / "improvements.csv") if (run_dir / "improvements.csv").is_file() else pd.DataFrame()

    lines: list[str] = []
    lines.append(f"# Stage-2 run summary — `{run_dir.name}`")
    lines.append("")
    dataset = manifest.get("dataset", {})
    lines.append(f"- status: **{manifest.get('status')}**")
    lines.append(f"- dataset: `{dataset.get('dataset_id')}` | label type: `{dataset.get('label_type')}` | unit: kW")
    lines.append(f"- anchor: `{dataset.get('anchor')}` | protocol: `{manifest.get('protocol', {}).get('protocol_id')}`")
    lines.append(f"- ensemble: `{manifest.get('ensemble', {}).get('method')}` weights `{manifest.get('ensemble', {}).get('weights')}`")
    lines.append(f"- research_result: **{manifest.get('research_result')}** | synthetic: **{manifest.get('synthetic')}**")
    lines.append("")
    for split in ("test", "development"):
        subset = metrics.loc[metrics["split"] == split]
        if subset.empty:
            continue
        lines.append(f"## {split} split")
        lines.append("")
        for horizon in [name for name in ("4h", "8h", "24h", "3d", "7d") if name in set(subset["horizon"])]:
            table = subset.loc[subset["horizon"] == horizon, ["model", "MAPE", "WAPE", "MAE", "RMSE", "n"]]
            lines.append(f"### {horizon}")
            lines.append("")
            lines.append(_md_table(table))
            lines.append("")
        if not improvements.empty:
            chosen = improvements.loc[(improvements["split"] == split) & (improvements["model"] == "power_ensemble")]
            if not chosen.empty:
                wide = chosen.pivot_table(
                    index=["horizon", "baseline"], columns="metric", values="improvement_pct"
                ).reset_index()
                lines.append("### power_ensemble improvement vs baselines (%)")
                lines.append("")
                lines.append(_md_table(wide, float_format="{:.2f}"))
                lines.append("")
    if not daily.empty:
        lines.append("## Per-forecast-day errors (24h daily mean / 3d day1..3 / 7d day1..7)")
        lines.append("")
        view = daily.loc[
            (daily["model"] == "power_ensemble")
            & daily["evaluation"].isin(["daily_mean_power", "forecast_day_curve", "forecast_day_mean_power"])
        ]
        columns = [column for column in ("split", "horizon", "evaluation", "forecast_day", "MAPE", "WAPE", "MAE", "RMSE") if column in view.columns]
        lines.append(_md_table(view.loc[:, columns].sort_values(["split", "horizon", "evaluation", "forecast_day"], na_position="first")))
        lines.append("")
    lines.append("## Provenance")
    lines.append("")
    checkpoint_lines = []
    for name, description in (manifest.get("model") or {}).items():
        checkpoint = description.get("checkpoint") if isinstance(description, dict) else None
        if checkpoint:
            checkpoint_lines.append(
                f"- `{name}`: `{checkpoint.get('model_id')}` revision `{checkpoint.get('revision')}` "
                f"sha256 `{list(checkpoint.get('file_sha256', {}).values())[-1][:16]}…`"
            )
    lines.extend(checkpoint_lines or ["- no foundation checkpoint recorded"])
    lines.append(f"- timings total: `{manifest.get('timings', {}).get('total_seconds')}` s")
    lines.append(f"- metric recomputation verified: `{manifest.get('metric_recomputation', {}).get('matches')}`")
    lines.append(f"- skipped windows: `{manifest.get('skipped_windows')}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a Markdown summary of a completed Stage-2 run.")
    parser.add_argument("--run", required=True, help="path to results/runs/<run_id>")
    parser.add_argument("--out", default=None, help="optional Markdown output path")
    args = parser.parse_args(argv)
    run_dir = resolve_path(args.run)
    if not (run_dir / "manifest.json").is_file():
        print(f"ERROR: {run_dir} is not a completed Stage-2 run directory.", file=sys.stderr)
        return 1
    text = summarize(run_dir)
    if args.out:
        destination = resolve_path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        print(f"written: {destination}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
