"""Stage-2 figures, generated exclusively from saved real experiment predictions.

Task book §7.3 requires seven figure families.  Nothing here plots synthetic or
hand-made curves: every line comes from a ``predictions.parquet`` table written by
a completed run, and titles state the real label type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REAL_BANNER = "REAL experiment — NLR ESIF facility_derived_kw (kW)"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, where: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{where} is missing columns {missing}.")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _subset(
    predictions: pd.DataFrame,
    *,
    split: str,
    horizon: str,
    model: str | None = None,
) -> pd.DataFrame:
    rows = predictions.loc[(predictions["split"] == split) & (predictions["horizon"] == horizon)]
    if model is not None:
        rows = rows.loc[rows["model"] == model]
    return rows.sort_values(["origin_time", "lead_step"])


def plot_history_and_forecast(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    path: Path,
    *,
    split: str = "test",
    horizon: str = "24h",
    model: str = "power_ensemble",
    context_points: int = 96,
    max_origins: int = 3,
) -> Path:
    """Figure 1: observed history + future truth + ensemble forecast."""
    _require_columns(frame, ["power_kw"], where="target frame")
    rows = _subset(predictions, split=split, horizon=horizon, model=model)
    origins = list(rows["origin_time"].drop_duplicates())[:max_origins]
    if not origins:
        raise ValueError(f"no rows for split={split!r} horizon={horizon!r} model={model!r}.")

    fig, axes = plt.subplots(len(origins), 1, figsize=(12, 3.0 * len(origins)), squeeze=False)
    index = frame.index
    for axis, origin in zip(axes[:, 0], origins):
        origin_ts = pd.Timestamp(origin)
        window = rows.loc[rows["origin_time"] == origin]
        history = frame.loc[index < origin_ts].tail(context_points)
        axis.plot(history.index, history["power_kw"], color="tab:gray", lw=1.0, label="history (observed)")
        axis.plot(window["forecast_time"], window["actual_kw"], color="tab:blue", lw=1.6, label="future truth")
        axis.plot(window["forecast_time"], window["prediction_kw"], color="tab:orange", ls="--", lw=1.6,
                  label=f"forecast ({model})")
        axis.axvline(origin_ts, color="black", ls=":", lw=0.8)
        axis.set_title(f"origin D{int(window['origin_day'].iloc[0])} = {origin_ts}  ({horizon}, {split})")
        axis.set_ylabel("kW")
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=8)
    fig.suptitle(f"Forecast curves — {REAL_BANNER}", fontsize=10)
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_model_vs_actual(
    predictions: pd.DataFrame,
    path: Path,
    *,
    model: str,
    split: str = "test",
    horizon: str = "24h",
    max_origins: int = 3,
) -> Path:
    """Figures 2/3: one foundation model against the real future curve."""
    rows = _subset(predictions, split=split, horizon=horizon, model=model)
    origins = list(rows["origin_time"].drop_duplicates())[:max_origins]
    if not origins:
        raise ValueError(f"no rows for model={model!r}.")
    fig, axes = plt.subplots(1, len(origins), figsize=(5.0 * len(origins), 3.4), squeeze=False)
    for axis, origin in zip(axes[0], origins):
        window = rows.loc[rows["origin_time"] == origin]
        axis.plot(window["forecast_time"], window["actual_kw"], color="tab:blue", lw=1.6, label="actual")
        axis.plot(window["forecast_time"], window["prediction_kw"], color="tab:red", ls="--", lw=1.6,
                  label=model)
        axis.set_title(f"D{int(window['origin_day'].iloc[0])} origin", fontsize=9)
        axis.set_ylabel("kW")
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=8)
        axis.tick_params(axis="x", labelrotation=25)
    fig.suptitle(f"{model} vs real power — {REAL_BANNER}", fontsize=10)
    return _save(fig, path)


def plot_ensemble_comparison(
    predictions: pd.DataFrame,
    path: Path,
    *,
    split: str = "test",
    horizon: str = "24h",
    origin_index: int = 0,
) -> Path:
    """Figure 4: ensemble against both members and the truth for one origin."""
    rows = _subset(predictions, split=split, horizon=horizon)
    origins = list(rows["origin_time"].drop_duplicates())
    if not origins:
        raise ValueError("no prediction rows for ensemble comparison.")
    origin = origins[min(origin_index, len(origins) - 1)]
    window = rows.loc[rows["origin_time"] == origin]
    fig, axis = plt.subplots(figsize=(12, 4.2))
    truth = window.loc[window["model"] == "power_ensemble"]
    axis.plot(truth["forecast_time"], truth["actual_kw"], color="black", lw=2.0, label="actual")
    colours = {"chronos2": "tab:red", "tirex2": "tab:green", "power_ensemble": "tab:orange"}
    styles = {"chronos2": "--", "tirex2": "-.", "power_ensemble": "-"}
    for model in ("chronos2", "tirex2", "power_ensemble"):
        member = window.loc[window["model"] == model]
        if member.empty:
            continue
        axis.plot(member["forecast_time"], member["prediction_kw"], color=colours.get(model),
                  ls=styles.get(model), lw=1.7, label=model)
    axis.set_title(f"Ensemble (0.5/0.5) vs members — D{int(truth['origin_day'].iloc[0])} origin, {horizon}")
    axis.set_ylabel("kW")
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    axis.tick_params(axis="x", labelrotation=25)
    fig.suptitle(REAL_BANNER, fontsize=9)
    return _save(fig, path)


def plot_metrics_by_horizon(aggregate: pd.DataFrame, path: Path, *, split: str = "test") -> Path:
    """Figure 5: MAE and WAPE comparison across horizons/models."""
    _require_columns(aggregate, ["horizon", "model", "MAE", "WAPE"], where="aggregate")
    rows = aggregate.loc[aggregate["split"] == split]
    if rows.empty:
        raise ValueError(f"aggregate has no rows for split={split!r}.")
    horizons = [item for item in ("4h", "8h", "24h", "3d", "7d") if item in set(rows["horizon"])]
    models = list(dict.fromkeys(rows["model"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    width = 0.8 / max(len(models), 1)
    positions = np.arange(len(horizons))
    for offset, model in enumerate(models):
        values_mae, values_wape = [], []
        for horizon in horizons:
            match = rows.loc[(rows["horizon"] == horizon) & (rows["model"] == model)]
            values_mae.append(float(match["MAE"].iloc[0]) if not match.empty else np.nan)
            values_wape.append(float(match["WAPE"].iloc[0]) if not match.empty else np.nan)
        axes[0].bar(positions + offset * width, values_mae, width=width, label=model)
        axes[1].bar(positions + offset * width, values_wape, width=width, label=model)
    axes[0].set_ylabel("MAE (kW)")
    axes[1].set_ylabel("WAPE (%)")
    for axis, title in zip(axes, ("MAE by horizon", "WAPE by horizon")):
        axis.set_xticks(positions + (len(models) - 1) * width / 2)
        axis.set_xticklabels(horizons)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=7)
    fig.suptitle(f"Metric comparison — {split} split — {REAL_BANNER}", fontsize=9)
    return _save(fig, path)


def plot_forecast_day_errors(daily: pd.DataFrame, path: Path) -> Path:
    """Figure 6: per-forecast-day errors for 3d (and 7d when available)."""
    _require_columns(daily, ["horizon", "model", "forecast_day", "MAE"], where="daily metrics")
    subset = daily.loc[(daily["horizon"].isin(["3d", "7d"])) & (daily["evaluation"] == "forecast_day_curve")]
    if subset.empty:
        raise ValueError("no per-forecast-day rows (3d/7d) available for plotting.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    for axis, metric in zip(axes, ("MAE", "WAPE")):
        for (horizon, model), group in subset.groupby(["horizon", "model"]):
            group = group.sort_values("forecast_day")
            axis.plot(group["forecast_day"], group[metric], marker="o", label=f"{model} {horizon}")
        axis.set_title(f"{metric} by forecast day")
        axis.set_xlabel("forecast day inside horizon")
        axis.set_ylabel("MAE (kW)" if metric == "MAE" else "WAPE (%)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    fig.suptitle(f"Multi-day error growth — {REAL_BANNER}", fontsize=9)
    return _save(fig, path)


def plot_error_by_lead_and_origin(
    predictions: pd.DataFrame,
    path: Path,
    *,
    model: str = "power_ensemble",
    split: str = "test",
    horizon: str = "24h",
) -> Path:
    """Figure 7: absolute error vs lead step and per-origin error spread."""
    rows = _subset(predictions, split=split, horizon=horizon, model=model)
    if rows.empty:
        raise ValueError(f"no rows for model={model!r} horizon={horizon!r}.")
    rows = rows.loc[rows["valid_label"].astype(bool)].copy()
    rows["abs_error"] = (rows["actual_kw"] - rows["prediction_kw"]).abs()
    by_lead = rows.groupby("lead_step")["abs_error"].mean()
    by_origin = rows.groupby("origin_day")["abs_error"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].plot(by_lead.index, by_lead.to_numpy(), marker=".", lw=1.2)
    axes[0].set_title(f"Mean absolute error vs lead step ({model}, {horizon}, {split})")
    axes[0].set_xlabel("lead step (15 min)")
    axes[0].set_ylabel("MAE (kW)")
    axes[0].grid(alpha=0.3)
    axes[1].bar(by_origin.index.astype(str), by_origin.to_numpy(), color="tab:purple")
    axes[1].set_title(f"Mean absolute error per origin ({model})")
    axes[1].set_xlabel("trace day Dk")
    axes[1].set_ylabel("MAE (kW)")
    axes[1].tick_params(axis="x", labelrotation=45)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(f"Error diagnostics — {REAL_BANNER}", fontsize=9)
    return _save(fig, path)


def generate_all_figures(
    *,
    predictions: pd.DataFrame,
    aggregate: pd.DataFrame,
    daily: pd.DataFrame,
    frame: pd.DataFrame,
    figures_dir: Path,
    context_points: int = 96,
    primary_horizon: str = "24h",
) -> list[Path]:
    """Generate the task-book §7.3 figure set, skipping only what the run lacks."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    split = "test" if (predictions["split"] == "test").any() else str(predictions["split"].iloc[0])
    available_models = set(str(model) for model in predictions["model"].unique())
    primary_model = "power_ensemble" if "power_ensemble" in available_models else sorted(available_models)[0]

    produced.append(
        plot_history_and_forecast(
            frame,
            predictions,
            figures_dir / "fig1_history_and_forecast.png",
            split=split,
            horizon=primary_horizon,
            model=primary_model,
            context_points=context_points,
        )
    )
    for model, filename in (("chronos2", "fig2_chronos2_vs_actual.png"), ("tirex2", "fig3_tirex2_vs_actual.png")):
        if model in available_models:
            produced.append(
                plot_model_vs_actual(
                    predictions,
                    figures_dir / filename,
                    model=model,
                    split=split,
                    horizon=primary_horizon,
                )
            )
    if "power_ensemble" in available_models:
        produced.append(
            plot_ensemble_comparison(
                predictions,
                figures_dir / "fig4_ensemble_vs_members.png",
                split=split,
                horizon=primary_horizon,
            )
        )
    produced.append(plot_metrics_by_horizon(aggregate, figures_dir / "fig5_metrics_by_horizon.png", split=split))
    multiday = (
        not daily.empty
        and daily["horizon"].isin(["3d", "7d"]).any()
        and daily["evaluation"].eq("forecast_day_curve").any()
    )
    if multiday:
        produced.append(plot_forecast_day_errors(daily, figures_dir / "fig6_multiday_per_day_error.png"))
    produced.append(
        plot_error_by_lead_and_origin(
            predictions,
            figures_dir / "fig7_error_by_lead_and_origin.png",
            model=primary_model,
            split=split,
            horizon=primary_horizon,
        )
    )
    return produced
