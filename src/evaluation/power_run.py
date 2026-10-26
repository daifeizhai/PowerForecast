"""Stage-2 run artefacts: the task-book §10 results directory for the power main model.

Deliberately separate from :mod:`src.evaluation.artifacts` (the earlier generic
skeleton run layout): this module implements the new task book's accepted layout
(``manifest.json``, ``metrics.csv``, ``origin_metrics.csv``,
``daily_metrics.csv``, ``predictions.parquet``, ``source_quality.json``,
``target_quality.json``, ``logs/``, ``figures/``).

Every run directory is new and never overwritten.  Predictions are stored so all
reported metrics can be recomputed independently.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.contracts import ContractError, ForecastBatch
from src.evaluation.metrics import forecast_metrics
from src.utils import (
    config_digest,
    jsonable,
    resolve_path,
    sha256_text,
    utc_timestamp,
    write_json,
)

#: Dataset of the Stage-2 main experiment (real NLR ESIF derived facility power).
POWER_DATASET_ID = "nlr_esif"

#: Exact required columns of ``predictions.parquet`` (task book §10) plus the
#: additive columns needed to recompute every layered metric without ambiguity.
PREDICTION_COLUMNS: tuple[str, ...] = (
    "dataset",
    "site_id",
    "target_kind",
    "model",
    "split",
    "horizon",
    "origin_time",
    "origin_day",
    "forecast_time",
    "lead_step",
    "actual_kw",
    "prediction_kw",
    "valid_label",
    "prediction_raw_kw",
    "clipped",
    "synthetic",
    "run_id",
)

METRIC_KEYS: tuple[str, ...] = (
    "MAPE",
    "WAPE",
    "MAE",
    "RMSE",
    "n",
    "mape_threshold",
    "mape_n",
    "mape_excluded",
    "wape_defined",
    "mape_defined",
)


class PowerRunError(RuntimeError):
    """A Stage-2 run cannot be created or completed safely."""


# ---------------------------------------------------------------------------
# run directory
# ---------------------------------------------------------------------------


@dataclass
class PowerRunDirectory:
    """A new, never-overwritten Stage-2 result directory."""

    path: Path
    run_id: str
    figures_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.figures_dir = self.path / "figures"
        self.logs_dir = self.path / "logs"

    @classmethod
    def create(cls, root: str | Path = "results/runs", *, run_id: str | None = None) -> "PowerRunDirectory":
        from src.utils import make_run_id

        root_path = resolve_path(root)
        new_id = make_run_id() if run_id is None else str(run_id)
        if not new_id or Path(new_id).name != new_id or new_id in {".", ".."}:
            raise PowerRunError(f"run_id must be a single path component, got {run_id!r}.")
        destination = root_path / new_id
        if destination.exists():
            raise PowerRunError(
                f"Run directory {destination} already exists; results evidence is never overwritten."
            )
        destination.mkdir(parents=True, exist_ok=False)
        run = cls(path=destination, run_id=new_id)
        run.figures_dir.mkdir()
        run.logs_dir.mkdir()
        return run

    # -- writers ---------------------------------------------------------
    def write_json(self, name: str, payload: Any) -> Path:
        return write_json(self.path / name, payload)

    def write_predictions(self, frame: pd.DataFrame) -> Path:
        if tuple(frame.columns) != PREDICTION_COLUMNS:
            raise PowerRunError(
                f"predictions frame must have exactly {PREDICTION_COLUMNS}; got {tuple(frame.columns)}."
            )
        destination = self.path / "predictions.parquet"
        frame.to_parquet(destination, index=False)
        return destination

    def write_table(self, name: str, frame: pd.DataFrame) -> Path:
        destination = self.path / name
        if destination.suffix != ".csv":
            destination = destination.with_suffix(".csv")
        frame.to_csv(destination, index=False)
        return destination

    def write_log(self, lines: Iterable[str], *, name: str = "run.log") -> Path:
        destination = self.logs_dir / name
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination

    def copy_quality(self, target_quality: str | Path | None, source_quality: str | Path | None) -> dict[str, str | None]:
        copied: dict[str, str | None] = {"target_quality.json": None, "source_quality.json": None}
        for destination_name, source in (
            ("target_quality.json", target_quality),
            ("source_quality.json", source_quality),
        ):
            if source is None:
                write_json(
                    self.path / destination_name,
                    {"available": False, "reason": "no quality artefact was configured for this run"},
                )
                copied[destination_name] = None
                continue
            resolved = resolve_path(source)
            if not resolved.is_file():
                write_json(
                    self.path / destination_name,
                    {"available": False, "reason": f"quality artefact not found: {resolved}"},
                )
                copied[destination_name] = None
                continue
            shutil.copyfile(resolved, self.path / destination_name)
            copied[destination_name] = str(resolved)
        return copied

    def finalize(
        self,
        *,
        manifest: Mapping[str, Any],
        status: str,
        error: str | None = None,
    ) -> Path:
        if status not in {"completed", "failed"}:
            raise PowerRunError(f"status must be 'completed' or 'failed', got {status!r}.")
        payload = dict(manifest)
        payload["run_id"] = self.run_id
        payload["status"] = status
        payload["finalized_at_utc"] = utc_timestamp()
        if error is not None:
            payload["error"] = error
        return write_json(self.path / "manifest.json", payload)


# ---------------------------------------------------------------------------
# prediction / metric tables
# ---------------------------------------------------------------------------


def power_predictions_frame(
    batch: ForecastBatch,
    predictions: np.ndarray,
    *,
    run_id: str,
    dataset_id: str,
    site_id: str,
    label_type: str,
    model: str,
    split: str,
    horizon: str,
    anchor: pd.Timestamp,
    prediction_raw: np.ndarray | None = None,
    clip_negative: bool = False,
) -> pd.DataFrame:
    """Long ``[B*H, ...]`` prediction table for one model/horizon/split."""
    if not isinstance(batch, ForecastBatch):
        raise PowerRunError("power_predictions_frame expects a ForecastBatch.")
    matrix = np.asarray(predictions, dtype=float)
    expected = batch.targets_kw.shape
    if matrix.shape != expected:
        raise PowerRunError(f"predictions must have shape {expected}, got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise PowerRunError("predictions contain non-finite values; refusing to save evidence.")
    raw = matrix if prediction_raw is None else np.asarray(prediction_raw, dtype=float)
    if raw.shape != expected:
        raise PowerRunError(f"prediction_raw must have shape {expected}, got {raw.shape}.")

    inputs = batch.inputs
    batch_size, horizon_points = matrix.shape
    origins = np.asarray(inputs.origin_timestamps).reshape(-1)
    target_times = np.asarray(inputs.target_timestamps).reshape(-1)
    anchor_ns = pd.Timestamp(anchor).value
    day_numbers = ((origins.astype("datetime64[ns]").astype("int64") - anchor_ns) // (96 * 15 * 60 * 1_000_000_000)) + 1
    repeated_origins = np.repeat(origins, horizon_points)
    repeated_days = np.repeat(day_numbers, horizon_points)
    domains = np.repeat(np.asarray(inputs.domain_ids, dtype=object), horizon_points)

    frame = pd.DataFrame(
        {
            "dataset": pd.array([dataset_id] * int(batch_size * horizon_points), dtype="string"),
            "site_id": pd.array([site_id] * int(batch_size * horizon_points), dtype="string"),
            "target_kind": pd.array([label_type] * int(batch_size * horizon_points), dtype="string"),
            "model": pd.array([str(model)] * int(batch_size * horizon_points), dtype="string"),
            "split": pd.array([split] * int(batch_size * horizon_points), dtype="string"),
            "horizon": pd.array([horizon] * int(batch_size * horizon_points), dtype="string"),
            "origin_time": pd.to_datetime(repeated_origins).astype("datetime64[ns]"),
            "origin_day": repeated_days.astype("int64"),
            "forecast_time": pd.to_datetime(target_times).astype("datetime64[ns]"),
            "lead_step": np.tile(np.arange(1, horizon_points + 1, dtype="int64"), batch_size),
            "actual_kw": np.asarray(batch.targets_kw, dtype="float64").reshape(-1),
            "prediction_kw": matrix.reshape(-1).astype("float64"),
            "valid_label": np.asarray(batch.target_valid_mask, dtype=bool).reshape(-1),
            "prediction_raw_kw": raw.reshape(-1).astype("float64"),
            "clipped": np.full(int(batch_size * horizon_points), bool(clip_negative), dtype=bool),
            "synthetic": np.full(int(batch_size * horizon_points), bool(inputs.target_spec.synthetic), dtype=bool),
            "run_id": pd.array([run_id] * int(batch_size * horizon_points), dtype="string"),
        }
    )
    # domain identity is retained in metadata; keep the required schema exact.
    del domains
    return frame.loc[:, list(PREDICTION_COLUMNS)]


def _metrics_row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS}


def aggregate_rows(
    *,
    report_rows: Sequence[Mapping[str, Any]],
    horizon_points: int,
    split: str,
    horizon: str,
) -> dict[str, Any]:
    """Aggregate per-origin metrics into one row (mean/std/min/max over origins)."""
    if not report_rows:
        raise PowerRunError("cannot aggregate zero origins.")
    aggregate: dict[str, Any] = {"split": split, "horizon": horizon, "n_origins": len(report_rows)}
    # Pooled metrics are recomputed by the caller from the saved predictions; here
    # we provide the distribution across origins.
    for key in ("MAPE", "WAPE", "MAE", "RMSE"):
        values = [float(row[key]) for row in report_rows if row.get(key) is not None]
        if values:
            aggregate[f"{key}_origin_mean"] = float(np.mean(values))
            aggregate[f"{key}_origin_std"] = float(np.std(values, ddof=0))
            aggregate[f"{key}_origin_min"] = float(np.min(values))
            aggregate[f"{key}_origin_max"] = float(np.max(values))
        else:
            aggregate[f"{key}_origin_mean"] = None
            aggregate[f"{key}_origin_std"] = None
            aggregate[f"{key}_origin_min"] = None
            aggregate[f"{key}_origin_max"] = None
    aggregate["horizon_points"] = int(horizon_points)
    return aggregate


def recompute_aggregate(
    predictions: pd.DataFrame,
    *,
    split: str,
    horizon: str,
    model: str,
    mape_relative_threshold: float,
) -> dict[str, Any]:
    """Pooled metrics recomputed straight from saved prediction rows."""
    rows = predictions.loc[
        (predictions["split"] == split)
        & (predictions["horizon"] == horizon)
        & (predictions["model"] == model)
    ]
    valid = rows.loc[rows["valid_label"].astype(bool)]
    if valid.empty:
        raise PowerRunError(
            f"no valid prediction rows for split={split!r} horizon={horizon!r} model={model!r}."
        )
    metrics = forecast_metrics(
        valid["actual_kw"].to_numpy(dtype=float),
        valid["prediction_kw"].to_numpy(dtype=float),
        mape_relative_threshold=mape_relative_threshold,
    )
    pooled = {"split": split, "horizon": horizon, "model": model, "n_origins": int(rows["origin_time"].nunique())}
    pooled.update(_metrics_row(metrics))
    return pooled


def improvement_rows(
    aggregate: pd.DataFrame,
    *,
    baselines: Sequence[str] = ("seasonal_naive", "three_day_mean"),
) -> pd.DataFrame:
    """Relative metric improvement of every model vs each baseline (percent)."""
    if not {"split", "horizon", "model"}.issubset(aggregate.columns):
        raise PowerRunError("aggregate table must contain split/horizon/model columns.")
    rows: list[dict[str, Any]] = []
    for (split, horizon), group in aggregate.groupby(["split", "horizon"], sort=True):
        index = group.set_index("model")
        for baseline in baselines:
            if baseline not in index.index:
                continue
            for metric in ("MAPE", "WAPE", "MAE", "RMSE"):
                base_value = index.loc[baseline, metric]
                if base_value is None or not np.isfinite(float(base_value)) or float(base_value) == 0.0:
                    continue
                for model, value in index[metric].items():
                    if model == baseline or value is None or not np.isfinite(float(value)):
                        continue
                    rows.append(
                        {
                            "split": split,
                            "horizon": horizon,
                            "model": model,
                            "baseline": baseline,
                            "metric": metric,
                            "model_value": float(value),
                            "baseline_value": float(base_value),
                            "improvement_pct": float((float(base_value) - float(value)) / float(base_value) * 100.0),
                        }
                    )
    return pd.DataFrame(rows)


def daily_rows(predictions: pd.DataFrame, *, mape_relative_threshold: float = 0.0) -> pd.DataFrame:
    """Daily-level metrics required by task book §7.2.

    Three evaluation kinds are produced per (split, horizon, model):

    * ``daily_mean_power`` — error of each origin-day's mean power, pooled over
      origins/days (the "complete day average power" error; 24h and multi-day).
    * ``forecast_day_curve`` — pooled 15-minute curve error restricted to the
      96-point block of forecast day 1, 2, … (3d/7d "第 N 天预测误差").
    * ``forecast_day_mean_power`` — per-forecast-day mean-power error (3d/7d
      "3 个日均误差分别统计").
    """
    rows: list[dict[str, Any]] = []
    for (split, horizon, model), group in predictions.groupby(["split", "horizon", "model"], sort=True):
        group = group.loc[group["valid_label"].astype(bool)].copy()
        if group.empty:
            continue
        group["forecast_day"] = ((group["lead_step"] - 1) // 96 + 1).astype(int)

        def pooled_curve(frame: pd.DataFrame) -> dict[str, Any]:
            metrics = forecast_metrics(
                frame["actual_kw"].to_numpy(dtype=float),
                frame["prediction_kw"].to_numpy(dtype=float),
                mape_relative_threshold=mape_relative_threshold,
            )
            return _metrics_row(metrics)

        def pooled_means(frame: pd.DataFrame) -> dict[str, Any]:
            means = (
                frame.groupby(["origin_time", "forecast_day"], observed=True)
                .agg(actual=("actual_kw", "mean"), predicted=("prediction_kw", "mean"))
                .reset_index()
            )
            metrics = forecast_metrics(
                means["actual"].to_numpy(dtype=float),
                means["predicted"].to_numpy(dtype=float),
                mape_relative_threshold=mape_relative_threshold,
            )
            return _metrics_row(metrics)

        rows.append(
            {
                "split": split,
                "horizon": horizon,
                "model": model,
                "forecast_day": None,
                "evaluation": "daily_mean_power",
                **pooled_means(group),
            }
        )
        multi_day = group["forecast_day"].nunique() > 1
        for day in sorted(group["forecast_day"].unique()):
            block = group.loc[group["forecast_day"] == day]
            rows.append(
                {
                    "split": split,
                    "horizon": horizon,
                    "model": model,
                    "forecast_day": int(day),
                    "evaluation": "forecast_day_curve",
                    **pooled_curve(block),
                }
            )
            if multi_day:
                rows.append(
                    {
                        "split": split,
                        "horizon": horizon,
                        "model": model,
                        "forecast_day": int(day),
                        "evaluation": "forecast_day_mean_power",
                        **pooled_means(block),
                    }
                )
    return pd.DataFrame(rows)


def verify_saved_predictions(
    predictions: pd.DataFrame,
    aggregate: pd.DataFrame,
    *,
    mape_relative_threshold: float,
) -> dict[str, Any]:
    """Recompute every aggregate from the saved rows and compare."""
    problems: list[str] = []
    checked = 0
    for row in aggregate.itertuples(index=False):
        saved = row._asdict()
        recomputed = recompute_aggregate(
            predictions,
            split=saved["split"],
            horizon=saved["horizon"],
            model=saved["model"],
            mape_relative_threshold=mape_relative_threshold,
        )
        checked += 1
        for key in ("MAPE", "WAPE", "MAE", "RMSE"):
            a, b = saved.get(key), recomputed.get(key)
            if a is None and b is None:
                continue
            if a is None or b is None or abs(float(a) - float(b)) > 1e-9:
                problems.append(
                    f"{saved['split']}/{saved['horizon']}/{saved['model']}: {key} saved={a} "
                    f"recomputed={b}"
                )
    return {"checked_rows": checked, "matches": not problems, "problems": problems}


# ---------------------------------------------------------------------------
# manifest helper
# ---------------------------------------------------------------------------


def manifest_payload(
    *,
    config: Mapping[str, Any],
    dataset_id: str,
    site_id: str,
    label_type: str,
    anchor: pd.Timestamp,
    target_source: Mapping[str, Any],
    protocol: Mapping[str, Any],
    model_descriptions: Mapping[str, Any],
    ensemble: Mapping[str, Any] | None,
    environment: Mapping[str, Any],
    origins: Mapping[str, Any],
    skipped: Sequence[Mapping[str, Any]],
    timings: Mapping[str, Any],
    figures: Sequence[str],
    quality_files: Mapping[str, Any],
    notes: Sequence[str] = (),
    command: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "power_main_v1",
        "task": "Stage 2: historical-power main model A (Chronos-2 + TiRex-2 equal-weight ensemble)",
        "status": "running",
        "created_at_utc": utc_timestamp(),
        "dataset": {
            "dataset_id": dataset_id,
            "site_id": site_id,
            "label_type": label_type,
            "anchor": str(anchor),
            "target_source": dict(target_source),
            "quality_files": dict(quality_files),
        },
        "label_policy": {
            "target_kind": label_type,
            "unit": "kW",
            "scope": "facility",
            "components_used_as_features": False,
            "future_operational_features_used": False,
        },
        "protocol": dict(protocol),
        "model": dict(model_descriptions),
        "ensemble": None if ensemble is None else dict(ensemble),
        "weights_frozen": True,
        "target_domain_training": False,
        "fusion_weight_search": False,
        "environment": dict(environment),
        "origins": dict(origins),
        "skipped_windows": [dict(item) for item in skipped],
        "timings": dict(timings),
        "figures": list(figures),
        "config": jsonable(config),
        "config_sha256": config_digest(config),
        "command": command,
        "notes": list(notes),
        "synthetic": False,
        "research_result": True,
    }


def write_json_safe(path: Path, payload: Any) -> Path:
    """Write JSON with numpy-safe conversion and a readable fallback for failures."""
    try:
        return write_json(path, payload)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        return write_json(path, {"serialization_error": str(exc), "repr": repr(payload)[:4000]})


def stable_sha256(payload: Any) -> str:
    return sha256_text(json.dumps(jsonable(payload), sort_keys=True, ensure_ascii=False))
