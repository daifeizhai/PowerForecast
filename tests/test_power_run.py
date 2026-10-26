"""Stage-2 orchestration tests (task book §9.3/9.4/9.6/§10) with deterministic stub models.

Real foundation weights are exercised in ``tests/test_model_adapters.py``; here the
goal is to verify the experiment layer itself: protocol execution, artefact layout,
metric recomputability, improvement arithmetic, per-day reporting, failure
handling and the fixed 0.5/0.5 policy — without a slow model load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_power_main import ExperimentError, build_parser, main, run_experiment
from src.evaluation.metrics import forecast_metrics
from src.evaluation.power_run import PREDICTION_COLUMNS
from support_stage2 import (
    MeanStub,
    NaiveStub,
    ProfileStub,
    make_experiment,
    only_run,
    run_stage2,
    stub_factory,
)
from src.utils import read_json


# ---------------------------------------------------------------------------
# happy path — 24h
# ---------------------------------------------------------------------------


def test_24h_stage2_run_writes_complete_auditable_artifacts(tmp_path: Path) -> None:
    exit_code, runs = run_stage2(
        tmp_path,
        "--horizons", "24h",
        "--splits", "test",
        "--models", "seasonal_naive", "three_day_mean", "chronos2", "tirex2", "power_ensemble",
    )
    assert exit_code == 0
    run = only_run(runs)
    for name in (
        "manifest.json",
        "metrics.csv",
        "origin_metrics.csv",
        "daily_metrics.csv",
        "improvements.csv",
        "predictions.parquet",
        "target_quality.json",
        "source_quality.json",
        "logs/run.log",
    ):
        assert (run / name).is_file(), f"missing artefact {name}"

    manifest = read_json(run / "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["synthetic"] is False
    assert manifest["research_result"] is True
    assert manifest["label_policy"]["target_kind"] == "facility_derived_kw"
    assert manifest["label_policy"]["unit"] == "kW"
    assert manifest["label_policy"]["components_used_as_features"] is False
    assert manifest["weights_frozen"] is True
    assert manifest["target_domain_training"] is False
    assert manifest["ensemble"]["weights"] == [0.5, 0.5]
    assert manifest["protocol"]["origin_days"]["test"]["24h"] == list(range(22, 31))
    assert manifest["metric_recomputation"]["matches"] is True
    assert manifest["origins"]["test"]["24h"]["n_origins"] == 9

    predictions = pd.read_parquet(run / "predictions.parquet")
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) == 9 * 96 * 5
    assert set(predictions["model"]) == {
        "seasonal_naive",
        "three_day_mean",
        "chronos2",
        "tirex2",
        "power_ensemble",
    }
    assert (predictions["target_kind"] == "facility_derived_kw").all()
    assert not predictions["synthetic"].any()
    assert predictions["valid_label"].all()
    assert np.isfinite(predictions["prediction_kw"]).all()

    # ensemble really is the exact 0.5/0.5 mean of its two members
    pivot = predictions.pivot_table(
        index=["split", "horizon", "origin_time", "lead_step"],
        columns="model",
        values="prediction_kw",
    )
    np.testing.assert_allclose(
        pivot["power_ensemble"].to_numpy(),
        0.5 * pivot["chronos2"].to_numpy() + 0.5 * pivot["tirex2"].to_numpy(),
        rtol=0,
        atol=1e-12,
    )

    # every reported metric is reproducible from the saved rows
    metrics = pd.read_csv(run / "metrics.csv")
    threshold = float(manifest["config"]["evaluation"]["mape_relative_threshold"])
    for row in metrics.itertuples(index=False):
        subset = predictions.loc[
            (predictions["split"] == row.split)
            & (predictions["horizon"] == row.horizon)
            & (predictions["model"] == row.model)
            & predictions["valid_label"]
        ]
        recomputed = forecast_metrics(
            subset["actual_kw"].to_numpy(float),
            subset["prediction_kw"].to_numpy(float),
            mape_relative_threshold=threshold,
        )
        for key in ("MAPE", "WAPE", "MAE", "RMSE"):
            assert float(getattr(row, key)) == pytest.approx(recomputed[key], rel=1e-9, abs=1e-9)

    # improvement arithmetic is consistent with the aggregate table
    improvements = pd.read_csv(run / "improvements.csv")
    ensemble_rows = improvements.loc[improvements["model"] == "power_ensemble"]
    assert not ensemble_rows.empty
    base = metrics.set_index(["split", "horizon", "model"])
    for row in ensemble_rows.itertuples(index=False):
        model_value = base.loc[(row.split, row.horizon, row.model), row.metric]
        baseline_value = base.loc[(row.split, row.horizon, row.baseline), row.metric]
        expected = (baseline_value - model_value) / baseline_value * 100.0
        assert row.improvement_pct == pytest.approx(expected, rel=1e-9)

    # figures were produced from the saved predictions
    manifest_figures = [run / name for name in manifest["figures"]]
    assert len(manifest_figures) >= 5
    assert all(path.is_file() and path.stat().st_size > 0 for path in manifest_figures)

    # a 24h run reports the complete-day mean-power error
    daily = pd.read_csv(run / "daily_metrics.csv")
    assert "daily_mean_power" in set(daily["evaluation"])


# ---------------------------------------------------------------------------
# 3d — per-day reporting and overlapping windows
# ---------------------------------------------------------------------------


def test_3d_run_reports_per_day_errors_and_overlap(tmp_path: Path) -> None:
    exit_code, runs = run_stage2(
        tmp_path, "--horizons", "3d", "--splits", "test", "--models", "seasonal_naive", "power_ensemble"
    )
    assert exit_code == 0
    run = only_run(runs)
    predictions = pd.read_parquet(run / "predictions.parquet")
    assert len(predictions) == 7 * 288 * 2
    assert int(predictions["lead_step"].max()) == 288

    daily = pd.read_csv(run / "daily_metrics.csv")
    curve = daily.loc[daily["evaluation"] == "forecast_day_curve"]
    assert set(curve["model"]) == {"seasonal_naive", "power_ensemble"}
    for model in ("seasonal_naive", "power_ensemble"):
        days = curve.loc[curve["model"] == model, "forecast_day"]
        assert sorted(days.astype(int)) == [1, 2, 3]
    mean_power = daily.loc[daily["evaluation"] == "forecast_day_mean_power"]
    assert not mean_power.empty

    manifest = read_json(run / "manifest.json")
    plan = manifest["origins"]["test"]["3d"]
    assert plan["days"] == [22, 23, 24, 25, 26, 27, 28]
    assert plan["overlap"] is True
    assert any("overlap" in note for note in manifest["notes"])


def test_development_split_is_reported_separately(tmp_path: Path) -> None:
    exit_code, runs = run_stage2(
        tmp_path, "--horizons", "24h", "--splits", "development", "--models", "seasonal_naive"
    )
    assert exit_code == 0
    run = only_run(runs)
    predictions = pd.read_parquet(run / "predictions.parquet")
    assert set(predictions["split"]) == {"development"}
    assert len(predictions) == 4 * 96
    manifest = read_json(run / "manifest.json")
    assert manifest["origins"]["development"]["24h"]["days"] == [14, 16, 18, 20]


# ---------------------------------------------------------------------------
# failure paths
# ---------------------------------------------------------------------------


def test_nan_model_output_fails_the_run(tmp_path: Path) -> None:
    def broken_factory(config, args, model_names):  # noqa: ANN001
        return {name: NaiveStub(name) for name in model_names}

    config_path, _ = make_experiment(tmp_path)
    args = build_parser().parse_args(
        ["--config", str(config_path), "--output", str(tmp_path / "runs"),
         "--horizons", "24h", "--splits", "test", "--models", "seasonal_naive"]
    )
    exit_code = run_experiment(args, forecaster_factory=broken_factory)
    assert exit_code == 1
    run = only_run(tmp_path / "runs")
    manifest = read_json(run / "manifest.json")
    assert manifest["status"] == "failed"
    assert "error" in manifest
    assert not (run / "predictions.parquet").exists()


def test_missing_target_artifact_is_reported_without_creating_a_run(tmp_path: Path) -> None:
    config_path, parquet_path = make_experiment(tmp_path)
    parquet_path.unlink()
    exit_code = main(
        ["--config", str(config_path), "--output", str(tmp_path / "runs"), "--horizons", "24h", "--splits", "test"]
    )
    assert exit_code == 1
    runs = tmp_path / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []


def test_illegal_label_type_is_refused(tmp_path: Path) -> None:
    config_path, _ = make_experiment(tmp_path, label_type="node_measured_kw")
    exit_code = main(
        ["--config", str(config_path), "--output", str(tmp_path / "runs"), "--horizons", "24h", "--splits", "test"]
    )
    assert exit_code == 1
    runs = tmp_path / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []


def test_ensemble_with_wrong_weights_is_refused(tmp_path: Path) -> None:
    class BadEnsemble(MeanStub):
        def __init__(self) -> None:
            super().__init__("power_ensemble", [ProfileStub("chronos2"), ProfileStub("tirex2")])
            self.weights = (0.7, 0.3)

        def describe(self) -> dict:
            payload = super().describe()
            payload["weights"] = [0.7, 0.3]
            return payload

    def bad_factory(config, args, model_names):  # noqa: ANN001
        return {name: (BadEnsemble() if name == "power_ensemble" else ProfileStub(name)) for name in model_names}

    config_path, _ = make_experiment(tmp_path)
    args = build_parser().parse_args(
        ["--config", str(config_path), "--output", str(tmp_path / "runs"),
         "--horizons", "24h", "--splits", "test", "--models", "power_ensemble"]
    )
    with pytest.raises(ExperimentError, match="weights"):
        run_experiment(args, forecaster_factory=bad_factory)
    runs = tmp_path / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []


def test_requires_fit_model_is_refused_by_the_zero_shot_stage(tmp_path: Path) -> None:
    class FitNeeded(ProfileStub):
        def __init__(self) -> None:
            super().__init__("chronos2")
            self.requires_fit = True

    def factory(config, args, model_names):  # noqa: ANN001
        return {name: (FitNeeded() if name == "chronos2" else ProfileStub(name)) for name in model_names}

    config_path, _ = make_experiment(tmp_path)
    args = build_parser().parse_args(
        ["--config", str(config_path), "--output", str(tmp_path / "runs"),
         "--horizons", "24h", "--splits", "test", "--models", "chronos2"]
    )
    with pytest.raises(ExperimentError, match="zero-shot"):
        run_experiment(args, forecaster_factory=factory)


def test_runs_are_never_overwritten(tmp_path: Path) -> None:
    first, runs = run_stage2(tmp_path, "--horizons", "24h", "--splits", "test", "--models", "seasonal_naive")
    second, runs = run_stage2(tmp_path, "--horizons", "24h", "--splits", "test", "--models", "seasonal_naive")
    assert first == 0 and second == 0
    assert len([path for path in runs.iterdir() if path.is_dir()]) == 2
