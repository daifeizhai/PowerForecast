"""End-to-end smoke acceptance tests (task book T02, T10, T11, T19, T20, T21, T22).

These tests drive the real CLI entry point (``scripts.run_experiment.main``) on
the synthetic dataset and verify the artefacts, the honesty flags, metric
recomputation, frozen-credential enforcement and reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.run_experiment import main
from src.evaluation.artifacts import PREDICTION_COLUMNS
from src.evaluation.metrics import forecast_metrics
from src.utils import read_json, resolve_path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"
FIXED_24H = REPO_ROOT / "configs" / "protocols" / "fixed_24h.yaml"
FIXED_7D = REPO_ROOT / "configs" / "protocols" / "fixed_7d.yaml"

REQUIRED_RUN_FILES = (
    "manifest.json",
    "data_audit.json",
    "split_audit.json",
    "frozen_before_test.json",
    "predictions.parquet",
    "metrics.json",
    "metrics.csv",
    "test_report.json",
)


def run_cli(output: Path, *extra: str) -> int:
    argv = ["--config", str(SMOKE_CONFIG), "--demo", "--output", str(output), *extra]
    return main(argv)


def newest_run(output: Path, run_ids_before: set[str]) -> Path:
    runs = sorted(path for path in output.iterdir() if path.is_dir())
    new = [path for path in runs if path.name not in run_ids_before]
    assert len(new) == 1, f"expected exactly one new run in {output}, got {[p.name for p in new]}"
    return new[0]


def run_ids(output: Path) -> set[str]:
    return {path.name for path in output.iterdir() if path.is_dir()}


def load_run(run: Path) -> dict:
    manifest = read_json(run / "manifest.json")
    metrics = read_json(run / "metrics.json")
    predictions = pd.read_parquet(run / "predictions.parquet")
    return {"manifest": manifest, "metrics": metrics, "predictions": predictions}


# ---------------------------------------------------------------------------
# T02 / CLI surface
# ---------------------------------------------------------------------------


def test_list_models_separates_implemented_from_planned(capsys) -> None:
    assert main(["--list-models"]) == 0
    captured = capsys.readouterr().out
    implemented = captured.split("planned")[0]
    assert "seasonal_naive" in implemented
    assert "three_day_mean" in implemented
    assert "lightgbm" in captured
    assert "lightgbm" not in implemented


def test_real_candidate_dataset_is_blocked_with_a_demo_hint(tmp_path: Path, capsys) -> None:
    config = tmp_path / "blocked.yaml"
    config.write_text(
        yaml.safe_dump({"inherit": str(SMOKE_CONFIG), "dataset": {"dataset_id": "nlr_esif_components"}}),
        encoding="utf-8",
    )
    exit_code = main(["--config", str(config), "--demo", "--horizon", "24h", "--output", str(tmp_path / "runs")])
    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "candidate" in stderr
    assert "configs/smoke.yaml" in stderr


def test_infeasible_origins_are_rejected_not_silently_shortened(tmp_path: Path, capsys) -> None:
    """Origins whose 7-day label window exceeds the data must be refused, not clipped."""
    output = tmp_path / "runs"
    exit_code = main(
        [
            "--config",
            str(SMOKE_CONFIG),
            "--demo",
            "--horizon",
            "7d",
            "--origins",
            "30",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 1
    assert "origin" in capsys.readouterr().err.lower()
    assert not output.exists() or list(output.iterdir()) == []


# ---------------------------------------------------------------------------
# T20 - real end-to-end synthetic runs for both baselines and both horizons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["seasonal_naive", "three_day_mean"])
@pytest.mark.parametrize("horizon,points", [("24h", 96), ("7d", 672)])
def test_smoke_run_completes_and_artifacts_are_honest(
    tmp_path: Path, model: str, horizon: str, points: int
) -> None:
    output = tmp_path / "runs"
    output.mkdir()
    assert run_cli(output, "--horizon", horizon, "--model", model) == 0
    run = newest_run(output, set())
    for name in REQUIRED_RUN_FILES:
        assert (run / name).is_file(), f"missing artefact {name}"
    assert (run / "model_state" / "state.json").is_file()

    data = load_run(run)
    manifest, metrics, predictions = data["manifest"], data["metrics"], data["predictions"]

    # Honesty flags: never a research result, always tagged synthetic.
    assert manifest["synthetic"] is True
    assert manifest["research_result"] is False
    assert manifest["status"] == "completed"
    assert manifest["phase"] == "smoke"
    # smoke freezes internally before scoring, then performs the test stage.
    assert "smoke" in manifest["phases_executed"]
    assert "final-test" in manifest["phases_executed"]
    assert manifest["demo"] is True
    assert manifest["horizon"] == horizon
    assert manifest["horizon_points"] == points
    assert manifest["label_policy"]["label_origin"] == "synthetic_simulated"
    assert manifest["label_policy"]["scope"] == "facility"
    assert manifest["label_policy"]["unit"] == "kW"
    assert manifest["frozen"]["model_name"] == model
    assert manifest["frozen"]["synthetic"] is True

    # Prediction table contract.
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert int(predictions["lead"].max()) == points
    assert predictions["synthetic"].all()
    assert predictions["valid_label"].all()
    assert (predictions["unit"] == "kW").all()
    assert np.isfinite(predictions["predicted"].to_numpy()).all()
    assert set(predictions["model"]) == {model}
    assert len(predictions) == manifest["horizon_points"] * len(manifest["origins"])

    # Metrics are reproducible from the saved predictions.
    recomputed = forecast_metrics(
        predictions.loc[predictions["valid_label"], "actual"].to_numpy(dtype=float),
        predictions.loc[predictions["valid_label"], "predicted"].to_numpy(dtype=float),
        mape_relative_threshold=metrics["labels"]["mape_relative_threshold"],
    )
    for key in ("MAPE", "WAPE", "MAE", "RMSE"):
        assert recomputed[key] == pytest.approx(metrics["aggregate"][key], rel=1e-9, abs=1e-9)
    assert recomputed["n"] == metrics["aggregate"]["n"]

    # The CLI also recorded the recomputation check it performed.
    test_report = read_json(run / "test_report.json")
    assert test_report["metric_recomputation"]["matches_metrics_json"] is True
    assert test_report["model_reloaded_from_disk"] is True
    assert test_report["hashes_verified"] is True

    # Multi-level reporting exists and 4h/8h naming rules are respected.
    assert len(metrics["per_origin"]) == len(manifest["origins"])
    assert len(metrics["per_lead"]) == points
    if points % 96 == 0:
        assert metrics["daily_mean_kind"] == "daily_mean_power"
        assert len(metrics["per_forecast_day"]) == points // 96
    else:  # pragma: no cover - horizons in this test are multiples of 96
        assert metrics["daily_mean_kind"] == "window_mean_power"


def test_seven_day_run_reports_overlapping_windows(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    output.mkdir()
    assert run_cli(output, "--horizon", "7d") == 0
    run = newest_run(output, set())
    manifest = read_json(run / "manifest.json")
    metrics = read_json(run / "metrics.json")
    split_audit = read_json(run / "split_audit.json")
    # demo_origins = 14,16,18,20,22,23,24,25..30; 7-day labels only fit up to D24
    # on a 31-trace-day frame, so exactly 7 origins are feasible and 6 are dropped.
    assert len(manifest["origins"]) == 7
    assert len(manifest["dropped_origins"]) == 6
    assert metrics["overlap"] is not None
    assert "NOT independent" in metrics["overlap"]["note"]
    assert split_audit["plan"]["statistically_independent_samples"] is False
    assert any("overlap" in warning for warning in manifest["warnings"])


# ---------------------------------------------------------------------------
# T10 / T11 - optional historical presets via the CLI
# ---------------------------------------------------------------------------


def test_fixed_24h_preset_uses_declared_origin_days(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    output.mkdir()
    assert run_cli(output, "--horizon", "24h", "--protocol", str(FIXED_24H)) == 0
    run = newest_run(output, set())
    manifest = read_json(run / "manifest.json")
    assert manifest["protocol"]["protocol_type"] == "fixed_24h"
    assert manifest["origin_days"] == list(range(22, 31))
    assert all(entry["horizon_points"] == 96 for entry in read_json(run / "split_audit.json")["plan"]["origins"])
    predictions = pd.read_parquet(run / "predictions.parquet")
    assert int(predictions["lead"].max()) == 96
    assert len(predictions) == 9 * 96


def test_fixed_7d_preset_uses_declared_origin_days(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    output.mkdir()
    assert run_cli(output, "--horizon", "7d", "--protocol", str(FIXED_7D)) == 0
    run = newest_run(output, set())
    manifest = read_json(run / "manifest.json")
    assert manifest["protocol"]["protocol_type"] == "fixed_7d"
    assert manifest["origin_days"] == [22, 23, 24]
    predictions = pd.read_parquet(run / "predictions.parquet")
    assert len(predictions) == 3 * 672
    split_audit = read_json(run / "split_audit.json")
    assert split_audit["plan"]["covered_label_days"][-1] == 31
    assert split_audit["plan"]["overlap"] is True


# ---------------------------------------------------------------------------
# T19 - frozen credential required for final-test
# ---------------------------------------------------------------------------


def test_final_test_without_frozen_run_is_refused(tmp_path: Path, capsys) -> None:
    output = tmp_path / "runs"
    output.mkdir()
    exit_code = main(
        [
            "--config",
            str(SMOKE_CONFIG),
            "--demo",
            "--phase",
            "final-test",
            "--horizon",
            "24h",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 1
    assert "frozen" in capsys.readouterr().err.lower()
    assert list(output.iterdir()) == []  # no run directory was created


def test_finalize_then_final_test_reloads_the_frozen_state(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    output.mkdir()
    assert main(
        ["--config", str(SMOKE_CONFIG), "--demo", "--phase", "finalize", "--horizon", "24h", "--output", str(output)]
    ) == 0
    frozen_run = newest_run(output, set())
    frozen_record = read_json(frozen_run / "frozen_before_test.json")
    assert frozen_record["config_sha256"]
    assert not (frozen_run / "predictions.parquet").exists()

    before = run_ids(output)
    assert main(
        [
            "--config",
            str(SMOKE_CONFIG),
            "--demo",
            "--phase",
            "final-test",
            "--horizon",
            "24h",
            "--frozen-run",
            str(frozen_run),
            "--output",
            str(output),
        ]
    ) == 0
    test_run = newest_run(output, before)
    test_manifest = read_json(test_run / "manifest.json")
    test_report = read_json(test_run / "test_report.json")
    assert test_manifest["phase"] == "final-test"
    assert test_report["model_reloaded_from_disk"] is True
    assert test_report["frozen_before_test_sha256"]
    assert test_report["frozen_run"].endswith(frozen_run.name)
    assert (test_run / "predictions.parquet").is_file()

    # The final-test run scores the same frozen model, so its aggregate must match
    # an independent smoke run of the same configuration and seed.
    assert main(
        ["--config", str(SMOKE_CONFIG), "--demo", "--horizon", "24h", "--output", str(output)]
    ) == 0
    smoke_run = newest_run(output, before | {test_run.name})
    smoke = read_json(smoke_run / "metrics.json")["aggregate"]
    tested = read_json(test_run / "metrics.json")["aggregate"]
    for key in ("MAPE", "WAPE", "MAE", "RMSE"):
        assert tested[key] == pytest.approx(smoke[key], rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# T22 - reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_and_config_reproduce_identical_predictions(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    output.mkdir()
    assert run_cli(output, "--horizon", "24h", "--model", "three_day_mean") == 0
    first = newest_run(output, set())
    before = run_ids(output)
    assert run_cli(output, "--horizon", "24h", "--model", "three_day_mean") == 0
    second = newest_run(output, before)

    left = pd.read_parquet(first / "predictions.parquet")
    right = pd.read_parquet(second / "predictions.parquet")
    for column in ("origin_timestamp", "timestamp", "lead", "actual", "predicted"):
        if column in ("actual", "predicted"):
            np.testing.assert_array_equal(left[column].to_numpy(), right[column].to_numpy())
        else:
            assert left[column].tolist() == right[column].tolist()

    left_metrics = read_json(first / "metrics.json")["aggregate"]
    right_metrics = read_json(second / "metrics.json")["aggregate"]
    for key in ("MAPE", "WAPE", "MAE", "RMSE", "n"):
        assert left_metrics[key] == pytest.approx(right_metrics[key], rel=1e-12, abs=1e-12)
    # Run identity must still differ: evidence is never overwritten.
    assert first.name != second.name
    assert read_json(first / "manifest.json")["run_id"] != read_json(second / "manifest.json")["run_id"]
