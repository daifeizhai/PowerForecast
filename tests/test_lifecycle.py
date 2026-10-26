"""Tests for the freeze/reload lifecycle and immutable run artifacts.

Covers T19 (and the artifact parts of §9):

* an unfrozen run cannot be used as ``final-test`` input (load/reload refused);
* ``smoke``/``finalize`` freeze writes ``model_state/state.json`` +
  ``frozen_before_test.json``, hashes verify, the reloaded model reproduces the
  pre-freeze predictions exactly;
* a tampered model-state byte raises :class:`FrozenRunError`;
* freezing twice into one directory raises :class:`FrozenRunError`;
* ``RunDirectory`` never overwrites, validates the exact prediction columns,
  writes JSON+CSV metrics, and records failure without claiming completion.

All files are written under pytest ``tmp_path``; the repo's real ``results/`` is
never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.contracts import (
    ForecastBatch,
    ForecastInputs,
    LabelOrigin,
    TargetKind,
    TargetScope,
    TargetSpec,
)
from src.evaluation.artifacts import PREDICTION_COLUMNS, RunDirectory, predictions_frame
from src.evaluation.backtest import score_batch
from src.evaluation.metrics import forecast_metrics
from src.models import build_forecaster
from src.models.power_backbone import PlannedPowerBackbone
from src.training.lifecycle import (
    FrozenRunError,
    freeze_model,
    load_frozen,
    phase_allows_fit,
    phase_allows_test_scoring,
    require_frozen_record,
)
from src.utils import config_digest, read_json, sha256_file

POINTS_PER_DAY = 96
ANCHOR = pd.Timestamp("2026-01-01T00:00:00")
SLOT = np.arange(POINTS_PER_DAY, dtype=float)
BASE_PROFILE = 120.0 + 40.0 * np.sin(2.0 * np.pi * SLOT / POINTS_PER_DAY) + 0.5 * SLOT

TARGET_SPEC = TargetSpec(
    kind=TargetKind.ACTIVE_POWER,
    scope=TargetScope.FACILITY,
    unit="kW",
    label_origin=LabelOrigin.SYNTHETIC_SIMULATED,
    measurement_boundary="synthetic facility main meter",
    sampling_semantics="interval_mean_15min",
    synthetic=True,
)

CONFIG = {
    "experiment_id": "lifecycle_test",
    "model": "seasonal_naive",
    "seed": 7,
    "evaluation": {"mape_relative_threshold": 0.0},
}
PROTOCOL = {
    "protocol_id": "test_configurable_v1",
    "horizon": "24h",
    "horizon_points": 96,
    "context_points": 480,
}
DATASET = {
    "dataset_id": "synthetic_facility",
    "synthetic": True,
    "target": {
        "kind": "active_power",
        "scope": "facility",
        "unit": "kW",
        "label_origin": "synthetic_simulated",
    },
}


# ---------------------------------------------------------------------------
# in-memory helpers (contracts only)
# ---------------------------------------------------------------------------


def _series(days: int = 5) -> np.ndarray:
    profiles = [BASE_PROFILE + 3.0 * day for day in range(days)]
    return np.concatenate(profiles)


def _grid(points: int) -> np.ndarray:
    return pd.date_range(ANCHOR, periods=points, freq="15min").to_numpy(dtype="datetime64[ns]")


def _make_inputs(series: np.ndarray, horizon: int, *, samples: int = 1) -> ForecastInputs:
    series = np.asarray(series, dtype=float)
    length = series.shape[0]
    grid = _grid(length + horizon)
    return ForecastInputs(
        past_power_kw=np.tile(series, (samples, 1)),
        past_power_valid=np.ones((samples, length), dtype=bool),
        past_timestamps=np.tile(grid[:length], (samples, 1)),
        origin_timestamps=np.repeat(grid[length], samples),
        target_timestamps=np.tile(grid[length:], (samples, 1)),
        target_spec=TARGET_SPEC,
        domain_ids=tuple(f"synthetic_domain_{index}" for index in range(samples)),
        dataset_id="synthetic_facility",
    )


def _future_values(horizon: int) -> np.ndarray:
    repeats = int(np.ceil(horizon / POINTS_PER_DAY))
    return np.tile(BASE_PROFILE, repeats)[:horizon]


def _make_batch(series: np.ndarray, horizon: int, *, samples: int = 1) -> ForecastBatch:
    inputs = _make_inputs(series, horizon, samples=samples)
    targets = np.tile(_future_values(horizon), (samples, 1))
    return ForecastBatch(
        inputs=inputs,
        targets_kw=targets,
        target_valid_mask=np.ones_like(targets, dtype=bool),
    )


def _make_batch_at(
    series: np.ndarray,
    horizon: int,
    origin_indices: list[int],
    context_points: int,
) -> ForecastBatch:
    """Batch with a distinct origin/target window per sample."""
    series = np.asarray(series, dtype=float)
    grid = _grid(series.shape[0])
    past = np.stack([series[index - context_points : index] for index in origin_indices])
    targets = np.stack([series[index : index + horizon] for index in origin_indices])
    inputs = ForecastInputs(
        past_power_kw=past,
        past_power_valid=np.ones_like(past, dtype=bool),
        past_timestamps=np.stack(
            [grid[index - context_points : index] for index in origin_indices]
        ),
        origin_timestamps=np.array(
            [grid[index] for index in origin_indices], dtype="datetime64[ns]"
        ),
        target_timestamps=np.stack([grid[index : index + horizon] for index in origin_indices]),
        target_spec=TARGET_SPEC,
        domain_ids=tuple(f"synthetic_domain_{index}" for index in range(len(origin_indices))),
        dataset_id="synthetic_facility",
    )
    return ForecastBatch(
        inputs=inputs,
        targets_kw=targets,
        target_valid_mask=np.ones_like(targets, dtype=bool),
    )


def _new_run(tmp_path: Path, run_id: str = "run-0001") -> RunDirectory:
    return RunDirectory.create(tmp_path / "runs", run_id=run_id)


_FREEZE_KWARGS = {
    "phase": "finalize",
    "seed": 7,
    "horizon": "24h",
    "extra": {"synthetic": True, "research_result": False},
}


def _freeze(run: RunDirectory, model=None, **overrides):
    arguments = {
        "run_dir": run,
        "config": CONFIG,
        "protocol": PROTOCOL,
        "dataset": DATASET,
        **_FREEZE_KWARGS,
    }
    arguments.update(overrides)
    return freeze_model(model if model is not None else build_forecaster("seasonal_naive"), **arguments)


# ---------------------------------------------------------------------------
# T19: freeze -> verify -> reload -> identical predictions
# ---------------------------------------------------------------------------


def test_freeze_and_reload_reproduces_predictions_exactly(tmp_path: Path) -> None:
    run = _new_run(tmp_path)
    model = build_forecaster("seasonal_naive")
    inputs = _make_inputs(_series(5), 96)
    before = model.predict(inputs)

    payload = _freeze(run, model)

    state_path = run.path / "model_state" / "state.json"
    frozen_path = run.path / "frozen_before_test.json"
    assert state_path.is_file()
    assert frozen_path.is_file()

    state = read_json(state_path)
    assert state["model_type"] == "seasonal_naive"
    assert state["name"] == "seasonal_naive"
    assert state["version"] == model.version
    assert state["requires_fit"] is False
    assert state["params"] == {"season_length": 96}

    record = read_json(frozen_path)
    assert record["phase"] == "finalize"
    assert record["synthetic"] is True
    assert record["research_result"] is False
    assert record["seed"] == 7
    assert record["horizon"] == "24h"
    assert record["model_name"] == "seasonal_naive"
    assert record["model_version"] == model.version
    assert record["model_params"] == {"season_length": 96}
    assert "freeze_time_utc" in record

    # Both credential hashes verify, and expected config/protocol/dataset match.
    assert record["model_state_sha256"] == sha256_file(state_path)
    assert record["config_sha256"] == config_digest(CONFIG)
    assert record["protocol_sha256"] == config_digest(PROTOCOL)
    assert record["dataset_sha256"] == config_digest(DATASET)
    assert payload["model_state_sha256"] == record["model_state_sha256"]

    frozen = load_frozen(
        run.path,
        expected_config=CONFIG,
        expected_protocol=PROTOCOL,
        expected_dataset=DATASET,
    )
    assert isinstance(frozen.run_dir, Path)
    assert frozen.model.name == "seasonal_naive"
    assert frozen.state["params"] == {"season_length": 96}
    np.testing.assert_array_equal(frozen.model.predict(inputs), before)

    # The same credential can be loaded by pointing directly at the JSON file.
    frozen_from_file = load_frozen(run.path / "frozen_before_test.json")
    np.testing.assert_array_equal(frozen_from_file.model.predict(inputs), before)

    # require_frozen_record is the CLI's "final-test must be frozen" gate.
    assert require_frozen_record(run.path)["model_name"] == "seasonal_naive"
    assert require_frozen_record(frozen_path)["phase"] == "finalize"


def test_tampered_model_state_is_rejected(tmp_path: Path) -> None:
    run = _new_run(tmp_path, "run-tamper")
    _freeze(run)
    state_path = run.path / "model_state" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["params"]["season_length"] = 48
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(FrozenRunError, match="SHA256"):
        load_frozen(run.path)


def test_expected_config_mismatch_is_rejected(tmp_path: Path) -> None:
    run = _new_run(tmp_path, "run-config-mismatch")
    _freeze(run)
    different_config = {**CONFIG, "seed": 8}
    with pytest.raises(FrozenRunError, match="config digest mismatch"):
        load_frozen(run.path, expected_config=different_config)
    with pytest.raises(FrozenRunError, match="protocol digest mismatch"):
        load_frozen(run.path, expected_protocol={**PROTOCOL, "horizon": "7d"})
    with pytest.raises(FrozenRunError, match="dataset digest mismatch"):
        load_frozen(run.path, expected_dataset={**DATASET, "dataset_id": "other"})


def test_unfrozen_run_is_refused_for_final_test(tmp_path: Path) -> None:
    run = _new_run(tmp_path, "run-unfrozen")
    assert not (run.path / "frozen_before_test.json").exists()

    with pytest.raises(FrozenRunError, match="frozen_before_test"):
        load_frozen(run.path)
    with pytest.raises(FrozenRunError, match="frozen_before_test"):
        require_frozen_record(run.path)
    with pytest.raises(FrozenRunError, match="does not exist"):
        require_frozen_record(tmp_path / "runs" / "never-created")


# ---------------------------------------------------------------------------
# phase policy and double-freeze protection
# ---------------------------------------------------------------------------


def test_phase_policy_is_explicit() -> None:
    assert phase_allows_fit("development") is True
    assert phase_allows_fit("finalize") is True
    assert phase_allows_fit("final-test") is False
    assert phase_allows_fit("smoke") is True  # internal prepare step only

    assert phase_allows_test_scoring("development") is False
    assert phase_allows_test_scoring("finalize") is False
    assert phase_allows_test_scoring("final-test") is True
    assert phase_allows_test_scoring("smoke") is True

    with pytest.raises(ValueError):
        phase_allows_fit("not-a-phase")
    with pytest.raises(ValueError):
        phase_allows_test_scoring("not-a-phase")


def test_freezing_twice_into_one_run_directory_is_refused(tmp_path: Path) -> None:
    run = _new_run(tmp_path, "run-double-freeze")
    _freeze(run)
    with pytest.raises(FrozenRunError, match="already exists"):
        _freeze(run)

    # Freezing a planned placeholder must never produce a credential either.
    planned_run = _new_run(tmp_path, "run-planned-freeze")
    with pytest.raises(FrozenRunError, match="implemented"):
        _freeze(planned_run, PlannedPowerBackbone())

    # A final-test run may not freeze at all (freezing happens before scoring).
    final_test_run = _new_run(tmp_path, "run-final-test-freeze")
    with pytest.raises(FrozenRunError, match="final-test"):
        _freeze(final_test_run, phase="final-test")


# ---------------------------------------------------------------------------
# RunDirectory: no overwrite, exact columns, metrics, failure status
# ---------------------------------------------------------------------------


def test_run_directory_never_overwrites_and_records_failure(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run = RunDirectory.create(root, run_id="abc123")
    assert run.run_id == "abc123"
    assert run.path == root / "abc123"
    assert run.path.is_dir()
    assert run.figures_dir.is_dir()

    with pytest.raises(FileExistsError):
        RunDirectory.create(root, run_id="abc123")

    manifest_path = run.write_manifest({"phase": "smoke", "experiment_id": "unit-test"})
    manifest = read_json(manifest_path)
    assert manifest["status"] == "running"
    assert "created_at" in manifest
    assert manifest["run_id"] == "abc123"

    finalized = run.finalize(status="failed", error="synthetic failure for the test")
    manifest = read_json(finalized)
    assert manifest["status"] == "failed"
    assert manifest["error"] == "synthetic failure for the test"
    assert manifest["status"] != "completed"
    assert "finalized_at" in manifest

    with pytest.raises(ValueError):
        run.finalize(status="done")

    completed_run = RunDirectory.create(root, run_id="def456")
    completed_run.write_manifest({"phase": "smoke"})
    completed_run.finalize(status="completed")
    assert read_json(completed_run.path / "manifest.json")["status"] == "completed"


def test_predictions_frame_matches_columns_and_writes_parquet(tmp_path: Path) -> None:
    batch = _make_batch_at(
        _series(5), 96, [2 * POINTS_PER_DAY, 3 * POINTS_PER_DAY], context_points=2 * POINTS_PER_DAY
    )
    model = build_forecaster("seasonal_naive")
    predictions = model.predict(batch.inputs)

    frame = predictions_frame(
        batch,
        predictions,
        run_id="run-pred",
        dataset_id="synthetic_facility",
        model="seasonal_naive",
        protocol_id="demo_v1",
        horizon="24h",
    )
    assert tuple(frame.columns) == PREDICTION_COLUMNS
    assert len(frame) == batch.batch_size * batch.horizon == 192
    assert frame["valid_label"].all()
    assert frame["synthetic"].all()
    assert (frame["unit"] == "kW").all()
    assert frame["lead"].min() == 1
    assert frame["lead"].max() == 96
    assert frame["domain_id"].nunique() == 2
    assert frame["origin_timestamp"].nunique() == 2

    run = _new_run(tmp_path, "run-predictions")
    path = run.write_predictions(frame)
    assert path.is_file()
    loaded = pd.read_parquet(path)
    assert tuple(loaded.columns) == PREDICTION_COLUMNS
    assert len(loaded) == 192
    np.testing.assert_allclose(
        loaded["predicted"].to_numpy(dtype=float), predictions.reshape(-1)
    )

    with pytest.raises(ValueError, match="PREDICTION_COLUMNS"):
        run.write_predictions(frame.drop(columns=["valid_label"]))
    with pytest.raises(ValueError, match="PREDICTION_COLUMNS"):
        run.write_predictions(frame.assign(unexpected=1))
    with pytest.raises(ValueError, match="PREDICTION_COLUMNS"):
        run.write_predictions(frame.loc[:, list(reversed(PREDICTION_COLUMNS))])

    with pytest.raises(ValueError):
        predictions_frame(
            batch,
            predictions[:, :-1],
            run_id="run-pred",
            dataset_id="synthetic_facility",
            model="seasonal_naive",
            protocol_id="demo_v1",
            horizon="24h",
        )


def test_write_metrics_json_csv_and_recomputability(tmp_path: Path) -> None:
    batch = _make_batch(_series(5), 96, samples=2)
    model = build_forecaster("seasonal_naive")
    predictions = np.asarray(batch.targets_kw) + 5.0
    report = score_batch(batch, predictions)

    run = _new_run(tmp_path, "run-metrics")
    json_path, csv_path = run.write_metrics(report)
    assert json_path.is_file() and csv_path.is_file()

    payload = read_json(json_path)
    assert payload["aggregate"]["MAE"] == pytest.approx(5.0)
    assert payload["aggregate"]["MAPE"] == pytest.approx(report.aggregate["MAPE"])
    assert payload["daily_mean_kind"] == "daily_mean_power"
    assert payload["labels"]["unit"] == "kW"

    csv_frame = pd.read_csv(csv_path)
    assert "origin_timestamp" in csv_frame.columns
    assert "MAPE" in csv_frame.columns
    assert len(csv_frame) == len(report.per_origin) == 2

    # metrics.json must be reproducible from the saved prediction table.
    frame = predictions_frame(
        batch,
        predictions,
        run_id=run.run_id,
        dataset_id="synthetic_facility",
        model="seasonal_naive",
        protocol_id="demo_v1",
        horizon="24h",
    )
    valid_rows = frame.loc[frame["valid_label"].astype(bool)]
    recomputed = forecast_metrics(
        valid_rows["actual"].to_numpy(dtype=float),
        valid_rows["predicted"].to_numpy(dtype=float),
        mape_relative_threshold=0.0,
    )
    for key in ("MAPE", "WAPE", "MAE", "RMSE"):
        assert recomputed[key] == pytest.approx(report.aggregate[key])

    test_report_path = run.write_test_report({"phase": "smoke", "hashes_verified": True})
    assert read_json(test_report_path)["hashes_verified"] is True
