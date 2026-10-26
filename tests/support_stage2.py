"""Shared fixtures for Stage-2 tests (not collected by pytest).

The helpers build a real-shaped processed target artefact and deterministic stub
forecasters, so the experiment layer can be exercised without loading foundation
weights (real weights are covered by ``tests/test_model_adapters.py``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.run_power_main import build_parser, run_experiment
from src.utils import load_config, read_json  # noqa: F401 - re-exported for tests

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiments" / "power_main_v1.yaml"
POINTS_PER_DAY = 96


# ---------------------------------------------------------------------------
# deterministic stub forecasters
# ---------------------------------------------------------------------------


class ProfileStub:
    """Repeats the last observed day profile (deterministic, label-free model)."""

    def __init__(self, name: str, delta: float = 0.0) -> None:
        self.name = name
        self.requires_fit = False
        self.delta = float(delta)

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        context = np.asarray(context, dtype=float)
        if context.ndim != 2 or context.shape[1] < POINTS_PER_DAY:
            raise ValueError("ProfileStub needs a [B, L>=96] context.")
        if not np.isfinite(context).all():
            raise ValueError("ProfileStub received non-finite context.")
        profile = context[:, -POINTS_PER_DAY:]
        repeats = int(np.ceil(horizon / POINTS_PER_DAY))
        tiled = np.tile(profile, (1, repeats))[:, :horizon]
        return tiled + self.delta

    def describe(self) -> dict:
        return {"name": self.name, "family": "stub", "requires_fit": False, "delta": self.delta}


class MeanStub:
    """Fixed 0.5/0.5 mean of two stub members."""

    def __init__(self, name: str, members: Sequence[ProfileStub]) -> None:
        self.name = name
        self.requires_fit = False
        self.members = list(members)
        self.weights = (0.5, 0.5)

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        stacked = np.stack([member.predict(context, horizon) for member in self.members])
        return np.tensordot(np.asarray(self.weights), stacked, axes=(0, 0))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": "stub",
            "requires_fit": False,
            "weights": list(self.weights),
            "members": [member.name for member in self.members],
        }


class NaiveStub(ProfileStub):
    """Stub whose prediction is deliberately broken (failure-path tests)."""

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        output = super().predict(context, horizon)
        output[0, 0] = np.nan
        return output


def stub_factory(config: Any, args: Any, model_names: Sequence[str]) -> dict[str, Any]:
    chronos = ProfileStub("chronos2", delta=5.0)
    tirex = ProfileStub("tirex2", delta=-5.0)
    ensemble = MeanStub("power_ensemble", [chronos, tirex])
    available = {
        "seasonal_naive": ProfileStub("seasonal_naive", delta=0.0),
        "three_day_mean": ProfileStub("three_day_mean", delta=10.0),
        "chronos2": chronos,
        "tirex2": tirex,
        "power_ensemble": ensemble,
    }
    return {name: available[name] for name in model_names}


# ---------------------------------------------------------------------------
# data / experiment fixtures
# ---------------------------------------------------------------------------


def make_target_frame(n_days: int = 31, *, anchor: str = "2016-06-12T06:00:00") -> pd.DataFrame:
    index = pd.date_range(anchor, periods=n_days * POINTS_PER_DAY, freq="15min", name="timestamp")
    slots = np.arange(len(index))
    power = (
        820.0
        + 60.0 * np.sin(2 * np.pi * slots / POINTS_PER_DAY)
        + 25.0 * np.sin(2 * np.pi * slots / (7 * POINTS_PER_DAY))
    )
    return pd.DataFrame(
        {"site_id": "nlr_esif", "power_kw": power, "power_valid": True, "coverage_ratio": 1.0},
        index=index,
    )


def _stage2_origins() -> dict[str, dict[str, list[int]]]:
    return {
        "4h": {"development": [14, 16, 18, 20], "test": list(range(22, 31))},
        "8h": {"development": [14, 16, 18, 20], "test": list(range(22, 31))},
        "24h": {"development": [14, 16, 18, 20], "test": list(range(22, 31))},
        "3d": {"development": [14, 16, 18], "test": list(range(22, 29))},
        "7d": {"development": [14, 15], "test": [22, 23, 24]},
    }


def make_experiment(
    tmp_path: Path,
    *,
    horizons: Sequence[str] = ("24h",),
    label_type: str | None = "facility_derived_kw",
    n_days: int = 31,
    frames: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    frame = make_target_frame(n_days) if frames is None else frames
    parquet_path = data_dir / "processed_target_power.parquet"
    frame.to_parquet(parquet_path)
    metadata = {
        "dataset_id": "nlr_esif",
        "site_id": "nlr_esif",
        "domain_id": "nlr_esif",
        "label_type": label_type if label_type is not None else "facility_derived_kw",
        "trace_anchor_timestamp": str(frame.index.min()),
        "timezone": "UTC",
        "timezone_verified": False,
        "measurement_boundary": "sum of ESIF facility sub-meter components",
        "sampling_semantics": "15-minute interval mean of ~1-minute samples",
    }
    metadata_path = data_dir / "target_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    config = copy.deepcopy(load_config(EXPERIMENT_CONFIG))
    config["data"]["target_power"] = str(parquet_path)
    config["data"]["target_metadata"] = str(metadata_path)
    config["data"]["target_quality"] = None
    config["data"]["source_quality"] = None
    config["protocol"]["horizons"] = list(horizons)
    config["protocol"]["origins"] = _stage2_origins()
    config["output"]["root"] = str(tmp_path / "runs")
    config["runtime"]["offline"] = True
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, parquet_path


def run_stage2(
    tmp_path: Path,
    *extra: str,
    factory=stub_factory,
    **kwargs: Any,
) -> tuple[int, Path]:
    config_path, _ = make_experiment(tmp_path, **kwargs)
    runs_root = tmp_path / "runs"
    args = build_parser().parse_args(
        ["--config", str(config_path), "--output", str(runs_root), *extra]
    )
    return run_experiment(args, forecaster_factory=factory), runs_root


def only_run(runs: Path) -> Path:
    directories = [path for path in runs.iterdir() if path.is_dir()]
    assert len(directories) == 1, f"expected exactly one run directory, got {[p.name for p in directories]}"
    return directories[0]
