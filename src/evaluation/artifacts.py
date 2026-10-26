"""Run-directory artifacts: predictions, metrics, audits, manifest, freeze file.

Every run gets a brand-new, never-overwritten directory::

    results/runs/<UTC timestamp>_<random id>/
    ├── manifest.json            # status/created_at defaults merged by RunDirectory
    ├── data_audit.json
    ├── split_audit.json
    ├── model_state/state.json   # written by src.training.lifecycle.freeze_model
    ├── frozen_before_test.json  # written once, never overwritten
    ├── metrics.json
    ├── metrics.csv              # per-origin table from MetricsReport
    ├── predictions.parquet      # one row per origin x lead, exactly PREDICTION_COLUMNS
    ├── test_report.json
    └── figures/                 # created, may stay empty

``results/`` is git-ignored and existing run directories are never deleted.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from src.data.contracts import ContractError, ForecastBatch
from src.utils import (
    make_run_id,
    read_json,
    resolve_path,
    sha256_file,
    utc_timestamp,
    write_json,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.evaluation.backtest import MetricsReport

__all__ = ["PREDICTION_COLUMNS", "RunDirectory", "predictions_frame"]

#: Exact column contract of ``predictions.parquet`` (task book §9.2).
PREDICTION_COLUMNS: tuple[str, ...] = (
    "run_id",
    "dataset_id",
    "domain_id",
    "scope",
    "label_origin",
    "synthetic",
    "model",
    "protocol_id",
    "horizon",
    "origin_timestamp",
    "timestamp",
    "lead",
    "actual",
    "predicted",
    "unit",
    "valid_label",
)

_STRING_COLUMNS = (
    "run_id",
    "dataset_id",
    "domain_id",
    "scope",
    "label_origin",
    "model",
    "protocol_id",
    "horizon",
    "unit",
)


def _as_prediction_matrix(batch: ForecastBatch, predictions: Any) -> np.ndarray:
    if not isinstance(batch, ForecastBatch):
        raise ContractError("predictions_frame expects a ForecastBatch (labelled data).")
    array = np.asarray(predictions, dtype=float)
    expected = batch.targets_kw.shape
    if array.shape != expected:
        raise ValueError(
            f"predictions must have shape {expected} [B, H]; got {array.shape}."
        )
    return array


def predictions_frame(
    batch: ForecastBatch,
    predictions: np.ndarray,
    *,
    run_id: str,
    dataset_id: str,
    model: str,
    protocol_id: str,
    horizon: str,
    unit: str = "kW",
) -> pd.DataFrame:
    """Long ``[B*H, 16]`` prediction table with exactly :data:`PREDICTION_COLUMNS`.

    One row per origin x lead: ``valid_label`` is the batch's
    ``target_valid_mask`` and ``synthetic`` comes from
    ``inputs.target_spec.synthetic``.  ``actual`` is stored as supplied by the
    batch (rows with ``valid_label=False`` must be filtered before recomputing
    metrics).  Dtypes are fixed so repeated runs of the same shape produce
    identical parquet schemas.
    """
    matrix = _as_prediction_matrix(batch, predictions)
    inputs = batch.inputs
    batch_size, horizon_points = matrix.shape
    rows = int(batch_size * horizon_points)

    repeated = lambda values: np.repeat(np.asarray(values), horizon_points, axis=0)  # noqa: E731
    origin_stamps = np.asarray(inputs.origin_timestamps).reshape(-1)
    target_stamps = np.asarray(inputs.target_timestamps).reshape(-1)
    origins = repeated(origin_stamps)
    domains = np.repeat(np.asarray(inputs.domain_ids, dtype=object), horizon_points)
    target = inputs.target_spec

    frame = pd.DataFrame(
        {
            "run_id": pd.array([str(run_id)] * rows, dtype="string"),
            "dataset_id": pd.array([str(dataset_id)] * rows, dtype="string"),
            "domain_id": pd.array(domains.tolist(), dtype="string"),
            "scope": pd.array([target.scope.value] * rows, dtype="string"),
            "label_origin": pd.array([target.label_origin.value] * rows, dtype="string"),
            "synthetic": np.full(rows, bool(target.synthetic), dtype=bool),
            "model": pd.array([str(model)] * rows, dtype="string"),
            "protocol_id": pd.array([str(protocol_id)] * rows, dtype="string"),
            "horizon": pd.array([str(horizon)] * rows, dtype="string"),
            "origin_timestamp": pd.to_datetime(origins).astype("datetime64[ns]"),
            "timestamp": pd.to_datetime(target_stamps).astype("datetime64[ns]"),
            "lead": np.tile(np.arange(1, horizon_points + 1, dtype=np.int64), batch_size),
            "actual": np.asarray(batch.targets_kw, dtype=np.float64).reshape(-1),
            "predicted": matrix.reshape(-1).astype(np.float64),
            "unit": pd.array([str(unit)] * rows, dtype="string"),
            "valid_label": np.asarray(batch.target_valid_mask, dtype=bool).reshape(-1),
        }
    )
    return frame.loc[:, list(PREDICTION_COLUMNS)]


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"run_id must be a nonempty string, got {run_id!r}.")
    candidate = run_id.strip()
    if candidate in {".", ".."} or Path(candidate).name != candidate:
        raise ValueError(
            f"run_id must be a single path component (no separators), got {run_id!r}."
        )
    return candidate


class RunDirectory:
    """A new, never-overwritten run directory with the standard artifact writers."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path: Path = path
        self.run_id: str = run_id
        self.figures_dir: Path = path / "figures"

    # -- construction --------------------------------------------------------
    @classmethod
    def create(cls, root: str | Path = "results/runs", *, run_id: str | None = None) -> "RunDirectory":
        """Create ``<root>/<run_id>/``; fail if it already exists (no overwrite)."""
        root_path = resolve_path(root)
        new_run_id = make_run_id() if run_id is None else _validate_run_id(str(run_id))
        destination = root_path / new_run_id
        if destination.exists():
            raise FileExistsError(
                f"Run directory already exists: {destination}.  Runs are immutable evidence; "
                "create a new run id instead of overwriting an old run."
            )
        destination.mkdir(parents=True, exist_ok=False)
        instance = cls(destination, new_run_id)
        instance.figures_dir.mkdir(exist_ok=False)
        return instance

    # -- low-level writers ---------------------------------------------------
    def _atomic_write_json(self, destination: Path, payload: Any) -> Path:
        temporary = destination.with_name(destination.name + ".tmp")
        write_json(temporary, payload)
        temporary.replace(destination)
        return destination

    def write_json(self, name: str, payload: Any) -> Path:
        """Write a named JSON artifact such as ``batch_summary.json``."""
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Artifact name must be a single file name, got {name!r}.")
        return self._atomic_write_json(self.path / name, payload)

    # -- standard artifacts --------------------------------------------------
    def write_manifest(self, payload: dict[str, Any]) -> Path:
        """Write ``manifest.json``, merging the ``status``/``created_at`` defaults."""
        if not isinstance(payload, Mapping):
            raise TypeError("manifest payload must be a mapping.")
        manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "status": "running",
            "created_at": utc_timestamp(),
        }
        manifest.update(dict(payload))
        return self._atomic_write_json(self.path / "manifest.json", manifest)

    def write_data_audit(self, payload: Any) -> Path:
        return self._atomic_write_json(self.path / "data_audit.json", payload)

    def write_split_audit(self, payload: Any) -> Path:
        return self._atomic_write_json(self.path / "split_audit.json", payload)

    def write_frozen(self, payload: dict[str, Any]) -> tuple[Path, str]:
        """Write ``frozen_before_test.json`` once and return ``(path, sha256)``."""
        if not isinstance(payload, Mapping):
            raise TypeError("frozen payload must be a mapping.")
        destination = self.path / "frozen_before_test.json"
        if destination.exists():
            raise FileExistsError(
                f"{destination} already exists; frozen credentials are never overwritten.  "
                "Start a new run instead."
            )
        frozen = {"run_id": self.run_id}
        frozen.update(dict(payload))
        self._atomic_write_json(destination, frozen)
        return destination, sha256_file(destination)

    def write_predictions(self, frame: pd.DataFrame) -> Path:
        """Write ``predictions.parquet`` after validating the exact column contract."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("predictions must be a pandas DataFrame.")
        actual_columns = tuple(frame.columns)
        if actual_columns != PREDICTION_COLUMNS:
            missing = [column for column in PREDICTION_COLUMNS if column not in actual_columns]
            extra = [column for column in actual_columns if column not in PREDICTION_COLUMNS]
            raise ValueError(
                "predictions frame must have exactly PREDICTION_COLUMNS in order; "
                f"missing={missing}, extra={extra}, got={list(actual_columns)}."
            )
        destination = self.path / "predictions.parquet"
        frame.to_parquet(destination, index=False)
        return destination

    def write_metrics(self, report: "MetricsReport") -> tuple[Path, Path]:
        """Write ``metrics.json`` and ``metrics.csv`` from a :class:`MetricsReport`."""
        payload = report.to_dict()
        json_path = self._atomic_write_json(self.path / "metrics.json", payload)
        csv_path = self.path / "metrics.csv"
        report.per_origin.to_csv(csv_path, index=False)
        return json_path, csv_path

    def write_test_report(self, payload: Any) -> Path:
        return self._atomic_write_json(self.path / "test_report.json", payload)

    # -- lifecycle -----------------------------------------------------------
    def finalize(self, *, status: str, error: str | None = None) -> Path:
        """Atomically update the manifest status (``completed`` / ``failed``)."""
        if status not in {"completed", "failed"}:
            raise ValueError(
                f"finalize status must be 'completed' or 'failed', got {status!r}."
            )
        manifest_path = self.path / "manifest.json"
        if manifest_path.is_file():
            existing = read_json(manifest_path)
            if not isinstance(existing, Mapping):
                raise ValueError(f"{manifest_path} does not contain a JSON object.")
            manifest = dict(existing)
        else:
            manifest = {"run_id": self.run_id, "created_at": utc_timestamp()}
        manifest["run_id"] = manifest.get("run_id", self.run_id)
        manifest["status"] = status
        manifest["finalized_at"] = utc_timestamp()
        if error is None:
            manifest.pop("error", None)
        else:
            manifest["error"] = str(error)
        return self._atomic_write_json(manifest_path, manifest)
