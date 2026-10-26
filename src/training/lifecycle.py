"""Training lifecycle: four phases, pre-test freezing, and frozen-run reload.

Phases (task book §8)
---------------------
``development``
    Pre-checks and diagnostics; fitting is allowed on legal training data only;
    formal test scoring is **not** allowed.
``finalize``
    Final legal fit/configuration, persist state and write the freeze credential;
    formal test scoring is **not** allowed.
``final-test``
    Must load an existing frozen run, verify hashes and rebuild the model; fitting
    is **not** allowed.  Only after the frozen credentials are verified may test
    truth be scored.
``smoke``
    One-command synthetic engineering validation: prepare/freeze/test inside the
    same run.  Fitting is allowed **only in the internal prepare step before the
    freeze**; after ``frozen_before_test.json`` exists the model must be reloaded
    from disk before scoring.  Output is always ``synthetic=true`` and
    ``research_result=false``.

The freeze credential is mandatory even for parameter-free baselines: it records
the model type/version/params, the effective config, protocol and dataset
digests, the model-state SHA256, seed, horizon and code SHA.  Reloading verifies
both the state-file hash and (when supplied) the expected config/protocol/dataset
digests.  ``frozen_before_test.json`` is never overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.data.contracts import ContractError
from src.evaluation.artifacts import RunDirectory
from src.models import build_forecaster
from src.models.interfaces import BasePowerForecaster, ModelStatus
from src.utils import (
    config_digest,
    environment_report,
    jsonable,
    read_json,
    resolve_path,
    sha256_file,
    utc_timestamp,
    write_json,
)

__all__ = [
    "FrozenModel",
    "FrozenRunError",
    "PHASES",
    "freeze_model",
    "load_frozen",
    "phase_allows_fit",
    "phase_allows_test_scoring",
    "require_frozen_record",
]

PHASES: tuple[str, ...] = ("development", "finalize", "final-test", "smoke")

FROZEN_FILENAME = "frozen_before_test.json"
MODEL_STATE_DIR = "model_state"
MODEL_STATE_FILENAME = "state.json"


class FrozenRunError(RuntimeError):
    """A frozen run is missing, corrupt, tampered with, or incompatible."""


@dataclass
class FrozenModel:
    """A model rebuilt from a verified frozen run directory."""

    run_dir: Path
    model: BasePowerForecaster
    record: dict[str, Any]
    state: dict[str, Any]


# ---------------------------------------------------------------------------
# phase policy
# ---------------------------------------------------------------------------


def _require_phase(phase: str) -> str:
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}.")
    return phase


def phase_allows_fit(phase: str) -> bool:
    """Whether the phase may fit/select parameters.

    ``development``/``finalize`` yes; ``final-test`` no; ``smoke`` yes **only**
    for the internal prepare step before freezing (the scoring step reloads the
    frozen model instead of refitting).
    """
    _require_phase(phase)
    return phase in {"development", "finalize", "smoke"}


def phase_allows_test_scoring(phase: str) -> bool:
    """Whether the phase may score formal test truth (after freezing)."""
    _require_phase(phase)
    return phase in {"final-test", "smoke"}


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _model_state_dict(model: BasePowerForecaster) -> dict[str, Any]:
    state_dict = getattr(model, "state_dict", None)
    if callable(state_dict):
        state = state_dict()
    else:  # pragma: no cover - future implemented models should provide one
        state = {
            "model_type": type(model).__name__,
            "name": getattr(model, "name", type(model).__name__),
            "version": getattr(model, "version", "unknown"),
            "requires_fit": bool(getattr(model, "requires_fit", False)),
            "params": dict(getattr(model, "params", {})),
        }
    if not isinstance(state, Mapping):
        raise ContractError("model.state_dict() must return a mapping.")
    return jsonable(dict(state))


def freeze_model(
    model: BasePowerForecaster,
    *,
    run_dir: RunDirectory,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    dataset: Mapping[str, Any],
    phase: str,
    seed: int,
    horizon: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist model state and write the immutable pre-test freeze credential.

    The credential is written **before** any test scoring and can never be
    overwritten: a second freeze into the same run directory raises
    :class:`FrozenRunError`.  The returned payload is exactly what was written
    (minus the ``run_id`` default that :meth:`RunDirectory.write_frozen` merges).
    """
    _require_phase(phase)
    if not isinstance(run_dir, RunDirectory):
        raise ContractError(f"run_dir must be a RunDirectory, got {type(run_dir).__name__}.")
    if phase == "final-test":
        raise FrozenRunError(
            "freeze_model cannot run in phase 'final-test': the model/configuration must be "
            "frozen before test scoring, never after seeing test truth."
        )
    status = getattr(model, "status", ModelStatus.IMPLEMENTED)
    if status is not ModelStatus.IMPLEMENTED:
        raise FrozenRunError(
            f"Cannot freeze model {getattr(model, 'name', type(model).__name__)!r} with status "
            f"{getattr(status, 'value', status)!r}: only implemented models may be frozen."
        )
    if not callable(getattr(model, "predict", None)):
        raise ContractError("freeze_model expects a model with a callable predict() method.")

    config_payload = _require_mapping("config", config)
    protocol_payload = _require_mapping("protocol", protocol)
    dataset_payload = _require_mapping("dataset", dataset)
    extra_payload = {} if extra is None else _require_mapping("extra", extra)

    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError(f"seed must be a nonnegative integer, got {seed}.")
    horizon_name = str(horizon)

    frozen_path = run_dir.path / FROZEN_FILENAME
    if frozen_path.exists():
        raise FrozenRunError(
            f"{frozen_path} already exists.  Frozen credentials are immutable evidence: create a "
            "new run directory instead of overwriting this run."
        )

    state = _model_state_dict(model)
    state_dir = run_dir.path / MODEL_STATE_DIR
    state_path = state_dir / MODEL_STATE_FILENAME
    if state_path.exists():
        raise FrozenRunError(
            f"{state_path} already exists; refusing to overwrite partially written model state. "
            "Create a new run directory."
        )
    write_json(state_path, state)
    model_state_sha256 = sha256_file(state_path)

    model_name = str(state.get("name") or getattr(model, "name", type(model).__name__))
    model_version = state.get("version")
    model_params = _require_mapping("model_state.params", state.get("params") or {})
    params_sha256 = config_digest(model_params)
    environment = environment_report(seed=seed_value)

    payload: dict[str, Any] = {
        "freeze_time_utc": utc_timestamp(),
        "phase": phase,
        "run_id": run_dir.run_id,
        "model_name": model_name,
        "model_type": state.get("model_type", model_name),
        "model_version": model_version,
        "model_requires_fit": bool(state.get("requires_fit", getattr(model, "requires_fit", False))),
        "model_params": jsonable(model_params),
        "model_params_sha256": params_sha256,
        "model_state_path": f"{MODEL_STATE_DIR}/{MODEL_STATE_FILENAME}",
        "model_state_sha256": model_state_sha256,
        "config_sha256": config_digest(config_payload),
        "protocol_sha256": config_digest(protocol_payload),
        "dataset_sha256": config_digest(dataset_payload),
        "seed": seed_value,
        "horizon": horizon_name,
        "code_sha": environment.get("git_head"),
        "synthetic": bool(extra_payload.get("synthetic", False)),
        "research_result": bool(extra_payload.get("research_result", False)),
    }
    run_dir.write_frozen(payload)
    return payload


# ---------------------------------------------------------------------------
# reload / verification
# ---------------------------------------------------------------------------


def _resolve_run_and_record(frozen_run: str | Path) -> tuple[Path, Path]:
    candidate = resolve_path(frozen_run)
    if not candidate.exists():
        raise FrozenRunError(f"Frozen run path does not exist: {candidate}")
    if candidate.is_dir():
        return candidate, candidate / FROZEN_FILENAME
    if candidate.is_file():
        if candidate.name != FROZEN_FILENAME:
            raise FrozenRunError(
                f"{candidate} is not a run directory and not a {FROZEN_FILENAME} file."
            )
        return candidate.parent, candidate
    raise FrozenRunError(f"Frozen run path is neither a file nor a directory: {candidate}")


def require_frozen_record(frozen_run: str | Path) -> dict[str, Any]:
    """Return the freeze record, or refuse an unfrozen run (``final-test`` gate).

    Used by the CLI to reject a ``final-test`` that was not frozen beforehand.
    """
    run_dir, record_path = _resolve_run_and_record(frozen_run)
    if not record_path.is_file():
        raise FrozenRunError(
            f"{run_dir} has no {FROZEN_FILENAME}; a 'final-test' run must load an existing frozen "
            "model/configuration first.  Run phase 'finalize' (or a smoke run) to produce the "
            "credential."
        )
    try:
        record = read_json(record_path)
    except (OSError, ValueError) as exc:
        raise FrozenRunError(f"Cannot read {record_path}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise FrozenRunError(f"{record_path} does not contain a JSON object.")
    return dict(record)


def load_frozen(
    frozen_run: str | Path,
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_protocol: Mapping[str, Any] | None = None,
    expected_dataset: Mapping[str, Any] | None = None,
) -> FrozenModel:
    """Verify a frozen run and rebuild its model.

    Verification steps:

    1. ``frozen_before_test.json`` and ``model_state/state.json`` must exist;
    2. the recorded ``model_state_sha256`` must match the state file on disk
       (detects any byte tampering/corruption);
    3. when ``expected_config`` / ``expected_protocol`` / ``expected_dataset``
       are supplied, their canonical digests must match the recorded ones;
    4. the model is rebuilt from the recorded name/params via
       :func:`src.models.build_forecaster`.

    Any failure raises :class:`FrozenRunError` with an actionable message.
    """
    run_dir, record_path = _resolve_run_and_record(frozen_run)
    if not record_path.is_file():
        raise FrozenRunError(
            f"{run_dir} has no {FROZEN_FILENAME}; cannot reload an unfrozen model."
        )
    try:
        record = read_json(record_path)
    except (OSError, ValueError) as exc:
        raise FrozenRunError(f"Cannot read {record_path}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise FrozenRunError(f"{record_path} does not contain a JSON object.")
    record = dict(record)

    recorded_state_sha = record.get("model_state_sha256")
    if not isinstance(recorded_state_sha, str) or not recorded_state_sha:
        raise FrozenRunError(
            f"{record_path} has no 'model_state_sha256'; the freeze credential is incomplete."
        )
    state_relative = record.get("model_state_path") or f"{MODEL_STATE_DIR}/{MODEL_STATE_FILENAME}"
    state_path = (run_dir / str(state_relative)).resolve()
    try:
        state_path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise FrozenRunError(
            f"Frozen record points outside its run directory: {state_relative!r}."
        ) from exc
    if not state_path.is_file():
        raise FrozenRunError(
            f"Frozen model state is missing: {state_path}.  The run record and state files must "
            "both exist."
        )
    actual_state_sha = sha256_file(state_path)
    if actual_state_sha != recorded_state_sha:
        raise FrozenRunError(
            f"SHA256 mismatch for {state_path}: recorded {recorded_state_sha}, actual "
            f"{actual_state_sha}.  The frozen model state was modified or corrupted; refusing to "
            "reload it.  Start a new run instead."
        )

    for key, label, expected in (
        ("config_sha256", "config", expected_config),
        ("protocol_sha256", "protocol", expected_protocol),
        ("dataset_sha256", "dataset", expected_dataset),
    ):
        if expected is None:
            continue
        expected_payload = _require_mapping(label, expected)
        expected_digest = config_digest(expected_payload)
        recorded_digest = record.get(key)
        if expected_digest != recorded_digest:
            raise FrozenRunError(
                f"{label} digest mismatch: frozen record {key}={recorded_digest!r}, expected "
                f"{expected_digest!r}.  The model was frozen for different {label} content."
            )

    try:
        state = read_json(state_path)
    except (OSError, ValueError) as exc:
        raise FrozenRunError(f"Cannot read frozen model state {state_path}: {exc}") from exc
    if not isinstance(state, Mapping):
        raise FrozenRunError(f"{state_path} does not contain a JSON object.")
    state = dict(state)

    # Second SHA256 credential: the recorded params digest must match the state
    # file content too (the state-file hash alone already protects the bytes).
    recorded_params_sha = record.get("model_params_sha256")
    state_params = state.get("params")
    if recorded_params_sha is not None:
        if not isinstance(state_params, Mapping):
            raise FrozenRunError(
                f"{state_path} has no 'params' mapping but the frozen record committed to a "
                "params hash; the credentials are inconsistent."
            )
        actual_params_sha = config_digest(dict(state_params))
        if actual_params_sha != recorded_params_sha:
            raise FrozenRunError(
                "SHA256 mismatch for the frozen model parameters: recorded "
                f"{recorded_params_sha}, actual {actual_params_sha}.  The frozen state was "
                "modified or the record is inconsistent."
            )

    record_name = record.get("model_name")
    state_name = state.get("name") or state.get("model_type")
    if record_name and state_name and str(record_name) != str(state_name):
        raise FrozenRunError(
            f"Frozen record model_name={record_name!r} disagrees with model state "
            f"name={state_name!r}; the credentials are inconsistent."
        )
    model_name = str(record_name or state_name)
    params = state.get("params")
    if params is None:
        params = record.get("model_params") or {}
    if not isinstance(params, Mapping):
        raise FrozenRunError(f"Frozen model params must be a mapping, got {type(params).__name__}.")
    try:
        model = build_forecaster(model_name, **dict(params))
    except (ContractError, TypeError) as exc:
        raise FrozenRunError(
            f"Cannot rebuild frozen model {model_name!r} from recorded params {dict(params)!r}: "
            f"{exc}"
        ) from exc
    return FrozenModel(run_dir=run_dir, model=model, record=record, state=state)
