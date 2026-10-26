"""Offline resolution of frozen foundation-model checkpoints + shared adapter validation.

The Stage-2 main model A runs **zero-shot, weight-frozen** foundation models on a
CPU.  This module makes those weights reproducible and auditable:

* :func:`resolve_checkpoint` locates a *complete* local Hugging-Face snapshot
  (``HF_HUB_CACHE`` layout) with :func:`huggingface_hub.try_to_load_from_cache`,
  refuses partial snapshots, records the snapshot path / revision / file sizes /
  file SHA256 digests, and **never downloads** unless ``allow_download=True`` is
  passed explicitly (weight preparation is a separate step, not part of a run).
* :func:`configure_offline_environment` forces ``HF_HUB_OFFLINE=1`` and friends
  once a local snapshot has been resolved, so the libraries cannot fall back to
  the network behind our back.
* :func:`validate_context` / :func:`validate_prediction` are the shared input /
  output guards of the two adapters: dense ``[B, L]`` finite kW context in,
  dense ``[B, H]`` finite float64 kW out.  Ragged, non-finite, too-short or
  label-shaped inputs are rejected instead of being silently repaired.

No random or partial weights are ever created here.  When the cache is missing
the call fails with :class:`FileNotFoundError` that names the missing files, the
cache directory and the expected layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from src.utils import resolve_path, sha256_file

__all__ = [
    "CheckpointSnapshot",
    "FOUNDATION_LIBRARIES",
    "MIN_CONTEXT_POINTS",
    "OFFLINE_ENVIRONMENT",
    "configure_offline_environment",
    "library_versions",
    "resolve_checkpoint",
    "snapshot_to_dict",
    "validate_context",
    "validate_prediction",
]

#: Environment variables that pin the Hugging-Face stack to local files only.
OFFLINE_ENVIRONMENT: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

#: Minimum number of 15-minute context points accepted by both adapters.
#: One trace day (96 points) is the smallest context that contains a full daily
#: profile; shorter windows are rejected explicitly rather than padded.
MIN_CONTEXT_POINTS = 96

#: Distribution names recorded in ``describe()`` / probe evidence.
FOUNDATION_LIBRARIES: tuple[str, ...] = (
    "chronos-forecasting",
    "tirex-2",
    "torch",
    "transformers",
    "huggingface-hub",
    "numpy",
)


@dataclass(frozen=True)
class CheckpointSnapshot:
    """One complete, locally resolved model snapshot (never a partial download).

    ``file_sha256`` / ``file_bytes`` follow symbolic links to the real blobs, so
    the recorded digest is the digest of model bytes actually used for inference.
    ``revision`` is the resolved snapshot revision (the snapshot directory name,
    normally a commit hash) rather than the requested branch name.
    """

    model_id: str
    snapshot_dir: Path
    files: tuple[str, ...]
    file_sha256: Mapping[str, str]
    file_bytes: Mapping[str, int]
    revision: str | None
    cache_dir: str
    source: Literal["local_cache", "downloaded"]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe evidence block for run manifests."""
        return snapshot_to_dict(self)


def snapshot_to_dict(snapshot: CheckpointSnapshot) -> dict[str, Any]:
    """Serialise a :class:`CheckpointSnapshot` into a JSON-safe manifest block."""
    if not isinstance(snapshot, CheckpointSnapshot):
        raise TypeError(
            f"snapshot_to_dict expects a CheckpointSnapshot, got {type(snapshot).__name__}."
        )
    return {
        "model_id": snapshot.model_id,
        "snapshot_dir": str(snapshot.snapshot_dir),
        "files": list(snapshot.files),
        "file_sha256": {name: snapshot.file_sha256[name] for name in snapshot.files},
        "file_bytes": {name: int(snapshot.file_bytes[name]) for name in snapshot.files},
        "revision": snapshot.revision,
        "cache_dir": snapshot.cache_dir,
        "source": snapshot.source,
    }


def configure_offline_environment() -> dict[str, str]:
    """Force the Hugging-Face stack into offline mode and return the applied flags.

    This is called after a complete local snapshot has been resolved.  The flags
    are set unconditionally (not ``setdefault``): once the run has committed to a
    frozen local checkpoint, a silent network fallback would break reproducibility.
    """
    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value
    return dict(OFFLINE_ENVIRONMENT)


def library_versions(names: Sequence[str] = FOUNDATION_LIBRARIES) -> dict[str, str | None]:
    """Installed versions of the inference stack (``None`` when not installed)."""
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


# ---------------------------------------------------------------------------
# checkpoint resolution
# ---------------------------------------------------------------------------


def _repo_cache_folder(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def _missing_snapshot_error(
    *,
    model_id: str,
    missing: Sequence[str],
    cache_dir: Path,
    reason: str,
) -> FileNotFoundError:
    expected = cache_dir / _repo_cache_folder(model_id) / "snapshots" / "<revision>" / "<file>"
    files = ", ".join(missing) if missing else "(none reported)"
    return FileNotFoundError(
        "Frozen foundation-model checkpoint files are missing from the offline cache.\n"
        f"  model_id: {model_id}\n"
        f"  weights cache: {cache_dir}\n"
        f"  missing files: {files}\n"
        f"  expected HF_HUB_CACHE layout: {expected}\n"
        f"  detail: {reason}\n"
        "Downloads are disabled for offline inference (allow_download=False); prepare the weights "
        "cache in a separate, explicit weight-preparation step (see docs/STAGE2_MAIN_MODEL.md and "
        "configs/models/foundation.yaml). No random, partial or substituted weights are created."
    )


def _validate_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"model_id must be a nonempty string, got {model_id!r}.")
    if model_id != model_id.strip() or "\\" in model_id:
        raise ValueError(f"model_id must be a canonical Hugging-Face repo id, got {model_id!r}.")
    return model_id


def _validate_filenames(filenames: Sequence[str]) -> tuple[str, ...]:
    if isinstance(filenames, (str, Path)):
        raise TypeError(
            "filenames must be a sequence of relative file names "
            f"(e.g. ('config.json', 'model.safetensors')), got a single {type(filenames).__name__}."
        )
    try:
        requested = tuple(filenames)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(f"filenames must be iterable, got {filenames!r}.") from exc
    if not requested:
        raise ValueError("filenames must contain at least one checkpoint file name.")
    normalized: list[str] = []
    for name in requested:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"checkpoint file names must be nonempty strings, got {name!r}.")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"checkpoint file name {name!r} must be a relative path inside the snapshot "
                "(absolute paths and '..' are rejected)."
            )
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


def _resolve_cached_files(
    *,
    model_id: str,
    cache_dir: Path,
    filenames: tuple[str, ...],
    revision: str | None,
) -> tuple[dict[str, Path], list[str]]:
    """Map each requested file to a cached path; return (found, missing)."""
    from huggingface_hub import try_to_load_from_cache

    found: dict[str, Path] = {}
    missing: list[str] = []
    for name in filenames:
        cached = try_to_load_from_cache(
            model_id, name, cache_dir=str(cache_dir), revision=revision
        )
        # ``try_to_load_from_cache`` returns a path string, ``None`` or the
        # ``_CACHED_NO_EXIST`` sentinel; only an existing regular file counts.
        if isinstance(cached, str) and Path(cached).is_file():
            found[name] = Path(cached)
        else:
            missing.append(name)
    return found, missing


def _snapshot_dir(found: Mapping[str, Path]) -> Path:
    parents = {path.parent for path in found.values()}
    if len(parents) != 1:
        detail = ", ".join(f"{name} -> {path.parent}" for name, path in found.items())
        raise FileNotFoundError(
            "Checkpoint files resolved to different snapshot directories; refusing to mix "
            f"revisions. Resolved: {detail}."
        )
    return next(iter(parents))


def _build_snapshot(
    *,
    model_id: str,
    snapshot_dir: Path,
    filenames: tuple[str, ...],
    cache_dir: Path,
    source: Literal["local_cache", "downloaded"],
) -> CheckpointSnapshot:
    digests: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for name in filenames:
        path = snapshot_dir / name
        if not path.is_file():
            raise _missing_snapshot_error(
                model_id=model_id,
                missing=[name],
                cache_dir=cache_dir,
                reason=f"the cached snapshot directory {snapshot_dir} does not contain it",
            )
        size = int(path.stat().st_size)
        if size <= 0:
            raise FileNotFoundError(
                f"Checkpoint file {path} is empty (0 bytes); the snapshot is incomplete. "
                "Refusing to load empty/partial weights."
            )
        digests[name] = _cached_sha256(path)
        sizes[name] = size
    return CheckpointSnapshot(
        model_id=model_id,
        snapshot_dir=snapshot_dir,
        files=filenames,
        file_sha256=digests,
        file_bytes=sizes,
        revision=snapshot_dir.name or None,
        cache_dir=str(cache_dir),
        source=source,
    )


#: In-process cache of ``(path, size, mtime_ns) -> sha256`` so repeated builder /
#: adapter constructions do not re-read multi-hundred-megabyte checkpoints.
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _cached_sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    digest = _HASH_CACHE.get(key)
    if digest is None:
        digest = sha256_file(path)
        _HASH_CACHE[key] = digest
    return digest


def resolve_checkpoint(
    model_id: str,
    weights_cache: str | Path,
    filenames: Sequence[str],
    *,
    allow_download: bool = False,
    revision: str | None = None,
) -> CheckpointSnapshot:
    """Resolve a complete frozen checkpoint snapshot from a local HF hub cache.

    Parameters
    ----------
    model_id:
        Canonical Hugging-Face repo id, e.g. ``"amazon/chronos-2"``.
    weights_cache:
        Directory in ``HF_HUB_CACHE`` layout, i.e. containing
        ``models--<org>--<name>/snapshots/<revision>/<files>``.  Relative paths
        are resolved against the project root.
    filenames:
        The checkpoint files that must all be present in **one** snapshot, e.g.
        ``("config.json", "model.safetensors")``.
    allow_download:
        ``False`` (default) means cache-only; a missing file raises
        :class:`FileNotFoundError` and no network request is made.  ``True`` may
        download the requested files into ``weights_cache`` (weight-preparation
        step only; experiment runs never pass ``True``).
    revision:
        Optional branch/tag/commit to resolve; defaults to the repository's
        ``main`` ref exactly as :func:`huggingface_hub.try_to_load_from_cache` does.

    Returns
    -------
    CheckpointSnapshot
        Immutable record with the snapshot directory, resolved revision, file
        sizes and SHA256 digests.
    """
    model_id = _validate_model_id(model_id)
    requested = _validate_filenames(filenames)
    if not isinstance(weights_cache, (str, Path)) or not str(weights_cache).strip():
        raise ValueError(f"weights_cache must be a nonempty path, got {weights_cache!r}.")
    cache_dir = resolve_path(weights_cache)

    source: Literal["local_cache", "downloaded"] = "local_cache"
    if not cache_dir.is_dir():
        if not allow_download:
            raise _missing_snapshot_error(
                model_id=model_id,
                missing=requested,
                cache_dir=cache_dir,
                reason="the weights cache directory does not exist",
            )
        found: dict[str, Path] = {}
        missing = list(requested)
    else:
        found, missing = _resolve_cached_files(
            model_id=model_id, cache_dir=cache_dir, filenames=requested, revision=revision
        )

    if missing:
        if not allow_download:
            raise _missing_snapshot_error(
                model_id=model_id,
                missing=missing,
                cache_dir=cache_dir,
                reason="no complete local snapshot contains every requested file",
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - huggingface_hub is a hard dependency
            raise FileNotFoundError(
                f"Cannot download checkpoint {model_id}: huggingface_hub is not installed ({exc})."
            ) from exc
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                revision=revision or "main",
                allow_patterns=list(requested),
            )
        except Exception as exc:  # network/OS errors surface with context
            raise FileNotFoundError(
                f"Explicit download of checkpoint {model_id} into {cache_dir} failed: "
                f"{type(exc).__name__}: {exc}. This run cannot continue without the frozen weights."
            ) from exc
        source = "downloaded"
        found, missing = _resolve_cached_files(
            model_id=model_id, cache_dir=cache_dir, filenames=requested, revision=revision
        )
        if missing:
            raise _missing_snapshot_error(
                model_id=model_id,
                missing=missing,
                cache_dir=cache_dir,
                reason="the download completed but the files are still not resolvable from the cache",
            )

    snapshot_dir = _snapshot_dir(found)
    for name, path in found.items():
        if path.parent != snapshot_dir:  # pragma: no cover - _snapshot_dir already guarantees this
            raise FileNotFoundError(f"Checkpoint file {name} resolved outside snapshot {snapshot_dir}.")
    return _build_snapshot(
        model_id=model_id,
        snapshot_dir=snapshot_dir,
        filenames=requested,
        cache_dir=cache_dir,
        source=source,
    )


# ---------------------------------------------------------------------------
# shared adapter input/output validation
# ---------------------------------------------------------------------------


def validate_context(
    context_kw: Any,
    horizon: Any,
    *,
    model_name: str,
    min_context_points: int = MIN_CONTEXT_POINTS,
    max_context_points: int | None = None,
) -> tuple[np.ndarray, int]:
    """Validate a raw ``[B, L]`` kW context and a positive integer horizon.

    Returns ``(context_float64, horizon)`` where ``context_float64`` is a dense,
    finite, C-contiguous copy.  Ragged/object arrays, non-finite values, too-short
    histories and non-positive/non-integer horizons are rejected with
    :class:`ValueError` instead of being padded, imputed or truncated.
    """
    if context_kw is None:
        raise ValueError(f"{model_name}.predict needs a [B, L] kW context array; got None.")
    try:
        array = np.asarray(context_kw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{model_name}.predict needs a dense numeric [B, L] kW context; the input is ragged "
            f"or non-numeric ({type(exc).__name__}: {exc})."
        ) from exc
    if array.ndim != 2:
        raise ValueError(
            f"{model_name}.predict needs a 2-D [B, L] kW context, got shape {array.shape} "
            f"(ndim={array.ndim})."
        )
    batch, length = array.shape
    if batch < 1 or length < 1:
        raise ValueError(f"{model_name}.predict needs B >= 1 and L >= 1, got shape {array.shape}.")
    if length < min_context_points:
        raise ValueError(
            f"{model_name}.predict needs at least {min_context_points} context points to form a "
            f"daily power profile, got L={length}. Short histories are rejected, never padded or "
            "silently repeated."
        )
    if max_context_points is not None and length > max_context_points:
        raise ValueError(
            f"{model_name}.predict accepts at most {max_context_points} context points (the frozen "
            f"checkpoint maximum), got L={length}. Truncating history would silently change the "
            "model input; shorten the context explicitly upstream instead."
        )
    non_finite = ~np.isfinite(array)
    count = int(non_finite.sum())
    if count:
        rows, columns = np.nonzero(non_finite)
        examples = ", ".join(
            f"[{int(row)}, {int(column)}]={array[row, column]!r}"
            for row, column in list(zip(rows, columns))[:5]
        )
        raise ValueError(
            f"{model_name}.predict received {count} non-finite context value(s) (NaN/Inf) at "
            f"{examples}. Missing power must be handled upstream and is never treated as 0 kW."
        )
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise ValueError(f"{model_name}.predict needs an integer horizon, got {horizon!r}.")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"{model_name}.predict needs horizon >= 1, got {horizon}.")
    return np.ascontiguousarray(array), horizon


def validate_prediction(
    model_name: str,
    output: Any,
    *,
    batch_size: int,
    horizon: int,
) -> np.ndarray:
    """Validate adapter output as ``[B, H]`` finite float64 absolute kW."""
    array = np.asarray(output, dtype=np.float64)
    expected = (int(batch_size), int(horizon))
    if array.shape != expected:
        raise RuntimeError(
            f"{model_name} produced shape {array.shape}, expected {expected} [B, H]. "
            "Silent truncation or repetition of forecasts is not allowed."
        )
    if not bool(np.isfinite(array).all()):
        raise RuntimeError(
            f"{model_name} produced non-finite kW predictions; refusing to return them."
        )
    return array
