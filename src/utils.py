"""Shared configuration, hashing, reproducibility, and filesystem helpers.

Everything in this module is CPU-only and free of project-specific business
assumptions, so data, model, and evaluation code can share it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import secrets
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORE_DEPENDENCIES = ("numpy", "pandas", "pyarrow", "PyYAML", "pytest")


# ---------------------------------------------------------------------------
# paths and configuration
# ---------------------------------------------------------------------------


def resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` against the project root unless it is absolute."""
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML mapping and optionally apply dotted-key overrides."""
    with resolve_path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``config`` with dotted keys such as ``protocol.horizons``."""
    result = json.loads(json.dumps(config))  # cheap deep copy of YAML scalars/lists/maps
    for dotted, value in overrides.items():
        target = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                raise ValueError(f"Override path {dotted!r} does not exist in the configuration.")
            target = child
        if parts[-1] not in target:
            raise ValueError(f"Override path {dotted!r} does not exist in the configuration.")
        target[parts[-1]] = value
    return result


def config_digest(config: Mapping[str, Any]) -> str:
    """Stable SHA256 of a JSON-serialisable configuration mapping."""
    payload = json.dumps(jsonable(config), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# JSON / hashing
# ---------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-serialisable Python objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, pd.DatetimeIndex):
        return [timestamp.isoformat() for timestamp in value]
    return value


def write_json(path: str | Path, data: Any, *, allow_nan: bool = False) -> Path:
    """Write pretty JSON, refusing non-finite numbers unless explicitly allowed."""
    destination = resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(data), handle, indent=2, ensure_ascii=False, allow_nan=allow_nan)
        handle.write("\n")
    return destination


def read_json(path: str | Path) -> Any:
    with resolve_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file hash so multi-gigabyte raw data never enters memory."""
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: str | Path, *, hash_contents: bool = True) -> dict[str, Any]:
    """Metadata sidecar entry for one source file (never loads it whole)."""
    resolved = resolve_path(path)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(resolved) if hash_contents else None,
    }


# ---------------------------------------------------------------------------
# time, seeds, logging
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(moment: datetime | None = None) -> str:
    """RFC3339 UTC timestamp, e.g. ``2026-10-26T14:29:47Z``."""
    value = (moment or utc_now()).astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(moment: datetime | None = None) -> str:
    """Sortable unique run identifier used as a new run-directory name."""
    value = (moment or utc_now()).astimezone(timezone.utc)
    return f"{value.strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(3)}"


def seed_everything(seed: int) -> None:
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    random.seed(int(seed))
    np.random.seed(int(seed))
    # Optional deep-learning frameworks are deliberately not imported here.


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# git / environment provenance
# ---------------------------------------------------------------------------


def _git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_rev(ref: str = "HEAD") -> str | None:
    return _git(["rev-parse", ref])


def git_branch() -> str | None:
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    return None if branch in (None, "HEAD") else branch


def git_is_dirty() -> bool | None:
    output = _git(["status", "--porcelain=v1", "--untracked-files=no"])
    return None if output is None else bool(output)


def dependency_versions(names: Iterable[str] = CORE_DEPENDENCIES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_report(*, seed: int | None = None) -> dict[str, Any]:
    """Provenance block stored in every run manifest."""
    return {
        "generated_at_utc": utc_timestamp(),
        "git_branch": git_branch(),
        "git_head": git_rev("HEAD"),
        "git_base_main": git_rev("origin/main"),
        "git_dirty_tracked": git_is_dirty(),
        "python": ".".join(str(part) for part in __import__("sys").version_info[:3]),
        "dependencies": dependency_versions(),
        "seed": seed,
    }
