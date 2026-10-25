"""Explicit, checksum-verified replay of the frozen pre-SAPF source."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from src.utils import PROJECT_ROOT

ARCHIVE = PROJECT_ROOT/"archive/pre_sapf_bc6a439"

LEGACY_SITE_HOOK = '''"""Trusted routing hook for explicit frozen-source processes and their runtime relaunches."""
import os
import sys
from pathlib import Path
if os.environ.get("POWERFORECAST_ARCHIVE_ROOT"):
    frozen = Path(os.environ["POWERFORECAST_ARCHIVE_ROOT"])
    workspace = Path(os.environ["POWERFORECAST_ARCHIVE_WORKSPACE"])
    sys.path.insert(0, str(frozen))
    supplemental = workspace/".venv/Lib/site-packages"
    if supplemental.is_dir():
        sys.path.append(str(supplemental))
    import src.utils as utils
    original_resolve = utils.resolve_path
    def resolve(path):
        value = Path(path)
        if not value.is_absolute() and value.parts and value.parts[0] in {"data", "results", "NLR_Dataset", ".venv"}:
            return (workspace/value).resolve()
        return original_resolve(value)
    utils.resolve_path = resolve
'''


def legacy_environment(frozen: Path) -> dict:
    """Keep routing active when archived Google switches to its original Python environment."""
    digest = sha256(LEGACY_SITE_HOOK.encode("utf-8")).hexdigest()
    directory = PROJECT_ROOT/"results/cache/legacy_bootstrap"/digest
    directory.mkdir(parents=True, exist_ok=True)
    hook = directory/"sitecustomize.py"
    if hook.exists() and sha256(hook.read_bytes()).hexdigest() != digest:
        raise ValueError("Cached legacy routing hook changed; refusing replay.")
    if not hook.exists():
        hook.write_bytes(LEGACY_SITE_HOOK.encode("utf-8"))
    environment = os.environ.copy()
    environment.update(POWERFORECAST_ARCHIVE_ROOT=str(frozen), POWERFORECAST_ARCHIVE_WORKSPACE=str(PROJECT_ROOT),
                       HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    environment["PYTHONPATH"] = os.pathsep.join([str(directory), str(frozen), environment.get("PYTHONPATH", "")])
    return environment


def verify_archive() -> dict:
    manifest = json.loads((ARCHIVE/"manifest.json").read_text(encoding="utf-8"))
    if sha256((ARCHIVE/"source.tar.gz").read_bytes()).hexdigest() != manifest["source_archive_sha256"]:
        raise ValueError("Frozen source archive checksum changed; refusing legacy replay.")
    return manifest


def materialize_archive() -> Path:
    manifest = verify_archive()
    target = PROJECT_ROOT/"results/cache/legacy_source"/manifest["source_archive_sha256"]
    marker = target/".verified-archive"
    if marker.exists():
        for relative, expected in manifest["source_files"].items():
            if sha256((target/relative).read_bytes()).hexdigest() != expected["sha256"]:
                raise ValueError(f"Cached legacy source changed: {relative}")
        return target
    if target.exists():
        raise FileExistsError(f"Incomplete legacy extraction exists: {target}; preserve it and use a fresh cache.")
    target.mkdir(parents=True)
    with tarfile.open(ARCHIVE/"source.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            if not (target/member.name).resolve().is_relative_to(target.resolve()):
                raise ValueError("Archive member escapes the frozen source root.")
        archive.extractall(target, filter="data")
    for relative, expected in manifest["source_files"].items():
        path = target/relative
        if sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Legacy extraction differs from frozen commit: {relative}")
    marker.write_text(manifest["source_archive_sha256"], encoding="ascii")
    return target


def run_legacy(config: str, arguments: list[str]) -> int:
    frozen = materialize_archive()
    path = Path(config)
    if path.is_absolute() or not path.parts or path.parts[0] != "configs" or not (frozen/path).is_file():
        raise ValueError("Use an original relative configs/...yaml path inside the frozen archive.")
    # Frozen code/configs and source hashes retain their own project root. Only
    # data/results/environment paths resolve to the shared workspace.
    bootstrap = """
import sys, runpy
from pathlib import Path
frozen, workspace = Path(sys.argv[1]), Path(sys.argv[2])
config, arguments = sys.argv[3], sys.argv[4:]
sys.path.insert(0, str(frozen))
import src.utils as utils
original_resolve = utils.resolve_path
def resolve(path):
    value = Path(path)
    if not value.is_absolute() and value.parts and value.parts[0] in {'data', 'results', 'NLR_Dataset', '.venv'}:
        return (workspace/value).resolve()
    return original_resolve(value)
utils.resolve_path = resolve
sys.argv = [str(frozen/'scripts/run_experiment.py'), '--config', config, *arguments]
runpy.run_path(str(frozen/'scripts/run_experiment.py'), run_name='__main__')
"""
    completed = subprocess.run([sys.executable, "-B", "-c", bootstrap, str(frozen), str(PROJECT_ROOT),
                                str(path), *arguments], cwd=PROJECT_ROOT, env=legacy_environment(frozen))
    return completed.returncode
