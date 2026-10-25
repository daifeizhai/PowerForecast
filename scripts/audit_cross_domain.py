"""Snapshot protected artifacts, or verify a previous snapshot without modifying them."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import PROJECT_ROOT, load_config, resolve_path, write_json


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot() -> dict:
    capture = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                   baseline_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   initial_status=subprocess.check_output(["git", "-c", "core.quotepath=false", "status", "--short"],
                                                          text=True, encoding="utf-8"),
                   index_entries=subprocess.check_output(["git", "ls-files", "--stage"], text=True, encoding="utf-8"))
    roots = [resolve_path(p) for p in ("data/raw", "data/processed", "data/interim", "results/runs", "results/cache")]
    weights_path = load_config("configs/models.yaml").get("foundation_defaults", {}).get("weights_cache")
    if weights_path:
        weights = resolve_path(weights_path)
        if weights.exists():
            roots.append(weights)
    records = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            record = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            # Hash all saved experiment outputs and existing prepared data. Large
            # raw archives, prediction caches and checkpoints use size/mtime only.
            hashed = root.name in {"runs", "processed", "interim"} or (
                root.name == "raw" and stat.st_size <= 128 * 1024**2)
            if hashed:
                record["sha256"] = file_digest(path)
            records.append(record)
    protected = [*PROJECT_ROOT.glob("configs/**/*.yaml"), *PROJECT_ROOT.glob("docs/**/*.md"),
                 PROJECT_ROOT / "README.md", *PROJECT_ROOT.glob("requirements*.txt")]
    return dict(**capture,
                files=records, source_files={str(p.relative_to(PROJECT_ROOT)): file_digest(p) for p in protected},
                policy="Existing files only; additions allowed. SHA256 when recorded, otherwise size/mtime. No protection claim beyond these checks.")


def verify(baseline: dict) -> dict:
    changes = []
    for record in baseline["files"]:
        path = Path(record["path"])
        if not path.is_file():
            changes.append(dict(path=str(path), reason="missing"))
            continue
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
            changes.append(dict(path=str(path), reason="size/mtime changed"))
        elif "sha256" in record and file_digest(path) != record["sha256"]:
            changes.append(dict(path=str(path), reason="content hash changed"))
    return dict(passed=not changes, checked_files=len(baseline["files"]), changes=changes,
                checked_at_utc=datetime.now(timezone.utc).isoformat(), policy=baseline["policy"])


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", metavar="BASELINE_JSON")
    parser.add_argument("--run", metavar="RUN_DIR", help="Recompute a completed cross-domain experiment without retraining.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = resolve_path(args.output)
    if output.exists():
        parser.error("Use a new output path; audit evidence is append-only.")
    if args.run and args.verify:
        parser.error("Choose either --run or --verify.")
    if args.run:
        from src.evaluation.cross_domain import audit_cross_domain_run
        data = audit_cross_domain_run(resolve_path(args.run))
    else:
        data = verify(json.loads(resolve_path(args.verify).read_text(encoding="utf-8"))) if args.verify else snapshot()
    write_json(output, data)
    print(f"Audit: {output}; files={data.get('checked_files', len(data.get('files', [])))}; passed={data.get('passed', True)}")
    return 0 if data.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
