"""Shared immutable-on-disk model record written before any test prediction."""
from datetime import datetime, timezone
from hashlib import sha256

from src.utils import write_json


def freeze_for_test(folder, **fields):
    path = folder / "frozen-before-test.json"
    if path.exists():
        raise FileExistsError(f"Do not overwrite an existing freeze record: {path}")
    record = dict(**fields,
                  model_sha256={str(p.relative_to(folder)): sha256(p.read_bytes()).hexdigest()
                                for p in folder.rglob("model.*")},
                  frozen_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, record)
    return record
