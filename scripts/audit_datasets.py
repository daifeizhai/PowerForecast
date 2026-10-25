"""Audit local heterogeneous schemas and official hashes; never download datasets."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.heterogeneous import audit_alibaba
from src.utils import load_config, resolve_path, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hashes", action="store_true", help="Stream all local Alibaba CSVs and compare published SHA256.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = resolve_path(args.output)
    if output.exists():
        parser.error("Audit output already exists; use a new path.")
    registry = load_config("configs/datasets/registry.yaml")
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), registry=registry,
                  alibaba=audit_alibaba(resolve_path("data/raw/Alibaba-GPU-v2020"), args.hashes),
                  local_availability={name: resolve_path(spec["local_path"]).is_dir()
                                      for name, spec in registry["sources"].items()},
                  eagle_policy="Only checked directory existence; no Eagle observations or labels are opened.")
    write_json(output, report)
    matches = [r.get("official_hash_matches") for r in report["alibaba"]["tables"] if "official_hash_matches" in r]
    print(f"Alibaba schema tables={sum(r['present'] for r in report['alibaba']['tables'])}; "
          f"official hash matches={sum(matches)}/{len(matches)}; report={output}")
    return 0 if all(matches) else 1


if __name__ == "__main__":
    raise SystemExit(main())
