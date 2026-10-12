from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import inspect_table, locate_source
from src.utils import load_config, resolve_path, write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect raw schema and sample rows without changing data.")
    parser.add_argument("--config", default="configs/datasets/nlr.yaml")
    parser.add_argument("--path", help="Inspect one file instead of configured sources.")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args(argv)
    try:
        paths = [resolve_path(args.path)] if args.path else [locate_source(source) for source in load_config(args.config)["sources"].values()]
        reports = [inspect_table(path) for path in paths]
        for report in reports:
            print(f"Data: {report['path']}")
            print(f"Rows: {report['rows']}")
            for name, dtype in report["schema"].items():
                print(f"  {name}: {dtype}")
            print(report["sample"])
        if args.output:
            write_json(resolve_path(args.output), reports)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
