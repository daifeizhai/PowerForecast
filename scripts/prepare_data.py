from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline import prepare_dataset
from src.utils import load_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare aligned 15-minute data and a quality/provenance report.")
    parser.add_argument("--config", default="configs/datasets/nlr.yaml")
    parser.add_argument("--output")
    parser.add_argument("--start", help="Inclusive first date (UTC after adaptation).")
    parser.add_argument("--end", help="Inclusive last date (UTC after adaptation).")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        frame, report = prepare_dataset(config, args.output, args.start, args.end)
        print(f"Data: {config['name']} target={config['target']} rows={len(frame)}")
        print(f"Time: {report['quality']['start']} -> {report['quality']['end']}")
        print(f"Saved: {args.output or config['processed_path']}")
        print(f"Timezone: {config['timezone']} verified={report['timezone_verified']}")
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
