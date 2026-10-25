"""The single SAPF experiment/replay entrypoint, with explicit frozen legacy access."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.reporting import configure_console, print_saved_results
from src.models import MODEL_NAMES
from src.utils import load_config


def run_experiment(args):
    config = load_config(args.config)
    if config.get("protocol") == "sapf_gain_v2_suite":
        from src.training.gain_v2_experiment import run_gain_v2
        return run_gain_v2(args, config)
    if config.get("protocol") == "sapf_transfer_suite_v1":
        from src.training.transfer_suite import run_transfer_suite
        return run_transfer_suite(args, config)
    from src.training.experiment import run_sapf
    return run_sapf(args, config)


def main(argv=None):
    configure_console()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--legacy-config" in arguments:
        from src.legacy import run_legacy
        legacy = argparse.ArgumentParser(add_help=False)
        legacy.add_argument("--legacy-config", required=True)
        selected, remaining = legacy.parse_known_args(arguments)
        return run_legacy(selected.legacy_config, remaining)
    parser = argparse.ArgumentParser(description="SAPF 24h/3d cross-domain forecasting; 14-day observed history.")
    parser.add_argument("--config", default="configs/sapf_google.yaml")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--horizons", nargs="+", choices=["4h", "24h", "3d"])
    parser.add_argument("--demo", action="store_true", help="Synthetic domain data; program verification only.")
    parser.add_argument("--demo-days", type=int, default=31)
    parser.add_argument("--results", metavar="RUN_DIR", help="Display a saved run; latest selects completed real SAPF runs.")
    parser.add_argument("--detailed", action="store_true", help="Print WAPE/RMSE/MAE and complete pairwise diagnostics.")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--legacy-config", help="Explicit replay of a configs/...yaml path in the frozen pre-SAPF archive.")
    args = parser.parse_args(arguments)
    try:
        if args.list_models:
            print("Models: "+", ".join(MODEL_NAMES))
            return 0
        config = load_config(args.config)
        if args.results:
            print_saved_results(args.results, results_dir=config["results_dir"], detailed=args.detailed)
        else:
            run_experiment(args)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
