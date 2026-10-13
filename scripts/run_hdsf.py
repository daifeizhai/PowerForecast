"""Default HDSF reproduction on the local NLR data."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_experiment import main

if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/hdsf.yaml", *sys.argv[1:]]))
