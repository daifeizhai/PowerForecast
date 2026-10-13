"""Validate that independent-origin batching preserves native serial predictions."""
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.synthetic import make_demo_data
from src.data.features import make_features
from src.data.splitting import build_windows
from src.models.foundation.forecaster import FoundationForecaster
from src.utils import load_config, write_json


def verify():
    frame = make_demo_data(90)
    history, known = make_features(frame, "facility_power_kw", [], "UTC")
    batch = build_windows(history, known, frame.facility_power_kw, (0, len(frame)), 1344, 672, 96)
    contexts = np.ascontiguousarray(batch.inputs.history[:4, :, 0])
    weights_cache = load_config("configs/hdsf_models.yaml")["models"]["hdsf"]["weights_cache"]
    rows = []
    for name in ("toto2_joint", "timesfm30", "timesfm30_uni"):
        model = FoundationForecaster(name, weights_cache, jobs=4)
        model._load()
        started = perf_counter()
        serial = np.concatenate([model.predictor.predict(row[None], 672) for row in contexts])
        serial_seconds = perf_counter()-started
        started = perf_counter()
        batched = model.predictor.predict_independent_origins(contexts, 672)
        batch_seconds = perf_counter()-started
        row = dict(model=name, serial_seconds=serial_seconds, batch_seconds=batch_seconds,
                   max_abs_delta_kw=float(np.max(np.abs(serial-batched))))
        print(row, flush=True)
        assert np.allclose(serial, batched, atol=.001, rtol=1e-5), f"Serial/batch mismatch: {name}"
        changed = contexts.copy()
        changed[1:] *= 7
        polluted = model.predictor.predict_independent_origins(changed, 672)
        assert np.allclose(polluted[0], batched[0], atol=.001, rtol=1e-5), f"Cross-origin leakage: {name}"
        row["cross_origin_pollution_max_delta_kw"] = float(np.max(np.abs(polluted[0]-batched[0])))
        rows.append(row)
        model.release()
    write_json("results/analysis/nlr/batch-equivalence.json", dict(data="synthetic verification only",
               origins=4, horizon=672, checks=rows))
    print("Batch equivalence passed", flush=True)


if __name__ == "__main__":
    verify()
