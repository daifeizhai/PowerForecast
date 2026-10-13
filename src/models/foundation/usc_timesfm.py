"""USC's native TimesFM 3.0 modes, matching the supplied harvest configuration."""
from pathlib import Path

import numpy as np

from .foundation import cached_model_directory, validate_forecast
from .prediction_cache import PredictionCache

MODEL_ID = "google/timesfm-3.0-pytorch"
INFERENCE_THREADS = 16


def mode_options(name):
    if name not in ("timesfm30", "timesfm30_long", "timesfm30_uni"):
        raise ValueError("Unknown USC TimesFM member")
    return dict(mode="univariate" if name == "timesfm30_uni" else "joint_all_series",
                context_days=14 if name == "timesfm30" else 21,
                per_core_batch_size=32 if name == "timesfm30_uni" else 8)


def prediction_cache(name, cache_dir, device):
    options = mode_options(name)
    return PredictionCache(Path(cache_dir), name, model_id=MODEL_ID,
                           batch_size=options["per_core_batch_size"], jobs=INFERENCE_THREADS, device=device,
                           package_names=["numpy", "torch", "timesfm"], adapter_file=Path(__file__),
                           extra={**options, "symmetric": False, "quantiles": False, "znorm": False,
                                  "padding": "none", "positive": False})


def load_predictor(name, *, cache_dir, device="cpu"):
    import torch
    from timesfm import TimesFM3Forecaster

    options = mode_options(name)
    torch.set_num_threads(INFERENCE_THREADS)
    snapshot = cached_model_directory(MODEL_ID, Path(cache_dir))
    print(f"[USC] Load {name}, {device}, all-series {options['mode']}.", flush=True)
    model = TimesFM3Forecaster.from_pretrained(
        str(snapshot) if snapshot else MODEL_ID, device=device,
        per_core_batch_size=options["per_core_batch_size"], local_files_only=snapshot is not None)

    class Predictor:
        def predict_independent_origins(self, contexts, horizon):
            with torch.inference_mode():
                outputs = list(model.predict_batch(contexts=[row.copy() for row in contexts], horizon=horizon,
                    return_quantiles=False, use_symmetric_averaging=False, make_positive=False,
                    use_znorm=False, padding_mode="none"))
                values = np.stack([np.asarray(output.forecast, dtype=np.float32) for output in outputs])
            return validate_forecast(values, len(contexts), horizon)

        def predict(self, context, horizon):
            context = np.ascontiguousarray(context, dtype=np.float32)
            with torch.inference_mode():
                flags = dict(return_quantiles=False, use_symmetric_averaging=False,
                             make_positive=False, use_znorm=False, padding_mode="none")
                if options["mode"] == "univariate":
                    outputs = list(model.predict_batch(contexts=[row.copy() for row in context], horizon=horizon, **flags))
                    prediction = np.stack([np.asarray(output.forecast, dtype=np.float32) for output in outputs])
                else:
                    prediction = np.asarray(model.predict(context=context, horizon=horizon, **flags).forecast, dtype=np.float32)
            return validate_forecast(prediction, len(context), horizon)
    return Predictor()
