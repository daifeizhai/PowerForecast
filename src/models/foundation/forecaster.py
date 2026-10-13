"""Read-only pretrained checkpoints, workspace-only input-addressed caches."""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

import numpy as np

from src.models.base import BaseForecaster
from .foundation import cached_model_directory, load_foundation
from .new_foundation import cached_checkpoint, load_new_foundation, NEW_MODEL_IDS, CHECKPOINT_FILES
from .prediction_cache import PredictionCache

MODEL_IDS = {"chronos2_long": "amazon/chronos-2", "tirex2": "NX-AI/TiRex-2",
             "timesfm25": "google/timesfm-2.5-200m-pytorch",
             "toto2_joint": "Datadog/Toto-2.0-313m", "toto2_single": "Datadog/Toto-2.0-313m",
             "timesfm30": "google/timesfm-3.0-pytorch", "timesfm30_long": "google/timesfm-3.0-pytorch",
             "timesfm30_uni": "google/timesfm-3.0-pytorch",
             "patchtstfm_long": "ibm-granite/granite-timeseries-patchtst-fm-r2"}


class FoundationForecaster(BaseForecaster):
    def __init__(self, name, weights_cache, prediction_cache="results/cache/foundation", jobs=4,
                 batch_size=32, context_days=14):
        self.name, self.context_days = name, int(context_days)
        self.jobs, self.batch_size = int(jobs), int(batch_size)
        self.weights_cache = Path(weights_cache).resolve()
        self.cache_root = Path(prediction_cache).resolve()
        self.predictor = None
        self.native = {"chronos2_long": "chronos2", "toto2_joint": "toto2",
                       "toto2_single": "toto2"}.get(name, name)
        filenames = CHECKPOINT_FILES["tirex2"] if name == "tirex2" else ("config.json", "model.safetensors")
        for root in (self.weights_cache, self.weights_cache/"hub"):
            snapshot = cached_checkpoint(MODEL_IDS[name], root, filenames)
            if snapshot is not None:
                self.load_root, self.snapshot = root, snapshot
                break
        else:
            raise FileNotFoundError(f"Complete offline checkpoint missing for {name} in {self.weights_cache}.")
        sources = {p.name: sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
        # Names with identical behavior share caches in the one-target NLR experiment.
        cache_name = ("toto2" if name.startswith("toto2") else
                      "timesfm30" if name in {"timesfm30", "timesfm30_long"} else name)
        self.cache = PredictionCache(self.cache_root, cache_name, model_id=MODEL_IDS[name],
            filenames=filenames, checkpoint_directory=self.snapshot, batch_size=batch_size, jobs=jobs,
            device="cpu", package_names=["numpy", "torch", "transformers", "timesfm", "chronos-forecasting",
                "tirex-2", "toto-2", "granite-tsfm"], adapter_file=Path(__file__),
            extra={"context_days": context_days, "sources": sources, "series": "single_target",
                   "cross_origin_attention": False})

    def fit(self, batch):
        if batch.inputs.history.shape[1] < self.context_days*96:
            raise ValueError(f"{self.name} needs {self.context_days} observed days.")
        return self

    def _load(self):
        if self.predictor is not None:
            return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = str(self.cache_root/"matplotlib")
        if self.native in {"chronos2", "timesfm25"}:
            self.predictor = load_foundation(self.native, cache_dir=self.load_root,
                batch_size=self.batch_size, jobs=self.jobs, device="cpu")
        elif self.native in NEW_MODEL_IDS:
            self.predictor = load_new_foundation(self.native, cache_dir=self.load_root,
                batch_size=self.batch_size, jobs=self.jobs, device="cpu", groups=[0] if self.native == "toto2" else None)
        elif self.name.startswith("timesfm30"):
            # One target per origin: origins are separate series, never variates of one sample.
            from .usc_timesfm import load_predictor
            self.predictor = load_predictor(self.name, cache_dir=self.load_root, device="cpu")
        else:
            from .hdsf_patchtst import load_predictor
            self.predictor = load_predictor(cache_dir=self.load_root)
        import torch
        torch.set_num_threads(self.jobs)

    def predict(self, inputs):
        contexts = np.ascontiguousarray(inputs.history[:, -self.context_days*96:, 0], dtype=np.float32)
        horizon = inputs.future_known.shape[1]
        values = [self.cache.load(row[None], horizon) for row in contexts]
        missing = [i for i, value in enumerate(values) if value is None]
        if missing:
            self._load()
            # These models treat rows independently. The joint adapters run one origin at a time.
            if (self.native in {"chronos2", "timesfm25", "tirex2"} or self.name == "patchtstfm_long"
                    or hasattr(self.predictor, "predict_independent_origins")):
                method = getattr(self.predictor, "predict_independent_origins", self.predictor.predict)
                predictions = method(contexts[missing], horizon)
                for i, prediction in zip(missing, predictions):
                    values[i] = prediction[None]
                    if not self.cache.save(contexts[i:i+1], horizon, values[i]):
                        raise OSError("Could not save reproducible foundation prediction cache.")
            else:
                for i in missing:
                    values[i] = self.predictor.predict(contexts[i:i+1], horizon)
                    if not self.cache.save(contexts[i:i+1], horizon, values[i]):
                        raise OSError("Could not save reproducible foundation prediction cache.")
        return np.concatenate(values)

    def release(self):
        self.predictor = None
        import gc
        gc.collect()

    def __getstate__(self):
        return {**self.__dict__, "predictor": None}
