"""CPU PatchTST-FM-r2 inference on observed 15-minute context only."""
from hashlib import sha256
import importlib
import importlib.util
import importlib.machinery
from pathlib import Path
import sys
import types

import numpy as np

from .foundation import cached_model_directory, validate_forecast
from .prediction_cache import PredictionCache

MODEL_ID = "ibm-granite/granite-timeseries-patchtst-fm-r2"
THREADS = 16


def checkpoint(cache_dir):
    root = Path(cache_dir)
    return cached_model_directory(MODEL_ID, root) or cached_model_directory(MODEL_ID, root/"hub")


def model_directory():
    spec = importlib.util.find_spec("tsfm_public")
    if spec is None:
        raise RuntimeError("Install granite-tsfm to run HDSF PatchTST-FM")
    return Path(next(iter(spec.submodule_search_locations)))/"models"/"patchtst_fm"


def model_class():
    # Load the unmodified model package under a private namespace. The toolkit's
    # top-level initializer imports training datasets that inference never uses.
    # This avoids fake datasets modules and changes to the shared environment.
    name = "_hdsf_patchtst_fm"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(model_directory())]
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    return importlib.import_module(name+".modeling_patchtst_fm").PatchTSTFMForPrediction


def prediction_cache(cache_dir):
    sources = {p.name: sha256(p.read_bytes()).hexdigest() for p in model_directory().glob("*.py")}
    return PredictionCache(Path(cache_dir), "hdsf_patchtstfm_long", model_id=MODEL_ID,
                           batch_size=57, jobs=THREADS, device="cpu",
                           package_names=["numpy", "torch", "transformers", "granite-tsfm", "einops"],
                           adapter_file=Path(__file__), checkpoint_directory=checkpoint(cache_dir),
                           extra={"context_days": 21, "mask": "all_observed_after_causal_imputation",
                                  "model_sources": sources})


def load_predictor(*, cache_dir):
    import torch
    torch.set_num_threads(THREADS)
    snapshot = checkpoint(cache_dir)
    model = model_class().from_pretrained(str(snapshot) if snapshot else MODEL_ID,
                                          local_files_only=snapshot is not None)
    model.to("cpu").eval()

    class Predictor:
        def predict(self, context, horizon):
            values = torch.tensor(np.ascontiguousarray(context, dtype=np.float32))
            with torch.no_grad():
                output = model(past_values=values, past_observed_mask=torch.ones_like(values, dtype=torch.bool),
                               prediction_length=horizon)
            prediction = output.prediction_outputs[..., 0].cpu().numpy().astype(np.float32)
            return validate_forecast(prediction, len(context), horizon)
    return Predictor()
