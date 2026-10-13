"""Official TiRex-2 and Toto 2.0 adapters with offline checkpoint reuse."""
from contextlib import nullcontext
from pathlib import Path
import os
import sys
from time import perf_counter

import numpy as np

from .config import FOUNDATION_BATCH_SIZE, FOUNDATION_JOBS, HISTORY_DAYS, POINTS_PER_DAY
from .foundation import validate_forecast

NEW_MODEL_NAMES = {"tirex2": "TiRex-2", "toto2": "Toto 2.0 313m"}
NEW_MODEL_IDS = {"tirex2": "NX-AI/TiRex-2", "toto2": "Datadog/Toto-2.0-313m"}
CHECKPOINT_FILES = {"tirex2": ("model-config.yaml", "model.ckpt"),
                    "toto2": ("config.json", "model.safetensors")}


def cached_checkpoint(model_id, cache_dir, filenames):
    from huggingface_hub import try_to_load_from_cache

    files = [try_to_load_from_cache(model_id, name, cache_dir=str(cache_dir)) for name in filenames]
    if not all(isinstance(file, str) for file in files):
        return None
    paths = [Path(file) for file in files]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return None
    return paths[0].parent if len({path.parent for path in paths}) == 1 else None


def tirex_recursive(model, context, horizon, *, batch_size, timeseries_type, torch_module):
    """Roll out <=320-step blocks using predicted medians, never future labels."""
    history = np.asarray(context, dtype=np.float32).copy()
    blocks = []
    max_block = min(int(model.future_len), 320)
    if max_block < 1:
        raise ValueError("TiRex-2 checkpoint has no valid forecast horizon")
    while sum(block.shape[1] for block in blocks) < horizon:
        remaining = horizon - sum(block.shape[1] for block in blocks)
        length = min(max_block, remaining)
        inputs = [timeseries_type(target=torch_module.as_tensor(row.copy()).unsqueeze(0),
                                  past_covariates=None, future_covariates=None) for row in history]
        # Residual blocks in the official package have @torch.compile even when
        # load_model(compile=False). Keep all CPU calls eager without changing weights.
        native = (torch_module.compiler.set_stance("force_eager")
                  if str(getattr(model, "device", "cpu")) == "cpu" else nullcontext())
        with native, torch_module.inference_mode():
            quantiles = model.forecast(inputs, prediction_length=length, output_type="numpy",
                                       batch_size=batch_size)
        if len(quantiles) != len(context) or any(np.asarray(q).shape != (1, 9, length) for q in quantiles):
            raise ValueError("TiRex-2 returned an unexpected quantile shape")
        median = validate_forecast(np.stack([q[0, 4] for q in quantiles]), len(context), length)
        blocks.append(median)
        # Bound both observed and generated context to the project's 14-day window.
        history = np.concatenate([history, median.astype(np.float32)], axis=1)[:, -HISTORY_DAYS * POINTS_PER_DAY:]
    return validate_forecast(np.concatenate(blocks, axis=1), len(context), horizon)


def grouped_toto(model, context, horizon, *, groups, torch_module, device):
    """Only channels in the same site/device group attend to each other."""
    if len(groups) != len(context):
        raise ValueError("Toto groups must match the input series")
    prediction = np.empty((len(context), horizon), dtype=float)
    for group in dict.fromkeys(groups):
        indices = [i for i, label in enumerate(groups) if label == group]
        target = torch_module.as_tensor(context[indices].copy(), dtype=torch_module.float32, device=device).unsqueeze(0)
        inputs = {"target": target, "target_mask": torch_module.ones_like(target, dtype=torch_module.bool),
                  "series_ids": torch_module.zeros((1, len(indices)), dtype=torch_module.long, device=device)}
        with torch_module.inference_mode():
            quantiles = model.forecast(inputs, horizon=horizon, decode_block_size=None, has_missing_values=False)
        if tuple(quantiles.shape) != (9, 1, len(indices), horizon):
            raise ValueError("Toto 2.0 returned an unexpected quantile shape")
        prediction[indices] = quantiles[4, 0].detach().cpu().numpy()
    return validate_forecast(prediction, len(context), horizon)


def load_new_foundation(name, *, cache_dir: Path, batch_size=FOUNDATION_BATCH_SIZE,
                        jobs=FOUNDATION_JOBS, device="cpu", groups=None):
    if name not in NEW_MODEL_NAMES:
        raise ValueError(f"Unknown foundation model: {name}")
    started = perf_counter()
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用，请选择 --device cpu")
        torch.set_num_threads(jobs)
        torch.set_float32_matmul_precision("high")
        # Check dependency imports before downloading large checkpoint files.
        if name == "tirex2":
            from tirex2 import TimeseriesType, load_model
        else:
            os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve() / "matplotlib"))
            os.environ.setdefault("MPLBACKEND", "Agg")
            from toto2 import Toto2Model
        from huggingface_hub import snapshot_download

        snapshot = cached_checkpoint(NEW_MODEL_IDS[name], cache_dir, CHECKPOINT_FILES[name])
        status = "读取本地缓存" if snapshot is not None else "本地缓存缺失，下载模型权重"
        print(f"加载 {NEW_MODEL_NAMES[name]}（{device}）：{status}。", flush=True)
        if snapshot is None:
            snapshot = Path(snapshot_download(NEW_MODEL_IDS[name], cache_dir=str(cache_dir),
                                               allow_patterns=list(CHECKPOINT_FILES[name])))
        if name == "tirex2":
            # Native CPU kernels avoid the Windows C++/CUDA compiler requirement.
            model = load_model(snapshot, device=device, use_flex_attention=False, compile=False)

            class Predictor:
                def predict(self, context, horizon):
                    return tirex_recursive(model, context, horizon, batch_size=batch_size,
                                           timeseries_type=TimeseriesType, torch_module=torch)
        else:
            model = Toto2Model.from_pretrained(str(snapshot), map_location=device,
                                               local_files_only=True).to(device).eval()

            class Predictor:
                def predict(self, context, horizon):
                    labels = groups if groups is not None else list(range(len(context)))
                    return grouped_toto(model, context, horizon, groups=labels, torch_module=torch, device=device)

                def predict_independent_origins(self, contexts, horizon):
                    # Native batch axis: each sample is one origin with one variable.
                    # Do not put rolling origins on the variate-attention axis.
                    target = torch.as_tensor(np.ascontiguousarray(contexts), dtype=torch.float32,
                                             device=device).unsqueeze(1)
                    inputs = {"target": target, "target_mask": torch.ones_like(target, dtype=torch.bool),
                              "series_ids": torch.zeros((len(contexts), 1), dtype=torch.long, device=device)}
                    with torch.inference_mode():
                        quantiles = model.forecast(inputs, horizon=horizon, decode_block_size=None,
                                                   has_missing_values=False)
                    if tuple(quantiles.shape) != (9, len(contexts), 1, horizon):
                        raise ValueError("Toto independent-origin batch returned an unexpected shape")
                    return validate_forecast(quantiles[4, :, 0].detach().cpu().numpy(), len(contexts), horizon)
        print(f"{NEW_MODEL_NAMES[name]} 加载完成，耗时 {perf_counter() - started:.1f} 秒。", flush=True)
        return Predictor()
    except ImportError as error:
        raise ValueError(f"新模型依赖缺失或不兼容：{error}。安装命令："
                         f' & "{sys.executable}" -m pip install -r requirements.txt') from error
