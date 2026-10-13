"""Local zero-shot adapters for the official Chronos-2 and TimesFM 2.5 models."""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .config import FOUNDATION_BATCH_SIZE, FOUNDATION_JOBS

MODEL_NAMES = {"chronos2": "Chronos-2", "timesfm25": "TimesFM 2.5"}
MODEL_IDS = {"chronos2": "amazon/chronos-2", "timesfm25": "google/timesfm-2.5-200m-pytorch"}


def cached_model_directory(model_id: str, cache_dir: Path) -> Path | None:
    """Find a complete cached snapshot without making any network requests."""
    from huggingface_hub import try_to_load_from_cache

    files = [try_to_load_from_cache(model_id, filename, cache_dir=str(cache_dir))
             for filename in ["config.json", "model.safetensors"]]
    if not all(isinstance(filename, str) for filename in files):
        return None
    paths = [Path(filename) for filename in files]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return None
    if paths[0].parent != paths[1].parent:
        return None
    # Preserve the snapshot directory rather than resolving symlinks into blobs.
    return paths[0].parent


def validate_forecast(values: np.ndarray, n_series: int, horizon: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (n_series, horizon) or not np.isfinite(values).all():
        raise ValueError(f"Model must return {n_series} x {horizon} finite predictions; got {values.shape}")
    return values


def load_foundation(name: str, *, cache_dir: Path, batch_size: int = FOUNDATION_BATCH_SIZE,
                    jobs: int = FOUNDATION_JOBS, device: str = "cpu"):
    """Import heavy dependencies only when the experiment command is used."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown foundation model: {name}")
    started = perf_counter()
    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用，请选择 --device cpu")
        torch.set_num_threads(jobs)
        torch.set_float32_matmul_precision("high")
        model_id = MODEL_IDS[name]
        snapshot = cached_model_directory(model_id, cache_dir)
        source = str(snapshot) if snapshot is not None else model_id
        load_options = {"cache_dir": str(cache_dir), "force_download": False,
                        "local_files_only": snapshot is not None}
        status = "读取本地缓存" if snapshot is not None else "本地缓存缺失，下载模型权重"
        print(f"加载 {MODEL_NAMES[name]}（{device}）：{status}。", flush=True)
        if name == "chronos2":
            from chronos import Chronos2Pipeline

            pipeline = Chronos2Pipeline.from_pretrained(
                source, device_map=device, dtype=torch.float32, **load_options,
            )
            print(f"{MODEL_NAMES[name]} 加载完成，耗时 {perf_counter() - started:.1f} 秒。", flush=True)

            class ChronosForecaster:
                def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
                    with torch.inference_mode():
                        _, points = pipeline.predict_quantiles(
                            inputs=[row.copy() for row in context], prediction_length=horizon,
                            quantile_levels=[0.5], batch_size=batch_size,
                            context_length=context.shape[1], cross_learning=False,
                        )
                    if len(points) != len(context):
                        raise ValueError("Chronos-2 returned a different number of series")
                    values = np.stack([point.detach().cpu().numpy()[0] for point in points])
                    return validate_forecast(values, len(context), horizon)

            return ChronosForecaster()

        import timesfm

        if snapshot is not None:
            from safetensors.torch import load_file

            # Avoid random initialization and copying a second full set of weights.
            with torch.device("meta"):
                model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
            model.model.load_state_dict(
                load_file(str(snapshot / "model.safetensors"), device=device), strict=True, assign=True,
            )
            model.model.eval()
        else:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                source, torch_compile=False, **load_options,
            )
        # TimesFM chooses CUDA automatically; honor an explicit CPU selection.
        model.model.device = torch.device(device)
        model.model.device_count = 1
        model.model.to(model.model.device)
        print(f"{MODEL_NAMES[name]} 加载完成，耗时 {perf_counter() - started:.1f} 秒。", flush=True)

        class TimesFMForecaster:
            configuration = None

            def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
                configuration = (context.shape[1], horizon)
                if configuration != self.configuration:
                    model.compile(timesfm.ForecastConfig(
                        max_context=context.shape[1], max_horizon=horizon,
                        normalize_inputs=True, per_core_batch_size=batch_size,
                        use_continuous_quantile_head=True, force_flip_invariance=True,
                        infer_is_positive=True, fix_quantile_crossing=True,
                    ))
                    self.configuration = configuration
                with torch.inference_mode():
                    points, _ = model.forecast(horizon=horizon, inputs=[row.copy() for row in context])
                return validate_forecast(points, len(context), horizon)

        return TimesFMForecaster()
    except ImportError as error:
        raise ValueError(
            f"时序模型依赖缺失或不兼容：{error}。请用当前解释器安装依赖："
            f' & "{sys.executable}" -m pip install -r requirements.txt'
        ) from error
