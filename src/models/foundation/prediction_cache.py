"""Cache model predictions by observed input, checkpoint and inference configuration."""
from contextlib import suppress
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .foundation import MODEL_IDS, cached_model_directory, validate_forecast


class PredictionCache:
    def __init__(self, cache_dir: Path, name: str, *, batch_size: int, jobs: int, device: str,
                 model_id: str | None = None, filenames: tuple[str, ...] | None = None,
                 package_names: list[str] | None = None, adapter_file: Path | None = None,
                 extra: dict | None = None, checkpoint_directory: Path | None = None):
        self.directory = cache_dir / "predictions" / name
        self.fingerprint = None
        model_id = model_id or MODEL_IDS[name]
        filenames = filenames or ("config.json", "model.safetensors")
        if checkpoint_directory is not None:
            snapshot = Path(checkpoint_directory)
        elif filenames == ("config.json", "model.safetensors"):
            snapshot = cached_model_directory(model_id, cache_dir)
        else:
            from .new_foundation import cached_checkpoint
            snapshot = cached_checkpoint(model_id, cache_dir, filenames)
        if snapshot is None:
            return
        packages = package_names or ["numpy", "torch", "safetensors", "transformers",
                                    "chronos-forecasting" if name == "chronos2" else "timesfm"]
        versions = {}
        for package in packages:
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        files = {}
        for filename in filenames:
            stat = (snapshot / filename).stat()
            files[filename] = [stat.st_size, stat.st_mtime_ns]
        identity = {
            "schema": 1, "model": model_id, "revision": snapshot.name,
            "files": files, "versions": versions, "batch_size": batch_size,
            "jobs": jobs, "device": device,
            "adapter": sha256((adapter_file or Path(__file__).with_name("foundation.py")).read_bytes()).hexdigest(),
        }
        if extra is not None:
            identity["options"] = extra
        self.fingerprint = json.dumps(identity, sort_keys=True).encode()

    def path(self, context: np.ndarray, horizon: int) -> Path | None:
        if self.fingerprint is None:
            return None
        context = np.ascontiguousarray(context)
        digest = sha256(self.fingerprint)
        digest.update(json.dumps([context.dtype.str, list(context.shape), horizon]).encode())
        digest.update(context.tobytes())
        return self.directory / f"{digest.hexdigest()}.npy"

    def load(self, context: np.ndarray, horizon: int) -> np.ndarray | None:
        path = self.path(context, horizon)
        if path is None:
            return None
        try:
            return validate_forecast(np.load(path, allow_pickle=False), len(context), horizon)
        except (OSError, ValueError, TypeError, EOFError):
            # An incomplete or invalid cache must be recomputed, never used for scoring.
            return None

    def save(self, context: np.ndarray, horizon: int, prediction: np.ndarray) -> bool:
        path = self.path(context, horizon)
        if path is None:
            return False
        prediction = validate_forecast(prediction, len(context), horizon)
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                np.save(output, prediction, allow_pickle=False)
            os.replace(temporary, path)
            return True
        except OSError:
            return False
        finally:
            if temporary is not None:
                # A locked optional cache must not turn successful inference into a failure.
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
