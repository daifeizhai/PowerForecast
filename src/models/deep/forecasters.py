"""Small CPU-compatible architectures for framework experiments.

These are configurable implementations, not claims of paper reproduction.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.data.splitting import ForecastInputs, WindowBatch
from src.models.base import BaseForecaster


class TorchForecaster(BaseForecaster):
    def __init__(self, name: str, seed: int = 42, hidden_size: int = 32, epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001, kernel_size: int = 25):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(f"{name} requires torch. Install requirements-optional.txt or pip install torch.") from exc
        if min(hidden_size, epochs, batch_size, kernel_size) <= 0 or learning_rate <= 0:
            raise ValueError("Deep model parameters must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("DLinear smoothing kernel_size must be odd.")
        self.torch, self.name, self.seed = torch, name, seed
        self.hidden_size, self.epochs, self.batch_size = hidden_size, epochs, batch_size
        self.learning_rate, self.kernel_size = learning_rate, kernel_size
        self.network = None

    def _build(self, context: int, features: int, horizon: int):
        torch, name, hidden, kernel = self.torch, self.name, self.hidden_size, self.kernel_size
        nn = torch.nn

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                if name == "lstm":
                    self.encoder = nn.LSTM(features, hidden, batch_first=True)
                    self.head = nn.Linear(hidden, horizon)
                elif name == "tcn":
                    self.convs = nn.ModuleList([
                        nn.Conv1d(features if i == 0 else hidden, hidden, 3, dilation=2**i)
                        for i in range(3)
                    ])
                    self.head = nn.Linear(hidden, horizon)
                else:
                    self.smoothing = nn.AvgPool1d(kernel, stride=1)
                    self.trend = nn.Linear(context, horizon)
                    self.seasonal = nn.Linear(context, horizon)

            def forward(self, x):
                if name == "lstm":
                    encoded, _ = self.encoder(x)
                    return self.head(encoded[:, -1])
                if name == "tcn":
                    encoded = x.transpose(1, 2)
                    for i, conv in enumerate(self.convs):
                        encoded = torch.relu(conv(nn.functional.pad(encoded, (2 * 2**i, 0))))
                    return self.head(encoded[:, :, -1])
                y = x[:, :, 0]
                padded = nn.functional.pad(y.unsqueeze(1), ((kernel-1)//2, (kernel-1)//2), mode="replicate")
                trend = self.smoothing(padded).squeeze(1)
                return self.trend(trend) + self.seasonal(y - trend)

        return Network()

    def fit(self, batch: WindowBatch) -> "TorchForecaster":
        torch = self.torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        history = batch.inputs.history
        # Every normalization statistic is fitted on training data only.
        self.mean = history.mean(axis=(0, 1), keepdims=True)
        self.std = np.maximum(history.std(axis=(0, 1), keepdims=True), 1e-6)
        self.target_mean, self.target_std = float(self.mean[0, 0, 0]), float(self.std[0, 0, 0])
        x = torch.from_numpy(((history - self.mean) / self.std).astype(np.float32))
        y = torch.from_numpy(((batch.targets - self.target_mean) / self.target_std).astype(np.float32))
        self.network = self._build(history.shape[1], history.shape[2], batch.targets.shape[1])
        self.shape = (history.shape[1], history.shape[2], batch.targets.shape[1])
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.network.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(self.network(xb), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, inputs: ForecastInputs) -> np.ndarray:
        if self.network is None:
            raise RuntimeError(f"Fit {self.name} before predicting.")
        self.network.eval()
        predictions = []
        with self.torch.no_grad():
            for start in range(0, len(inputs), self.batch_size):
                x = ((inputs.history[start:start+self.batch_size] - self.mean) / self.std).astype(np.float32)
                predictions.append(self.network(self.torch.from_numpy(x)).numpy())
        return np.concatenate(predictions) * self.target_std + self.target_mean

    def save(self, path: str | Path) -> Path:
        if self.network is None:
            raise RuntimeError(f"Fit {self.name} before saving.")
        path = Path(path).with_suffix(".pt")
        self.torch.save({
            "name": self.name, "shape": self.shape, "seed": self.seed,
            "hidden_size": self.hidden_size, "kernel_size": self.kernel_size,
            "state_dict": self.network.state_dict(),
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "target_mean": self.target_mean, "target_std": self.target_std,
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TorchForecaster":
        import torch
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(state["name"], seed=state["seed"], hidden_size=state["hidden_size"],
                    kernel_size=state["kernel_size"])
        model.shape = tuple(state["shape"])
        model.network = model._build(*model.shape)
        model.network.load_state_dict(state["state_dict"])
        model.mean = np.asarray(state["mean"], dtype=np.float32)
        model.std = np.asarray(state["std"], dtype=np.float32)
        model.target_mean, model.target_std = state["target_mean"], state["target_std"]
        return model
