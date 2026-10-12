from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(table: pd.DataFrame, output: str | Path, title: str, unit: str = "kW") -> None:
    first = table[table["origin"] == table["origin"].iloc[0]]
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    ax.plot(first["timestamp"], first["actual"], label="Observed", linewidth=1.5)
    ax.plot(first["timestamp"], first["predicted"], label="Forecast", linewidth=1.3)
    ax.set(title=title, xlabel="Time (UTC)", ylabel=f"Load ({unit})")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.savefig(output, dpi=160)
    plt.close(fig)
