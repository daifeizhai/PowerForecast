"""Fixed-cutoff supervised windows over separate Google power domains.

Datetime values are coordinates for the public interface only. Known features
use trace-relative day/slot phases, never fabricated real weekdays or weather.
"""
import numpy as np
import pandas as pd

from src.data.splitting import ForecastInputs, WindowBatch


def panel_windows(data, origins, context_days, horizon):
    context = context_days*96
    flat = data.target.reshape(data.n_series, -1)
    histories, futures, labels, times, metadata, origin_times = [], [], [], [], [], []
    base = pd.Timestamp("2000-01-01", tz="UTC")

    def phases(points):
        day = points/96
        return np.stack([np.sin(2*np.pi*day), np.cos(2*np.pi*day),
                         np.sin(2*np.pi*day/7), np.cos(2*np.pi*day/7)], axis=-1).astype(np.float32)

    for origin in origins:
        position = int(origin)*96
        if position < context or position+horizon > flat.shape[1]:
            raise ValueError(f"Invalid Google context/horizon at D{origin}.")
        past_known = phases(np.arange(position-context, position))
        future = phases(np.arange(position, position+horizon))
        target_times = pd.date_range(base+pd.Timedelta(minutes=15*position), periods=horizon, freq="15min").to_numpy()
        for series_id in range(data.n_series):
            histories.append(np.column_stack([flat[series_id, position-context:position], past_known]))
            futures.append(future)
            labels.append(flat[series_id, position:position+horizon])
            times.append(target_times)
            origin_times.append(base+pd.Timedelta(days=int(origin)))
            metadata.append((series_id, int(origin)))
    if not labels:
        raise ValueError("No Google supervised windows before the declared cutoff.")
    batch = WindowBatch(ForecastInputs(np.asarray(histories, np.float32), np.asarray(futures, np.float32),
        pd.DatetimeIndex(origin_times), np.asarray(times)), np.asarray(labels, np.float32))
    return batch, np.asarray(metadata, dtype=int)


def training_panel(data, cutoff, context_days, horizon):
    origins = range(context_days, cutoff-horizon//96+1)
    batch, metadata = panel_windows(data, origins, context_days, horizon)
    if (metadata[:, 1]+horizon//96 > cutoff).any():
        raise AssertionError("Training labels cross the fixed Google cutoff.")
    return batch, metadata


def assign_predictions(frame, name, predictions, metadata):
    if predictions.shape[0] != len(metadata) or not np.isfinite(predictions).all():
        raise ValueError(f"Invalid panel predictions for {name}.")
    lookup = {(int(series), int(origin)): i for i, (series, origin) in enumerate(metadata)}
    rows = np.asarray([lookup[(int(s), int(o))] for s, o in zip(frame.series_id, frame.origin_day)])
    leads = frame.horizon_step.to_numpy(dtype=int)-1
    frame[name] = predictions[rows, leads]
