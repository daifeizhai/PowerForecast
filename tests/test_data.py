from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.adapters import adapt_frame
from src.data.cleaning import clean_frame
from src.data.features import make_features
from src.data.pipeline import prepare_dataset
from src.data.resampling import resample_frame
from src.data.splitting import build_windows, chronological_splits


def small_frame(points=120):
    index = pd.date_range("2024-01-01", periods=points, freq="15min", tz="UTC")
    return pd.DataFrame({"load": np.arange(points, dtype=float)+1, "weather": np.arange(points, dtype=float)}, index=index)


def test_adapter_maps_units_and_duplicates():
    raw = pd.DataFrame({"ts": ["2024-01-01", "2024-01-01"], "temp": [32.0, 50.0]})
    result = adapt_frame(raw, {"timestamp": "ts", "columns": {"temperature_c": "temp"}, "scales": {"temperature_c": 5/9}, "offsets": {"temperature_c": -32*5/9}}, "UTC")
    assert len(result) == 1
    assert result.iloc[0, 0] == pytest.approx(5.0)
    assert str(result.index.tz) == "UTC"


def test_adapter_drops_sentinel_before_resampling():
    raw = pd.DataFrame({"ts": ["1970-01-01", "2024-01-01"], "load": [0, 1]})
    result = adapt_frame(raw, {"timestamp": "ts", "columns": {"load": "load"}, "minimum_timestamp": "2015-01-01"}, "UTC")
    assert len(result) == 1
    assert result.index[0].year == 2024


def test_microsecond_sources_join_on_complete_quarter_hour_grid():
    frames = []
    for offset, column in [(0, "power"), (1, "weather")]:
        timestamps = pd.date_range("2024-01-01", periods=6, freq="15min").as_unit("us")[offset:]
        raw = pd.DataFrame({"ts": timestamps, "value": np.ones(len(timestamps))})
        adapted = adapt_frame(raw, {"timestamp": "ts", "columns": {column: "value"}}, "UTC")
        assert adapted.index.unit == "ns"
        frames.append(resample_frame(adapted))
    merged = pd.concat(frames, axis=1)
    assert len(merged) == 6
    assert merged.index.equals(pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC"))


def test_cleaning_never_backfills_or_fills_labels():
    frame = small_frame(6)
    frame["weather"] = [np.nan, 2, np.nan, np.nan, np.nan, 100]
    frame.loc[frame.index[2], "load"] = np.nan
    cleaned = clean_frame(frame, "load", {"ffill_limit": 2, "bounds": {"weather": [0, 80]}})
    assert np.isnan(cleaned.weather.iloc[0])
    assert cleaned.weather.iloc[2:4].tolist() == [2, 2]
    assert np.isnan(cleaned.weather.iloc[4])
    assert np.isnan(cleaned.weather.iloc[5])
    assert np.isnan(cleaned.load.iloc[2])


def test_resampling_keeps_missing_intervals():
    index = pd.to_datetime(["2024-01-01 00:00Z", "2024-01-01 00:14Z", "2024-01-01 00:45Z"])
    result = resample_frame(pd.DataFrame({"load": [2, 4, 9]}, index=index))
    assert result.load.tolist()[0] == 3
    assert result.iloc[1:3].load.isna().all()
    assert len(result) == 4


def test_split_labels_stay_inside_boundaries_and_history_precedes_origin():
    frame = small_frame()
    history, known = make_features(frame, "load", ["weather"], "UTC")
    splits = chronological_splits(len(frame), 0.6, 0.2)
    for bounds in splits.values():
        batch = build_windows(history, known, frame.load, bounds, context=8, horizon=4, stride=4)
        assert pd.Timestamp(batch.inputs.target_times[0, 0]) >= frame.index[bounds[0]]
        assert pd.Timestamp(batch.inputs.target_times[-1, -1]) <= frame.index[bounds[1]-1]
        for i, origin in enumerate(batch.inputs.origins):
            position = frame.index.get_loc(origin)
            np.testing.assert_array_equal(batch.inputs.history[i, :, 0], frame.load.iloc[position-8:position])
        assert not hasattr(batch.inputs, "targets")


def test_future_observed_covariates_do_not_enter_forecast_inputs():
    frame = small_frame(40)
    history, known = make_features(frame, "load", ["weather"], "UTC")
    before = build_windows(history, known, frame.load, (16, 20), 8, 4, 4)
    frame.loc[frame.index[16:], "weather"] = 99999
    changed_history, changed_known = make_features(frame, "load", ["weather"], "UTC")
    after = build_windows(changed_history, changed_known, frame.load, (16, 20), 8, 4, 4)
    np.testing.assert_array_equal(before.inputs.history, after.inputs.history)
    np.testing.assert_array_equal(before.inputs.future_known, after.inputs.future_known)


def test_windows_skip_missing_labels():
    frame = small_frame(40)
    frame.loc[frame.index[16], "load"] = np.nan
    history, known = make_features(frame, "load", [], "UTC")
    batch = build_windows(history, known, frame.load, (16, 28), 4, 4, 4)
    assert batch.skipped_windows == 2  # label at origin 16; history at origin 20
    assert len(batch) == 1
    assert batch.inputs.origins[0] == frame.index[24]


def test_not_enough_data_has_actionable_error():
    frame = small_frame(20)
    history, known = make_features(frame, "load", [], "UTC")
    with pytest.raises(ValueError, match="No valid windows"):
        build_windows(history, known, frame.load, (0, 20), 16, 8, 1)


def toy_config(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    path = raw_dir / "power.csv"
    pd.DataFrame({"ts": ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:30"], "it": [10, 20, 30], "aux": [2, np.nan, -100]}).to_csv(path, index=False)
    return {
        "name": "toy", "timezone": "UTC", "frequency": "15min", "target": "total",
        "processed_path": str(tmp_path / "processed" / "toy.parquet"),
        "sources": {"power": {"path": str(path), "timestamp": "ts", "columns": {"it": "it", "aux": "aux"}}},
        "derived_columns": {"total": {"sum": ["it", "aux"]}},
        "cleaning": {"ffill_limit": 4, "bounds": {"it": [0, None], "aux": [0, None]}},
    }


def test_pipeline_preserves_raw_and_does_not_impute_derived_labels(tmp_path):
    config = toy_config(tmp_path)
    raw_path = tmp_path / "raw" / "power.csv"
    original = raw_path.read_bytes()
    frame, report = prepare_dataset(config)
    assert raw_path.read_bytes() == original
    assert frame.total.iloc[0] == 12
    assert frame.total.iloc[1:].isna().all()
    assert frame.aux.tolist() == [2, 2, 2]
    assert report["quality"]["missing_fraction"]["total"] == pytest.approx(2/3)
    assert (tmp_path / "processed" / "toy.metadata.json").is_file()


def test_pipeline_refuses_to_write_into_raw_directory(tmp_path):
    config = toy_config(tmp_path)
    with pytest.raises(ValueError, match="raw source directory"):
        prepare_dataset(config, tmp_path / "raw" / "prepared.parquet")
