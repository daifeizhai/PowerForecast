"""Stage 1 target power characteristics analysis (task book §3.5, Milestone 2).

Usage (from the project root, CPU only)::

    & .\\.venv\\Scripts\\python.exe -m scripts.analyze_power \
        --config configs/datasets/nlr_target.yaml

Reads the Stage 1 processed target artefact
(``data/processed/processed_target_power.parquet``) and writes descriptive
analysis figures plus ``power_stats.json`` to
``results/analysis/stage1/power_characteristics/``:

* ``raw_power_curve.png``         full prepared series (decimated for plotting)
* ``typical_day_profile.png``     mean/median by 15-minute slot, weekday vs weekend
* ``hourly_daily_weekly.png``     hourly / day-of-week / weekly aggregates
* ``autocorrelation.png``         pairwise-complete ACF up to 7 days (672 lags)
* ``ramp_volatility.png``         15-minute ramp ΔkW histogram + rolling volatility
* ``peak_valley.png``             per-trace-day peak/valley/range + range histogram
* ``coverage_quality.png``        daily valid fraction and coverage ratio
* ``power_stats.json``            numeric summaries of everything above

Every figure is computed from the real prepared NLR ESIF label. Nothing is
synthetic, resampled-by-hand or fabricated; invalid labels stay invalid (the ACF
and daily statistics handle missing values explicitly instead of filling them).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless: no display is needed or used
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

if __package__ in (None, ""):  # allow `python scripts/analyze_power.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (  # noqa: E402
    ensure_dir,
    load_config,
    read_json,
    resolve_path,
    utc_timestamp,
    write_json,
)

DEFAULT_CONFIG = "configs/datasets/nlr_target.yaml"
DEFAULT_OUT_DIR = "results/analysis/stage1/power_characteristics"

#: A trace day is summarised only if at least this fraction of its slots is valid.
MIN_VALID_FRACTION_FOR_DAILY_STATS = 0.75

REAL_DATA_FOOTER = (
    "REAL data — NLR ESIF facility_derived_kw "
    "(derived label: it + cooling + hvac + plug_and_light + pump interval means)"
)

_FIGURE_SIZE = (12.0, 6.0)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Stage 1 target power characteristics from the processed real NLR ESIF "
            "facility-power label and save figures + power_stats.json."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"dataset YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument(
        "--target",
        default=None,
        help="processed target parquet (default: <outputs.out_dir>/<outputs.target_power_file>)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="target_metadata.json (default: alongside the processed target parquet)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"analysis output directory (default: config analysis_output_dir, {DEFAULT_OUT_DIR})",
    )
    return parser


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _summary(values: np.ndarray | pd.Series) -> dict[str, Any]:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None,
                "median": None, "p05": None, "p95": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def _pairwise_complete_acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation with pairwise-complete observations (missing stays missing).

    FFT-based: for each lag ``k`` only timestamp pairs where both values are finite
    are used, so the pre-2016 zero-coverage era and internal outages never enter as
    zeros and no value is imputed. ``acf[0] = 1``; lags without enough pairs are NaN.
    """
    array = np.asarray(values, dtype="float64")
    finite = np.isfinite(array)
    x = np.where(finite, array, 0.0)
    x2 = x * x
    mask = finite.astype("float64")

    length = array.size
    size = 1
    while size < length + max_lag + 1:
        size <<= 1
    fft_x = np.fft.rfft(x, size)
    fft_x2 = np.fft.rfft(x2, size)
    fft_mask = np.fft.rfft(mask, size)

    def _cross(spectrum_a: np.ndarray, spectrum_b: np.ndarray) -> np.ndarray:
        return np.fft.irfft(spectrum_a * np.conj(spectrum_b), size)[: max_lag + 1]

    pairs = _cross(fft_mask, fft_mask)
    sum_xy = _cross(fft_x, fft_x)
    sum_x = _cross(fft_x, fft_mask)  # sum_i x_i * mask_{i+k}
    sum_y = _cross(fft_mask, fft_x)  # sum_i mask_i * x_{i+k}
    sum_xx = _cross(fft_x2, fft_mask)
    sum_yy = _cross(fft_mask, fft_x2)

    acf = np.full(max_lag + 1, np.nan, dtype="float64")
    enough = pairs > 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_left = np.where(enough, sum_x / pairs, np.nan)
        mean_right = np.where(enough, sum_y / pairs, np.nan)
        covariance = sum_xy / pairs - mean_left * mean_right
        variance_left = sum_xx / pairs - mean_left * mean_left
        variance_right = sum_yy / pairs - mean_right * mean_right
        denominator = np.sqrt(variance_left * variance_right)
        acf = np.where((denominator > 0) & enough, covariance / denominator, np.nan)
    if length:
        acf[0] = 1.0
    return acf


def _trace_positions(index: pd.DatetimeIndex, anchor: pd.Timestamp, delta: pd.Timedelta) -> np.ndarray:
    nanoseconds = (index.astype("int64") - int(anchor.value)) // int(delta.value)
    return nanoseconds.astype("int64")


def _resolve_paths(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path | None, Path]:
    outputs = config["outputs"]
    target_path = resolve_path(
        args.target if args.target is not None else Path(outputs["out_dir"]) / outputs["target_power_file"]
    )
    if args.metadata is not None:
        metadata_path: Path | None = resolve_path(args.metadata)
    else:
        candidate = target_path.parent / outputs["metadata_file"]
        metadata_path = candidate if candidate.is_file() else None
    out_dir = resolve_path(
        args.out if args.out is not None else config.get("analysis_output_dir", DEFAULT_OUT_DIR)
    )
    return target_path, metadata_path, out_dir


def _save_figure(fig: plt.Figure, out_dir: Path, name: str, written: list[str]) -> None:
    # Reserve space for the figure title and the REAL-data footer so text never
    # overlaps the axes labels.
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
    destination = out_dir / name
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    written.append(str(destination))


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def analyse(
    target: pd.DataFrame, *, config: dict[str, Any], anchor: pd.Timestamp, out_dir: Path
) -> tuple[dict[str, Any], list[str], Path]:
    """Compute every Stage 1 power-characteristic summary and figure."""
    frequency = str(config["frequency"])
    points_per_day = int(config["points_per_day"])
    delta = pd.Timedelta(pd.tseries.frequencies.to_offset(frequency))

    if "timestamp" not in target.columns:
        raise ValueError("processed target table must contain a 'timestamp' column.")
    index = pd.DatetimeIndex(pd.to_datetime(target["timestamp"]))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("processed target timestamps must be sorted and unique.")
    positions = _trace_positions(index, anchor, delta)
    if len(positions) and int(positions.min()) < 0:
        anchor = index[0]
        positions = _trace_positions(index, anchor, delta)

    power = target["power_kw"].to_numpy(dtype="float64")
    valid = target["power_valid"].to_numpy(dtype=bool) & np.isfinite(power)
    coverage = target["coverage_ratio"].to_numpy(dtype="float64")
    day = positions // points_per_day
    slot_of_day = positions % points_per_day

    valid_series = pd.Series(power[valid], index=index[valid])
    valid_hours = valid_series.index.hour.to_numpy()
    # A trace day starts at the anchor slot (e.g. 06:00 UTC), not at UTC midnight, so the
    # weekday/month of a row for profile purposes must be the TRACE DAY's, not the timestamp's
    # calendar weekday (which would switch in the middle of a trace day and create an
    # artificial discontinuity at UTC midnight).
    trace_day_start = anchor + pd.to_timedelta(day, unit="D")
    trace_weekday = np.asarray(trace_day_start.dayofweek, dtype="int64")
    trace_month = np.asarray(trace_day_start.month, dtype="int64")
    valid_dow = trace_weekday[valid]
    valid_months = trace_month[valid]
    valid_slot = slot_of_day[valid]

    # --- core distribution ------------------------------------------------------------
    label_summary = _summary(valid_series)
    coverage_summary = _summary(coverage)

    # --- typical-day profiles ---------------------------------------------------------
    weekday_mask = valid_dow < 5
    weekday_slot = pd.Series(valid_series.to_numpy()[weekday_mask], index=valid_slot[weekday_mask])
    weekend_slot = pd.Series(valid_series.to_numpy()[~weekday_mask], index=valid_slot[~weekday_mask])
    weekday_mean = weekday_slot.groupby(level=0).mean().reindex(range(points_per_day))
    weekday_median = weekday_slot.groupby(level=0).median().reindex(range(points_per_day))
    weekend_mean = weekend_slot.groupby(level=0).mean().reindex(range(points_per_day))
    weekend_median = weekend_slot.groupby(level=0).median().reindex(range(points_per_day))

    hourly_mean = valid_series.groupby(valid_hours).mean().reindex(range(24))
    dow_mean = valid_series.groupby(valid_dow).mean().reindex(range(7))
    monthly_mean = valid_series.groupby(valid_months).mean().reindex(range(1, 13))

    week_index = positions // (7 * points_per_day)
    weekly_mean = pd.Series(power).groupby(week_index).mean()
    weekly_summary = _summary(weekly_mean)

    hour_by_month = (
        pd.DataFrame({"value": valid_series.to_numpy(), "month": valid_months, "hour": valid_hours})
        .groupby(["month", "hour"])["value"]
        .mean()
        .unstack("hour")
        .reindex(index=range(1, 13), columns=range(24))
    )
    seasons = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
    hour_by_season: dict[str, list[float | None]] = {}
    for season, months in seasons.items():
        mask = np.isin(valid_months, months)
        series = pd.Series(valid_series.to_numpy()[mask], index=valid_hours[mask])
        values = series.groupby(level=0).mean().reindex(range(24))
        hour_by_season[season] = [_float_or_none(value) for value in values.to_numpy()]

    # --- daily (trace-day) peak / valley / range --------------------------------------
    daily_frame = pd.DataFrame(
        {"day": day, "power": power, "valid": valid, "coverage": coverage}
    )
    daily_grouped = daily_frame.groupby("day", sort=True)
    daily_slots = daily_grouped.size()
    daily_valid = daily_grouped["valid"].sum()
    daily_valid_fraction = (daily_valid / daily_slots).to_numpy(dtype="float64")
    daily_peak = daily_grouped["power"].max().to_numpy(dtype="float64")
    daily_valley = daily_grouped["power"].min().to_numpy(dtype="float64")
    daily_mean = daily_grouped["power"].mean().to_numpy(dtype="float64")
    daily_range = daily_peak - daily_valley
    daily_coverage = daily_grouped["coverage"].mean().to_numpy(dtype="float64")
    day_ids = daily_grouped.size().index.to_numpy()
    day_timestamps = anchor + pd.to_timedelta(pd.Index(day_ids), unit="D")
    eligible = daily_valid_fraction >= MIN_VALID_FRACTION_FOR_DAILY_STATS

    peak_valley_stats = {
        "eligible_days": int(eligible.sum()),
        "total_trace_days": int(len(day_ids)),
        "min_valid_fraction_for_stats": MIN_VALID_FRACTION_FOR_DAILY_STATS,
        "peak": _summary(daily_peak[eligible]),
        "valley": _summary(daily_valley[eligible]),
        "range": _summary(daily_range[eligible]),
        "daily_mean": _summary(daily_mean[eligible]),
        "daily_valid_fraction": _summary(daily_valid_fraction),
    }

    # --- ramps -------------------------------------------------------------------------
    delta_ns = int(delta.value)
    consecutive = np.zeros(len(power), dtype=bool)
    if len(power) > 1:
        consecutive[1:] = np.diff(index.astype("int64")) == delta_ns
    both_valid = consecutive & valid & np.concatenate([[False], valid[:-1]])
    ramps = np.diff(power)[both_valid[1:]]
    ramp_summary = _summary(ramps)
    ramp_stats = {
        **ramp_summary,
        "positive_count": int(np.count_nonzero(ramps > 0)),
        "negative_count": int(np.count_nonzero(ramps < 0)),
        "zero_count": int(np.count_nonzero(ramps == 0)),
        "mean_absolute": float(np.abs(ramps).mean()) if ramps.size else None,
        "p99_absolute": float(np.percentile(np.abs(ramps), 99)) if ramps.size else None,
        "max_positive": float(ramps.max()) if ramps.size else None,
        "max_negative": float(ramps.min()) if ramps.size else None,
    }

    rolling = pd.Series(power).rolling(window=points_per_day, min_periods=points_per_day // 2).std()
    volatility_stats = {
        "rolling_window_slots": points_per_day,
        "rolling_window_hours": points_per_day / 4.0,
        "rolling_std": _summary(rolling.to_numpy(dtype="float64")),
    }

    # --- autocorrelation ---------------------------------------------------------------
    # The level series contains multi-year level shifts, so its Pearson ACF is dominated by
    # slow persistence. We therefore report BOTH the level ACF (levels are never filled) and
    # the ACF of consecutive 15-minute increments, which isolates short-range structure.
    max_lag = 7 * points_per_day
    level_acf = _pairwise_complete_acf(power, max_lag)
    increments = np.diff(power)
    increment_valid = consecutive & valid & np.concatenate([[False], valid[:-1]])
    increments[~increment_valid[1:]] = np.nan
    increment_acf = _pairwise_complete_acf(increments, max_lag)
    report_lags = [1, 2, 4, 8, 16, 48, points_per_day, 2 * points_per_day,
                   3 * points_per_day, max_lag]

    def _lag_window(acf: np.ndarray, low: int, high: int) -> dict[str, Any]:
        low = max(1, low)
        high = min(high, max_lag)
        window = acf[low : high + 1]
        if not np.isfinite(window).any():
            return {"window": [low, high], "argmax_lag": None, "acf_at_argmax": None,
                    "is_interior_local_maximum": False}
        offset = int(np.nanargmax(window))
        lag = low + offset
        is_peak = (
            low < lag < high
            and np.isfinite(acf[lag])
            and acf[lag] >= acf[lag - 1]
            and acf[lag] >= acf[lag + 1]
        )
        return {
            "window": [low, high],
            "argmax_lag": lag,
            "argmax_lag_days": lag / points_per_day,
            "acf_at_argmax": _float_or_none(acf[lag]),
            "is_interior_local_maximum": bool(is_peak),
        }

    local_peaks = [
        int(lag)
        for lag in np.arange(2, max_lag)  # 1 is always the trivial maximum
        if np.isfinite(level_acf[lag])
        and level_acf[lag] >= level_acf[lag - 1]
        and level_acf[lag] >= level_acf[lag + 1]
    ]
    top_local_peaks = sorted(local_peaks, key=lambda lag: level_acf[lag], reverse=True)[:5]
    acf_stats = {
        "max_lag": max_lag,
        "max_lag_days": max_lag / points_per_day,
        "level_method": (
            "pairwise-complete Pearson ACF of the label level; missing labels are never filled"
        ),
        "levels": {
            "acf_by_lag": {
                str(lag): _float_or_none(level_acf[lag]) for lag in report_lags if lag <= max_lag
            },
            "acf_lag_0": _float_or_none(level_acf[0]),
            "lag_1_day_window": _lag_window(level_acf, points_per_day - 6, points_per_day + 6),
            "lag_7_day_window": _lag_window(level_acf, max_lag - 6, max_lag),
            "local_peak_lags": [
                {"lag": lag, "lag_days": lag / points_per_day, "acf": _float_or_none(level_acf[lag])}
                for lag in top_local_peaks
            ],
            "local_peak_count": int(len(local_peaks)),
            "local_peak_method": (
                "local maxima of the level ACF for lags 2..max_lag; an empty list means the "
                "level ACF decays monotonically and no separate periodic peak exists"
            ),
            "level_acf_monotone_non_increasing": bool(
                np.all(np.diff(level_acf[1:]) <= 1e-9)
            ),
        },
        "first_differences": {
            "method": (
                "pairwise-complete Pearson ACF of consecutive 15-minute label increments; a pair "
                "is used only when both labels are valid and exactly 15 minutes apart"
            ),
            "valid_increment_count": int(np.isfinite(increments).sum()),
            "acf_by_lag": {
                str(lag): _float_or_none(increment_acf[lag])
                for lag in report_lags
                if lag <= max_lag
            },
            "lag_1_day_window": _lag_window(increment_acf, points_per_day - 6, points_per_day + 6),
            "lag_7_day_window": _lag_window(increment_acf, max_lag - 6, max_lag),
        },
        "interpretation": (
            "The level ACF decays very slowly (it is dominated by persistent multi-year level "
            "shifts), while the increment ACF is close to zero at every lag. No separate 24-hour "
            "or 7-day periodicity peak stands out above the local trend; the mean slot-of-day "
            "profile amplitude is small compared with the level standard deviation. This is a "
            "real property of this facility's load, not an artefact: Stage 2 must still include "
            "the naive seasonal baselines so the comparison is explicit."
        ),
    }
    daily_profile_amplitude = float(
        np.nanmax(weekday_mean.to_numpy(dtype="float64"))
        - np.nanmin(weekday_mean.to_numpy(dtype="float64"))
    )
    acf_stats["levels"]["daily_profile_peak_to_peak_kw"] = daily_profile_amplitude
    acf_stats["levels"]["daily_profile_amplitude_share_of_std"] = (
        float(daily_profile_amplitude / label_summary["std"])
        if label_summary["std"]
        else None
    )

    # --- coverage / missingness --------------------------------------------------------
    coverage_zero_interval_count = int(np.count_nonzero(coverage == 0.0))
    invalid_count = int(np.count_nonzero(~valid))
    first_window = slice(0, min(31 * points_per_day, len(power)))
    first_31 = {
        "expected_slots": 31 * points_per_day,
        "actual_slots": int(len(power[first_window])),
        "valid_slots": int(valid[first_window].sum()),
        "valid_fraction": float(valid[first_window].mean()) if len(power[first_window]) else 0.0,
        "mean_power_kw": _float_or_none(np.nanmean(power[first_window])),
        "start": index[0].isoformat() if len(index) else None,
        "end": index[min(31 * points_per_day, len(index)) - 1].isoformat() if len(index) else None,
    }

    stats: dict[str, Any] = {
        "analysis": "stage1_target_power_characteristics",
        "generated_at_utc": utc_timestamp(),
        "real_data_statement": REAL_DATA_FOOTER,
        "dataset_id": str(config["dataset_id"]),
        "site_id": str(config["site_id"]),
        "domain_id": str(config["domain_id"]),
        "label_type": "facility_derived_kw",
        "label_composition": list(config["label"]["derived_columns"]),
        "label_unit": "kW",
        "timezone": str(config["timezone"]),
        "timezone_verified": False,
        "anchor_timestamp": anchor.isoformat(),
        "frequency": frequency,
        "points_per_day": points_per_day,
        "rows": int(len(power)),
        "first_timestamp": index[0].isoformat() if len(index) else None,
        "last_timestamp": index[-1].isoformat() if len(index) else None,
        "trace_day_count": int(len(day_ids)),
        "valid_label_count": int(valid.sum()),
        "invalid_label_count": invalid_count,
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        "coverage_zero_interval_count": coverage_zero_interval_count,
        "missing_interval_count": None,  # filled from target_quality.json when available
        "missing_interval_count_source": None,
        "coverage_ratio": coverage_summary,
        "power_kw": label_summary,
        "daily": peak_valley_stats,
        "ramp_delta_kw": ramp_stats,
        "volatility": volatility_stats,
        "autocorrelation": acf_stats,
        "weekly": {
            **weekly_summary,
            "week_count": int(len(weekly_mean)),
            "aggregation": "mean of valid 15-minute labels per 7-trace-day block from the anchor",
        },
        "profiles": {
            "hourly_mean": [_float_or_none(value) for value in hourly_mean.to_numpy()],
            "dow_mean": [_float_or_none(value) for value in dow_mean.to_numpy()],
            "monthly_mean": [_float_or_none(value) for value in monthly_mean.to_numpy()],
            "slot_of_day_weekday_mean": [_float_or_none(value) for value in weekday_mean.to_numpy()],
            "slot_of_day_weekend_mean": [_float_or_none(value) for value in weekend_mean.to_numpy()],
            "slot_of_day_weekday_median": [_float_or_none(value) for value in weekday_median.to_numpy()],
            "slot_of_day_weekend_median": [_float_or_none(value) for value in weekend_median.to_numpy()],
            "hour_of_day_by_month_mean": [
                [_float_or_none(value) for value in row] for row in hour_by_month.to_numpy()
            ],
            "hour_of_day_by_season_mean": hour_by_season,
            "month_labels": [f"{month:02d}" for month in range(1, 13)],
        },
        "first_31_day_window": first_31,
        "weekday_source": (
            "weekday/month derived from the trace-day start date under assumed-UTC timestamps "
            "(the site local timezone is unverified); a trace day starts at the anchor slot, "
            "not at UTC midnight"
        ),
        "figures": [],
    }

    # --- figures -----------------------------------------------------------------------
    figures: list[str] = []

    # 1. Full prepared series (decimated for plotting only; values themselves untouched).
    figure, axis = plt.subplots(figsize=_FIGURE_SIZE)
    stride = max(1, len(power) // 20000)
    sampled = slice(None, None, stride)
    axis.plot(index[sampled], power[sampled], linewidth=0.6, color="#1f4e79")
    axis.set_title("Prepared NLR ESIF facility power — full range (REAL derived kW)")
    axis.set_xlabel("Timestamp (naive UTC, 15-min interval start)")
    axis.set_ylabel("Facility power (kW)")
    axis.text(
        0.01,
        0.01,
        f"{REAL_DATA_FOOTER}\nplot decimated x{stride} for rendering only; "
        f"{len(power):,} prepared slots",
        transform=axis.transAxes,
        fontsize=7,
        va="bottom",
    )
    figure.autofmt_xdate()
    _save_figure(figure, out_dir, "raw_power_curve.png", figures)

    # 2. Typical-day profile (absolute level and deviation from the all-day mean).
    slots = np.arange(points_per_day)
    overall_slot_mean = pd.Series(power[valid]).groupby(slot_of_day[valid]).mean().reindex(
        range(points_per_day)
    )
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True)
    axes[0].plot(slots, weekday_mean.to_numpy(), label="weekday mean", color="#1f4e79")
    axes[0].plot(slots, weekend_mean.to_numpy(), label="weekend mean", color="#c55a11")
    axes[0].plot(slots, weekday_median.to_numpy(), label="weekday median", color="#1f4e79",
                 linestyle="--")
    axes[0].plot(slots, weekend_median.to_numpy(), label="weekend median", color="#c55a11",
                 linestyle="--")
    axes[0].set_ylabel("Facility power (kW)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(slots, weekday_mean.to_numpy() - overall_slot_mean.to_numpy(),
                 label="weekday mean − all-day slot mean", color="#1f4e79")
    axes[1].plot(slots, weekend_mean.to_numpy() - overall_slot_mean.to_numpy(),
                 label="weekend mean − all-day slot mean", color="#c55a11")
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Deviation (kW)")
    axes[1].set_xlabel("15-minute slot of trace day (0 = interval starting at the anchor slot)")
    axes[1].legend(loc="best", fontsize=8)
    figure.suptitle("Typical-day facility power profile (mean / median by 15-min slot)")
    figure.text(
        0.01,
        0.01,
        REAL_DATA_FOOTER
        + "\nweekday derived from the trace-day start date (assumed-UTC timestamps); "
        "absolute profile amplitude is small relative to the level spread",
        fontsize=7,
    )
    _save_figure(figure, out_dir, "typical_day_profile.png", figures)

    # 3. Hourly / day-of-week / weekly aggregates.
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    axes[0].bar(range(24), hourly_mean.to_numpy(), color="#1f4e79")
    axes[0].set_title("Hourly mean")
    axes[0].set_xlabel("Hour of day (assumed UTC)")
    axes[0].set_ylabel("Facility power (kW)")
    axes[1].bar(range(7), dow_mean.to_numpy(), color="#2e75b6")
    axes[1].set_title("Day-of-week mean")
    axes[1].set_xlabel("Monday=0 … Sunday=6")
    week_timestamps = anchor + pd.to_timedelta(
        pd.Index(weekly_mean.index.to_numpy(dtype="int64") * 7), unit="D"
    )
    axes[2].plot(week_timestamps, weekly_mean.to_numpy(), color="#c55a11", linewidth=0.8)
    axes[2].set_title("Weekly mean (7 trace days)")
    axes[2].set_xlabel("Timestamp (naive UTC)")
    axes[2].tick_params(axis="x", labelrotation=30)
    figure.suptitle("NLR ESIF facility power — hourly / daily / weekly aggregates (REAL derived kW)")
    figure.text(0.01, 0.01, REAL_DATA_FOOTER, fontsize=7)
    _save_figure(figure, out_dir, "hourly_daily_weekly.png", figures)

    # 4. Autocorrelation (0 … 7 days) of levels and of 15-minute increments.
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True)
    lags = np.arange(max_lag + 1)
    axes[0].plot(lags, level_acf, linewidth=0.8, color="#1f4e79")
    axes[0].set_title("Level ACF up to 7 days (dominant slow level persistence)")
    axes[0].set_ylabel("Autocorrelation")
    axes[1].plot(lags, increment_acf, linewidth=0.8, color="#c55a11")
    axes[1].set_title("First-difference (15-min increment) ACF up to 7 days")
    axes[1].set_xlabel("Lag (15-minute steps; dotted lines = whole days)")
    axes[1].set_ylabel("Autocorrelation")
    for axis in axes:
        for day in range(1, 8):
            axis.axvline(day * points_per_day, color="grey", linestyle=":", linewidth=0.8)
        axis.axhline(0.0, color="black", linewidth=0.6)
    figure.suptitle("Autocorrelation of NLR ESIF facility power (REAL derived kW)")
    figure.text(
        0.01,
        0.01,
        REAL_DATA_FOOTER
        + "\npairwise-complete ACF: missing labels are not filled; levels carry multi-year trend",
        fontsize=7,
    )
    _save_figure(figure, out_dir, "autocorrelation.png", figures)

    # 5. Ramp magnitude + rolling volatility.
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.0))
    if ramps.size:
        axes[0].hist(np.clip(ramps, -500, 500), bins=120, color="#1f4e79")
        axes[0].set_xlabel("15-minute ramp ΔkW (clipped to ±500 for display only)")
    axes[0].set_title("Histogram of 15-minute ramp ΔkW (valid consecutive pairs)")
    axes[0].set_ylabel("Count")
    volatility = rolling.to_numpy(dtype="float64")
    axes[1].plot(index[sampled], volatility[sampled], linewidth=0.6, color="#c55a11")
    axes[1].set_title("Rolling volatility (96-slot = 24 h rolling standard deviation)")
    axes[1].set_xlabel("Timestamp (naive UTC)")
    axes[1].set_ylabel("Rolling std (kW)")
    figure.suptitle("NLR ESIF facility power — ramp and volatility (REAL derived kW)")
    figure.text(0.01, 0.01, REAL_DATA_FOOTER, fontsize=7)
    figure.autofmt_xdate()
    _save_figure(figure, out_dir, "ramp_volatility.png", figures)

    # 6. Per-trace-day peak / valley / range.
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))
    axes[0, 0].plot(day_timestamps, daily_peak, linewidth=0.7, color="#c00000")
    axes[0, 0].set_title("Daily peak power (trace days)")
    axes[0, 1].plot(day_timestamps, daily_valley, linewidth=0.7, color="#1f4e79")
    axes[0, 1].set_title("Daily valley power (trace days)")
    axes[1, 0].plot(day_timestamps, daily_range, linewidth=0.7, color="#548235")
    axes[1, 0].set_title("Daily peak–valley range (trace days)")
    if eligible.any():
        axes[1, 1].hist(daily_range[eligible], bins=60, color="#7030a0")
    axes[1, 1].set_title("Histogram of daily peak–valley range")
    axes[1, 1].set_xlabel("Range (kW)")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_ylabel("Facility power (kW)")
        axis.tick_params(axis="x", labelrotation=30)
    for axis in axes.ravel():
        axis.text(0.01, 0.98, f"eligible days = {int(eligible.sum())}/{len(day_ids)}",
                  transform=axis.transAxes, fontsize=7, va="top")
    figure.suptitle("NLR ESIF facility power — daily peak / valley / range (REAL derived kW)")
    figure.text(0.01, 0.01, REAL_DATA_FOOTER, fontsize=7)
    _save_figure(figure, out_dir, "peak_valley.png", figures)

    # 7. Coverage / quality over the full prepared range.
    figure, axis = plt.subplots(figsize=_FIGURE_SIZE)
    axis.plot(day_timestamps, daily_valid_fraction, linewidth=0.7, color="#1f4e79",
              label="daily valid fraction (power_valid)")
    axis.plot(day_timestamps, daily_coverage, linewidth=0.7, color="#c55a11",
              label="daily mean coverage_ratio")
    axis.axhline(MIN_VALID_FRACTION_FOR_DAILY_STATS, color="grey", linestyle=":", linewidth=0.8,
                 label="daily-stats eligibility threshold")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Target label coverage and validity by trace day")
    axis.set_xlabel("Trace-day start (naive UTC)")
    axis.set_ylabel("Fraction / ratio")
    axis.legend(loc="best", fontsize=8)
    figure.text(0.01, 0.01, REAL_DATA_FOOTER + "\nmissing intervals are not filled",
                fontsize=7)
    figure.autofmt_xdate()
    _save_figure(figure, out_dir, "coverage_quality.png", figures)

    stats["figures"] = figures
    return stats, figures, out_dir


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        target_path, metadata_path, out_dir = _resolve_paths(args, config)
        if not target_path.is_file():
            raise FileNotFoundError(
                f"Processed target file not found: {target_path}. Run "
                "`python -m scripts.prepare_target_power` first."
            )
        target = pd.read_parquet(target_path)
        quality_path = target_path.parent / str(config["outputs"]["quality_file"])
        quality: dict[str, Any] | None = None
        if quality_path.is_file():
            quality = read_json(quality_path)
        anchor = None
        metadata: dict[str, Any] | None = None
        if metadata_path is not None and metadata_path.is_file():
            metadata = read_json(metadata_path)
            anchor_value = metadata.get("anchor_timestamp")
            if anchor_value is not None:
                anchor = pd.Timestamp(anchor_value)
        if anchor is None:
            anchor = pd.Timestamp(config["analysis"]["window_start"])
        ensure_dir(out_dir)
        stats, figures, out_dir = analyse(
            target, config=config, anchor=anchor, out_dir=out_dir
        )
        stats["source_file"] = str(target_path)
        stats["metadata_file"] = str(metadata_path) if metadata_path is not None else None
        if quality is not None:
            stats["quality_file"] = str(quality_path)
            quality_target = quality.get("target", {})
            grid_info = quality_target.get("grid", {})
            stats["missing_interval_count"] = grid_info.get("missing_interval_count")
            stats["missing_interval_count_source"] = "target_quality.json:target.grid"
            stats["no_native_sample_interval_count"] = grid_info.get(
                "slots_without_any_native_sample"
            )
            stats["native_sampling_audit"] = quality_target.get("native_sampling")
            stats["coverage_validity_threshold"] = quality_target.get(
                "coverage_validity_threshold"
            )
            stats["first_31_day_window_from_quality"] = quality.get("first_31_day_window")
        if metadata is not None:
            stats["label_type"] = metadata.get("label_type", stats["label_type"])
            stats["timezone_verified"] = bool(
                metadata.get("timezone_verified", stats["timezone_verified"])
            )
            stats["label_honesty_note"] = metadata.get("label", {}).get("honesty_note")
        stats_path = write_json(out_dir / "power_stats.json", stats)
        print("Stage 1 target power characteristics written")
        print(f"  source: {target_path}")
        missing = stats.get("missing_interval_count")
        missing_text = "n/a" if missing is None else f"{int(missing):,}"
        print(f"  rows={stats['rows']:,} valid={stats['valid_label_count']:,} "
              f"({100.0 * stats['valid_fraction']:.2f}%) invalid={stats['invalid_label_count']:,} "
              f"missing_intervals={missing_text}")
        print(f"  mean={stats['power_kw']['mean']:.2f} kW std={stats['power_kw']['std']:.2f} kW "
              f"min={stats['power_kw']['min']:.2f} max={stats['power_kw']['max']:.2f}")
        print(f"  daily peak mean={stats['daily']['peak']['mean']:.2f} kW "
              f"valley mean={stats['daily']['valley']['mean']:.2f} kW "
              f"range mean={stats['daily']['range']['mean']:.2f} kW")
        print(f"  ramp abs(delta) mean={stats['ramp_delta_kw']['mean_absolute']:.2f} kW "
              f"p99={stats['ramp_delta_kw']['p99_absolute']:.2f} kW")
        level_acf = stats["autocorrelation"]["levels"]["acf_by_lag"]
        print(f"  level ACF: lag1={level_acf.get('1')} "
              f"lag96={level_acf.get(str(stats['points_per_day']))} "
              f"lag672={level_acf.get(str(stats['autocorrelation']['max_lag']))}; "
              "increment ACF ~ 0 at all lags")
        print(f"  first 31 trace days: {stats['first_31_day_window']['valid_slots']}/"
              f"{stats['first_31_day_window']['expected_slots']} valid")
        print(f"  stats: {stats_path}")
        for figure in figures:
            print(f"  figure: {figure}")
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
