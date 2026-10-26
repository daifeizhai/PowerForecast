"""Stage-1 source-characteristics analysis for the REAL M100 source panel.

Usage (after ``scripts.prepare_source_features``)::

    & .\\.venv\\Scripts\\python.exe -m scripts.analyze_sources ^
        --config configs/datasets/sources.yaml

Produces ``results/analysis/stage1/source_characteristics/``:

* ``m100_feature_distributions.png`` -- observed-value histogram grid per curated metric;
* ``m100_availability_heatmap.png`` -- observed fraction per feature x node/domain;
* ``m100_power_and_temp_timeseries.png`` -- one real node: node total power + thermal telemetry;
* ``m100_periodicity.png`` -- slot-of-day profile and pairwise-complete autocorrelation;
* ``m100_15min_feasibility.png`` -- native interval (delta seconds) and samples per 15-min slot;
* ``ucc_audit.png`` -- UCC bounded audit: native interval + column missingness;
* ``source_stats.json`` -- row/domain/feature counts, observed fractions, distributions,
  native-frequency and aggregation verdicts, alignment counts;
* ``alignment.json`` -- full field-alignment routes/reasons from ``FieldRegistry.align``.

Every figure is computed from the real processed panel (or the real raw timestamp audit)
and labelled as REAL M100 node telemetry; no curves are fabricated and gaps are kept as
NaN (never interpolated).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/analyze_sources.py` too
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data import source_features as sf
from src.data.loaders import assert_not_raw_path
from src.utils import ensure_dir, resolve_path, utc_timestamp, write_json

DEFAULT_CONFIG = sf.SOURCES_CONFIG
DEFAULT_PANEL = f"{sf.DEFAULT_PROCESSED_DIR}/{sf.PANEL_FILENAME}"
DEFAULT_OUT = "results/analysis/stage1/source_characteristics"
DEFAULT_METADATA = f"{sf.DEFAULT_PROCESSED_DIR}/{sf.METADATA_FILENAME}"

FIGURE_NAMES = (
    "m100_feature_distributions.png",
    "m100_availability_heatmap.png",
    "m100_power_and_temp_timeseries.png",
    "m100_periodicity.png",
    "m100_15min_feasibility.png",
    "ucc_audit.png",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _feature_units(panel: pd.DataFrame) -> dict[str, str]:
    mapping = (
        panel[["feature_name", "unit"]]
        .drop_duplicates()
        .set_index("feature_name")["unit"]
        .astype(str)
        .to_dict()
    )
    return {str(key): value for key, value in mapping.items()}


def _feature_names(panel: pd.DataFrame) -> list[str]:
    if isinstance(panel["feature_name"].dtype, pd.CategoricalDtype):
        return [str(item) for item in panel["feature_name"].cat.categories]
    return sorted(str(item) for item in panel["feature_name"].unique())


def _domain_names(panel: pd.DataFrame) -> list[str]:
    if isinstance(panel["domain_id"].dtype, pd.CategoricalDtype):
        return [str(item) for item in panel["domain_id"].cat.categories]
    return sorted(str(item) for item in panel["domain_id"].unique())


def _load_panel(path: str | Path) -> pd.DataFrame:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Source panel {resolved} not found. Run `python -m scripts.prepare_source_features` "
            "first (or pass --rebuild)."
        )
    panel = pd.read_parquet(resolved)
    if list(panel.columns) != list(sf.PANEL_COLUMNS):
        raise ValueError(
            f"Panel {resolved} does not follow the logical contract {list(sf.PANEL_COLUMNS)}."
        )
    return panel


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def plot_feature_distributions(
    panel: pd.DataFrame, config: dict[str, Any], destination: Path
) -> Path:
    plt = _pyplot()
    features = _feature_names(panel)
    units = _feature_units(panel)
    sample_limit = int(config["analysis"]["distribution_sample_rows_per_feature"])
    figure, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes_flat = axes.reshape(-1)
    for axis, name in zip(axes_flat, features):
        values = panel.loc[panel["feature_name"] == name, "feature_value"].dropna().to_numpy()
        if len(values) > sample_limit:
            stride = max(1, len(values) // sample_limit)
            values = values[::stride]
        axis.hist(values, bins=60, color="#31688e", alpha=0.85)
        axis.set_title(f"{name} [{units.get(name, '?')}]", fontsize=9)
        axis.set_ylabel("count")
        axis.tick_params(labelsize=7)
    for axis in axes_flat[len(features):]:
        axis.axis("off")
    figure.suptitle(
        "M100 node hardware telemetry -- REAL observed feature distributions (all nodes, gaps excluded)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def plot_availability_heatmap(
    panel: pd.DataFrame, config: dict[str, Any], destination: Path
) -> Path:
    plt = _pyplot()
    fractions = (
        panel.groupby(["feature_name", "domain_id"], observed=True)["observed_mask"]
        .mean()
        .unstack("domain_id")
    )
    fractions = fractions.reindex(
        index=_feature_names(panel), columns=_domain_names(panel)
    )
    figure, axis = plt.subplots(figsize=(max(9.0, 0.55 * fractions.shape[1] + 4), 6.5))
    image = axis.imshow(fractions.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(fractions.shape[1]))
    axis.set_xticklabels(fractions.columns, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(fractions.shape[0]))
    axis.set_yticklabels(fractions.index, fontsize=8)
    for row in range(fractions.shape[0]):
        for column in range(fractions.shape[1]):
            value = fractions.iat[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value < 0.6 else "black",
                )
    figure.colorbar(image, ax=axis, label="observed fraction")
    axis.set_title(
        "M100 REAL node telemetry: observed fraction per feature x node/domain\n"
        "(missing = telemetry gap; never zero-filled)"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def plot_power_and_temperature_timeseries(
    panel: pd.DataFrame, config: dict[str, Any], destination: Path
) -> Path:
    plt = _pyplot()
    analysis = config["analysis"]
    domain = str(analysis["example_domain"])
    start = pd.Timestamp(str(analysis["timeseries_start"]))
    end = pd.Timestamp(str(analysis["timeseries_end"]))
    features = ["node_total_power_w", "node_ambient_temp_c", "gpu0_core_temp_c", "gpu1_core_temp_c"]
    subset = panel[
        (panel["domain_id"] == domain)
        & (panel["timestamp"] >= start)
        & (panel["timestamp"] <= end)
        & (panel["feature_name"].isin(features))
    ]
    wide = subset.pivot(index="timestamp", columns="feature_name", values="feature_value")
    wide = wide.sort_index()
    figure, axis_power = plt.subplots(figsize=(14, 5.5))
    axis_power.plot(
        wide.index, wide.get("node_total_power_w"), color="#1f77b4", lw=1.0, label="node total power (W)"
    )
    axis_power.set_ylabel("node total power [W]", color="#1f77b4")
    axis_power.tick_params(axis="y", labelcolor="#1f77b4")
    axis_power.grid(alpha=0.25)
    axis_temperature = axis_power.twinx()
    colors = {
        "node_ambient_temp_c": "#2ca02c",
        "gpu0_core_temp_c": "#d62728",
        "gpu1_core_temp_c": "#ff7f0e",
    }
    for name, color in colors.items():
        if name in wide:
            axis_temperature.plot(
                wide.index, wide[name], color=color, lw=0.8, alpha=0.9, label=name
            )
    axis_temperature.set_ylabel("temperature [degC]")
    lines_left, labels_left = axis_power.get_legend_handles_labels()
    lines_right, labels_right = axis_temperature.get_legend_handles_labels()
    axis_power.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right", fontsize=8)
    axis_power.set_title(
        f"REAL M100 node telemetry, domain={domain}: node power + thermal sensors\n"
        "gaps are preserved as NaN (no interpolation, no fabricated values)"
    )
    if len(wide.index):
        # Keep the full requested window visible even where the node telemetry is absent:
        # matplotlib autoscaling would otherwise crop at the last finite value.
        axis_power.set_xlim(wide.index.min(), wide.index.max())
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def _pairwise_autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    mask = np.isfinite(values)
    result = np.full(max_lag + 1, np.nan)
    for lag in range(max_lag + 1):
        if lag == 0:
            pair_mask = mask
            left = values[pair_mask]
            right = left
        else:
            pair_mask = mask[:-lag] & mask[lag:]
            left = values[:-lag][pair_mask]
            right = values[lag:][pair_mask]
        if len(left) < 30:
            continue
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        denominator = np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
        if denominator > 0 and np.isfinite(denominator):
            result[lag] = float(np.sum(left_centered * right_centered) / denominator)
    return result


def plot_periodicity(panel: pd.DataFrame, config: dict[str, Any], destination: Path) -> Path:
    plt = _pyplot()
    analysis = config["analysis"]
    domain = str(analysis["example_domain"])
    start = pd.Timestamp(str(analysis["periodicity_start"]))
    end = pd.Timestamp(str(analysis["periodicity_end"]))
    features = [str(item) for item in analysis["periodicity_features"]]
    max_lag = int(analysis["periodicity_max_lag"])
    subset = panel[
        (panel["domain_id"] == domain)
        & (panel["timestamp"] >= start)
        & (panel["timestamp"] <= end)
        & (panel["feature_name"].isin(features))
    ]
    wide = subset.pivot(index="timestamp", columns="feature_name", values="feature_value").sort_index()

    figure, (axis_profile, axis_acf) = plt.subplots(1, 2, figsize=(15, 5.2))
    slots = wide.index.hour * 4 + wide.index.minute // 15
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for color, name in zip(colors, features):
        if name not in wide:
            continue
        series = wide[name]
        profile = series.groupby(slots).mean()
        profile = profile.reindex(range(96))
        values = profile.to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 9 or np.nanstd(values) == 0:
            continue
        normalized = (values - np.nanmean(values)) / np.nanstd(values)
        axis_profile.plot(np.arange(96) / 4.0, normalized, color=color, lw=1.1, label=name)
    axis_profile.set_xlabel("hour of day (UTC)")
    axis_profile.set_ylabel("slot-of-day mean, z-scored per feature")
    axis_profile.set_title(f"Slot-of-day profile, domain={domain}, window={start.date()}..{end.date()}")
    axis_profile.grid(alpha=0.25)
    axis_profile.legend(fontsize=8)

    for color, name in zip(colors, features):
        if name not in wide:
            continue
        acf = _pairwise_autocorrelation(wide[name].to_numpy(dtype=float), max_lag)
        axis_acf.plot(np.arange(len(acf)) / 96.0, acf, color=color, lw=1.0, label=name)
    for day in (1, 2, 7):
        axis_acf.axvline(day, color="gray", ls=":", lw=0.8)
    axis_acf.set_xlabel("lag [days] (96 points/day)")
    axis_acf.set_ylabel("pairwise-complete autocorrelation")
    axis_acf.set_title(f"Autocorrelation (gaps pairwise-excluded), domain={domain}")
    axis_acf.set_xlim(0, max(7.0, max_lag / 96.0))
    axis_acf.grid(alpha=0.25)
    axis_acf.legend(fontsize=8)
    figure.suptitle("M100 REAL node telemetry: periodicity and volatility", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def plot_15min_feasibility(
    panel: pd.DataFrame, config: dict[str, Any], destination: Path
) -> tuple[Path, dict[str, Any]]:
    plt = _pyplot()
    audit = sf.audit_m100_timestamps(config)
    global_deltas = audit["global_delta_seconds_counts"]
    labels = [int(key) for key in global_deltas]
    counts = [global_deltas[key] for key in global_deltas]
    figure, (axis_delta, axis_samples) = plt.subplots(1, 2, figsize=(14, 5.2))
    axis_delta.bar([str(label) for label in labels], counts, color="#31688e")
    axis_delta.set_yscale("log")
    axis_delta.set_xlabel("consecutive timestamp delta [seconds]")
    axis_delta.set_ylabel("count (log scale)")
    axis_delta.set_title(
        "M100 native interval audit: delta seconds\n"
        f"verdict={audit['verdict']} (target 15min = 900 s)"
    )
    axis_delta.tick_params(axis="x", rotation=60, labelsize=7)

    slot_counts = audit["global_samples_per_15min_slot_counts"]
    keys = sorted(slot_counts, key=lambda item: int(item))
    axis_samples.bar([str(key) for key in keys], [slot_counts[key] for key in keys], color="#35b779")
    axis_samples.set_yscale("log")
    axis_samples.set_xlabel("raw samples per 15-minute slot")
    axis_samples.set_ylabel("slot count (log scale)")
    axis_samples.set_title(
        "Samples per 15-minute slot (native preserved)\n0 = preserved telemetry gap, never filled"
    )
    figure.suptitle("M100 REAL node telemetry: 15-minute aggregation feasibility", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination, audit


def plot_ucc_audit(
    config: dict[str, Any], destination: Path
) -> tuple[Path, dict[str, Any]]:
    plt = _pyplot()
    audit = sf.build_ucc_audit(config)
    primary = audit["tables"]["primary_file"]
    secondary = audit["tables"]["secondary_file"]
    delta_counts = primary["native_delta_seconds_counts"]
    delta_keys = [int(key) for key in delta_counts]
    # dataset1 has no missing values inside the bounded subset; dataset2 does, so the
    # missingness panel shows dataset2 (and reports dataset1's zero explicitly).
    missing = secondary["missing_fraction_by_column"]
    missing_columns = sorted(missing, key=lambda key: float(missing[key]), reverse=True)
    figure, (axis_delta, axis_missing, axis_series) = plt.subplots(1, 3, figsize=(16, 5.2))
    axis_delta.bar(
        [str(key) for key in delta_keys],
        [delta_counts[str(key)] for key in delta_keys],
        color="#440154",
    )
    axis_delta.set_yscale("log")
    axis_delta.set_xlabel("consecutive timestamp delta [seconds]")
    axis_delta.set_ylabel("count (log scale)")
    axis_delta.set_title(
        f"UCC dataset1 native deltas (bounded {audit['bounded_audit_rows_per_file']} rows)\n"
        f"native={audit['native_frequency']}, aggregation={audit['aggregation']['status']}"
    )
    axis_delta.tick_params(axis="x", rotation=60, labelsize=7)

    axis_missing.bar(
        range(len(missing_columns)),
        [float(missing[column]) for column in missing_columns],
        color="#fde725",
    )
    axis_missing.set_xticks(range(len(missing_columns)))
    axis_missing.set_xticklabels(missing_columns, rotation=75, fontsize=7)
    axis_missing.set_ylabel("missing fraction")
    dataset1_missing = float(
        max(primary["missing_fraction_by_column"].values(), default=0.0)
    )
    axis_missing.set_title(
        "UCC dataset2 missing fraction by column (bounded subset)\n"
        f"dataset1 max missing fraction in subset = {dataset1_missing:.3f}"
    )

    series_rows = primary["series_rows_in_subset"]
    names = list(series_rows)
    axis_series.bar(range(len(names)), [series_rows[name] for name in names], color="#21918c")
    axis_series.set_xticks(range(len(names)))
    axis_series.set_xticklabels(names, rotation=75, fontsize=6)
    axis_series.set_ylabel("rows in bounded subset")
    axis_series.set_title(f"UCC series in bounded subset (n={len(names)})")
    figure.suptitle(
        "UCC 2023 bounded audit: native 10-minute sampling preserved (no fake 15-minute product)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination, audit


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def build_source_stats(
    panel: pd.DataFrame,
    config: dict[str, Any],
    *,
    panel_path: str | Path = DEFAULT_PANEL,
    metadata_path: str | Path = DEFAULT_METADATA,
    figure_paths: list[str] | None = None,
    native_audit: dict[str, Any] | None = None,
    ucc_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features = _feature_names(panel)
    domains = _domain_names(panel)
    observed_overall = (
        panel.groupby("feature_name", observed=True)["observed_mask"].mean().to_dict()
    )
    observed_by_domain = (
        panel.groupby(["domain_id", "feature_name"], observed=True)["observed_mask"]
        .mean()
        .to_dict()
    )
    quality = sf.build_source_quality(panel, config)
    alignment = sf.build_alignment_results(config)
    registry_payload = sf.build_feature_registry_payload(config)
    m100_native = native_audit or sf.audit_m100_timestamps(config)
    ucc = ucc_audit or sf.build_ucc_audit(config)
    ucc_primary = ucc["tables"]["primary_file"]
    stats: dict[str, Any] = {
        "generated_at_utc": utc_timestamp(),
        "generated_by": "scripts.analyze_sources",
        "panel": {
            "path": str(resolve_path(panel_path)),
            "rows": int(len(panel)),
            "columns": list(sf.PANEL_COLUMNS),
            "domains": domains,
            "domain_count": len(domains),
            "features": features,
            "feature_count": len(features),
            "contains_label_columns": False,
        },
        "native_frequency": {
            "m100_exadata": {
                "verdict": m100_native["verdict"],
                "declared_frequency": m100_native["declared_frequency"],
                "target_frequency": m100_native["target_frequency"],
                "aggregation_status": m100_native["aggregation_status"],
                "per_domain": {
                    domain: {
                        "raw_rows": item["raw_rows"],
                        "native_interval_seconds": item["native_interval_seconds"],
                        "gap_count": item["gap_count"],
                        "max_gap_seconds": item["max_gap_seconds"],
                        "duplicate_timestamps": item["duplicate_timestamps"],
                        "observed_15min_slots": item["observed_15min_slots"],
                        "expected_15min_slots": item["expected_15min_slots"],
                    }
                    for domain, item in m100_native["domains"].items()
                },
            },
            "ucc2023": {
                "native_frequency": ucc["native_frequency"],
                "target_frequency": ucc["target_frequency"],
                "aggregation_status": ucc["aggregation"]["status"],
                "aggregation_verdict": ucc["aggregation"]["verdict"],
                "fifteen_minute_product_created": ucc["aggregation"]["fifteen_minute_product_created"],
                "bounded_audit_rows_per_file": ucc["bounded_audit_rows_per_file"],
                "series_in_bounded_subset": ucc_primary["series_in_subset"],
                "native_delta_seconds_counts": ucc_primary["native_delta_seconds_counts"],
                "missing_fraction_by_column": ucc_primary["missing_fraction_by_column"],
            },
        },
        "observed": {
            "overall_fraction_by_feature": {
                name: float(value) for name, value in observed_overall.items()
            },
            "mean_observed_fraction": float(np.mean(list(observed_overall.values()))),
            "missing_fraction_overall": float(1.0 - np.mean(list(observed_overall.values()))),
            "by_domain": {
                domain: {
                    name: float(observed_by_domain[(domain, name)])
                    for name in features
                    if (domain, name) in observed_by_domain
                }
                for domain in domains
            },
        },
        "distributions": quality["features"],
        "distributions_summary": quality["distribution_summary"],
        "aggregation_verdicts": {
            "m100_exadata": {
                "verdict": m100_native["verdict"],
                "note": "native 15-minute sampling confirmed; no resampling performed",
            },
            "ucc2023": {
                "status": ucc["aggregation"]["status"],
                "verdict": ucc["aggregation"]["verdict"],
                "note": "native 10-minute sampling preserved; no 15-minute product fabricated",
            },
            "alibaba_gpu_v2020": {
                "status": "not_ingested",
                "verdict": "candidate_only",
                "note": "headerless machine-metric copy + event tables; no continuous series fabricated",
            },
        },
        "field_alignment": {
            key: {
                "source_id": result.source_id,
                "target_id": result.target_id,
                "status": result.status.value,
                "is_transferable": result.is_transferable,
                "shared_features": list(result.shared_features),
                "source_only_features": list(result.source_only_features),
                "target_only_features": list(result.target_only_features),
                "incomparable_features": list(result.incomparable_features),
                "target_only_local_weight": result.target_only_local_weight,
                "global_fusion_weight": result.global_fusion_weight,
                "blocked_reasons": list(result.blocked_reasons),
            }
            for key, result in alignment.items()
        },
        "field_alignment_status_counts": registry_payload["alignment_status_counts"],
        "field_registry": {
            "path": str(config["paths"]["feature_registry"]),
            "sources": list(registry_payload["sources"]),
            "fusion": registry_payload["fusion"],
        },
        "training_approval": registry_payload["approval"],
        "metadata_path": str(resolve_path(metadata_path)),
        "figures": list(figure_paths or []),
    }
    return stats


def build_alignment_payload(config: dict[str, Any]) -> dict[str, Any]:
    registry_payload = sf.build_feature_registry_payload(config)
    return {
        "generated_at_utc": utc_timestamp(),
        "generated_by": "scripts.analyze_sources",
        "fusion": registry_payload["fusion"],
        "alignment_status_counts": registry_payload["alignment_status_counts"],
        "alignment": registry_payload["alignment"],
        "validation": registry_payload["approval"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the Stage-1 REAL M100 source panel and UCC bounded audit; write the "
            "source-characteristics figures and JSON statistics."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"source config (default: {DEFAULT_CONFIG})")
    parser.add_argument("--panel", default=DEFAULT_PANEL, help=f"processed panel (default: {DEFAULT_PANEL})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"analysis output dir (default: {DEFAULT_OUT})")
    parser.add_argument("--metadata", default=DEFAULT_METADATA, help="source metadata JSON path")
    parser.add_argument("--skip-ucc", action="store_true", help="skip the bounded UCC audit figure")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the panel if the parquet is missing")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = sf.load_source_config(args.config)
        destination = ensure_dir(assert_not_raw_path(args.out))
        try:
            panel = _load_panel(args.panel)
        except FileNotFoundError:
            if not args.rebuild:
                raise
            panel, _ = sf.build_source_panel(config)
        print(f"Analyzing REAL M100 source panel: {len(panel)} rows, "
              f"{panel['domain_id'].nunique()} domains, {panel['feature_name'].nunique()} features")

        written: list[str] = []
        written.append(str(plot_feature_distributions(panel, config, destination / FIGURE_NAMES[0])))
        print(f"  wrote {FIGURE_NAMES[0]}")
        written.append(str(plot_availability_heatmap(panel, config, destination / FIGURE_NAMES[1])))
        print(f"  wrote {FIGURE_NAMES[1]}")
        written.append(
            str(plot_power_and_temperature_timeseries(panel, config, destination / FIGURE_NAMES[2]))
        )
        print(f"  wrote {FIGURE_NAMES[2]}")
        written.append(str(plot_periodicity(panel, config, destination / FIGURE_NAMES[3])))
        print(f"  wrote {FIGURE_NAMES[3]}")
        feasibility_path, native_audit = plot_15min_feasibility(
            panel, config, destination / FIGURE_NAMES[4]
        )
        written.append(str(feasibility_path))
        print(f"  wrote {FIGURE_NAMES[4]}")

        ucc_audit: dict[str, Any] | None = None
        if args.skip_ucc:
            ucc_audit = sf.build_ucc_audit(config)
        else:
            ucc_path, ucc_audit = plot_ucc_audit(config, destination / FIGURE_NAMES[5])
            written.append(str(ucc_path))
            print(f"  wrote {FIGURE_NAMES[5]}")

        stats = build_source_stats(
            panel,
            config,
            panel_path=args.panel,
            metadata_path=args.metadata,
            figure_paths=[Path(item).name for item in written],
            native_audit=native_audit,
            ucc_audit=ucc_audit,
        )
        stats_path = write_json(destination / "source_stats.json", stats)
        print(f"  wrote {Path(stats_path).name}")
        alignment_path = write_json(destination / "alignment.json", build_alignment_payload(config))
        print(f"  wrote {Path(alignment_path).name}")

        print(f"Output directory: {destination}")
        print(
            "  alignment counts: "
            + "; ".join(
                f"{key}={item['shared']}/{item['source_only']}/{item['target_only']}/{item['incomparable']} "
                f"({item['status']})"
                for key, item in stats["field_alignment_status_counts"].items()
            )
        )
        print(
            "  (shared/source_only/target_only/incomparable; M100 vs UCC direction: UCC is the "
            "source side)"
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
