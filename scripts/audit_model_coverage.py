"""Require real curve/daily outputs for all requested models on both datasets."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.models import COMPARISON_MODELS
from src.utils import resolve_path, write_json


def independent_scores(actual, predicted, domains=None):
    actual, predicted = np.asarray(actual, float).reshape(-1), np.asarray(predicted, float).reshape(-1)
    error = np.abs(actual-predicted)
    nonzero = np.abs(actual) > 1e-8
    wape = 100*error.sum()/np.abs(actual).sum()
    if domains is not None:
        groups = pd.Series(np.asarray(domains))
        wape = float((100*pd.Series(error).groupby(groups).sum()/pd.Series(np.abs(actual)).groupby(groups).sum()).mean())
    return dict(MAPE=float((100*error[nonzero]/np.abs(actual[nonzero])).mean()),
                WAPE=float(wape), RMSE=float(np.sqrt(np.square(error).mean())), MAE=float(error.mean()),
                n=len(actual), mape_n=int(nonzero.sum()), mape_excluded=int((~nonzero).sum()))


def audit_run(folder):
    folder = resolve_path(folder)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Run is not complete: {folder}")
    metrics = pd.read_csv(folder/"metrics.csv")
    panel = manifest.get("protocol") == "google_hdsf_d14_d22"
    checks = []
    for horizon in ("24h", "7d"):
        window_info = json.loads((folder/horizon/"windows.json").read_text(encoding="utf-8"))
        horizon_points = 96 if horizon == "24h" else 672
        for split in ("validation", "test"):
            table = pd.read_parquet(folder/horizon/(split+"_panel.parquet")) if panel else None
            expected_windows = window_info[split]["windows"]
            if panel:
                if len(table) != expected_windows*57*horizon_points or table.duplicated(
                        ["series_id", "origin_day", "horizon_step"]).any():
                    raise ValueError(f"Incomplete or duplicate Google panel: {horizon} {split}")
                if table.series_id.nunique() != 57 or table.origin_day.nunique() != expected_windows:
                    raise ValueError(f"Wrong Google domain/origin coverage: {horizon} {split}")
            reference = None
            for model in COMPARISON_MODELS:
                rows = metrics[(metrics.horizon == horizon) & (metrics.split == split) & (metrics.model == model)]
                valid = set(rows.evaluation) == {"curve", "daily_mean"} and len(rows) == 2
                if valid:
                    valid = bool(np.isfinite(rows[["MAPE", "WAPE", "RMSE"]].to_numpy()).all() and (rows.n > 0).all())
                prediction_path = folder/horizon/model/(split+"_predictions.parquet")
                if panel:
                    prediction = bool(model in table and np.isfinite(table[model].to_numpy()).all())
                    if prediction:
                        selected = table.loc[table.target_valid]
                        daily = selected.groupby(["series_key", "origin_day", "target_day"])[["y_true", model]].mean().reset_index()
                        computed = {"curve": independent_scores(selected.y_true, selected[model], selected.series_key),
                            "daily_mean": independent_scores(daily.y_true, daily[model], daily.series_key)}
                else:
                    prediction = prediction_path.is_file()
                    if prediction:
                        values = pd.read_parquet(prediction_path).sort_values(["origin", "lead"])
                        prediction = bool(len(values) == expected_windows*horizon_points
                            and not values.duplicated(["origin", "lead"]).any()
                            and values.origin.nunique() == expected_windows
                            and np.isfinite(values[["actual", "predicted"]].to_numpy()).all())
                        coordinates = values[["origin", "timestamp", "lead", "actual"]].reset_index(drop=True)
                        if reference is None:
                            reference = coordinates
                        elif not reference.equals(coordinates):
                            raise ValueError(f"Methods use different labels or origins: {horizon} {split} {model}")
                        actual = values.actual.to_numpy(dtype=np.float32).reshape(-1, horizon_points)
                        predicted = values.predicted.to_numpy(dtype=float).reshape(-1, horizon_points)
                        computed = {"curve": independent_scores(actual, predicted),
                            "daily_mean": independent_scores(actual.reshape(len(actual), -1, 96).mean(-1),
                                                             predicted.reshape(len(predicted), -1, 96).mean(-1))}
                recomputed = bool(valid and prediction)
                if recomputed:
                    for _, row in rows.iterrows():
                        expected = computed[row.evaluation]
                        for metric in ("MAPE", "WAPE", "RMSE", "MAE", "n", "mape_n", "mape_excluded"):
                            if not np.isclose(row[metric], expected[metric], rtol=1e-12, atol=1e-10):
                                recomputed = False
                checks.append(dict(model=model, horizon=horizon, split=split,
                    metrics_complete=valid, predictions_present=prediction, metrics_recomputed=recomputed,
                    passed=valid and prediction and recomputed))
    return dict(run=str(folder), data=manifest["data"], passed=all(c["passed"] for c in checks), checks=checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nlr", required=True)
    parser.add_argument("--google", required=True)
    parser.add_argument("--output", default="results/analysis/all-models/coverage.json")
    args = parser.parse_args(argv)
    runs = {name: audit_run(path) for name, path in (("nlr", args.nlr), ("google", args.google))}
    passed = all(run["passed"] for run in runs.values())
    write_json(resolve_path(args.output), dict(status="PASS" if passed else "FAIL", models=list(COMPARISON_MODELS), runs=runs))
    for name, run in runs.items():
        print(f"Data: {name}; model/task/split checks={sum(c['passed'] for c in run['checks'])}/{len(run['checks'])}")
        for check in run["checks"]:
            if not check["passed"]:
                print(f"Missing: {check['model']} {check['horizon']} {check['split']}")
    print("Coverage: "+("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
