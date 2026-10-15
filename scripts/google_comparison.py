"""Additional fixed-D14/D22 controls; the historical HDSF is left intact."""
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import joblib
import numpy as np

from src.data.google_panel import panel_windows, training_panel, assign_predictions
from src.models import COMPARISON_MODELS, FOUNDATION_NAMES, create_model
from src.utils import load_config, write_json


def fit_sarima_domain(series_id, batch, params, seed):
    """Process worker: fit and prepare a domain's frozen Kalman-gain plan."""
    model = create_model("sarima", params, seed).fit(batch)
    model.predict(batch.inputs)
    return int(series_id), model


class PanelSarima:
    """One independent SARIMA parameter vector for every power domain."""
    def fit(self, batch, metadata, params, seed, jobs=8):
        from threadpoolctl import threadpool_limits
        self.models = {}
        self.jobs = int(jobs)
        def domain_batch(series_id):
            indices = np.flatnonzero(metadata[:, 0] == series_id)
            last = indices[-1:]
            part = type(batch)(batch.inputs.take(last), batch.targets[last])
            return part
        domains = np.unique(metadata[:, 0])
        # Independent domain fits; each worker uses one BLAS thread. No samples
        # or fitted coefficients are shared between domain-specific models.
        with threadpool_limits(limits=1, user_api="blas"), ProcessPoolExecutor(max_workers=self.jobs) as pool:
            futures = [pool.submit(fit_sarima_domain, series_id, domain_batch(series_id), params, seed)
                       for series_id in domains]
            for completed, future in enumerate(as_completed(futures), 1):
                series_id, model = future.result()
                self.models[series_id] = model
                if completed == 1 or completed % 10 == 0 or completed == len(domains):
                    print(f"[Google] SARIMA fit: {completed}/{len(domains)} domains", flush=True)
        return self

    def predict(self, inputs, metadata):
        from threadpoolctl import threadpool_limits
        values = np.empty((len(inputs), inputs.future_known.shape[1]))
        def predict_domain(series_id, model):
            indices = np.flatnonzero(metadata[:, 0] == series_id)
            return indices, model.predict(inputs.take(indices))
        with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futures = [pool.submit(predict_domain, series_id, model) for series_id, model in self.models.items()
                       if (metadata[:, 0] == series_id).any()]
            for future in as_completed(futures):
                indices, prediction = future.result()
                values[indices] = prediction
        return values

    def save(self, path):
        path = Path(path).with_suffix(".joblib")
        joblib.dump(self, path)
        return path


def collect_foundation_controls(collector, frames):
    aliases = {"chronos2": "chronos2_long", "tirex2": "tirex2", "timesfm25": "timesfm25",
               "toto2": "toto2_joint", "patchtstfm": "patchtstfm_long"}
    for task, frame in frames.items():
        for canonical in FOUNDATION_NAMES:
            source = ("timesfm30" if task == "24h" else "timesfm30_long") if canonical == "timesfm30" else aliases[canonical]
            if source not in frame:
                frame[source] = np.nan
                for origin in frame.origin_day.unique():
                    mask = frame.origin_day == origin
                    part = frame.loc[mask]
                    values = collector._predict_member(source, task, int(origin))
                    frame.loc[mask, source] = values[part.series_id.to_numpy(int), part.horizon_step.to_numpy(int)-1]
                collector._release_predictors()
            frame[canonical] = frame[source]


def fit_controls(data, tasks, development, config, folder):
    from power_forecast.config import TASKS

    requested = config["comparison_models"]
    if set(requested) != set(COMPARISON_MODELS):
        raise ValueError("The all-model Google preset requires the 18 declared comparison models.")
    model_config = load_config(config["comparison_model_config"])
    context_days = int(config["comparison_context_days"])
    seed = int(config.get("seed", 42))
    prepared, records = {}, {}
    for task in tasks:
        horizon = TASKS[task].horizon_days*96
        validation, val_meta = panel_windows(data, TASKS[task].dev_origins, context_days, horizon)
        trained = {}
        records[task] = {}
        for cutoff in (14, 22):
            training, train_meta = training_panel(data, cutoff, context_days, horizon)
            stage = "development" if cutoff == 14 else "final"
            records[task][stage] = dict(label_end_day=cutoff, train_windows=len(training),
                first_origin_day=int(train_meta[:, 1].min()), last_origin_day=int(train_meta[:, 1].max()),
                context_days=context_days, horizon_points=horizon, domains=data.n_series, models={})
            for name in requested:
                if name in FOUNDATION_NAMES:
                    continue
                print(f"[Google {task}] Fit {name} D{cutoff}; windows={len(training)}", flush=True)
                started = perf_counter()
                params = {**model_config["models"].get(name, {}),
                          **config.get("comparison_model_overrides", {}).get(name, {})}
                if name == "sarima":
                    model = PanelSarima().fit(training, train_meta, params, seed, jobs=int(config.get("jobs", 8)))
                else:
                    model = create_model(name, params, seed).fit(training)
                directory = folder/task/name
                directory.mkdir(parents=True, exist_ok=True)
                model_path = model.save(directory/("development-D14" if cutoff == 14 else "model"))
                records[task][stage]["models"][name] = dict(path=str(model_path.relative_to(folder)),
                    sha256=sha256(model_path.read_bytes()).hexdigest(), train_seconds=perf_counter()-started,
                    params=params)
                if cutoff == 14:
                    values = model.predict(validation.inputs, val_meta) if name == "sarima" else model.predict(validation.inputs)
                    assign_predictions(development[task], name, np.maximum(values, 0), val_meta)
                else:
                    trained[name] = model
            del training
        prepared[task] = trained
    write_json(folder/"comparison-training.json", records)
    return prepared, records


def predict_controls(data, frames, prepared, context_days):
    from power_forecast.config import TASKS

    for task, frame in frames.items():
        batch, metadata = panel_windows(data, TASKS[task].test_origins, context_days, TASKS[task].horizon_days*96)
        for name, model in prepared[task].items():
            values = model.predict(batch.inputs, metadata) if name == "sarima" else model.predict(batch.inputs)
            assign_predictions(frame, name, np.maximum(values, 0), metadata)


def score_controls(results):
    from power_forecast.evaluation import summarize_predictions, daily_mean_metrics

    for result in results.values():
        methods = list(dict.fromkeys([*result["test"], *COMPARISON_MODELS]))
        for stage in ("development", "test"):
            result[stage], _, _ = summarize_predictions(result[stage+"_predictions"], methods)
        result["test_daily_mean"] = daily_mean_metrics(result["test_predictions"], methods)
