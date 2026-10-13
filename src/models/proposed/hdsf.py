"""HDSF for adaptive day splits, with explicit train/validation separation."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import forecast_metrics
from src.models.base import BaseForecaster
from src.models.foundation.forecaster import FoundationForecaster
from .local import Local24Forecaster, LocalCurveForecaster
from .simplex import fit_segments, apply_segments, purged_folds

MEMBERS = {
    96: ("chronos2_long", "tirex2", "timesfm25", "toto2_joint", "toto2_single",
         "timesfm30", "timesfm30_long", "timesfm30_uni"),
    672: ("timesfm30_long", "toto2_joint", "timesfm30_uni", "patchtstfm_long"),
}


class HDSFForecaster(BaseForecaster):
    name = "hdsf"

    def __init__(self, weights_cache, prediction_cache="results/cache/foundation", jobs=4, batch_size=32,
                 context_days=14, segment_points=48, cv_mode="blocked", cv_folds=5,
                 loo_tolerance=.005, mape_threshold=1e-8, local_rounds=350):
        self.options = dict(weights_cache=weights_cache, prediction_cache=prediction_cache,
            jobs=jobs, batch_size=batch_size, context_days=context_days)
        self.segment_points, self.cv_mode, self.cv_folds = segment_points, cv_mode, cv_folds
        self.loo_tolerance, self.mape_threshold = loo_tolerance, mape_threshold
        self.local_rounds = local_rounds
        self.frozen = False

    def fit(self, batch):
        if self.frozen:
            raise RuntimeError("Cannot replace frozen members; explicit pre-test refit only.")
        horizon = batch.targets.shape[1]
        if horizon not in MEMBERS:
            raise ValueError("HDSF supports complete 96-point / 672-point horizons.")
        self.member_names = (*MEMBERS[horizon], "local24" if horizon == 96 else "local7")
        self.models = {name: FoundationForecaster(name, **self.options).fit(batch) for name in MEMBERS[horizon]}
        self.models[self.member_names[-1]] = (Local24Forecaster(jobs=self.options["jobs"], rounds=self.local_rounds)
                                             if horizon == 96 else LocalCurveForecaster()).fit(batch)
        self.training_end = str(batch.inputs.target_times[-1, -1])
        return self

    def collect(self, inputs):
        parts = []
        for name in self.member_names:
            model = self.models[name]
            chunks = []
            size = self.options["batch_size"]
            for start in range(0, len(inputs), size):
                chunks.append(model.predict(inputs.take(slice(start, start+size))))
                if start % (size*5) == 0:
                    print(f"[HDSF] {name}: {min(start+size, len(inputs))}/{len(inputs)} origins", flush=True)
            parts.append(np.concatenate(chunks))
            if hasattr(model, "release"):
                model.release()
        return np.stack(parts, axis=-1)

    def calibrate(self, validation):
        import pandas as pd
        if pd.Timestamp(self.training_end) >= validation.inputs.origins[0]:
            raise ValueError("Local model training labels must precede validation origins.")
        members = self.collect(validation.inputs)
        truth = validation.targets
        self.weights_a = fit_segments(members[:, :, :-1], truth, self.segment_points)
        self.weights_b = fit_segments(members, truth, self.segment_points)
        self.weights_global = fit_segments(members, truth, truth.shape[1])
        a, b = np.empty_like(truth, dtype=float), np.empty_like(truth, dtype=float)
        covered = np.zeros(len(truth), bool)
        folds = purged_folds(validation.inputs.origins, truth.shape[1], self.cv_folds, self.cv_mode)
        fold_info = []
        for training, held in folds:
            wa = fit_segments(members[training, :, :-1], truth[training], self.segment_points)
            wb = fit_segments(members[training], truth[training], self.segment_points)
            a[held] = apply_segments(members[held, :, :-1], wa)
            b[held] = apply_segments(members[held], wb)
            covered[held] = True
            fold_info.append({"fit_origins": [str(o) for o in validation.inputs.origins[training]],
                              "held_origins": [str(o) for o in validation.inputs.origins[held]]})
        full_a = apply_segments(members[:, :, :-1], self.weights_a)
        full_b = apply_segments(members, self.weights_b)
        candidates = []
        for mix in np.linspace(0., 1., 41):
            mixed = (1-mix)*a[covered]+mix*b[covered]
            scores = [forecast_metrics(y, p, self.mape_threshold)["MAPE"] for y, p in zip(truth[covered], mixed)]
            scores = [score for score in scores if score is not None]
            if not scores:
                raise ValueError("No nonzero validation labels for MAPE selection.")
            development = forecast_metrics(truth, (1-mix)*full_a+mix*full_b, self.mape_threshold)["MAPE"]
            candidates.append({"mix": float(mix), "cv_mape": float(np.mean(scores)), "fit_mape": development})
        best = min(row["cv_mape"] for row in candidates)
        plateau = [row for row in candidates if row["cv_mape"] <= best+self.loo_tolerance]
        self.mix = min(plateau, key=lambda row: row["fit_mape"])["mix"]
        cv_prediction = (1-self.mix)*a[covered]+self.mix*b[covered]
        cv_rows = [forecast_metrics(y, p, self.mape_threshold) for y, p in zip(truth[covered], cv_prediction)]
        self.configuration = {
            "members": list(self.member_names), "context_days": self.options["context_days"],
            "segment_points": self.segment_points, "segments": len(self.weights_b), "mix": self.mix,
            "optimizer": "original pooled L1 simplex; 25 iterations; deterministic max 20000 points",
            "selection": "MAPE; purged validation CV; 41-point grid; fit-MAPE plateau tie-break",
            "validation_fit_score_is_in_sample": True, "cv_mode": self.cv_mode, "cv_folds": fold_info,
            "cv_covered_origins": int(covered.sum()), "candidates": candidates,
            "cv_origin_mean": {metric: float(np.mean([row[metric] for row in cv_rows if row[metric] is not None]))
                               for metric in ("MAPE", "WAPE", "RMSE")},
            "weights_a": self.weights_a.tolist(), "weights_b": self.weights_b.tolist(),
            "weights_global": self.weights_global.tolist(), "training_end": self.training_end,
            "validation_label_end": str(validation.inputs.target_times[-1, -1]),
            "checkpoint_identities": {name: model.cache.fingerprint.decode() for name, model in self.models.items()
                                      if isinstance(model, FoundationForecaster)},
        }
        self.validation_members = members
        self.frozen = True
        return self

    def refit(self, batch):
        # Fusion weights retain the validation-only calibration. Only the local supervised member refits.
        self.models[self.member_names[-1]].fit(batch)
        self.configuration["final_training_end"] = str(batch.inputs.target_times[-1, -1])
        return self

    def outputs(self, members):
        if not self.frozen:
            raise RuntimeError("Calibrate and freeze HDSF before inference.")
        a = apply_segments(members[:, :, :-1], self.weights_a)
        b = apply_segments(members, self.weights_b)
        return {**{name: members[:, :, i] for i, name in enumerate(self.member_names)},
                "hdsf": (1-self.mix)*a+self.mix*b, "hdsf_foundation_only": a,
                "hdsf_with_local": b, "hdsf_equal_weights": members.mean(-1),
                "hdsf_global_weights": apply_segments(members, self.weights_global)}

    def predict(self, inputs):
        return self.outputs(self.collect(inputs))["hdsf"]
