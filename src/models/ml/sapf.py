"""SAPF mechanism study: normalized source experts, similarity and validation-only reliability."""
from __future__ import annotations

from copy import deepcopy
from itertools import product

import numpy as np

from src.data.shape_features import SIGNATURE_NAMES, load_signature
from src.data.splitting import WindowBatch
from src.evaluation.metrics import forecast_metrics
from src.models.base import BaseForecaster
from src.models.ml.normalized import HistoryNormalizedForecaster
from src.models.statistical.baselines import ThreeDayMeanForecaster


class SelectiveTransferForecaster(BaseForecaster):
    def __init__(self, mode="reliable", base_model="ridge", base_params=None, scale_fraction=0.01,
                 epsilon=1e-6, top_k=(1, 3, 7), temperatures=(0.5, 1.0, 2.0),
                 feature_groups=("shape", "level_shape"), transfer_strengths=(0.0, 0.25, 0.5, 0.75, 1.0),
                 reliability_powers=(0.0, 1.0), worst_domain_penalty=0.1,
                 mape_threshold=1e-8, seed=42):
        if mode not in {"similarity", "reliable"}:
            raise ValueError("SAPF mode must be similarity or reliable.")
        self.mode = mode
        self.name = "sapf_similarity" if mode == "similarity" else "sapf"
        self.base_model, self.base_params = base_model, dict(base_params or {})
        self.scale_fraction, self.epsilon = float(scale_fraction), float(epsilon)
        self.top_k = tuple(int(k) for k in top_k)
        self.temperatures = tuple(float(t) for t in temperatures)
        self.feature_groups = tuple(feature_groups)
        self.transfer_strengths = tuple(float(v) for v in transfer_strengths)
        self.reliability_powers = tuple(float(v) for v in reliability_powers)
        self.worst_domain_penalty, self.mape_threshold, self.seed = worst_domain_penalty, mape_threshold, seed
        if (not self.top_k or min(self.top_k) < 1 or not self.temperatures or min(self.temperatures) <= 0
                or set(self.feature_groups)-{"shape", "level_shape"} or not self.feature_groups
                or not self.transfer_strengths or any(v < 0 or v > 1 for v in self.transfer_strengths)
                or any(v < 0 for v in self.reliability_powers)):
            raise ValueError("Invalid predeclared SAPF candidate grid.")
        self.calibration = None

    def _fit_sources(self, batch):
        if batch.inputs.domain_ids is None:
            raise ValueError("SAPF needs explicit training domain identities.")
        domains = sorted(set(batch.inputs.domain_ids))
        if len(domains) < 2:
            raise ValueError("SAPF requires at least two source domains.")
        self.source_domains = domains
        self.experts, profiles = [], []
        for domain in domains:
            indices = np.flatnonzero(np.array(batch.inputs.domain_ids) == domain)
            part = WindowBatch(batch.inputs.take(indices), batch.targets[indices])
            expert = HistoryNormalizedForecaster(self.base_model, self.base_params, self.scale_fraction,
                                                  self.epsilon, self.seed).fit(part)
            self.experts.append(expert)
            # Source training labels are observed by its declared cutoff. No
            # held-out domain or future target label contributes to this profile.
            last = max(range(len(part)), key=lambda i: part.inputs.origins[i])
            observed = np.r_[part.inputs.history[last, :, 0], part.targets[last]]
            profiles.append(load_signature(observed[-part.inputs.history.shape[1]:], batch.inputs.points_per_day))
        self.source_profiles = np.stack(profiles)
        self.profile_location = np.median(self.source_profiles, axis=0)
        interquartile = np.percentile(self.source_profiles, 75, axis=0)-np.percentile(self.source_profiles, 25, axis=0)
        self.profile_scale = np.maximum(np.maximum(interquartile, np.std(self.source_profiles, axis=0)), 0.05)
        return self

    def fit(self, batch):
        self.calibration = None
        return self._fit_sources(batch)

    def refit(self, batch):
        if self.calibration is None:
            raise RuntimeError("Calibrate on independent validation domains before final source refit.")
        if sorted(set(batch.inputs.domain_ids or ())) != self.source_domains:
            raise ValueError("Final refit cannot change the frozen source pool.")
        calibration = deepcopy(self.calibration)
        self._fit_sources(batch)
        self.calibration = calibration
        return self

    def expert_predictions(self, inputs):
        return np.stack([expert.predict(inputs) for expert in self.experts], axis=1)  # [N,S,H]

    def _weights(self, inputs, candidate, reliability):
        profiles = np.stack([load_signature(past[:, 0], inputs.points_per_day) for past in inputs.history])
        mask = np.arange(len(SIGNATURE_NAMES)) if candidate["features"] == "level_shape" else np.arange(1, len(SIGNATURE_NAMES))
        differences = (profiles[:, None, mask]-self.source_profiles[None, :, mask])/self.profile_scale[mask]
        distances = np.sqrt(np.mean(differences**2, axis=-1))
        k = min(candidate["top_k"], len(self.source_domains))
        nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
        score = -distances/candidate["temperature"]
        score += candidate.get("reliability_power", 0.0)*np.log(np.maximum(reliability, 1e-12))[None, :]
        keep = np.zeros_like(score, dtype=bool)
        np.put_along_axis(keep, nearest, True, axis=1)
        score = np.where(keep, score, -np.inf)
        weights = np.exp(score-np.max(score, axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        confidence = 1/(1+distances.min(axis=1))
        return weights, distances, confidence

    def _combine(self, inputs, expert_values, local, candidate, reliability):
        weights, distances, confidence = self._weights(inputs, candidate, reliability)
        transferred = np.einsum("ns,nsh->nh", weights, expert_values)
        strength = candidate.get("strength", 1.0)
        gate = np.ones(len(inputs)) if self.mode == "similarity" else strength*confidence
        predictions = gate[:, None]*transferred+(1-gate[:, None])*local
        return predictions, dict(weights=weights, distances=distances, transfer_gate=gate)

    def _domain_scores(self, actual, prediction, domains):
        values = []
        for domain in sorted(set(domains)):
            mask = np.array(domains) == domain
            metric = forecast_metrics(actual[mask], np.maximum(prediction[mask], 0), self.mape_threshold)["MAPE"]
            if metric is None:
                raise ValueError("Validation MAPE is undefined; do not select on missing scores.")
            values.append(metric)
        mean, worst = float(np.mean(values)), float(np.max(values))
        return mean, worst, mean+self.worst_domain_penalty*worst

    def calibrate(self, validation):
        domains = validation.inputs.domain_ids
        if domains is None or set(domains) & set(self.source_domains):
            raise ValueError("Calibration domains must be independent of all source fitting domains.")
        expert_values = self.expert_predictions(validation.inputs)
        local = ThreeDayMeanForecaster().predict(validation.inputs)
        expert_losses = np.array([self._domain_scores(validation.targets, expert_values[:, s], domains)[0]
                                  for s in range(len(self.source_domains))])
        reliability = np.clip(np.median(expert_losses)/np.maximum(expert_losses, 1e-8), 0.01, 10.0)
        strengths = (1.0,) if self.mode == "similarity" else self.transfer_strengths
        powers = (0.0,) if self.mode == "similarity" else self.reliability_powers
        candidates = []
        for features, k, temperature, strength, power in product(
                self.feature_groups, self.top_k, self.temperatures, strengths, powers):
            candidate = dict(features=features, top_k=k, temperature=temperature,
                             strength=strength, reliability_power=power)
            prediction, _ = self._combine(validation.inputs, expert_values, local, candidate, reliability)
            mean, worst, objective = self._domain_scores(validation.targets, prediction, domains)
            candidates.append(dict(**candidate, mean_MAPE=mean, worst_MAPE=worst, objective=objective))
        # Candidate order breaks exact ties in favor of the smaller transfer strength.
        selected = min(candidates, key=lambda c: (c["objective"], c["strength"], c["top_k"]))
        self.calibration = dict(selected=selected, candidates=candidates, source_domains=self.source_domains,
            validation_domains=sorted(set(domains)), validation_origins=sorted({str(o) for o in validation.inputs.origins}),
            expert_validation_MAPE=expert_losses.tolist(), reliability=reliability.tolist(),
            criterion=f"equal-domain mean MAPE + {self.worst_domain_penalty} * worst-domain MAPE",
            local_reference="three_day_mean", fitted_before_test=True)
        return self

    def predict_with_diagnostics(self, inputs):
        if self.calibration is None:
            raise RuntimeError("SAPF requires validation-only calibration before prediction.")
        return self._combine(inputs, self.expert_predictions(inputs), ThreeDayMeanForecaster().predict(inputs),
                             self.calibration["selected"], np.asarray(self.calibration["reliability"]))

    def predict(self, inputs):
        return self.predict_with_diagnostics(inputs)[0]
