"""One-mechanism interventions, conditional on a validation-selected SAPF model."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.data.splitting import WindowBatch
from src.models.base import BaseForecaster
from src.models.statistical.baselines import ThreeDayMeanForecaster

ABLATION_MODES = {
    "sapf_no_normalization": "normalization",
    "sapf_no_selection": "selection",
    "sapf_no_fallback": "fallback",
    "sapf_no_reliability": "reliability",
}


class ControlledAblationForecaster(BaseForecaster):
    def __init__(self, name, reference_model=None, seed=42):
        if name not in ABLATION_MODES:
            raise ValueError("Choose an explicitly registered one-mechanism ablation.")
        self.name, self.disabled, self.seed = name, ABLATION_MODES[name], seed
        self._reference_model = reference_model

    def fit(self, batch):
        from src.models import create_model
        reference = self._reference_model
        if reference is None or reference.calibration is None or reference.mode != "reliable":
            raise ValueError("Controlled ablations require a fitted, independently calibrated full SAPF reference.")
        if sorted(set(batch.inputs.domain_ids or ())) != reference.source_domains:
            raise ValueError("Ablations must use the exact same source pool as full SAPF.")
        self.reference = deepcopy(reference)
        self._reference_model = None
        self.source_domains = list(reference.source_domains)
        self.calibration = deepcopy(reference.calibration)
        self.calibration["intervention"] = dict(disabled=self.disabled, reference="sapf",
            independently_retuned=False, reference_selected=deepcopy(reference.calibration["selected"]))
        self.raw_experts = []
        if self.disabled == "normalization":
            for domain in self.source_domains:
                indices = np.flatnonzero(np.asarray(batch.inputs.domain_ids) == domain)
                part = WindowBatch(batch.inputs.take(indices), batch.targets[indices])
                self.raw_experts.append(create_model(reference.base_model, reference.base_params, self.seed).fit(part))
        return self

    def predict_with_diagnostics(self, inputs):
        if not hasattr(self, "reference"):
            raise RuntimeError("Fit a controlled ablation with its frozen full SAPF reference first.")
        candidate = deepcopy(self.reference.calibration["selected"])
        if self.disabled == "reliability":
            candidate["reliability_power"] = 0.
        reliability = np.asarray(self.reference.calibration["reliability"])
        weights, distances, confidence = self.reference._weights(inputs, candidate, reliability)
        if self.disabled == "selection":
            weights = np.full_like(weights, 1/len(self.source_domains))
        expert_values = (np.stack([expert.predict(inputs) for expert in self.raw_experts], axis=1)
                         if self.disabled == "normalization" else self.reference.expert_predictions(inputs))
        gate = candidate["strength"]*confidence
        if self.disabled == "fallback":
            gate = np.ones(len(inputs))
        transferred = np.einsum("ns,nsh->nh", weights, expert_values)
        local = ThreeDayMeanForecaster().predict(inputs)
        return gate[:, None]*transferred+(1-gate[:, None])*local, dict(
            weights=weights, distances=distances, transfer_gate=gate)

    def predict(self, inputs):
        return self.predict_with_diagnostics(inputs)[0]
