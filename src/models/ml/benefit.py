"""Source-only learned gain/risk, with an explicit zero-transfer decision."""
from copy import deepcopy
from itertools import product

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from src.data.shape_features import load_signature, SIGNATURE_NAMES
from src.evaluation.metrics import forecast_metrics
from src.models.base import BaseForecaster
from src.models.statistical.baselines import ThreeDayMeanForecaster

POLICY_NAMES = ("uniform_transfer", "fixed_blend", "sapf_gain", "sapf_gain_no_veto")
PRIOR_FEATURES = ("expert_past_MAPE", "peer_local_past_MAPE", "past_gain", "past_negative_fraction", "past_worst_MAPE")
FEATURE_NAMES = tuple(f"{kind}_{name}" for kind in ("source", "target", "signed_difference", "absolute_difference")
                      for name in SIGNATURE_NAMES)+PRIOR_FEATURES+(
    "horizon_days", "log_training_windows", "forecast_disagreement", "forecast_level_difference", "forecast_relative_std")


def pair_features(inputs, bank, priors, experts, local):
    target = np.stack([load_signature(past[:, 0], inputs.points_per_day) for past in inputs.history])
    n, s = len(inputs), len(bank.source_domains)
    source = np.broadcast_to(bank.source_profiles[None], (n, s, len(SIGNATURE_NAMES)))
    target = np.broadcast_to(target[:, None], source.shape)
    differences = source-target
    history_level = np.maximum(np.abs(inputs.history[:, :, 0].mean(1)), 1e-8)[:, None]
    prior = np.broadcast_to(np.asarray([priors[d] for d in bank.source_domains])[None], (n, s, len(PRIOR_FEATURES)))
    other = np.stack([
        np.full((n, s), inputs.future_known.shape[1]/inputs.points_per_day),
        np.full((n, s), np.log1p(bank.training_windows_per_expert)),
        np.abs(experts-local[:, None]).mean(-1)/history_level,
        (experts.mean(-1)-inputs.history[:, :, 0].mean(1)[:, None])/history_level,
        experts.std(-1)/history_level], axis=-1)
    features = np.concatenate([source, target, differences, np.abs(differences), prior, other], axis=-1)
    if features.shape[-1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("Benefit features must be finite history/model-output-only inputs.")
    return features


class JointBenefitPredictor:
    def __init__(self, params, seed=42):
        self.regressor = ExtraTreesRegressor(**params, random_state=seed)
        self.classifier = ExtraTreesClassifier(**params, random_state=seed, class_weight="balanced")

    def fit(self, table, source_domains):
        self.source_domains = sorted(source_domains)
        if (set(table.expert_domain) | set(table.pseudo_target))-set(self.source_domains):
            raise ValueError("Benefit training must remain wholly inside source domains.")
        if (table.expert_domain == table.pseudo_target).any():
            raise ValueError("Leave-domain gain samples cannot use a self-trained expert.")
        if (table.label_end_day > 50).any() or (table.prior_label_end_day > table.origin_day).any():
            raise ValueError("Gain labels or performance priors cross their training-time boundary.")
        x = table[["feature_"+name for name in FEATURE_NAMES]].to_numpy()
        gain = table.actual_gain_pp.to_numpy()
        self.regressor.fit(x, gain)
        self.classifier.fit(x, gain < -1e-10)
        self.negative_prevalence = float(np.mean(gain < -1e-10))
        self.training_gain_mean = float(np.mean(gain))
        self.training_rows = len(table)
        return self

    def predict(self, features):
        shape = features.shape[:2]
        x = features.reshape(-1, features.shape[-1])
        gain = self.regressor.predict(x).reshape(shape)
        classes = list(self.classifier.classes_)
        risk = (self.classifier.predict_proba(x)[:, classes.index(True)].reshape(shape)
                if True in classes else np.zeros(shape))
        return gain, risk


class BenefitTransferPolicy(BaseForecaster):
    def __init__(self, name, bank=None, predictor=None, priors=None, fixed_strength=.5, policy_grid=None, seed=42):
        if name not in POLICY_NAMES:
            raise ValueError("Unknown transfer policy.")
        self.name, self.bank, self.predictor = name, deepcopy(bank), deepcopy(predictor)
        self.priors, self.fixed_strength, self.policy_grid = deepcopy(priors), fixed_strength, deepcopy(policy_grid)
        self.calibration = None

    def fit(self, batch):
        if self.bank is None or sorted(set(batch.inputs.domain_ids or ())) != self.bank.source_domains:
            raise ValueError("A transfer policy needs an explicitly source-fitted expert bank and identical source IDs.")
        self.source_domains = list(self.bank.source_domains)
        if self.name.startswith("sapf_gain") and (self.predictor is None or self.predictor.source_domains != self.source_domains):
            raise ValueError("The joint gain predictor must be trained on the same source identities.")
        return self

    def components(self, inputs):
        experts = np.maximum(self.bank.expert_predictions(inputs), 0.)
        local = ThreeDayMeanForecaster().predict(inputs)
        if self.name.startswith("sapf_gain"):
            features = pair_features(inputs, self.bank, self.priors, experts, local)
            gain, risk = self.predictor.predict(features)
        else:
            gain, risk = np.full(experts.shape[:2], np.nan), np.full(experts.shape[:2], np.nan)
        return experts, local, gain, risk

    @staticmethod
    def route(gain, risk, candidate, veto=True):
        eligible = (gain > candidate["minimum_gain_pp"]) & (risk <= candidate["maximum_negative_risk"])
        weights = np.zeros_like(gain)
        gate = np.zeros(len(gain))
        for i in range(len(gain)):
            indices = np.flatnonzero(eligible[i])
            if not len(indices) and not veto:
                indices = np.array([int(np.argmax(gain[i]))])
            if len(indices):
                indices = indices[np.argsort(-gain[i, indices], kind="stable")[:candidate["top_k"]]]
                positive = np.maximum(gain[i, indices], 1e-8)
                weights[i, indices] = positive/positive.sum()
                gate[i] = candidate["strength"]
        return weights, gate, eligible

    def calibrate(self, validation):
        if self.name != "sapf_gain" or set(validation.inputs.domain_ids) & set(self.source_domains):
            raise ValueError("Only the veto policy calibrates, on independent source-validation nodes.")
        experts, local, gain, risk = self.components(validation.inputs)
        candidates = []
        grid = self.policy_grid
        for margin, maximum, strength, top_k in product(grid["minimum_gain_pp"], grid["maximum_negative_risk"], grid["strengths"], grid["top_k"]):
            c = dict(minimum_gain_pp=margin, maximum_negative_risk=maximum, strength=strength, top_k=top_k)
            weights, gate, _ = self.route(gain, risk, c)
            prediction = gate[:, None]*np.einsum("ns,nsh->nh", weights, experts)+(1-gate[:, None])*local
            values = [forecast_metrics(validation.targets[np.asarray(validation.inputs.domain_ids) == d],
                                      prediction[np.asarray(validation.inputs.domain_ids) == d])["MAPE"]
                      for d in sorted(set(validation.inputs.domain_ids))]
            candidates.append(dict(**c, mean_MAPE=float(np.mean(values)), worst_MAPE=float(max(values)),
                                   objective=float(np.mean(values)+grid["worst_domain_penalty"]*max(values))))
        selected = min(candidates, key=lambda c: (c["objective"], c["strength"], -c["minimum_gain_pp"], c["maximum_negative_risk"], c["top_k"]))
        self.calibration = dict(selected=selected, candidates=candidates, validation_domains=sorted(set(validation.inputs.domain_ids)),
            source_domains=self.source_domains, predictor_training_domains=self.predictor.source_domains,
            predictor_training_rows=self.predictor.training_rows, predictor_joint_horizons=["24h", "3d"],
            zero_transfer_rule="No predicted positive admissible source => exact local forecast.")
        return self

    def predict_with_diagnostics(self, inputs):
        experts, local, gain, risk = self.components(inputs)
        if self.name in {"uniform_transfer", "fixed_blend"}:
            weights = np.full(experts.shape[:2], 1/len(self.source_domains))
            gate = np.full(len(inputs), 1. if self.name == "uniform_transfer" else self.fixed_strength)
            eligible = np.ones_like(weights, dtype=bool)
        else:
            if self.calibration is None:
                raise RuntimeError("Freeze validation-only policy choices before test inference.")
            weights, gate, eligible = self.route(gain, risk, self.calibration["selected"], self.name == "sapf_gain")
        prediction = gate[:, None]*np.einsum("ns,nsh->nh", weights, experts)+(1-gate[:, None])*local
        return prediction, dict(weights=weights, transfer_gate=gate, predicted_gain_pp=gain,
            negative_risk=risk, admissible=eligible, expert_predictions=experts, local_prediction=local)

    def predict(self, inputs):
        return self.predict_with_diagnostics(inputs)[0]
