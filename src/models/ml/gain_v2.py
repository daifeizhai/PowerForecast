"""Four transfer-value objectives, independent calibration, and sequential fusion."""
from collections import defaultdict
from copy import deepcopy

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from src.models.base import BaseForecaster
from src.models.ml.benefit import FEATURE_NAMES, pair_features
from src.models.statistical.baselines import ThreeDayMeanForecaster

EXTRA_FEATURES = ('stage', 'current_local_disagreement', 'current_level_offset', 'expert_current_disagreement', 'current_relative_std')
VALUE_FEATURES = FEATURE_NAMES+EXTRA_FEATURES
OBJECTIVES = ('single', 'fixed', 'marginal', 'rank')
GAIN_METHODS = ('gain_v2_single', 'gain_v2_fixed', 'gain_v2_marginal', 'gain_v2_rank', 'gain_v2_forced', 'gain_v2_uncalibrated')


def value_features(inputs, bank, priors, experts, local, current, stage):
    base = pair_features(inputs, bank, priors, experts, local)
    n, s = base.shape[:2]
    if current.ndim == 2:
        current = np.broadcast_to(current[:, None], experts.shape)
    level = np.maximum(np.abs(inputs.history[:,:,0].mean(1)), 1e-8)[:,None]
    extra = np.stack([np.full((n,s), stage), np.abs(current-local[:,None]).mean(-1)/level,
        (current.mean(-1)-local.mean(-1)[:,None])/level,
        np.abs(experts-current).mean(-1)/level, current.std(-1)/level], axis=-1)
    return np.concatenate([base, extra], axis=-1)


def _probability(model, x):
    classes = list(model.classes_)
    return model.predict_proba(x)[:,classes.index(True)] if True in classes else np.zeros(len(x))


def _domain_weights(table):
    counts = table.target_domain.value_counts()
    return np.asarray([len(table)/len(counts)/counts[d] for d in table.target_domain])


def _logit(values):
    q = np.clip(values, 1e-5, 1-1e-5)
    return np.log(q/(1-q)).reshape(-1,1)


class TransferValueLearner:
    def __init__(self, objective='marginal', params=None, seed=42):
        if objective not in OBJECTIVES:
            raise ValueError('Declare a transfer-value objective.')
        self.objective, self.params, self.seed = objective, params or {}, seed

    def fit(self, table, sources):
        self.source_domains = sorted(sources)
        if not set(table.target_domain).issubset(sources) or not set(table.expert_domain).issubset(sources):
            raise ValueError('Meta-fitting may only use source-internal pseudo-targets.')
        if (table.expert_domain == table.target_domain).any() or (table.context_domain == table.target_domain).any():
            raise ValueError('A pseudo-target cannot train its expert or current-ensemble context.')
        if (table.expert_fit_cutoff_day > table.origin_day).any() or (table.prior_label_end_day > table.origin_day).any() or (table.label_end_day > 42).any():
            raise ValueError('Teacher labels or historical priors cross the predeclared boundary.')
        x = table[['feature_'+k for k in VALUE_FEATURES]].to_numpy()
        label = table['gain_'+('marginal' if self.objective == 'rank' else self.objective)].to_numpy()
        weights = _domain_weights(table)
        self.training_rows, self.marginal_negative_prior = len(table), float(np.average(table.gain_marginal < -1e-10, weights=weights))
        self.classifier = ExtraTreesClassifier(**self.params, random_state=self.seed, class_weight='balanced').fit(x, label < -1e-10, sample_weight=weights)
        if self.objective != 'rank':
            self.regressor = ExtraTreesRegressor(**self.params, random_state=self.seed).fit(x, label, sample_weight=weights)
            self.rank_training_pairs = 0
        else:
            pair_x, pair_y, pair_w = [], [], []
            for indices in table.groupby(['target_domain','horizon','origin_day','stage'], sort=False).indices.values():
                for pos, i in enumerate(indices):
                    for step in (1,2):
                        if len(indices) <= step:
                            continue
                        j = indices[(pos+step)%len(indices)]
                        if abs(label[i]-label[j]) <= 1e-10:
                            continue
                        pair_x.extend([np.r_[x[i], x[j]], np.r_[x[j], x[i]]])
                        pair_y.extend([label[i] > label[j], label[j] > label[i]])
                        pair_w.extend([weights[i], weights[i]])
            if len(set(pair_y)) != 2:
                raise ValueError('Pairwise ranking needs non-tied positive and negative training comparisons.')
            self.ranker = ExtraTreesClassifier(**self.params, random_state=self.seed).fit(np.asarray(pair_x), pair_y, sample_weight=pair_w)
            self.rank_training_pairs = len(pair_y)
        self.calibration = None
        return self

    def _rank(self, features, available=None):
        n,s,f = features.shape
        if available is None:
            available = np.ones((n,s), dtype=bool)
        left = np.broadcast_to(features[:,:,None], (n,s,s,f))
        right = np.broadcast_to(features[:,None], (n,s,s,f))
        x = np.concatenate([left,right], axis=-1).reshape(-1,2*f)
        q = _probability(self.ranker, x).reshape(n,s,s)
        pairs = available[:,:,None] & available[:,None,:] & ~np.eye(s,dtype=bool)[None]
        return np.divide((q*pairs).sum(2), pairs.sum(2), out=np.full((n,s),.5), where=pairs.sum(2)>0)

    def raw(self, features, available=None):
        shape = features.shape[:2]; x = features.reshape(-1,features.shape[-1])
        gain = self._rank(features, available) if self.objective == 'rank' else self.regressor.predict(x).reshape(shape)
        return gain, _probability(self.classifier,x).reshape(shape)

    def table_raw(self, table):
        x = table[['feature_'+k for k in VALUE_FEATURES]].to_numpy()
        if self.objective != 'rank':
            return self.regressor.predict(x), _probability(self.classifier,x)
        values = np.empty(len(table)); buckets = defaultdict(list)
        for indices in table.groupby(['target_domain','horizon','origin_day','stage'],sort=False).indices.values():
            buckets[len(indices)].append(indices)
        for groups in buckets.values():
            indices = np.stack(groups)
            values[indices.reshape(-1)] = self._rank(x[indices]).reshape(-1)
        return values, _probability(self.classifier,x)

    @staticmethod
    def _fit_sigmoid(raw, y, weights):
        if np.unique(y).size < 2:
            return float(np.average(y,weights=weights))
        return LogisticRegression(C=1., max_iter=1000).fit(_logit(raw), y, sample_weight=weights)

    @staticmethod
    def _sigmoid(model, raw):
        return np.full(len(raw),model) if isinstance(model,float) else model.predict_proba(_logit(raw))[:,list(model.classes_).index(True)]

    def calibrate(self, table, calibration_domains):
        if set(table.target_domain) != set(calibration_domains) or set(calibration_domains) & set(self.source_domains):
            raise ValueError('Independent calibration identities must be disjoint from meta training.')
        if (table.origin_day<44).any() or (table.label_end_day>49).any():
            raise ValueError('Calibration must follow source meta fitting and end before policy selection.')
        gain, raw = self.table_raw(table)
        label = table['gain_'+('marginal' if self.objective == 'rank' else self.objective)].to_numpy()
        y = table.gain_marginal.to_numpy() < -1e-10
        weights = _domain_weights(table)
        # Affine gain scale correction is learned only on calibration identities.
        self.gain_calibrator = Ridge(alpha=10.).fit(gain[:,None],label,sample_weight=weights)
        oof = np.empty(len(table))
        for domain in calibration_domains:
            keep = table.target_domain.to_numpy() != domain
            model = self._fit_sigmoid(raw[keep], y[keep], weights[keep])
            oof[~keep] = self._sigmoid(model,raw[~keep])
        scores = {mix:float(np.average((mix*oof+(1-mix)*self.marginal_negative_prior-y)**2,weights=weights)) for mix in [0.,.5,1.]}
        self.risk_mix = min(scores,key=lambda v:(scores[v],v))
        self.risk_calibrator = self._fit_sigmoid(raw,y,weights)
        self.calibration = dict(calibration_domains=sorted(calibration_domains), source_domains=self.source_domains,
            independent_rows=len(table), risk_cross_domain_oof_brier=scores, selected_risk_mix=self.risk_mix,
            source_marginal_negative_prior=self.marginal_negative_prior, gain_slope=float(self.gain_calibrator.coef_[0]),
            gain_intercept=float(self.gain_calibrator.intercept_), risk_target='negative marginal fusion gain',
            risk_calibration='sigmoid, convex shrinkage to source marginal prevalence selected by leave-calibration-domain-out Brier')
        return self

    def apply_calibration(self, gain, risk):
        if self.calibration is None:
            raise RuntimeError('Freeze independent calibration before policy validation/test.')
        shape = gain.shape
        calibrated_gain = self.gain_calibrator.predict(gain.reshape(-1,1)).reshape(shape)
        calibrated_risk = self.risk_mix*self._sigmoid(self.risk_calibrator,risk.reshape(-1)).reshape(shape)+(1-self.risk_mix)*self.marginal_negative_prior
        return calibrated_gain, calibrated_risk


class FixedExpertBlend(BaseForecaster):
    def __init__(self, bank=None, strength=.5, name='fixed_blend_50', seed=42):
        self.bank, self.strength, self.name = bank, strength, name

    def fit(self,batch):
        if self.bank is None or sorted(set(batch.inputs.domain_ids)) != self.bank.source_domains:
            raise ValueError('Use the same declared source bank for controlled mixtures.')
        self.source_domains = self.bank.source_domains
        return self

    def predict(self,inputs):
        local = ThreeDayMeanForecaster().predict(inputs)
        return (1-self.strength)*local+self.strength*self.bank.predict(inputs)


class GainV2Policy(BaseForecaster):
    def __init__(self,name='gain_v2_marginal',bank=None,learner=None,priors=None,rule=None,seed=42):
        # Fitted banks and learners are immutable during policy selection/inference.
        self.name,self.bank,self.learner,self.priors,self.rule = name,bank,learner,deepcopy(priors),deepcopy(rule)

    def fit(self,batch):
        self.source_domains = sorted(set(batch.inputs.domain_ids or ()))
        if self.bank is None or self.learner is None or self.source_domains != self.bank.source_domains or self.source_domains != self.learner.source_domains:
            raise ValueError('Policy, expert and learner source identities must agree.')
        return self

    def predict_with_diagnostics(self,inputs):
        if self.rule is None:
            raise RuntimeError('Select the policy on separate validation identities before test.')
        experts = self.bank.expert_predictions(inputs)
        local = ThreeDayMeanForecaster().predict(inputs)
        current = local.astype(float).copy()
        n,s = experts.shape[:2]
        weights,local_weight = np.zeros((n,s)),np.ones(n)
        available,active = np.ones((n,s),bool),np.ones(n,bool)
        records=[]
        for stage in (0,1):
            features = value_features(inputs,self.bank,self.priors,experts,local,current,stage)
            raw_gain,raw_risk = self.learner.raw(features,available)
            gain,risk = self.learner.apply_calibration(raw_gain,raw_risk)
            if self.name == 'gain_v2_uncalibrated':
                gain,risk = raw_gain,raw_risk
            eligible = available & (gain > self.rule['margin_pp']) & (risk <= self.rule['max_risk']) & active[:,None]
            candidates = eligible if self.name != 'gain_v2_forced' else np.where(eligible.any(1)[:,None],eligible,available & active[:,None])
            has = candidates.any(1)
            chosen = np.argmax(np.where(candidates,gain,-np.inf),axis=1)
            records.append(dict(stage=stage, current=current.copy(), available=available & active[:,None],
                gain=gain,risk=risk,raw_gain=raw_gain,raw_risk=raw_risk,eligible=eligible,chosen=np.where(has,chosen,-1)))
            rows = np.flatnonzero(has)
            current[rows] = .5*current[rows]+.5*experts[rows,chosen[rows]]
            weights[rows] *= .5
            weights[rows,chosen[rows]] += .5
            local_weight[rows] *= .5
            available[rows,chosen[rows]] = False
            active = has
        return current,dict(experts=experts,local=local,weights=weights,local_weight=local_weight,steps=records)

    def predict(self,inputs):
        return self.predict_with_diagnostics(inputs)[0]
