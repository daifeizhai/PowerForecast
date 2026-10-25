"""Controlled Ridge representations, with history-only local scale restoration."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.shape_features import load_signature
from src.data.splitting import WindowBatch
from src.models.base import BaseForecaster

REPRESENTATIONS = ('raw', 'zscore', 'level_shape', 'scale_metadata')


class RepresentationRidge(BaseForecaster):
    def __init__(self, representation='level_shape', alpha=100., name=None, seed=42):
        if representation not in REPRESENTATIONS:
            raise ValueError('Unknown controlled representation.')
        self.representation, self.alpha = representation, alpha
        self.name = name or 'ridge_'+representation

    def features(self, inputs):
        raw = inputs.history[:, :, 0].astype(float)
        mu = raw.mean(1, keepdims=True)
        sigma = np.maximum(raw.std(1, keepdims=True), np.maximum(.01*np.abs(mu), 1e-6))
        shape = (raw-mu)/sigma
        known = np.concatenate([inputs.history[:, -1, 1:], inputs.future_known[:, 0], inputs.future_known[:, -1]], axis=1)
        if self.representation == 'raw':
            features = raw
        elif self.representation == 'zscore':
            features = shape
        elif self.representation == 'scale_metadata':
            features = np.concatenate([shape, mu, sigma], axis=1)
        else:
            # Explicit raw-level lag channel plus locally normalized shape lag
            # channel. Ridge regularizes their collinearity; scaler is source-fit.
            features = np.concatenate([raw, shape], axis=1)
        return np.concatenate([features, known], axis=1), mu, sigma

    def fit(self, batch):
        x, mu, sigma = self.features(batch.inputs)
        y = batch.targets.astype(float)
        if self.representation != 'raw':
            y = (y-mu)/sigma
        self.estimator = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(x, y)
        self.source_domains = sorted(set(batch.inputs.domain_ids or ()))
        return self

    def predict(self, inputs):
        x, mu, sigma = self.features(inputs)
        y = np.asarray(self.estimator.predict(x)).reshape(len(inputs), -1)
        return y if self.representation == 'raw' else y*sigma+mu


class RepresentationBank(BaseForecaster):
    def __init__(self, representation='level_shape', alpha=100., name='uniform_transfer', seed=42):
        self.representation, self.alpha, self.name = representation, alpha, name

    def fit(self, batch):
        self.source_domains = sorted(set(batch.inputs.domain_ids or ()))
        if len(self.source_domains) < 3:
            raise ValueError('Leave-domain experts need at least three training identities.')
        self.experts, profiles = [], []
        for domain in self.source_domains:
            indices = np.flatnonzero(np.asarray(batch.inputs.domain_ids) == domain)
            part = WindowBatch(batch.inputs.take(indices), batch.targets[indices])
            self.experts.append(RepresentationRidge(self.representation, self.alpha).fit(part))
            last = max(range(len(part)), key=lambda i: part.inputs.origins[i])
            past = np.r_[part.inputs.history[last,:,0], part.targets[last]][-part.inputs.history.shape[1]:]
            profiles.append(load_signature(past, batch.inputs.points_per_day))
        self.source_profiles = np.stack(profiles)
        self.training_windows_per_expert = len(batch)//len(self.source_domains)
        return self

    def expert_predictions(self, inputs):
        return np.maximum(np.stack([e.predict(inputs) for e in self.experts], axis=1), 0.)

    def predict(self, inputs):
        return self.expert_predictions(inputs).mean(1)
