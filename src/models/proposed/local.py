"""NLR-compatible causal local members using exactly the supplied history."""
import numpy as np
from sklearn.linear_model import LinearRegression

from src.models.base import BaseForecaster


def local_features(inputs):
    values = inputs.history[:, :, 0].reshape(len(inputs), -1, 96).astype(float)
    days = values.shape[1]
    fields = [values[:, -lag] for lag in (1, 2, 3, 7, 14) if lag <= days]
    for length in (3, 7, 14):
        part = values[:, -min(days, length):]
        fields.extend([part.mean(1), np.median(part, axis=1), part.std(1)])
    for length in (1, 3, 7, 14):
        level = values[:, -min(days, length):].mean((1, 2))
        fields.append(np.repeat(level[:, None], 96, axis=1))
    fields.extend([values[:, -1]-values[:, -2], values[:, -1]-values[:, -3:].mean(1)])
    for i in range(inputs.future_known.shape[2]):
        fields.append(inputs.future_known[:, :96, i])
    return np.stack(fields, axis=-1), values


class Local24Forecaster(BaseForecaster):
    name = "local24"

    def __init__(self, jobs=4, rounds=350, seeds=(101, 202, 303, 404)):
        self.jobs, self.rounds, self.seeds = jobs, rounds, seeds

    def fit(self, batch):
        import lightgbm as lgb
        X, values = local_features(batch.inputs)
        y = batch.targets[:, :96].astype(float)
        residual = (y-values[:, -1]).reshape(-1)
        X = X.reshape(-1, X.shape[-1])
        options = dict(objective="l1", num_leaves=15, learning_rate=.04, n_estimators=self.rounds,
            min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.8, reg_lambda=1.,
            n_jobs=self.jobs, verbosity=-1, deterministic=True, force_col_wise=True)
        self.models = [lgb.LGBMRegressor(random_state=seed, **options).fit(X, residual) for seed in self.seeds]
        self.lags = tuple(lag for lag in (1, 2, 3, 4, 5, 6, 7, 14) if lag <= values.shape[1])
        lags = np.stack([values[:, -lag] for lag in self.lags], axis=-1).reshape(-1, len(self.lags))
        self.linear = LinearRegression(fit_intercept=False).fit(lags, y.reshape(-1), sample_weight=np.abs(y).reshape(-1))
        return self

    def predict(self, inputs):
        X, values = local_features(inputs)
        residual = np.mean([m.predict(X.reshape(-1, X.shape[-1])).reshape(len(inputs), 96) for m in self.models], axis=0)
        lags = np.stack([values[:, -lag] for lag in self.lags], axis=-1)
        linear = self.linear.predict(lags.reshape(-1, len(self.lags))).reshape(len(inputs), 96)
        prediction = .5*(residual+values[:, -1])+.5*linear
        prediction = np.clip(prediction, values.min(1), values.max(1))
        return np.tile(prediction, (1, inputs.future_known.shape[1]//96))


class LocalCurveForecaster(BaseForecaster):
    name = "local7"

    def fit(self, batch):
        return self

    def predict(self, inputs):
        history = inputs.history[:, :, 0].reshape(len(inputs), -1, 96).astype(float)[:, -24:]
        daily = history.mean(-1)
        levels = []
        for half_life in (3., 5.):
            weights = .5**(np.arange(history.shape[1])/half_life)
            levels.append(daily[:, ::-1] @ (weights/weights.sum()))
        level = np.mean(levels, axis=0)
        # A zero daily level contributes a flat shape; no division by zero.
        shape = np.divide(history, daily[:, :, None], out=np.ones_like(history), where=np.abs(daily[:, :, None]) > 1e-8)
        weights = .5**(np.arange(history.shape[1])[::-1]/10.)
        shape = np.einsum("ods,d->os", shape, weights/weights.sum())
        return np.tile(level[:, None]*shape, (1, inputs.future_known.shape[1]//96))
