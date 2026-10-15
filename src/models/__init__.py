"""Lazy model registry: optional libraries are loaded only when requested."""

from __future__ import annotations

from src.models.base import BaseForecaster

FOUNDATION_NAMES = {"chronos2": "chronos2_long", "tirex2": "tirex2", "timesfm25": "timesfm25",
                    "timesfm30": "timesfm30", "toto2": "toto2_joint", "patchtstfm": "patchtstfm_long"}
COMPARISON_MODELS = ("naive", "seasonal_naive", "three_day_mean", "seasonal_ar", "sarima",
                     "ridge", "extra_trees", "lightgbm", "xgboost", "lstm", "tcn", "dlinear",
                     *FOUNDATION_NAMES)
MODEL_NAMES = (*COMPARISON_MODELS, "hdsf", "proposed")


def create_model(name: str, params: dict | None = None, seed: int = 42) -> BaseForecaster:
    params = dict(params or {})
    if name in FOUNDATION_NAMES:
        from src.models.foundation.forecaster import FoundationForecaster
        return FoundationForecaster(FOUNDATION_NAMES[name], **params)
    if name == "hdsf":
        from src.models.proposed.hdsf import HDSFForecaster
        return HDSFForecaster(**params)
    if name in {"three_day_mean", "seasonal_ar"}:
        from src.models.statistical.baselines import ThreeDayMeanForecaster, SeasonalARForecaster
        return ThreeDayMeanForecaster(**params) if name == "three_day_mean" else SeasonalARForecaster(**params)
    if name in {"naive", "seasonal_naive"}:
        from src.models.statistical.baselines import NaiveForecaster, SeasonalNaiveForecaster
        return NaiveForecaster(**params) if name == "naive" else SeasonalNaiveForecaster(**params)
    if name in {"ridge", "extra_trees", "lightgbm", "xgboost"}:
        from src.models.ml.tabular import TabularForecaster
        return TabularForecaster(name, params, seed)
    if name == "sarima":
        from src.models.statistical.sarima import SarimaForecaster
        return SarimaForecaster(**params)
    if name in {"lstm", "tcn", "dlinear"}:
        from src.models.deep.forecasters import TorchForecaster
        return TorchForecaster(name, seed=seed, **params)
    if name == "proposed":
        from src.models.proposed import ProposedForecaster
        return ProposedForecaster(**params)
    raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODEL_NAMES)}")
