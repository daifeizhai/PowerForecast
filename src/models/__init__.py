"""Lazy model registry: optional libraries are loaded only when requested."""

from __future__ import annotations

from src.models.base import BaseForecaster

COMPARISON_MODELS = ("naive", "seasonal_naive", "three_day_mean", "seasonal_ar", "sarima",
                     "ridge", "extra_trees", "lightgbm", "xgboost", "lstm", "tcn", "dlinear")
MODEL_NAMES = (*COMPARISON_MODELS, "normalized_ridge", "normalized_extra_trees", "sapf_similarity", "sapf")
from src.models.ml.ablation import ABLATION_MODES
MODEL_NAMES = (*MODEL_NAMES, *ABLATION_MODES)
MODEL_NAMES = (*MODEL_NAMES, "uniform_transfer", "fixed_blend", "sapf_gain", "sapf_gain_no_veto")


def create_model(name: str, params: dict | None = None, seed: int = 42) -> BaseForecaster:
    params = dict(params or {})
    if name in {"uniform_transfer", "fixed_blend", "sapf_gain", "sapf_gain_no_veto"}:
        from src.models.ml.benefit import BenefitTransferPolicy
        return BenefitTransferPolicy(name=name, seed=seed, **params)
    if name in ABLATION_MODES:
        from src.models.ml.ablation import ControlledAblationForecaster
        return ControlledAblationForecaster(name=name, seed=seed, **params)
    if name in {"normalized_ridge", "normalized_extra_trees"}:
        from src.models.ml.normalized import HistoryNormalizedForecaster
        params.setdefault("base_model", "extra_trees" if name == "normalized_extra_trees" else "ridge")
        return HistoryNormalizedForecaster(seed=seed, name=name, **params)
    if name in {"sapf_similarity", "sapf"}:
        from src.models.ml.sapf import SelectiveTransferForecaster
        return SelectiveTransferForecaster(mode="similarity" if name == "sapf_similarity" else "reliable", seed=seed, **params)
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
    if name in {"hdsf", "proposed", "chronos2", "tirex2", "timesfm25", "timesfm30", "toto2", "patchtstfm"}:
        raise ValueError(f"{name} is archived; use --legacy-config through the unified entrypoint.")
    raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODEL_NAMES)}")
