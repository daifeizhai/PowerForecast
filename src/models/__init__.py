"""Lazy model registry: optional libraries are loaded only when requested."""

from __future__ import annotations

from src.models.base import BaseForecaster

COMPARISON_MODELS = ("naive", "seasonal_naive", "three_day_mean", "seasonal_ar", "sarima",
                     "ridge", "extra_trees", "lightgbm", "xgboost", "lstm", "tcn", "dlinear")
MODEL_NAMES = (*COMPARISON_MODELS, "normalized_ridge", "normalized_extra_trees", "sapf_similarity", "sapf")
from src.models.ml.ablation import ABLATION_MODES
MODEL_NAMES = (*MODEL_NAMES, *ABLATION_MODES)
MODEL_NAMES = (*MODEL_NAMES, "uniform_transfer", "fixed_blend", "sapf_gain", "sapf_gain_no_veto")
MODEL_NAMES = (*MODEL_NAMES, "ridge_raw", "ridge_zscore", "ridge_level_shape", "ridge_scale_metadata",
               "fixed_blend_25", "fixed_blend_50", "fixed_blend_75", "gain_v2_single", "gain_v2_fixed",
               "gain_v2_marginal", "gain_v2_rank", "gain_v2_forced", "gain_v2_uncalibrated")


def create_model(name: str, params: dict | None = None, seed: int = 42) -> BaseForecaster:
    params = dict(params or {})
    if name in {"ridge_raw", "ridge_zscore", "ridge_level_shape", "ridge_scale_metadata"}:
        from src.models.ml.level_shape import RepresentationRidge
        return RepresentationRidge(representation=name[6:], name=name, seed=seed, **params)
    if name in {"fixed_blend_25", "fixed_blend_50", "fixed_blend_75"}:
        from src.models.ml.gain_v2 import FixedExpertBlend
        return FixedExpertBlend(name=name, strength=int(name.rsplit('_',1)[1])/100, seed=seed, **params)
    if name.startswith("gain_v2_"):
        from src.models.ml.gain_v2 import GainV2Policy
        return GainV2Policy(name=name, seed=seed, **params)
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
