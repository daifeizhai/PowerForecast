"""Predeclared temporal cutoffs and leave-one-domain-out membership."""
from __future__ import annotations

import numpy as np

from src.data.domains import validate_compatible
from src.data.google_panel import DomainPanel, domain_windows


def training_origins(context: int, horizon: int, cutoff: int) -> list[int]:
    if context <= 0 or horizon not in {96, 672} or context % 96:
        raise ValueError("Cross-domain pilot requires whole-day history and 24h/7d horizons.")
    origins = list(range(context//96, cutoff-horizon//96+1))
    if not origins:
        raise ValueError(f"No training windows: {context//96} history days + {horizon//96} label days exceed D{cutoff}.")
    return origins


def validate_protocol(config: dict, panel: DomainPanel, tasks: list[str]) -> dict:
    """Validate the full plan before fitting or consulting any test errors."""
    context = int(config["context_length"])
    if config.get("split_protocol") != "leave_one_power_domain_out":
        raise ValueError("Declare leave_one_power_domain_out explicitly.")
    if config.get("normalization") != "source_training_only":
        raise ValueError("This pilot supports source-training-only normalization; no target fitting.")
    targets, singles = config["target_domains"], config["single_source_domains"]
    if (not targets or len(set(targets)) != len(targets) or not singles or len(set(singles)) != len(singles)
            or set(targets) & set(singles)):
        raise ValueError("Predeclare distinct target and single-source domains with no overlap.")
    unknown = (set(targets) | set(singles))-set(panel.series)
    if unknown:
        raise ValueError(f"Unknown domains: {sorted(unknown)}")
    validate_compatible(list(panel.metadata.values()))
    first = next(iter(panel.metadata.values()))
    if (config["target_kind"], config["target_scope"]) != (first.target_kind, first.target_scope):
        raise ValueError("Experiment physical target differs from domain contracts.")
    plans = {}
    for task in tasks:
        horizon = int(config["horizons"][task])
        spec = config["temporal_protocol"][task]
        final_cutoff = int(spec["final_cutoff_day"])
        development = list(spec["development_origins"])
        test = list(spec["test_origins"])
        if (not test or test != sorted(set(test)) or development != sorted(set(development))
                or any(not isinstance(day, int) for day in [*development, *test])):
            raise ValueError("Use sorted distinct integer origin days.")
        if min(test) < final_cutoff or max(test)*96+horizon > len(next(iter(panel.series.values()))):
            raise ValueError("Test origins must follow final training and have complete labels.")
        final_origins = training_origins(context, horizon, final_cutoff)
        dev_cutoff = spec.get("development_cutoff_day")
        if development:
            if dev_cutoff is None or min(development) < dev_cutoff or max(development)*96+horizon > final_cutoff*96:
                raise ValueError("Source development labels must follow development training and end by final cutoff.")
            development_training = training_origins(context, horizon, int(dev_cutoff))
        else:
            if dev_cutoff is not None or not spec.get("no_validation_reason"):
                raise ValueError("Missing development windows need an explicit reason and no fitted selection stage.")
            development_training = []
        eligible, excluded = [], {}
        for domain in panel.series:
            y = panel.series[domain].measured_power_util.to_numpy()
            valid = [day for day in final_origins if np.isfinite(y[day*96-context:day*96+horizon]).all()]
            if len(valid) < int(config["minimum_train_windows_per_domain"]):
                excluded[domain] = f"Only {len(valid)} complete final training windows."
                continue
            if development and any(not np.isfinite(y[day*96-context:day*96+horizon]).all()
                                   for day in [*development_training, *development]):
                excluded[domain] = "Incomplete source development training or validation window."
                continue
            eligible.append(domain)
        if (set(targets) | set(singles))-set(eligible):
            raise ValueError(f"Predeclared pilot/reference domains fail training coverage: {excluded}")
        folds = {}
        for target in targets:
            # Eligibility uses only fixed missingness masks, never target test errors.
            mask = np.isfinite(panel.series[target].measured_power_util.to_numpy())
            if any(not mask[day*96-context:day*96+horizon].all() for day in test):
                raise ValueError(f"{target}: incomplete fixed test window; do not silently replace target/origins.")
            requested = config["source_domains"]
            if requested == "all_other":
                sources = [d for d in eligible if d != target]
            elif isinstance(requested, list) and len(requested) == len(set(requested)):
                if target in requested or set(requested)-set(eligible):
                    raise ValueError("Explicit sources include target or unavailable domains.")
                sources = list(requested)
            else:
                raise ValueError("source_domains must be all_other or an explicit distinct list.")
            if len(sources) < 2 or not set(singles).issubset(sources):
                raise ValueError("Multi-source pool must contain each predeclared single-source baseline and >=2 sources.")
            mapped_overlap = set(panel.audit.get(target, {}).get("mapping_overlap_domains", [])) & set(sources)
            if mapped_overlap:
                raise ValueError(f"{target}: ambiguous machine mapping overlaps sources {sorted(mapped_overlap)}; audit or explicitly revise the source pool.")
            folds[target] = dict(source_domains=sources, target_domain=target)
        plans[task] = dict(final_origins=final_origins, development_training_origins=development_training,
                           development_origins=development, test_origins=test, final_cutoff_day=final_cutoff,
                           development_cutoff_day=dev_cutoff, excluded_domains=excluded, folds=folds,
                           no_validation_reason=spec.get("no_validation_reason"))
    return plans


def fit_batch(panel, domains, origins, context, horizon, *, forbidden_domain=None):
    """Reject a target domain at the actual training boundary, before scaling/fitting."""
    if forbidden_domain is not None and forbidden_domain in domains:
        raise ValueError(f"Held-out target {forbidden_domain} cannot enter source fitting.")
    batch = domain_windows(panel, domains, origins, context, horizon)
    if forbidden_domain in (batch.inputs.domain_ids or ()):
        raise AssertionError("Held-out target reached a training batch.")
    return batch
