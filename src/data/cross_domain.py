"""Disjoint source/validation/test domains and chronological SAPF boundaries."""
from __future__ import annotations

import re

import numpy as np

from src.data.domains import assert_data_usage, validate_compatible
from src.data.google_panel import DomainPanel, domain_windows


def training_origins(context: int, horizon: int, cutoff: int) -> list[int]:
    if context <= 0 or context % 96 or horizon not in {96, 288}:
        raise ValueError("SAPF supports 14-day histories and 24h/3d complete-day forecasts.")
    origins = list(range(context//96, cutoff-horizon//96+1))
    if not origins:
        raise ValueError(f"No training windows before D{cutoff}: context={context}, horizon={horizon}.")
    return origins


def _natural_key(value):
    return [int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", value)]


def assign_domain_roles(panel: DomainPanel, config: dict) -> dict:
    if config.get("domain_roles"):
        roles = {role: list(config["domain_roles"][role]) for role in ("source", "validation", "test")}
        excluded = {}
    else:
        roles = {"source": [], "validation": [], "test": []}
        excluded = {}
        eligible = {}
        reserved = set(config.get("previously_evaluated_pilot_domains", []))
        for domain in sorted(panel.series, key=_natural_key):
            meta, frame = panel.metadata[domain], panel.series[domain]
            if domain in reserved:
                excluded[domain] = "Previous pilot target; excluded from this mechanism experiment."
            elif not meta.power_domain_id or not meta.power_domain_id.startswith("pdu"):
                excluded[domain] = "Different MVPP structure; separate future protocol."
            elif panel.audit.get(domain, {}).get("ambiguous_mapping_machine_count", 0):
                excluded[domain] = "Ambiguous machine membership; do not claim equipment independence."
            elif not np.isfinite(frame.measured_power_util.to_numpy()).all():
                excluded[domain] = "Full-trace native 3/3 coverage mask is incomplete."
            else:
                eligible.setdefault(meta.site_id, []).append(domain)
        for domains in eligible.values():
            for position, domain in enumerate(domains):
                roles[("source", "validation", "test")[position % 3]].append(domain)
    sets = {role: set(domains) for role, domains in roles.items()}
    if any(len(sets[role]) != len(roles[role]) or len(sets[role]) < 2 for role in roles):
        raise ValueError("Use at least two distinct domains in each source/validation/test role.")
    if sets["source"] & sets["validation"] or sets["source"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError("Source, independent validation and test domains must be disjoint.")
    if set.union(*sets.values())-set(panel.series):
        raise ValueError("Domain roles name unavailable local domains.")
    for role, domains in roles.items():
        for domain in domains:
            assert_data_usage(panel.metadata[domain], "training" if role == "source" else role)
    validate_compatible([panel.metadata[d] for d in set.union(*sets.values())])
    return dict(**roles, excluded_domains=excluded,
                rule=config.get("domain_assignment", "Natural PDU order within each cell, position modulo 3; full-coverage masks only."))


def validate_protocol(config: dict, panel: DomainPanel, tasks: list[str]) -> dict:
    if config.get("protocol") not in {"sapf_google_holdout_v1", "sapf_m100_nodes_v1"} or config.get("context_length") != 1344:
        raise ValueError("Use the declared SAPF protocol and exactly 14 days (1344 points) of history.")
    if config.get("frequency") != "15min" or config.get("horizons") != {"24h": 96, "3d": 288}:
        raise ValueError("SAPF phase one uses 15min sampling with 24h=96 and 3d=288; 7d is archived.")
    roles = assign_domain_roles(panel, config)
    metadata = panel.metadata[roles["source"][0]]
    if (config["target_kind"], config["target_scope"]) != (metadata.target_kind, metadata.target_scope):
        raise ValueError("Experiment and dataset physical target boundaries disagree.")
    if config.get("target_unit", metadata.target_unit) != metadata.target_unit:
        raise ValueError("Experiment and dataset target units disagree.")
    if config["protocol"] == "sapf_m100_nodes_v1":
        if set.union(*(set(roles[r]) for r in ("source", "validation", "test"))) != set(panel.series):
            raise ValueError("Declare a role for every M100 node; no performance-based exclusion.")
        if (metadata.target_kind, metadata.target_scope, metadata.target_unit) != ("active_power", "node", "kW"):
            raise ValueError("M100 node experiments require measured node active power in kW.")
    plans = {}
    for task in tasks:
        horizon = config["horizons"][task]
        spec = config["temporal_protocol"][task]
        development, final = spec["development_cutoff_day"], spec["final_cutoff_day"]
        validation, test = list(spec["validation_origins"]), list(spec["test_origins"])
        if (not validation or not test or validation != sorted(set(validation)) or test != sorted(set(test))
                or min(validation) < development or max(validation)+horizon//96 > final or min(test) < final):
            raise ValueError("Validation labels must end by final fit cutoff, before test labels.")
        if max(test)*96+horizon > len(next(iter(panel.series.values()))):
            raise ValueError("Full test horizon is unavailable; never pad target labels.")
        plans[task] = dict(development_cutoff_day=development, final_cutoff_day=final,
            development_origins=training_origins(1344, horizon, development),
            final_origins=training_origins(1344, horizon, final),
            validation_origins=validation, test_origins=test)
        for role, origins in (("source", plans[task]["final_origins"]),
                              ("validation", validation), ("test", test)):
            for domain in roles[role]:
                values = panel.series[domain][panel.target_field].to_numpy()
                if any(not np.isfinite(values[day*96-1344:day*96+horizon]).all() for day in origins):
                    raise ValueError(f"{domain}: incomplete fixed {role} windows for {task}; no silent replacement.")
        for target in roles["validation"]+roles["test"]:
            overlap = set(panel.audit.get(target, {}).get("mapping_overlap_domains", [])) & set(roles["source"])
            if overlap:
                raise ValueError(f"{target}: ambiguous mapping overlaps training sources {sorted(overlap)}.")
    return dict(roles=roles, tasks=plans)


def fit_batch(panel, domains, origins, context, horizon, *, forbidden_domains=()):
    if set(domains) & set(forbidden_domains):
        raise ValueError("Held-out validation/test domains cannot enter source fitting or scaling.")
    for domain in domains:
        assert_data_usage(panel.metadata[domain], "training")
    return domain_windows(panel, domains, origins, context, horizon)
