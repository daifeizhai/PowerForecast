"""Global tests stay sealed until every direction and source-only meta-model freezes."""
from hashlib import sha256
import json

import pandas as pd


def audit_transfer_suite(folder):
    from src.evaluation.cross_domain import audit_cross_domain_run
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((folder/"frozen-before-test.json").read_text(encoding="utf-8"))
    unchanged = all(sha256((folder/path).read_bytes()).hexdigest() == expected for path, expected in frozen["model_sha256"].items())
    artifacts = all(sha256((folder/path).read_bytes()).hexdigest() == expected for path, expected in frozen["training_artifact_sha256"].items())
    roles = manifest["experiment"]["domain_roles"]
    globally_test = set(roles["m100"]["test"]+roles["ucc"]["test"])
    retired = set(roles["retired_m100_previous_test"])
    exclusions = not retired & set(sum((roles[g][r] for g in ("m100", "ucc") for r in ("source", "validation", "test")), []))
    meta_valid, meta_rows = True, 0
    for group in ("m100", "ucc"):
        table = pd.read_csv(folder/"groups"/group/"benefit/gain-training.csv")
        priors = pd.read_csv(folder/"groups"/group/"benefit/past-performance.csv")
        legal = set(roles[group]["source"])
        identities = set(table.expert_domain) | set(table.pseudo_target) | set(priors.expert_domain) | set(priors.target_domain)
        meta_valid &= (identities <= legal and not identities & globally_test and not (table.expert_domain == table.pseudo_target).any()
                       and bool((table.prior_label_end_day <= table.origin_day).all()) and bool((table.expert_fit_cutoff_day <= table.origin_day).all())
                       and bool((table.label_end_day <= 50).all()) and bool((priors.label_end_day <= 24).all()))
        meta_rows += len(table)
    children, freeze_first, target_blind, declared_roles = {}, True, True, True
    for study in manifest["studies"]:
        child = folder/"studies"/study["name"]
        children[study["name"]] = audit_cross_domain_run(child)
        started = json.loads((child/"test-start.json").read_text(encoding="utf-8"))
        freeze_first &= pd.Timestamp(frozen["frozen_at_utc"]) <= pd.Timestamp(started["started_at_utc"])
        cf = json.loads((child/"frozen-before-test.json").read_text(encoding="utf-8"))
        cm = json.loads((child/"manifest.json").read_text(encoding="utf-8"))
        declared_roles &= (set(cm["domain_roles"]["source"]) == set(roles[study["source_group"]]["source"])
            and set(cm["domain_roles"]["validation"]) == set(roles[study["source_group"]]["validation"])
            and set(cm["domain_roles"]["test"]) == set(roles[study["test_group"]]["test"]))
        expected = set(roles[study["source_group"]]["validation"])
        target_blind &= all(set(r["calibration_domains"]) <= expected for r in cf["training_records"])
        for task in manifest["experiment"]["horizons"]:
            records = {r["model"]: r for r in cf["training_records"] if r["horizon"] == task}
            target_blind &= records["sapf_gain"]["selected"]["selected"] == records["sapf_gain_no_veto"]["selected"]["selected"]
        for filename in ("gain-training.csv", "past-performance.csv"):
            artifacts &= sha256((child/"benefit"/filename).read_bytes()).hexdigest() == sha256((folder/"groups"/study["source_group"]/"benefit"/filename).read_bytes()).hexdigest()
    shared_equal = True
    for task in manifest["experiment"]["horizons"]:
        for name in manifest["model_config"]["default_models"]:
            a, b = (folder/"studies"/s/task/name/"model.joblib" for s in ("m100_gain", "m100_to_ucc"))
            shared_equal &= sha256(a.read_bytes()).hexdigest() == sha256(b.read_bytes()).hexdigest()
    passed = unchanged and artifacts and exclusions and meta_valid and freeze_first and target_blind and declared_roles and shared_equal and all(c["passed"] for c in children.values())
    return dict(passed=bool(passed), all_models_frozen_before_any_test=bool(freeze_first), model_hashes_unchanged=bool(unchanged),
        source_training_artifacts_unchanged=bool(artifacts), retired_m100_tests_excluded=bool(exclusions),
        source_only_leave_domain_meta_labels=bool(meta_valid), meta_training_rows=meta_rows,
        target_dataset_labels_not_used_for_selection=bool(target_blind), m100_source_models_shared_without_retuning=bool(shared_equal),
        globally_declared_roles_preserved=bool(declared_roles),
        prediction_checks=sum(c["prediction_checks"] for c in children.values()), restore_checks=sum(c["restore_checks"] for c in children.values()),
        children=children)
