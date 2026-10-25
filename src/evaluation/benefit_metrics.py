"""Held-out expert-pair gain/risk diagnostics; never used for fitting or selection."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def risk_summary(table):
    table = table[(table.model == "sapf_gain") & (table.split == "test")]
    result = []
    for task, part in table.groupby("horizon"):
        actual_negative = part.actual_gain_pp.to_numpy() < -1e-10
        risk = part.negative_risk.to_numpy()
        predicted_negative = risk >= .5
        positive_gain = part.predicted_gain_pp.to_numpy() > 0
        actually_useful = part.actual_gain_pp.to_numpy() > 1e-10
        selected = part.weight.to_numpy() > 0
        true_positive = int((actual_negative & predicted_negative).sum())
        false_positive = int((~actual_negative & predicted_negative).sum())
        false_negative = int((actual_negative & ~predicted_negative).sum())
        true_negative = int((~actual_negative & ~predicted_negative).sum())
        safe_divide = lambda numerator, denominator: float(numerator/denominator) if denominator else None
        decisions = part.drop_duplicates(["target_domain", "origin"])
        spearman = part[["predicted_gain_pp", "actual_gain_pp"]].corr(method="spearman").iloc[0, 1]
        result.append(dict(horizon=task, expert_pairs=len(part), negative_pairs=int(actual_negative.sum()),
            negative_prevalence=float(actual_negative.mean()),
            risk_AUROC=float(roc_auc_score(actual_negative, risk)) if np.unique(actual_negative).size == 2 else None,
            risk_Brier=float(np.mean((risk-actual_negative)**2)),
            risk_precision=safe_divide(true_positive, true_positive+false_positive),
            risk_recall=safe_divide(true_positive, true_positive+false_negative),
            risk_TP=true_positive, risk_FP=false_positive, risk_FN=false_negative, risk_TN=true_negative,
            gain_MAE_pp=float(np.abs(part.predicted_gain_pp-part.actual_gain_pp).mean()),
            gain_spearman=float(spearman) if np.isfinite(spearman) else None,
            predicted_positive_gain_pairs=int(positive_gain.sum()),
            positive_gain_precision=safe_divide(int((positive_gain & actually_useful).sum()), int(positive_gain.sum())),
            selected_pairs=int(selected.sum()), selected_negative_fraction=safe_divide(int((selected & actual_negative).sum()), int(selected.sum())),
            test_origins=len(decisions), zero_transfer_fraction=float((decisions.transfer_gate == 0).mean()),
            mean_transfer_gate=float(decisions.transfer_gate.mean()),
            interpretation="Repeated source/target/origin pairs and overlapping horizons; descriptive, not IID samples. Risk threshold fixed at 0.5; policy admission threshold selected on source-validation nodes."))
    return pd.DataFrame(result)
