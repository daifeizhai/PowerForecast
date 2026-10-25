"""Descriptive value recognition, calibration, and transfer-use summaries."""
import numpy as np
import pandas as pd

from src.evaluation.statistics import weighted_auc


def value_summary(table):
    rows=[]
    stage0=table[table.stage==0]
    for (task,name),part in stage0.groupby(['horizon','model'],sort=False):
        actual=part.actual_gain_pp.to_numpy();gain=part.predicted_gain_pp.to_numpy();risk=part.negative_risk.to_numpy()
        positive=actual>1e-10;negative=actual < -1e-10;predicted=gain>0;selected=part.selected.to_numpy(bool)
        prior=float(part.source_marginal_negative_prior.iloc[0]);divide=lambda a,b:float(a/b) if b else np.nan
        decisions=part.drop_duplicates(['target_domain','origin'])
        used=decisions.transfer_gate.to_numpy()>0
        success=(decisions.final_MAPE.to_numpy()<decisions.local_MAPE.to_numpy()-1e-10)&used
        opportunities=part.assign(useful=positive).groupby(['target_domain','origin']).useful.any()
        decision_index=pd.MultiIndex.from_frame(decisions[['target_domain','origin']]);opportunity=opportunities.reindex(decision_index).to_numpy()
        rows.append(dict(horizon=task,model=name,expert_pairs=len(part),actual_positive_pairs=int(positive.sum()),predicted_positive_pairs=int(predicted.sum()),
            negative_prevalence=float(negative.mean()),risk_AUROC=weighted_auc(negative,risk,np.ones(len(part))),
            risk_Brier=float(np.mean((risk-negative)**2)),raw_risk_Brier=float(np.mean((part.raw_risk.to_numpy()-negative)**2)),
            source_constant_Brier=float(np.mean((prior-negative)**2)),gain_MAE_pp=float(np.abs(gain-actual).mean()),
            risk_TP=int(((risk>=.5)&negative).sum()),risk_FP=int(((risk>=.5)&~negative).sum()),
            risk_FN=int(((risk<.5)&negative).sum()),risk_TN=int(((risk<.5)&~negative).sum()),
            negative_risk_precision=divide(((risk>=.5)&negative).sum(),(risk>=.5).sum()),
            negative_risk_recall=divide(((risk>=.5)&negative).sum(),negative.sum()),
            positive_gain_precision=divide((predicted&positive).sum(),predicted.sum()),positive_gain_recall=divide((predicted&positive).sum(),positive.sum()),
            selected_pairs=int(selected.sum()),selected_positive_precision=divide((selected&positive).sum(),selected.sum()),
            selected_positive_recall=divide((selected&positive).sum(),positive.sum()),origins=len(decisions),
            zero_transfer_fraction=float((~used).mean()),successful_transfer_fraction_all_origins=float(success.mean()),
            successful_transfer_fraction_used_origins=divide(success.sum(),used.sum()),
            one_step_positive_opportunity_origins=int(opportunity.sum()),successful_opportunity_capture=divide((success&opportunity).sum(),opportunity.sum()),
            mean_transfer_gate=float(decisions.transfer_gate.mean()),
            definition='Stage-0 marginal 50% fusion labels shared across objectives; success compares final curve MAPE with three-day local history.'))
    return pd.DataFrame(rows)


def calibration_curves(table):
    """Fixed bins; never choose test quantiles or recalibrate after testing."""
    rows=[]
    for (task,model),part in table[table.stage==0].groupby(['horizon','model'],sort=False):
        actual=part.actual_gain_pp.to_numpy();negative=actual < -1e-10
        variants=[('risk','raw',part.raw_risk.to_numpy(),negative,np.linspace(0,1,11)),
                  ('risk','calibrated',part.negative_risk.to_numpy(),negative,np.linspace(0,1,11)),
                  ('gain_pp','calibrated',part.predicted_gain_pp.to_numpy(),actual,[-np.inf,-5,-1,-.1,0,.1,1,5,np.inf])]
        if model!='gain_v2_rank':variants.append(('gain_pp','raw',part.raw_gain_pp.to_numpy(),actual,[-np.inf,-5,-1,-.1,0,.1,1,5,np.inf]))
        for kind,variant,predicted,observed,edges in variants:
            codes=np.searchsorted(edges[1:-1],predicted,side='right')
            for code in np.unique(codes):
                keep=codes==code
                rows.append(dict(horizon=task,model=model,kind=kind,variant=variant,bin=int(code),lower=edges[code],upper=edges[code+1],
                    pairs=int(keep.sum()),target_domains=part.loc[keep,'target_domain'].nunique(),mean_prediction=float(predicted[keep].mean()),
                    mean_observed=float(observed[keep].mean()),mean_absolute_error=float(np.abs(predicted[keep]-observed[keep]).mean())))
    return pd.DataFrame(rows)
