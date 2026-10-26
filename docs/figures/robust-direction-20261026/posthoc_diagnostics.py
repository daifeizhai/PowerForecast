"""Descriptive diagnostics of frozen predictions only; never refit or select."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd


def primary(frame,task):
    if task!='3d':return frame
    clock=pd.to_datetime(frame.origin,utc=True)
    return frame[np.asarray((clock-clock.min())/pd.Timedelta(days=1))%3==0]


def main(folder):
    output=folder/'descriptive';output.mkdir(exist_ok=True)
    numeric=[];cohorts=[];experts=[]
    for child in (folder/'studies').iterdir():
        if not child.is_dir():continue
        route=pd.read_parquet(child/'routing.parquet')
        for task in ['4h','24h','3d']:
            d=pd.read_parquet(child/task/'test-expert-diagnostics.parquet');d=primary(d,task)
            numeric.append(dict(study=child.name,horizon=task,expert_origin_pairs=len(d),
                mean_negative_fraction=d.negative_fraction.mean(),max_abs_z=d.max_abs_z.max(),
                max_mean_level_ratio=d.mean_level_ratio.max(),
                max_raw_MAPE=d.MAPE.max(),p99_raw_MAPE=d.MAPE.quantile(.99),
                mean_history_envelope_outside_fraction=d.beyond_history_envelope_fraction.mean(),
                min_prediction_kw=d.prediction_min_kw.min(),max_prediction_kw=d.prediction_max_kw.max()))
            for (target,source),part in d.groupby(['target_domain','source_domain'],sort=False):
                experts.append(dict(study=child.name,horizon=task,target_domain=target,source_domain=source,
                    MAPE=float(np.average(part.MAPE,weights=part.mape_n)),WAPE_mean=part.WAPE.mean(),MAE_mean=part.MAE.mean(),RMSE_mean=part.RMSE.mean()))
            history=d.drop_duplicates(['target_domain','origin'])[['target_domain','origin','history_state','history_cv','hardware_group']]
            r=primary(route[route.horizon==task],task).merge(history,on=['target_domain','origin'],validate='many_to_one')
            for (model,hardware,state),g in r.groupby(['model','hardware_group','history_state'],sort=False):
                op=g.positive_opportunity
                cohorts.append(dict(study=child.name,horizon=task,model=model,published_group=hardware,history_state=state,windows=len(g),
                    mean_origin_MAPE=g.final_MAPE.mean(),mean_selected_local_MAPE=g.local_MAPE.mean(),
                    mean_delta_MAPE_pp=(g.final_MAPE-g.local_MAPE).mean(),
                    zero_transfer_pct=100*g.zero_transfer.mean() if g.alpha.notna().any() else np.nan,
                    opportunity_utilization_pct=100*(g.successful&op).sum()/op.sum() if op.any() and model in ['selected_fixed','selected_bounded','dynamic'] else np.nan))
    pd.DataFrame(numeric).to_csv(output/'numerical-distributions.csv',index=False)
    pd.DataFrame(cohorts).to_csv(output/'history-state-summary.csv',index=False)
    pd.DataFrame(experts).to_csv(output/'single-source-error-matrix.csv',index=False)
    metrics=pd.read_csv(folder/'metrics.csv');p=metrics[(metrics.evaluation=='curve')&(metrics.horizon!='3d')|(metrics.evaluation=='curve_nonoverlap')]
    p.sort_values('MAPE',ascending=False).groupby(['study','horizon','model'],sort=False).head(5).to_csv(output/'worst-domains.csv',index=False)
    (output/'README.txt').write_text('Post-test descriptive analysis only. No model fitting or parameter selection. D1/D2 are published dataset groups, not verified hardware types. History CV is a statistical proxy, not observed job state. Opportunity utilization is interpreted only for the identical-pool fixed/bounded/dynamic family.\n',encoding='utf-8')
    print(output)


if __name__=='__main__':main(Path(sys.argv[1]))
