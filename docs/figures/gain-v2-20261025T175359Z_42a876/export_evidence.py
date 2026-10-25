"""Export frozen results and posthoc representation diagnostics, without fitting."""
from pathlib import Path
import sys,json,hashlib,shutil,tarfile
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from src.data.domains import DomainMetadata
from src.data.google_panel import DomainPanel,domain_windows
from src.evaluation.metrics import forecast_metrics

run=ROOT/'results/runs/20261025T175359Z_sapf_gain_v2_42a876';analysis=ROOT/'results/analysis/gain_v2'
out=ROOT/'docs/figures/gain-v2-20261025T175359Z_42a876';out.mkdir(parents=True,exist_ok=False)
m=json.loads((run/'manifest.json').read_text(encoding='utf-8'));assert m['status']=='complete'
assert json.loads((analysis/'independent-verification.json').read_text(encoding='utf-8'))['passed']
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if src.suffix=='.log':
        raw=src.read_bytes();t=raw.decode('utf-16') if raw[:2]==b'\xff\xfe' else raw.decode('utf-8')
        dst.write_text('\n'.join(line.rstrip() for line in t.splitlines())+'\n',encoding='utf-8')
    else:shutil.copyfile(src,dst)
for name in ['manifest.json','predeclared-plan.json','preregistration.json','frozen-before-test.json','audit.json','audit-amendment.json',
             'failed-manifest-before-audit-repair.json','metrics.csv','statistics.csv','risk-statistics.csv']:
    copy(run/name,out/name)
for name in ['independent-verification.json','verify_independently.py','pytest-final.log','pytest-audit-repair.log','real-run.log','results-latest.log']:
    copy(analysis/name,out/name)
copy(Path(__file__),out/'export_evidence.py')
summaries=[];risks=[];mixtures=[];matrices=[];curves=[];features=[];calibrations=[];training=[]
for group in ['m100_30','m100_15','ucc_30']:
    source=run/'groups'/group
    for name in ['gain-training.parquet','calibration-data.parquet','calibration-predictions.parquet','calibration-curves.csv','training-records.json','validation-metrics.csv']:
        copy(source/name,out/'groups'/group/name)
    for task in m['experiment']['horizon_days']:
        copy(source/task/'validation-selection.json',out/'groups'/group/task/'validation-selection.json')
        copy(source/task/'past-performance.csv',out/'groups'/group/task/'past-performance.csv')
    for objective in ['single','fixed','marginal','rank']:
        learner=joblib.load(source/'learners'/objective/'model.joblib')
        cal=learner.calibration;calibrations.append(dict(group=group,objective=objective,**{k:cal[k] for k in ['selected_risk_mix','gain_slope','gain_intercept','source_marginal_negative_prior']}))
        copy(source/'learners'/objective/'calibration.json',out/'groups'/group/'learners'/objective/'calibration.json')
        training.append(dict(group=group,objective=objective,rows=learner.training_rows,rank_pairs=learner.rank_training_pairs))
for study in m['studies']:
    child=run/'studies'/study['name'];cm=json.loads((child/'manifest.json').read_text(encoding='utf-8'));period=96 if study['frequency']=='15min' else 48
    for name in ['manifest.json','test-start.json','metrics.csv','domain-summary.csv','risk-summary.csv','calibration-curves.csv',
                 'gain-diagnostics.parquet','representation-pairs.parquet','oracle-choices.parquet','saved-model-audit.json']:
        copy(child/name,out/'studies'/study['name']/name)
    summaries.append(pd.read_csv(child/'domain-summary.csv').assign(study=study['name']))
    risks.append(pd.read_csv(child/'risk-summary.csv').assign(study=study['name']))
    curves.append(pd.read_csv(child/'calibration-curves.csv').assign(study=study['name']))
    rep=pd.read_parquet(child/'representation-pairs.parquet').groupby(['horizon','representation','source_domain','target_domain']).MAPE.mean().reset_index()
    matrices.append(rep.assign(study=study['name']))
    for (task,representation),rows in rep.groupby(['horizon','representation']):
        p=out/'matrices'/study['name'];p.mkdir(parents=True,exist_ok=True)
        rows.pivot(index='target_domain',columns='source_domain',values='MAPE').to_csv(p/(task+'_'+representation+'.csv'))
    frames={d:pd.read_parquet(child/'adapted-data'/(d+'.parquet')) for d in cm['domain_roles']['test']}
    panel=DomainPanel(frames,{d:DomainMetadata(**v) for d,v in cm['domain_metadata'].items()},{},{},pd.Timestamp(m['experiment']['protocol_anchor']),cm['source_clock_audit'])
    with threadpool_limits(1):
        for task,days in m['experiment']['horizon_days'].items():
            h=int(round(days*period));batch=domain_windows(panel,cm['domain_roles']['test'],m['experiment']['test_origins'][task],14*period,h)
            local=batch.inputs.history[:,-3*period:,0].reshape(len(batch),3,period).mean(1)[:,np.arange(h)%period].astype(float)
            for representation in ['raw','zscore','level_shape','scale_metadata']:
                bank=joblib.load(run/'groups'/cm['source_group']/task/'expert-banks'/representation/'model.joblib')
                transferred=bank.predict(batch.inputs)
                for alpha in [0.,.25,.5,.75,1.]:
                    prediction=(1-alpha)*local+alpha*transferred
                    for domain in cm['domain_roles']['test']:
                        idx=np.asarray(batch.inputs.domain_ids)==domain
                        mixtures.append(dict(study=study['name'],horizon=task,representation=representation,alpha=alpha,target_domain=domain,
                            **forecast_metrics(batch.targets[idx],prediction[idx]),exploratory_only=True,
                            scope='Post-test crossing of frozen representations and fixed alpha grid; no tuning or deployment selection.'))
summary=pd.concat(summaries,ignore_index=True);risk=pd.concat(risks,ignore_index=True);mix=pd.DataFrame(mixtures)
summary.to_csv(out/'all-domain-summary.csv',index=False);risk.to_csv(out/'all-risk-summary.csv',index=False)
pd.concat(curves,ignore_index=True).to_csv(out/'all-calibration-curves.csv',index=False)
pd.concat(matrices,ignore_index=True).to_csv(out/'source-target-matrices-long.csv',index=False)
pd.DataFrame(training).to_csv(out/'value-training-summary.csv',index=False);pd.DataFrame(calibrations).to_csv(out/'calibration-parameters.csv',index=False)
mix.to_csv(out/'posthoc-representation-mixtures-per-domain.csv',index=False)
mix_summary=mix.groupby(['study','horizon','representation','alpha']).agg(mean_MAPE=('MAPE','mean'),worst_MAPE=('MAPE','max'),mean_WAPE=('WAPE','mean'),mean_RMSE_kw=('RMSE','mean'),mean_MAE_kw=('MAE','mean')).reset_index()
mix_summary['exploratory_only']=True;mix_summary.to_csv(out/'posthoc-representation-mixtures.csv',index=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
names=[s['name'] for s in m['studies']];titles=['M100 internal | 30min','M100 -> UCC | 30min','UCC -> M100 | 30min','M100 internal | 15min']
stats=pd.read_csv(run/'statistics.csv');fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for ax,name,title in zip(axes.flat,names,titles):
    rows=[]
    for task in ['4h','24h','3d']:
        grid='nonoverlap_3d' if task=='3d' else 'daily_rolling'
        rows.append(stats[(stats.study==name)&(stats.horizon==task)&(stats.kind=='domain_and_origin_block_bootstrap')&(stats.baseline=='three_day_mean')&(stats.evaluation_grid==grid)].iloc[0])
    v=np.asarray([r.delta_MAPE_pp for r in rows]);lo=np.asarray([r.ci_low for r in rows]);hi=np.asarray([r.ci_high for r in rows])
    ax.errorbar(v,range(3),xerr=np.vstack([v-lo,hi-v]),fmt='o',color='#2563eb',capsize=4)
    ax.axvline(0,color='#6b7280',ls='--');ax.set_yticks(range(3),['4h','24h','3d nonoverlap']);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(title)
    ax.set_xlabel('MAPE difference vs 3-day local (pp; negative is better)')
fig.suptitle('Prespecified marginal Gain-v2 | Conditional node/block 95% intervals',fontsize=13);fig.savefig(out/'paired-effects.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for ax,name,title in zip(axes.flat,names,titles):
    part=mix_summary[(mix_summary.study==name)&(mix_summary.horizon=='24h')]
    matrix=part.pivot(index='representation',columns='alpha',values='mean_MAPE').reindex(['raw','zscore','level_shape','scale_metadata'])
    pic=ax.imshow(matrix,aspect='auto',cmap='YlOrRd',norm=LogNorm(vmin=max(.1,matrix.min().min()),vmax=matrix.max().max()))
    ax.set_xticks(range(5),['0%','25%','50%','75%','100%']);ax.set_yticks(range(4),['Raw','Z-score','Level + shape','Z-score + mu/sigma']);ax.set_title(title);ax.set_xlabel('Fixed transfer weight')
    for i in range(4):
        for j in range(5):ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if pic.norm(matrix.iloc[i,j])>.6 else 'black')
    fig.colorbar(pic,ax=ax,label='Mean MAPE (%)')
fig.suptitle('Post-test diagnostic only | Frozen 24h expert banks | No model selection',fontsize=13);fig.savefig(out/'representation-mixtures.png',dpi=170);plt.close(fig)
curve=pd.concat(curves,ignore_index=True);fig,axes=plt.subplots(3,4,figsize=(15,9),layout='constrained')
for j,(name,title) in enumerate(zip(names,titles)):
    for i,task in enumerate(['4h','24h','3d']):
        ax=axes[i,j];part=curve[(curve.study==name)&(curve.horizon==task)&(curve.model=='gain_v2_marginal')&(curve.kind=='risk')]
        for variant,color in [('raw','#6b7280'),('calibrated','#059669')]:
            rows=part[part.variant==variant];ax.plot(rows.mean_prediction,rows.mean_observed,'o-',color=color,label=variant,markersize=4)
        ax.plot([0,1],[0,1],'--',color='#9ca3af');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title(title+' | '+task,fontsize=9)
        if j==0:ax.set_ylabel('Observed negative marginal gain')
        if i==2:ax.set_xlabel('Predicted risk')
        if i==0 and j==0:ax.legend(fontsize=8)
fig.suptitle('Held-out risk reliability | Fixed bins | Pair counts and cluster intervals saved separately',fontsize=12);fig.savefig(out/'risk-calibration.png',dpi=160);plt.close(fig)
print(out)
print(mix_summary[(mix_summary.alpha==.5)&mix_summary.representation.isin(['zscore','level_shape'])].pivot_table(index=['study','horizon'],columns='representation',values='mean_MAPE').round(6).to_string())
