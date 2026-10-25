"""Independent saved-score, source-clock, routing and label-boundary checks; never fit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import argparse,json,hashlib
import numpy as np,pandas as pd,joblib
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from src.data.domains import DomainMetadata
from src.data.google_panel import DomainPanel,domain_windows
import src.models.ml.tabular

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',default='results/runs/20261025T143438Z_sapf_transfer_771b85')
parser.add_argument('--output',required=True)
args=parser.parse_args();run=ROOT/args.run;out=ROOT/args.output
if out.exists():raise FileExistsError('Use a new verification output path.')
m=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
assert m['status']=='complete'
prereg=json.loads((ROOT/'results/analysis/benefit_transfer_v1/preregistration.json').read_text(encoding='utf-8'))
metric_checks=[];mixtures=[];risk_checks=[];zero_origins=0
for study in m['studies']:
 child=run/'studies'/study['name'];cm=json.loads((child/'manifest.json').read_text(encoding='utf-8'))
 data=pd.read_csv(child/'metrics.csv')
 for (domain,task,name,split),part in data.groupby(['target_domain','horizon','model','split']):
  t=pd.read_parquet(child/domain/task/name/(split+'_predictions.parquet')).sort_values(['origin','lead'])
  h=m['experiment']['horizons'][task]
  a=t.actual.to_numpy().reshape(-1,h);p=t.predicted.to_numpy().reshape(-1,h)
  cache=pd.read_parquet(child/'adapted-data'/(domain+'.parquet'))
  expected=cache.loc[pd.DatetimeIndex(t.timestamp)]
  assert np.array_equal(expected.node_power_kw.to_numpy(np.float32),a.reshape(-1))
  assert np.array_equal(expected.source_clock_timestamp.to_numpy(),t.source_clock_timestamp.to_numpy())
  for row in part.itertuples():
   x,y=a,p
   if row.evaluation=='daily_mean':x=x.reshape(len(x),-1,48).mean(-1);y=y.reshape(len(y),-1,48).mean(-1)
   x=x.astype(float);y=y.astype(float);error=x-y;mask=np.abs(x)>1e-8
   value=dict(MAPE=100*np.mean(np.abs(error[mask])/np.abs(x[mask])),WAPE=100*np.abs(error).sum()/np.abs(x).sum(),
       MAE=np.abs(error).mean(),RMSE=np.sqrt((error**2).mean()),n=x.size,mape_n=int(mask.sum()),mape_excluded=int((~mask).sum()))
   assert all(np.isclose(getattr(row,k),v,rtol=1e-10,atol=1e-10) for k,v in value.items()),(study['name'],domain,task,name,split)
   metric_checks.append(dict(study=study['name'],domain=domain,horizon=task,model=name,split=split,evaluation=row.evaluation,passed=True))
 print(study['name'],'independent score/clock checks complete',flush=True)
 frames={d:pd.read_parquet(child/'adapted-data'/(d+'.parquet')) for d in cm['domain_metadata']}
 metadata={d:DomainMetadata(**v) for d,v in cm['domain_metadata'].items()}
 panel=DomainPanel(frames,metadata,{}, {},pd.Timestamp(cm['trace_anchor_utc']),{})
 diag=pd.read_csv(child/'gain-diagnostics.csv');diag.origin=pd.to_datetime(diag.origin,utc=True)
 risk_summary=pd.read_csv(child/'risk-summary.csv').set_index('horizon')
 with threadpool_limits(1):
  for task,h in m['experiment']['horizons'].items():
   batch=domain_windows(panel,cm['domain_roles']['test'],cm['temporal_plan'][task]['test_origins'],672,h)
   model=joblib.load(child/task/'sapf_gain/model.joblib')
   pieces=[model.components(batch.inputs.take(slice(i,i+32))) for i in range(0,len(batch),32)]
   experts,local,predicted_gain,risk=[np.concatenate([piece[k] for piece in pieces]) for k in range(4)]
   n,s=experts.shape[:2];sources=model.source_domains
   coordinates=pd.MultiIndex.from_tuples([(d,o,source) for d,o in zip(batch.inputs.domain_ids,batch.inputs.origins) for source in sources],names=['target_domain','origin','source_domain'])
   actual=batch.targets.astype(float)
   local_score=100*np.mean(np.abs(actual-local)/np.abs(actual),axis=1)
   expert_score=100*np.mean(np.abs(actual[:,None]-experts)/np.abs(actual[:,None]),axis=2)
   actual_gain=local_score[:,None]-expert_score
   subset=diag[(diag.horizon==task)&(diag.split=='test')]
   gain_rows=subset[subset.model=='sapf_gain'].set_index(['target_domain','origin','source_domain']).reindex(coordinates)
   assert np.allclose(gain_rows.actual_gain_pp.to_numpy().reshape(n,s),actual_gain,rtol=1e-10,atol=1e-10)
   assert np.allclose(gain_rows.predicted_gain_pp.to_numpy().reshape(n,s),predicted_gain,rtol=1e-10,atol=1e-10)
   assert np.allclose(gain_rows.negative_risk.to_numpy().reshape(n,s),risk,rtol=1e-12,atol=1e-12)
   y=(actual_gain.reshape(-1)<-1e-10);q=risk.reshape(-1)
   if np.unique(y).size==2:
    ranks=rankdata(q);pos=int(y.sum());neg=len(y)-pos
    auc=(ranks[y].sum()-pos*(pos+1)/2)/(pos*neg)
    assert np.isclose(auc,risk_summary.loc[task,'risk_AUROC'],atol=1e-12)
   assert np.isclose(np.mean((q-y)**2),risk_summary.loc[task,'risk_Brier'],atol=1e-12)
   gate_gain=gain_rows.transfer_gate.to_numpy().reshape(n,s)[:,0]
   for name in ['uniform_transfer','fixed_blend','sapf_gain','sapf_gain_no_veto']:
    rows=subset[subset.model==name].set_index(['target_domain','origin','source_domain']).reindex(coordinates)
    weights=rows.weight.to_numpy().reshape(n,s);gate=rows.transfer_gate.to_numpy().reshape(n,s)[:,0]
    expected=gate[:,None]*np.einsum('ns,nsh->nh',weights,experts)+(1-gate[:,None])*local
    if name=='sapf_gain':
     no_positive=(predicted_gain<=0).all(1)
     assert np.array_equal(gate[no_positive],np.zeros(no_positive.sum()))
     empty=gate==0;zero_origins+=int(empty.sum())
     assert np.array_equal(expected[empty],local[empty].astype(float))
    elif name=='uniform_transfer':assert np.allclose(weights,1/s) and np.all(gate==1)
    elif name=='fixed_blend':assert np.allclose(weights,1/s) and np.all(gate==.5)
    for domain in cm['domain_roles']['test']:
     idx=np.flatnonzero(np.asarray(batch.inputs.domain_ids)==domain)
     table=pd.read_parquet(child/domain/task/name/'test_predictions.parquet').sort_values(['origin','lead'])
     difference=float(np.max(np.abs(expected[idx]-table.predicted.to_numpy().reshape(-1,h))))
     assert difference<1e-6,(study['name'],task,name,domain,difference)
     mixtures.append(dict(study=study['name'],horizon=task,model=name,domain=domain,max_absolute_kw_difference=difference,passed=True))
   risk_checks.append(dict(study=study['name'],horizon=task,pairs=n*s,passed=True))
 print(study['name'],'independent gain/risk/mixture checks complete',flush=True)
meta_checks=[]
for group in ['m100','ucc']:
 table=pd.read_csv(run/'groups'/group/'benefit/gain-training.csv')
 assert set(table.pseudo_target)<=set(m['experiment']['domain_roles'][group]['source'])
 assert not (table.expert_domain==table.pseudo_target).any()
 assert (table.expert_fit_cutoff_day<=table.origin_day).all()
 assert (table.prior_label_end_day<=table.origin_day).all() and (table.label_end_day<=50).all()
 learner=joblib.load(run/'groups'/group/'joint_predictor/model.joblib')
 assert learner.training_rows==len(table) and learner.source_domains==sorted(m['experiment']['domain_roles'][group]['source'])
 assert np.isclose(learner.training_gain_mean,table.actual_gain_pp.mean())
 assert np.isclose(learner.negative_prevalence,(table.actual_gain_pp < -1e-10).mean())
 meta_checks.append(dict(group=group,rows=len(table),source_domains=learner.source_domains,passed=True))
for path,value in prereg['computational_source_sha256'].items():
 assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
frozen=json.loads((run/'frozen-before-test.json').read_text(encoding='utf-8'))
assert pd.Timestamp(prereg['created_at_utc'])<pd.Timestamp(frozen['frozen_at_utc'])
report=dict(passed=True,metric_checks=len(metric_checks),mixture_checks=len(mixtures),risk_pair_checks=sum(r['pairs'] for r in risk_checks),
 zero_transfer_origins_verified=zero_origins,maximum_mixture_difference_kw=max(r['max_absolute_kw_difference'] for r in mixtures),
 computational_source_and_fit_configs_match_preregistration=True,all_gain_training_source_only=True,
 metric_records=metric_checks,mixture_records=mixtures,risk_records=risk_checks,meta_records=meta_checks,
 scope='Saved models only, no refit or tuning; independent NumPy metrics and rank-sum AUROC, original source clocks, standalone expert mixture reconstruction.')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print({k:v for k,v in report.items() if not isinstance(v,list)},flush=True)
