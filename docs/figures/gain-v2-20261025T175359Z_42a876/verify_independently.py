"""Read-only NumPy metrics, raw-native labels, expert matrices and route reconstruction."""
from pathlib import Path
from datetime import datetime,timezone
from io import BytesIO
import argparse,hashlib,json,sys,tarfile
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import joblib,numpy as np,pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from src.data.domains import DomainMetadata
from src.data.google_panel import DomainPanel,domain_windows

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',default='results/runs/20261025T175359Z_sapf_gain_v2_42a876')
parser.add_argument('--output',required=True);args=parser.parse_args()
run=ROOT/args.run;out=ROOT/args.output
if out.exists():raise FileExistsError('Verification output must be a new file.')
manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'));assert manifest['status']=='complete'
config=manifest['experiment'];prereg=json.loads((run/'preregistration.json').read_text(encoding='utf-8'))
frozen=json.loads((run/'frozen-before-test.json').read_text(encoding='utf-8'))
amend=json.loads((run/'audit-amendment.json').read_text(encoding='utf-8')) if (run/'audit-amendment.json').exists() else None
changed=[]
for name,digest in prereg['computational_source_sha256'].items():
    actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if actual!=digest:
        assert name=='src/evaluation/gain_v2_audit.py' and amend
        assert amend['allowed_changes'][name]==dict(before_sha256=digest,after_sha256=actual)
        changed.append(name)
if amend:
    assert all(hashlib.sha256((run/name).read_bytes()).hexdigest()==digest for name,digest in amend['preserved_outcome_sha256'].items())
assert len(frozen['model_sha256'])==237 and pd.Timestamp(prereg['created_at_utc'])<pd.Timestamp(frozen['frozen_at_utc'])

native_records=[]
def check_native(raw,time,power,domain,group,native):
    for cadence in (['15','30'] if group=='m100' else ['30']):
        path=run/'adapted-data'/(group+'_'+cadence)/(domain+'.parquet')
        if not path.exists():continue
        cache=pd.read_parquet(path);minutes=int(cadence)
        original=pd.DatetimeIndex(pd.to_datetime(cache.source_clock_timestamp)).as_unit('ns')
        times=pd.DatetimeIndex(pd.to_datetime(raw[time])).as_unit('ns')
        start=original[0];stop=start+pd.Timedelta(days=config['days'])
        keep=(times>=start)&(times<stop);times=times[keep]
        values=pd.to_numeric(raw.loc[keep,power]).to_numpy(dtype=float)
        grid=pd.date_range(start,periods=config['days']*1440//native,freq=f'{native}min')
        positions=grid.get_indexer(times);assert (positions>=0).all()
        count=np.bincount(positions,minlength=len(grid))
        flags=np.where(count==0,1,0).astype(np.uint8);flags[count>1]|=2
        raw_flags=np.where(~np.isfinite(values),4,np.where(values<0,8,0)).astype(np.uint8)
        np.bitwise_or.at(flags,positions,raw_flags)
        slots=np.full(len(grid),np.nan);slots[positions]=values
        size=minutes//native;mask=np.bitwise_or.reduce(flags.reshape(-1,size),axis=1)
        independent=slots.reshape(-1,size).mean(1)/1000.;independent[mask!=0]=np.nan
        assert np.array_equal(mask,cache.quality_mask.to_numpy())
        assert np.allclose(independent,cache.node_power_kw.to_numpy(),rtol=0,atol=1e-12,equal_nan=True)
        assert [t.isoformat() for t in grid[::size]]==cache.source_clock_timestamp.tolist()
        assert [(t+pd.Timedelta(minutes=minutes)).isoformat() for t in grid[::size]]==cache.source_available_at.tolist()
        assert np.array_equal(pd.DatetimeIndex(cache.available_at).asi8,(cache.index+pd.Timedelta(minutes=minutes)).asi8)
        native_records.append(dict(domain=domain,cadence=cadence,labels=len(cache),bad_labels=int((mask!=0).sum()),passed=True))
assigned={g:set(sum(roles.values(),[])) for g,roles in config['domain_roles'].items()}
for rack in [0,1]:
    path=ROOT/'data/raw/M100'/f'{rack}.tar'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['data_provenance']['M100/'+path.name]['sha256']
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            domain=f'm100_rack{rack}_node{Path(member.name).stem}'
            if domain not in assigned['m100']:continue
            raw=pd.read_parquet(BytesIO(archive.extractfile(member).read()),columns=['timestamp','total_power_avg'])
            check_native(raw,'timestamp','total_power_avg',domain,'m100',15)
for cohort in ['d1','d2']:
    path=ROOT/'data/raw/UCC2023'/f'dataset{cohort[-1]}_combined.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['data_provenance']['UCC/'+path.name]['sha256']
    raw=pd.read_csv(path,usecols=['Series','Time','Power']);raw.Series=raw.Series.str.strip()
    for host,part in raw.groupby('Series'):
        domain='ucc_'+cohort+'_'+host.replace('-','_')
        if domain in assigned['ucc']:check_native(part,'Time','Power',domain,'ucc',10)
print('Independent native-power, quality-bit and source-clock checks complete.',flush=True)

def metric(actual,predicted):
    a=np.asarray(actual,float);p=np.asarray(predicted,float);error=np.abs(a-p);valid=np.abs(a)>1e-8
    return dict(MAPE=100*np.mean(error[valid]/np.abs(a[valid])) if valid.any() else None,WAPE=100*error.sum()/np.abs(a).sum() if np.abs(a).sum()>1e-8 else None,
        RMSE=np.sqrt(np.mean((a-p)**2)),MAE=error.mean(),n=a.size,mape_n=int(valid.sum()),mape_excluded=int((~valid).sum()))

def losses(actual,prediction):
    actual=np.asarray(actual,float);prediction=np.asarray(prediction,float)
    valid=np.abs(actual)>1e-8
    return 100*np.divide(np.abs(actual-prediction),np.abs(actual),out=np.zeros_like(prediction),where=valid).sum(-1)/valid.sum(-1)

def expert_predictions(bank,inputs):
    raw=inputs.history[:,:,0].astype(float);mu=raw.mean(1,keepdims=True);sigma=np.maximum(raw.std(1,keepdims=True),np.maximum(.01*np.abs(mu),1e-6))
    z=(raw-mu)/sigma;known=np.concatenate([inputs.history[:,-1,1:],inputs.future_known[:,0],inputs.future_known[:,-1]],axis=1)
    predictions=[]
    for expert in bank.experts:
        representation=expert.representation
        x=raw if representation=='raw' else z if representation=='zscore' else np.concatenate([z,mu,sigma],axis=1) if representation=='scale_metadata' else np.concatenate([raw,z],axis=1)
        values=np.asarray(expert.estimator.predict(np.concatenate([x,known],axis=1))).reshape(len(inputs),-1)
        predictions.append(np.maximum(values if representation=='raw' else values*sigma+mu,0.))
    return np.stack(predictions,axis=1)

metric_checks=0;route_checks=0;risk_pairs=0;zero_origins=0;representation_pairs=0;max_route_difference=0.;oracle_rounding_pp=0.
for study in config['studies']:
    child=run/'studies'/study['name'];cm=json.loads((child/'manifest.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(child/'metrics.csv');period=96 if study['frequency']=='15min' else 48
    frames={d:pd.read_parquet(child/'adapted-data'/(d+'.parquet')) for d in cm['domain_roles']['test']}
    metas={d:DomainMetadata(**m) for d,m in cm['domain_metadata'].items()}
    panel=DomainPanel(frames,metas,{}, {},pd.Timestamp(config['protocol_anchor']),cm['source_clock_audit'])
    pair=pd.read_parquet(child/'gain-diagnostics.parquet');pair.origin=pd.to_datetime(pair.origin,utc=True)
    rep=pd.read_parquet(child/'representation-pairs.parquet');rep.origin=pd.to_datetime(rep.origin,utc=True)
    for (task,name),part in metrics.groupby(['horizon','model']):
        table=pd.read_parquet(child/'predictions'/task/(name+'.parquet'));h=int(round(config['horizon_days'][task]*period))
        for domain,rows in part.groupby('target_domain'):
            t=table[table.domain_id==domain].sort_values(['origin','lead'])
            a=t.actual.to_numpy(np.float32).reshape(-1,h);p=t.predicted.to_numpy().reshape(-1,h)
            expected=frames[domain].loc[pd.DatetimeIndex(t.timestamp)]
            assert np.array_equal(a.reshape(-1),expected.node_power_kw.to_numpy(np.float32))
            assert np.array_equal(t.source_clock_timestamp.to_numpy(),expected.source_clock_timestamp.to_numpy())
            for row in rows.itertuples():
                x,y=a,p
                if row.evaluation=='daily_mean':
                    if h<period:values=dict(MAPE=None,WAPE=None,RMSE=None,MAE=None,n=0,mape_n=0,mape_excluded=0)
                    else:values=metric(a.reshape(len(a),-1,period).mean(-1),p.reshape(len(p),-1,period).mean(-1))
                else:
                    if row.evaluation=='curve_nonoverlap':x,y=a[::3],p[::3]
                    values=metric(x,y)
                assert all((v is None and pd.isna(getattr(row,k))) or (v is not None and np.isclose(v,getattr(row,k),rtol=1e-10,atol=1e-10)) for k,v in values.items()),(study['name'],task,name,domain,row.evaluation)
                metric_checks+=1
    print(study['name'],'NumPy metric checks complete',flush=True)
    with threadpool_limits(1):
        for task,days in config['horizon_days'].items():
            h=int(round(days*period));batch=domain_windows(panel,cm['domain_roles']['test'],config['test_origins'][task],14*period,h)
            local=batch.inputs.history[:,-3*period:,0].reshape(len(batch),3,period).mean(1)[:,np.arange(h)%period]
            coordinates=pd.MultiIndex.from_tuples([(d,o,lead) for d,o in zip(batch.inputs.domain_ids,batch.inputs.origins) for lead in range(1,h+1)],names=['domain_id','origin','lead'])
            stored={}
            for name in manifest['model_config']['default_models']+['oracle_posthoc']:
                t=pd.read_parquet(child/'predictions'/task/(name+'.parquet')).set_index(['domain_id','origin','lead']).reindex(coordinates)
                assert not t.predicted.isna().any();stored[name]=t.predicted.to_numpy().reshape(len(batch),h)
            for representation in ['raw','zscore','level_shape','scale_metadata']:
                bank=joblib.load(run/'groups'/cm['source_group']/task/'expert-banks'/representation/'model.joblib')
                manual=expert_predictions(bank,batch.inputs)
                index=pd.MultiIndex.from_tuples([(d,o,s) for d,o in zip(batch.inputs.domain_ids,batch.inputs.origins) for s in bank.source_domains],names=['target_domain','origin','source_domain'])
                recorded=rep[(rep.horizon==task)&(rep.representation==representation)].set_index(['target_domain','origin','source_domain']).reindex(index)
                actual=losses(batch.targets[:,None],manual).reshape(-1)
                assert np.allclose(actual,recorded.MAPE.to_numpy(),atol=1e-9,rtol=1e-10)
                representation_pairs+=len(actual)
                if representation=='level_shape':experts=manual;sources=bank.source_domains
            n,s=experts.shape[:2]
            index=pd.MultiIndex.from_tuples([(d,o,source) for d,o in zip(batch.inputs.domain_ids,batch.inputs.origins) for source in sources],names=['target_domain','origin','source_domain'])
            for name in ['gain_v2_single','gain_v2_fixed','gain_v2_marginal','gain_v2_rank','gain_v2_forced','gain_v2_uncalibrated']:
                model=joblib.load(run/cm['model_paths'][task][name]);current=local.astype(float).copy();available=np.ones((n,s),bool);active=np.ones(n,bool)
                weights=np.zeros((n,s));local_weight=np.ones(n)
                for stage in [0,1]:
                    logged=pair[(pair.horizon==task)&(pair.model==name)&(pair.stage==stage)].set_index(['target_domain','origin','source_domain']).reindex(index)
                    exists=logged.actual_gain_pp.notna().to_numpy().reshape(n,s)
                    assert np.array_equal(exists,available&active[:,None])
                    gain=logged.predicted_gain_pp.fillna(-np.inf).to_numpy().reshape(n,s);risk=logged.negative_risk.fillna(1.).to_numpy().reshape(n,s)
                    true_gain=losses(batch.targets,current)[:,None]-losses(batch.targets[:,None],.5*current[:,None]+.5*experts)
                    assert np.allclose(true_gain[exists],logged.actual_gain_pp.to_numpy().reshape(n,s)[exists],rtol=1e-10,atol=1e-9)
                    if stage==0:risk_pairs+=int(exists.sum())
                    eligible=exists&(gain>model.rule['margin_pp'])&(risk<=model.rule['max_risk'])
                    candidates=eligible.copy()
                    if name=='gain_v2_forced':
                        candidates[~eligible.any(1)]=exists[~eligible.any(1)]
                    has=candidates.any(1);chosen=np.argmax(np.where(candidates,gain,-np.inf),axis=1)
                    expected=np.zeros((n,s),bool);ii=np.flatnonzero(has);expected[ii,chosen[ii]]=True
                    assert np.array_equal(expected[exists],logged.selected.fillna(False).to_numpy(bool).reshape(n,s)[exists])
                    current[ii]=.5*current[ii]+.5*experts[ii,chosen[ii]];weights[ii]*=.5;weights[ii,chosen[ii]]+=.5;local_weight[ii]*=.5
                    available[ii,chosen[ii]]=False;active=has
                difference=float(np.max(np.abs(current-stored[name])));max_route_difference=max(max_route_difference,difference)
                assert difference<1e-8,(study['name'],task,name,difference)
                zero=local_weight==1;zero_origins+=int(zero.sum());assert np.array_equal(current[zero],local[zero].astype(float))
                reconstructed=local_weight[:,None]*local+np.einsum('ns,nsh->nh',weights,experts)
                assert np.allclose(reconstructed,current,atol=1e-12,rtol=1e-12)
                route_checks+=n
            # Verify the complete predeclared oracle family, allowing only the
            # stored float32 local-baseline dtype's rounding at output assignment.
            best_loss=losses(batch.targets,local)
            for name,values in stored.items():
                if name!='oracle_posthoc':best_loss=np.minimum(best_loss,losses(batch.targets,values))
            for i in range(s):
                for alpha in [0.,.25,.5,.75,1.]:best_loss=np.minimum(best_loss,losses(batch.targets,(1-alpha)*local+alpha*experts[:,i]))
                for j in range(s):
                    if i!=j:best_loss=np.minimum(best_loss,losses(batch.targets,.25*local+.25*experts[:,i]+.5*experts[:,j]))
            difference=float(np.max(np.abs(losses(batch.targets,stored['oracle_posthoc'])-best_loss)))
            oracle_rounding_pp=max(oracle_rounding_pp,difference);assert difference<1e-4
    print(study['name'],'independent expert/benefit/routing/oracle checks complete',flush=True)

statistics=pd.read_csv(run/'statistics.csv')
assert set(statistics[statistics.kind=='domain_bootstrap'].holm_family_size)=={48}
assert set(statistics[statistics.kind=='dm_hac'].holm_family_size)=={1392}
assert statistics[(statistics.horizon=='3d')&(statistics.kind=='dm_hac')].p_value.isna().all()
assert metric_checks==14616
report=dict(passed=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),run=args.run,metric_checks=metric_checks,
 independent_native_caches=len(native_records),native_labels=sum(r['labels'] for r in native_records),representation_expert_pairs=representation_pairs,
 route_origin_checks=route_checks,stage0_risk_pairs=risk_pairs,zero_transfer_origin_checks=zero_origins,maximum_route_difference_kw=max_route_difference,
 maximum_oracle_finite_family_difference_pp=oracle_rounding_pp,oracle_tolerance_pp=1e-4,
 scientific_source_and_fit_configs_match_preregistration_except_documented_auditor=True,amended_audit_files=changed,statistical_families_and_short_series_DM_guard_verified=True,
 scope='Read-only native-slot scatter/count aggregation, NumPy metrics, manual Ridge feature construction, logged-action routing and finite-family oracle; no fit or policy change.',native_records=native_records)
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print({k:v for k,v in report.items() if k!='native_records'},flush=True)
