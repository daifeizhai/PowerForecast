"""Read-only robust study checks; recompute every saved domain metric."""
from hashlib import sha256
import json
import joblib
import numpy as np,pandas as pd
from src.evaluation.metrics import forecast_metrics
from src.utils import PROJECT_ROOT


def audit_robust(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));freeze=json.loads((folder/'frozen-before-test.json').read_text(encoding='utf-8'))
    failures=[];hashes=0;checks=0;replays=0
    for field in ['model_sha256','adapted_data_sha256','training_artifact_sha256']:
        for path,digest in freeze[field].items():
            hashes+=1
            if sha256((folder/path).read_bytes()).hexdigest()!=digest:failures.append('Frozen hash changed: '+path)
    for p,digest in manifest['source_sha256'].items():
        if sha256((PROJECT_ROOT/p).read_bytes()).hexdigest()!=digest:failures.append('Computational source changed: '+p)
    roles=freeze['global_domain_roles'];all_ids=sum([sum(r.values(),[]) for r in roles.values()],[])
    if len(all_ids)!=len(set(all_ids)):failures.append('Role overlap.')
    for key,records in freeze['group_training_records'].items():
        group=key.split('_')[0]
        for record in records:
            if set(record['source_domains'])!=set(roles[group]['source']) or record['cutoff_day']!=72:failures.append('Source fitting boundary changed.')
            task=record['horizon'];directory=folder/'groups'/key/task
            reference=joblib.load(directory/'expert-bank'/'model.joblib')
            signature=joblib.hash(reference)
            for name in record['model_names']:
                if name in ['seasonal_naive','three_day_mean','ridge','sapf','gain_v2_reference']:continue
                model=joblib.load(directory/name/'model.joblib')
                if joblib.hash(model.bank)!=signature:failures.append('Fusion methods changed expert pool: '+key+'/'+task+'/'+name)
            if pd.Timestamp(record['last_training_label'])>=pd.Timestamp(manifest['experiment']['protocol_anchor'])+pd.Timedelta(days=72):failures.append('Final source training used future labels.')
    for study in manifest['studies']:
        child=folder/'studies'/study['name'];start=json.loads((child/'test-start.json').read_text(encoding='utf-8'))
        if start['global_freeze_sha256']!=sha256((folder/'frozen-before-test.json').read_bytes()).hexdigest() or start['started_at_utc']<=freeze['frozen_at_utc']:failures.append('Test started without global freeze.')
        restored=json.loads((child/'saved-model-audit.json').read_text(encoding='utf-8'));replays+=len(restored)
        if not all(r['passed'] for r in restored):failures.append('Saved-model replay failed.')
        metrics=pd.read_csv(child/'metrics.csv');matched={}
        for (task,model),part in metrics.groupby(['horizon','model'],sort=False):
            table=pd.read_parquet(child/'predictions'/task/(model+'.parquet'))
            cols=['domain_id','origin','timestamp','lead'];keys=pd.util.hash_pandas_object(table[cols],index=False).to_numpy()
            truth=table.actual.to_numpy()
            if task in matched:
                if not np.array_equal(keys,matched[task][0]) or not np.array_equal(truth,matched[task][1]):failures.append('Unmatched evaluation grid: '+task+'/'+model)
            else:matched[task]=(keys,truth)
            for domain,saved_domain in table.groupby('domain_id',sort=False):
                target_key=study['test_group']+'_'+study['frequency'].replace('min','')
                adapted=pd.read_parquet(folder/'adapted-data'/target_key/(domain+'.parquet'))
                a=adapted.loc[pd.DatetimeIndex(saved_domain.timestamp)]
                if not np.array_equal(a.node_power_kw.to_numpy(np.float32),saved_domain.actual.to_numpy(np.float32)):failures.append('Labels differ from frozen adapted kW data.')
                if not np.array_equal(a.source_clock_timestamp.to_numpy(),saved_domain.source_clock_timestamp.to_numpy()):failures.append('Published clock changed.')
            for row in part.itertuples():
                t=table[table.domain_id==row.target_domain]
                if row.evaluation=='daily_mean':
                    period=96 if study['frequency']=='15min' else 48
                    if task=='4h':
                        if row.n!=0 or np.isfinite(row.MAPE):failures.append('4h falsely reported daily mean.')
                        continue
                    a=t.actual.to_numpy().reshape(-1,period).mean(1);p=t.predicted.to_numpy().reshape(-1,period).mean(1)
                else:
                    if row.evaluation=='curve_nonoverlap':
                        clock=pd.to_datetime(t.origin,utc=True);t=t[np.asarray((clock-clock.min())/pd.Timedelta(days=1))%3==0]
                    a=t.actual.to_numpy();p=t.predicted.to_numpy()
                values=forecast_metrics(a,p)
                for field in ['MAPE','WAPE','MAE','RMSE','n','mape_excluded','mape_n']:
                    checks+=1
                    if not np.isclose(getattr(row,field),values[field],rtol=1e-10,atol=1e-10):failures.append(f'Metric mismatch: {study["name"]}/{task}/{model}/{row.target_domain}/{row.evaluation}/{field}')
    return dict(passed=not failures,frozen_hash_checks=hashes,metric_checks=checks,saved_model_replays=replays,failures=failures,
        scope='Hashes, source roles/cutoff, global freeze chronology, identical node/origin/label grids, every saved curve/daily/nonoverlap metric, saved-model replay records.')
