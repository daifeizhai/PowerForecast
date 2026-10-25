"""Read-only Gain-v2 source-role, clock, freeze, reconstruction and score audit."""
from hashlib import sha256
import json

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import forecast_metrics
from src.models.ml.gain_v2 import GAIN_METHODS, OBJECTIVES, VALUE_FEATURES
from src.utils import PROJECT_ROOT


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_gain_v2(folder):
    manifest=_read(folder/'manifest.json');config=manifest['experiment'];frozen=_read(folder/'frozen-before-test.json')
    model_hashes=all(sha256((folder/p).read_bytes()).hexdigest()==v for p,v in frozen['model_sha256'].items())
    data_hashes=all(sha256((folder/p).read_bytes()).hexdigest()==v for p,v in frozen['adapted_data_sha256'].items())
    training_hashes=all(sha256((folder/p).read_bytes()).hexdigest()==v for p,v in frozen['training_artifact_sha256'].items())
    original_source_hashes=all(sha256((PROJECT_ROOT/p).read_bytes()).hexdigest()==v for p,v in manifest['source_sha256'].items())
    source_hashes=original_source_hashes
    amended_files=[]
    if not original_source_hashes and (folder/'audit-amendment.json').exists():
        amendment=_read(folder/'audit-amendment.json')
        allowed=amendment['allowed_changes']
        assert set(allowed)=={'src/evaluation/gain_v2_audit.py'},'Only this documented audit-only repair may differ.'
        assert amendment['no_refit'] and amendment['global_freeze_sha256']==sha256((folder/'frozen-before-test.json').read_bytes()).hexdigest()
        assert amendment['original_failed_manifest_sha256']==sha256((folder/'failed-manifest-before-audit-repair.json').read_bytes()).hexdigest()
        assert all(sha256((folder/p).read_bytes()).hexdigest()==v for p,v in amendment['preserved_outcome_sha256'].items())
        source_hashes=all((sha256((PROJECT_ROOT/p).read_bytes()).hexdigest()==v) if p not in allowed else
            (allowed[p]['before_sha256']==v and allowed[p]['after_sha256']==sha256((PROJECT_ROOT/p).read_bytes()).hexdigest()) for p,v in manifest['source_sha256'].items())
        amended_files=list(allowed)
    meta=[]
    for key,records in frozen['group_training_records'].items():
        roles=config['domain_roles'][key.split('_')[0]]
        table=pd.read_parquet(folder/'groups'/key/'gain-training.parquet')
        assert set(table.target_domain)<=set(roles['source']) and set(table.expert_domain)<=set(roles['source'])
        assert not (table.expert_domain==table.target_domain).any() and not (table.context_domain==table.target_domain).any()
        assert not (table.context_domain==table.expert_domain).any()
        assert (table.prior_label_end_day<=table.origin_day).all() and (table.expert_fit_cutoff_day<=table.origin_day).all() and (table.label_end_day<=42).all()
        assert table[["feature_"+k for k in VALUE_FEATURES]].shape[1]==47
        assert np.allclose(table.loc[table.stage==0,'gain_fixed'],table.loc[table.stage==0,'gain_marginal'],rtol=0,atol=1e-10)
        cal=pd.read_parquet(folder/'groups'/key/'calibration-data.parquet')
        assert set(cal.target_domain)==set(roles['calibration']) and (cal.label_end_day<=min(config['validation_origins'])).all()
        for objective in OBJECTIVES:
            learner=joblib.load(folder/'groups'/key/'learners'/objective/'model.joblib')
            assert learner.source_domains==sorted(roles['source']) and learner.training_rows==len(table)
            assert learner.calibration['calibration_domains']==sorted(roles['calibration'])
            assert not set(learner.calibration['calibration_domains'])&set(roles['validation'])
        for record in records:
            assert set(record['source_domains'])<=set(roles['source'])
            assert set(record['calibration_domains'])<=set(roles['calibration'])
            assert set(record['selection_domains'])<=set(roles['validation'])
            assert pd.Timestamp(record['last_training_label'])<pd.Timestamp(config['protocol_anchor'])+pd.Timedelta(days=config['final_cutoff_day'])
        for task in config['horizon_days']:
            paths={r['model']:r['model_path'] for r in records if r['horizon']==task}
            reference=joblib.load(folder/paths['gain_v2_marginal'])
            for name in ['gain_v2_forced','gain_v2_uncalibrated']:
                model=joblib.load(folder/paths[name]);assert model.rule==reference.rule
                assert np.array_equal(model.bank.source_profiles,reference.bank.source_profiles)
                assert model.learner.calibration==reference.learner.calibration
        meta.append(dict(group=key,source_only_rows=len(table),independent_calibration_rows=len(cal),passed=True))
    checks=[];restores=[];first_starts=[];diagnostic_pairs=0
    for study in config['studies']:
        child=folder/'studies'/study['name'];cm=_read(child/'manifest.json');start=_read(child/'test-start.json')
        first_starts.append(pd.Timestamp(start['started_at_utc']))
        assert start['global_freeze_sha256']==sha256((folder/'frozen-before-test.json').read_bytes()).hexdigest()
        assert pd.Timestamp(frozen['frozen_at_utc'])<first_starts[-1]
        period=96 if study['frequency']=='15min' else 48
        frames={d:pd.read_parquet(child/'adapted-data'/(d+'.parquet')) for d in cm['domain_roles']['test']}
        metrics=pd.read_csv(child/'metrics.csv')
        for (task,name),group in metrics.groupby(['horizon','model']):
            table=pd.read_parquet(child/'predictions'/task/(name+'.parquet'))
            h=int(round(config['horizon_days'][task]*period))
            for domain,part in group.groupby('target_domain'):
                saved=table[table.domain_id==domain].sort_values(['origin','lead']);cache=frames[domain]
                assert list(pd.DatetimeIndex(saved.origin.drop_duplicates()))==[pd.Timestamp(config['protocol_anchor'])+pd.Timedelta(days=d) for d in config['test_origins'][task]]
                times=pd.DatetimeIndex(pd.to_datetime(saved.timestamp,utc=True))
                assert np.array_equal(cache.loc[times,'node_power_kw'].to_numpy(np.float32),saved.actual.to_numpy(np.float32))
                assert np.array_equal(cache.loc[times,'source_clock_timestamp'].to_numpy(),saved.source_clock_timestamp.to_numpy())
                assert (cache.loc[times,'quality_mask']==0).all()
                origin=pd.DatetimeIndex(pd.to_datetime(saved.origin,utc=True))
                assert np.array_equal(times.asi8,(origin+pd.to_timedelta((saved.lead.to_numpy()-1)*1440/period,unit='min')).asi8)
                actual=saved.actual.to_numpy(np.float32).reshape(-1,h);predicted=saved.predicted.to_numpy().reshape(-1,h)
                assert np.isfinite(predicted).all() and (predicted>=0).all()
                for row in part.itertuples():
                    x,y=actual,predicted
                    if row.evaluation=='daily_mean':
                        computed=dict(MAPE=None,WAPE=None,MAE=None,RMSE=None,n=0,mape_n=0,mape_excluded=0) if h<period else forecast_metrics(x.reshape(len(x),-1,period).mean(-1),y.reshape(len(y),-1,period).mean(-1))
                    else:
                        if row.evaluation=='curve_nonoverlap':x,y=x[::3],y[::3]
                        computed=forecast_metrics(x,y)
                    assert all((value is None and pd.isna(getattr(row,k))) or (value is not None and np.isclose(value,getattr(row,k),rtol=1e-10,atol=1e-10)) for k,value in computed.items()),(study['name'],domain,task,name,row.evaluation)
                    checks.append(dict(study=study['name'],domain=domain,horizon=task,model=name,evaluation=row.evaluation,passed=True))
                if name=='oracle_posthoc':assert saved.oracle_only.all()
                else:assert not saved.oracle_only.any()
        pair=pd.read_parquet(child/'gain-diagnostics.parquet');diagnostic_pairs+=len(pair[pair.stage==0])
        for (task,name),part in pair.groupby(['horizon','model']):
            final=part.drop_duplicates(['target_domain','origin']);zero=final.transfer_gate==0
            if zero.any():
                # Exact fallback is a curve contract. Equal curves can have
                # slightly different reductions from C/F array memory layouts.
                keys=pd.MultiIndex.from_frame(final.loc[zero,['target_domain','origin']]);keys.names=['domain_id','origin']
                prediction=pd.read_parquet(child/'predictions'/task/(name+'.parquet'))
                local=pd.read_parquet(child/'predictions'/task/'three_day_mean.parquet')
                prediction=prediction[pd.MultiIndex.from_frame(prediction[['domain_id','origin']]).isin(keys)].sort_values(['domain_id','origin','lead'])
                local=local[pd.MultiIndex.from_frame(local[['domain_id','origin']]).isin(keys)].sort_values(['domain_id','origin','lead'])
                assert prediction[['domain_id','origin','lead']].reset_index(drop=True).equals(local[['domain_id','origin','lead']].reset_index(drop=True))
                assert np.array_equal(prediction.predicted.to_numpy(),local.predicted.to_numpy())
            if name=='gain_v2_forced':assert not zero.any()
        # Target periods are fresh forecast labels, even though identities are known.
        for domain,cache in frames.items():
            first=pd.Timestamp(cache.source_clock_timestamp.iloc[config['test_origins']['4h'][0]*period])
            stop=pd.Timestamp(cache.source_clock_timestamp.iloc[config['test_origins']['24h'][-1]*period])+pd.Timedelta(days=1)
            if study['test_group']=='m100':assert first>=pd.Timestamp('2022-02-18T00:00Z')
            else:assert stop<=pd.Timestamp('2019-04-07') and first>=pd.Timestamp('2019-03-29')
        records=_read(child/'saved-model-audit.json');assert len(records)==len(config['horizon_days'])*len(manifest['model_config']['default_models'])
        assert all(r['passed'] for r in records);restores.extend(records)
    passed=model_hashes and data_hashes and training_hashes and source_hashes and bool(checks)
    return dict(passed=passed,all_studies_frozen_before_test=True,model_states=len(frozen['model_sha256']),model_hashes_unchanged=model_hashes,
        adapted_data_hashes_unchanged=data_hashes,source_training_artifacts_unchanged=training_hashes,computational_source_unchanged=original_source_hashes,
        scientific_source_checks_passed=source_hashes,documented_audit_only_amendment=amended_files,
        independent_calibration_and_selection=True,source_only_meta=meta,fresh_forecast_label_periods_verified=True,
        controlled_interventions_inherited=True,oracle_analysis_only=True,metric_checks=len(checks),restore_checks=len(restores),
        stage0_risk_pairs=diagnostic_pairs,checks=checks)
