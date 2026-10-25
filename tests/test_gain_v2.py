from copy import deepcopy
from types import SimpleNamespace
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from src.data.contracts import CanonicalDataAdapter, PublishedPowerAdapter, attach_panel_contract, DUPLICATE, MISSING
from src.data.domains import DomainMetadata, validate_compatible
from src.data.google_panel import DomainPanel, domain_windows
from src.evaluation.backtest import score_predictions
from src.evaluation.statistics import dm_hac, holm, paired_statistics, weighted_auc
from src.models import create_model
from src.models.ml.gain_v2 import GainV2Policy, TransferValueLearner, VALUE_FEATURES
from src.training.gain_v2_learning import ape, source_batch
from src.utils import load_config


def metadata(domain='a',**kwargs):
    values=dict(dataset_id='m100_fixture',site_id='fixture',domain_id=domain,target_field='node_power_kw',target_kind='active_power',
        target_scope='node',target_unit='kW',label_origin='measured',timezone='UTC',timezone_verified=True,raw_frequency='15min',
        evaluation_frequency='30min',sampling_semantics='interval_mean',interval_definition='Complete published slots.',coverage='Fixture.',source_version='fixture')
    values.update(kwargs);return DomainMetadata(**values)


def fixture_panels():
    config=load_config('configs/sapf_gain_v2.yaml');roles={};panels={}
    for group in ['m100','ucc']:
        roles[group]={r:[f'{group}_{r}_{n}' for n in range(count)] for r,count in [('source',3),('calibration',2),('validation',2),('test',2)]}
    for group,frequency in [('m100','30min'),('ucc','30min'),('m100','15min')]:
        ppd=48 if frequency=='30min' else 96
        grid=pd.date_range(config['protocol_anchor'],periods=64*ppd,freq=frequency,name='timestamp');t=np.arange(len(grid))
        series,metas,audit,coverage={},{},{},{}
        for i,(role,domains) in enumerate(roles[group].items()):
            for j,domain in enumerate(domains):
                rng=np.random.default_rng(400+i*10+j)
                values=.3+.02*(i*3+j)+.02*np.sin(t/ppd*2*np.pi+(i+j)/3)+.01*np.cos(t/ppd/7*2*np.pi+j)+rng.normal(0,.004,len(t))
                start=config['source_clock_periods'][group]['test_start' if role=='test' else 'development_start']
                clock=pd.date_range(start,periods=len(grid),freq=frequency)
                series[domain]=pd.DataFrame(dict(node_power_kw=values,source_clock_timestamp=[c.isoformat() for c in clock]),index=grid)
                metas[domain]=metadata(domain,dataset_id=group+'_fixture',timezone='UTC' if group=='m100' else 'unknown_source_clock',timezone_verified=group=='m100',
                    raw_frequency='15min' if group=='m100' else '10min',evaluation_frequency=frequency,usage_role='final_holdout' if role=='test' else 'development')
                audit[domain]=dict(source_clock_start=clock[0].isoformat(),source_clock_timezone_verified=group=='m100',history_availability_upper_bound_minutes=1440/ppd)
                coverage[domain]=pd.Series(1,index=grid)
        panels[group+'_'+frequency.replace('min','')]=attach_panel_contract(DomainPanel(series,metas,coverage,{},grid[0],audit))
    return panels,roles


def test_native_quality_masks_and_unknown_clock_are_preserved():
    raw=pd.DataFrame({'time':pd.date_range('2019-02-01',periods=6,freq='10min'),'power':[100,200,300,400,500,600]})
    adapter=PublishedPowerAdapter('time','power',10)
    frame=adapter.adapt(raw,start='2019-02-01',days=1,frequency='30min')
    assert frame.node_power_kw.iloc[0]==.2 and frame.node_power_kw.iloc[1]==.5
    assert not frame.source_clock_timestamp.iloc[0].endswith('+00:00')
    duplicated=pd.concat([raw,raw.iloc[[1]]]);bad=adapter.adapt(duplicated,start='2019-02-01',days=1,frequency='30min')
    assert np.isnan(bad.node_power_kw.iloc[0]) and bad.quality_mask.iloc[0]&DUPLICATE
    assert bad.quality_mask.iloc[2]&MISSING
    with pytest.raises(ValueError,match='interpolation'):adapter.adapt(raw,start='2019-02-01',days=1,frequency='15min')


def test_canonical_contract_blocks_unit_mixing_and_silent_utc():
    panel,_=fixture_panels();m=next(iter(panel['m100_30'].metadata.values()))
    google=metadata('g',target_kind='power_utilization',target_scope='power_domain',target_unit='utilization_fraction')
    with pytest.raises(ValueError,match='Incompatible'):validate_compatible([m,google])
    frame=next(iter(panel['m100_30'].series.values())).copy()
    with pytest.raises(ValueError,match='unknown source clock'):
        CanonicalDataAdapter().adapt_domain(frame,metadata=metadata(timezone_verified=False),source_hashes={},covariate_schema={})


def test_actual_availability_and_bad_quality_prevent_history_use():
    panels,roles=fixture_panels();panel=panels['m100_30'];domain=roles['m100']['test'][0]
    frame=panel.series[domain];frame.loc[frame.index[53*48],'available_at']=frame.index[55*48]
    with pytest.raises(ValueError,match='available'):domain_windows(panel,[domain],[54],672,8)
    frame.loc[frame.index[53*48],'available_at']=frame.index[53*48]+pd.Timedelta(minutes=30)
    frame.loc[frame.index[53*48],'quality_mask']=DUPLICATE
    with pytest.raises(ValueError,match='bad-quality'):domain_windows(panel,[domain],[54],672,8)


@pytest.mark.parametrize('frequency,period,horizon',[('30',48,8),('15',96,16)])
def test_4h_never_crosses_fit_cutoff_or_claims_daily_mean(frequency,period,horizon):
    panels,roles=fixture_panels();panel=panels['m100_'+frequency]
    batch=source_batch(panel,roles['m100']['source'],20,horizon)
    assert max(pd.Timestamp(x) for x in batch.inputs.target_times[:,-1])<panel.anchor+pd.Timedelta(days=20)
    assert len(batch)==3*6
    scored=score_predictions(batch,batch.targets+.01)
    assert scored.daily_metrics['MAPE'] is None and scored.daily_metrics['n']==0
    assert scored.metrics['n']==len(batch)*horizon


def test_complementary_expert_and_stage_zero_gain_identity():
    actual=np.array([100.,100.]);local=np.array([90.,110.]);expert=np.array([120.,90.])
    assert ape(actual,local)-ape(actual,expert)<0
    fixed=ape(actual,local)-ape(actual,.5*local+.5*expert)
    marginal=ape(actual,local)-ape(actual,.5*local+.5*expert)
    assert fixed==marginal and fixed>0


@pytest.mark.parametrize('name',['ridge_raw','ridge_zscore','ridge_level_shape','ridge_scale_metadata'])
def test_representations_are_registered_readonly_and_restore(name,tmp_path):
    panels,roles=fixture_panels();batch=source_batch(panels['m100_30'],roles['m100']['source'],20,8)
    before=batch.inputs.history.copy();model=create_model(name,{'alpha':100.}).fit(batch)
    prediction=model.predict(batch.inputs)
    np.testing.assert_array_equal(before,batch.inputs.history)
    path=model.save(tmp_path/name);np.testing.assert_array_equal(prediction,joblib.load(path).predict(batch.inputs))


def test_zero_veto_is_exact_and_forcing_changes_only_rejection():
    panels,roles=fixture_panels();batch=source_batch(panels['m100_30'],roles['m100']['source'],20,8)
    domains=sorted(roles['m100']['source'])
    class Bank:
        source_domains=domains;training_windows_per_expert=6
        source_profiles=np.zeros((3,8))
        def expert_predictions(self,inputs):return np.full((len(inputs),3,8),.9)
    class Learner:
        source_domains=domains;marginal_negative_prior=.5
        def raw(self,features,available=None):return np.full(features.shape[:2],-1.),np.full(features.shape[:2],.9)
        def apply_calibration(self,gain,risk):return gain,risk
    priors={d:[1.,1.,0.,.5,1.] for d in domains}
    policy=GainV2Policy(bank=Bank(),learner=Learner(),priors=priors,rule={'margin_pp':0.,'max_risk':.5}).fit(batch)
    prediction,info=policy.predict_with_diagnostics(batch.inputs)
    np.testing.assert_array_equal(prediction,info['local'])
    assert not info['weights'].any()
    policy.name='gain_v2_forced';forced,info=policy.predict_with_diagnostics(batch.inputs)
    assert np.all(info['local_weight']==.25) and not np.allclose(forced,prediction)


def test_holm_weighted_auc_and_dm_match_independent_definitions():
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    y=np.array([False,True,False,True,True]);p=np.array([.3,.5,.5,.2,.8]);w=np.array([1,2,3,1,2])
    assert weighted_auc(y,p,w)==pytest.approx(roc_auc_score(y,p,sample_weight=w))
    assert dm_hac([1,2,3],3)['p_value'] is None
    assert dm_hac(np.zeros(9),1)['p_value']==1.
    values=np.array([1,2,1,4,3,1,2,4,3.]);center=values-values.mean()
    variance=center@center/9+sum(2*(1-k/3)*(center[k:]@center[:-k]/9) for k in [1,2])
    assert dm_hac(values,1)['dm_statistic']==pytest.approx(values.mean()/np.sqrt(variance/9))


@pytest.mark.parametrize('bad',['external_target','self_expert','self_context','future_teacher','future_prior','future_label'])
def test_value_learning_rejects_domain_and_time_leakage(bad):
    row=dict(target_domain='a',expert_domain='b',context_domain='',origin_day=24,expert_fit_cutoff_day=24,prior_label_end_day=24,label_end_day=25)
    if bad=='external_target':row['target_domain']='held_out'
    elif bad=='self_expert':row['expert_domain']='a'
    elif bad=='self_context':row['context_domain']='a'
    elif bad=='future_teacher':row['expert_fit_cutoff_day']=25
    elif bad=='future_prior':row['prior_label_end_day']=25
    else:row['label_end_day']=43
    with pytest.raises(ValueError):TransferValueLearner().fit(pd.DataFrame([row]),['a','b','c'])


def test_global_v2_fixture_freezes_all_cadences_and_keeps_oracle_out_of_models(tmp_path,monkeypatch):
    import src.training.gain_v2_experiment as suite
    panels,roles=fixture_panels();config=load_config('configs/sapf_gain_v2.yaml');config['domain_roles']=roles
    config['results_dir']=str(tmp_path/'runs');config['preregistration']=str(tmp_path/'no-prereg.json')
    config['statistics']['replicates']=20
    model=load_config(config['model_config']);model['benefit'].update(n_estimators=4,max_depth=3,min_samples_leaf=2)
    model['sapf'].update(top_k=[1],feature_groups=['shape'],transfer_strengths=[.5],reliability_powers=[1.])
    model['policy_grid'].update(margin_pp=[0.],max_risk=[1.])
    path=tmp_path/'models.yaml';path.write_text(yaml.safe_dump(model),encoding='utf-8');config['model_config']=str(path)
    monkeypatch.setattr(suite,'raw_power_frames',lambda *args:({},{}))
    monkeypatch.setattr(suite,'load_gain_v2_panel',lambda config,group,frequency,*args:panels[group+'_'+frequency.replace('min','')])
    original=suite._evaluate;seen=[]
    def evaluate(folder,panel,domains,origins,task,models,banks,config,split,*args):
        if split=='test':
            root=folder.parents[1]
            frozen=json.loads((root/'frozen-before-test.json').read_text(encoding='utf-8'))
            assert len(frozen['group_training_records'])==3 and len(frozen['studies'])==4
            assert all('oracle' not in name for name in models)
            seen.append((folder.name,task))
        return original(folder,panel,domains,origins,task,models,banks,config,split,*args)
    monkeypatch.setattr(suite,'_evaluate',evaluate)
    folder=suite.run_gain_v2(SimpleNamespace(test_fixture=True),config)
    audit=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
    assert audit['passed'] and audit['restore_checks']==4*3*17
    assert len(seen)==12
    child=folder/'studies/m100_to_ucc_30';metrics=pd.read_csv(child/'metrics.csv')
    assert metrics.query("horizon=='4h' and evaluation=='daily_mean'").MAPE.isna().all()
    assert not pd.read_csv(folder/'statistics.csv').empty
    oracle=pd.read_parquet(child/'predictions/4h/oracle_posthoc.parquet')
    assert oracle.oracle_only.all() and not oracle.source_clock_timestamp.str.endswith('+00:00').any()
    records=json.loads((folder/'groups/m100_30/training-records.json').read_text(encoding='utf-8'))
    assert not any('oracle' in r['model'] for r in records)
