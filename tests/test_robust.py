from copy import deepcopy
from types import SimpleNamespace
import numpy as np,pandas as pd,pytest
from src.data.features import source_clock_calendar_features
from src.data.google_panel import domain_windows
from src.models.ml.robust_transfer import RobustFixedFusion,RobustDynamicFusion,AnalogFusionMap,history_scale,local_prediction
from src.training.gain_v2_learning import source_batch
from src.training.robust_final import dynamic_admitted,validate_plan
from src.training.robust_experiment import fit_experts,expanded_origins
from src.utils import load_config
from test_gain_v2 import fixture_panels


def test_published_calendar_preserves_naive_clock_weekday():
    clock=pd.date_range('2019-05-01',periods=48,freq='30min');idx=pd.date_range('2000-01-03',periods=48,freq='30min',tz='UTC')
    frame=source_clock_calendar_features(clock,idx)
    assert clock.tz is None and frame.index.equals(idx)
    assert frame.day_sin.iloc[12]==pytest.approx(1.)
    assert frame.week_sin.iloc[0]==pytest.approx(np.sin(2*np.pi*2/7))


def prepared():
    panels,roles=fixture_panels();panel=panels['m100_30']
    for audit in panel.audit.values():audit.update(all_4h_origins=True,source_clock_calendar=True)
    return panel,roles['m100']


def test_all_six_4h_origins_and_causal_training():
    panel,roles=prepared();batch=source_batch(panel,roles['source'],20,8)
    assert len(batch)==3*6*6
    assert max(pd.Timestamp(t) for t in batch.inputs.target_times[:,-1])<panel.anchor+pd.Timedelta(days=20)
    test=domain_windows(panel,roles['test'],expanded_origins([54], '4h'),672,8)
    assert len(test)==12 and test.inputs.origins[5].hour==20
    with pytest.raises(ValueError,match='sampling grid'):domain_windows(panel,roles['test'],[54.001],672,8)


def test_shared_experts_fixed_zero_and_bounded_dynamic_are_history_only():
    panel,roles=prepared();bank,training=fit_experts(panel,roles['source'],20,8,100.)
    batch=domain_windows(panel,roles['test'],[54,55],672,8)
    local=local_prediction(batch.inputs,'three_day_mean').astype(float)
    zero=RobustFixedFusion(bank=bank).fit(training);assert np.array_equal(zero.predict(batch.inputs),local)
    fixed=RobustFixedFusion(bank=bank,alpha=.5,bounded=True).fit(training)
    values,info=fixed.predict_with_diagnostics(batch.inputs);_,sigma=history_scale(batch.inputs)
    assert np.all(np.abs(values-local)<=sigma[:,None]+1e-12)
    class FixedMap:
        def predict(self,x):return np.full(len(x),.5),np.zeros((len(x),4))
    dynamic=RobustDynamicFusion(bank=bank,analog=FixedMap()).fit(training)
    assert np.array_equal(dynamic.predict(batch.inputs),values)
    from src.data.splitting import WindowBatch
    changed=WindowBatch(batch.inputs,np.full_like(batch.targets,999999.))
    assert np.array_equal(dynamic.predict(batch.inputs),dynamic.predict(changed.inputs))
    assert np.array_equal(info['raw_experts'],bank.expert_predictions(batch.inputs))


def analogy_table():
    return pd.DataFrame([dict(target_domain=d,expert_domains=[x for x in ['a','b','c'] if x!=d],teacher_cutoff_day=24,origin_day=24,label_stop_day=25,
        shape_distance=0.,forecast_disagreement=0.,gain_0=0.,gain_0_1=.01, gain_0_25=.03,gain_0_5=.04) for d in ['a','b','c']]).rename(columns={'gain_0_1':'gain_0.1','gain_0_25':'gain_0.25','gain_0_5':'gain_0.5'})


def test_analog_leave_source_guards_and_far_case_zero_transfer():
    table=analogy_table();model=AnalogFusionMap().fit(table,['a','b','c'],[0,.1,.25,.5])
    alpha,_=model.predict(np.array([[0.,0.],[1e6,1e6]]));assert np.array_equal(alpha,[.5,0.])
    with pytest.raises(ValueError,match='source-internal'):AnalogFusionMap().fit(table,['b','c'],[0,.1,.25,.5])
    bad=table.copy();bad.loc[0,'label_stop_day']=45
    with pytest.raises(ValueError,match='precede'):AnalogFusionMap().fit(bad,['a','b','c'],[0,.1,.25,.5])
    bad=table.copy();bad.at[0,'expert_domains']=['a','b']
    with pytest.raises(ValueError,match='self-trained'):AnalogFusionMap().fit(bad,['a','b','c'],[0,.1,.25,.5])


def test_conditional_gate_does_not_admit_4h_alone_or_failed_3d():
    gates={h:dict(stable_positive_signal=h=='4h') for h in ['4h','24h','3d']}
    assert not dynamic_admitted(gates,'4h')
    gates['24h']['stable_positive_signal']=True
    assert dynamic_admitted(gates,'4h') and dynamic_admitted(gates,'24h') and not dynamic_admitted(gates,'3d')


def test_final_rejects_development_config_changes():
    config=load_config('configs/sapf_robust.yaml');dev=dict(status='development_complete',no_new_target_forecast_errors_accessed=True,experiment=deepcopy(config))
    validate_plan(config,dev);config['alpha_candidates']=[0,.5]
    with pytest.raises(ValueError,match='changed'):validate_plan(config,dev)


def test_full_source_fit_global_freeze_test_audit_and_reporting(tmp_path):
    import json
    from threadpoolctl import threadpool_limits
    from src.data.contracts import attach_panel_contract
    from src.training.robust_final import build_group,evaluate_study,digest
    from src.training.lifecycle import freeze_for_test
    from src.evaluation.robust_audit import audit_robust
    from src.evaluation.robust_report import produce_analysis,print_robust_results
    from src.utils import write_json
    panel,roles=prepared()
    for domain,frame in panel.series.items():
        extra=frame.iloc[:46*48][['node_power_kw','source_clock_timestamp']].copy();extra.index=extra.index+pd.Timedelta(days=64)
        extra.source_clock_timestamp=[(pd.Timestamp(t)+pd.Timedelta(days=64)).isoformat() for t in extra.source_clock_timestamp]
        panel.series[domain]=pd.concat([frame[['node_power_kw','source_clock_timestamp']],extra]);panel.coverage[domain]=pd.Series(1,index=panel.series[domain].index)
    attach_panel_contract(panel)
    config=load_config('configs/sapf_robust.yaml');config['domain_roles']={'m100':roles};config['statistics']['replicates']=20
    config['dataset_clocks']['m100'].update(test_start_day=83,test_stop_day=89)
    config['studies']=[dict(name='m100_internal_15',source_group='m100',test_group='m100',frequency='30min')]
    gates={h:dict(stable_positive_signal=h in ['4h','24h'],local_kind='three_day_mean',selected_alphas=dict(fixed=.25,bounded=.25)) for h in ['4h','24h','3d']}
    with threadpool_limits(1,user_api='blas'):
        built,records=build_group(panel,roles,config,gates,tmp_path/'groups'/'m100_30')
        adapted={}
        for domain,frame in panel.series.items():
            path=tmp_path/'adapted-data'/'m100_30'/(domain+'.parquet');path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path);adapted[path.relative_to(tmp_path).as_posix()]=digest(path)
        write_json(tmp_path/'manifest.json',dict(protocol=config['protocol'],experiment=config,studies=config['studies'],source_sha256={}))
        freeze_for_test(tmp_path,global_domain_roles=config['domain_roles'],group_training_records={'m100_30':records},adapted_data_sha256=adapted,
            training_artifact_sha256={p.relative_to(tmp_path).as_posix():digest(p) for p in (tmp_path/'groups').rglob('*') if p.is_file()})
        metric,origin,route=evaluate_study(tmp_path,config['studies'][0],{'m100_30':panel},{'m100_30':built},config)
    produce_analysis(tmp_path,metric,origin,route,config);print_robust_results(tmp_path)
    result=audit_robust(tmp_path)
    assert result['passed'],result['failures'][:5]
    assert result['saved_model_replays']==50 and result['metric_checks']>100
