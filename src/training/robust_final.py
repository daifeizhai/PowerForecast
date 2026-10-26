"""One global freeze, then a single matched new-date evaluation. No target tuning."""
from copy import deepcopy
from datetime import datetime,timezone
from hashlib import sha256
import json,uuid

import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits

from src.data.google_panel import domain_windows
from src.data.robust_panel import load_robust_panels
from src.evaluation.metrics import forecast_metrics
from src.models import create_model
from src.models.ml.robust_transfer import RobustFixedFusion,RobustDynamicFusion,history_scale,local_prediction
from src.training.gain_v2_experiment import _save_predictions
from src.training.gain_v2_learning import ape,source_batch
from src.training.lifecycle import freeze_for_test
from src.training.robust_experiment import (horizon_points,expanded_origins,test_origins,fit_experts,
    train_analog,fit_reference_gain,numerical_diagnostics)
from src.utils import PROJECT_ROOT,resolve_path,write_json,load_config


def digest(path):return sha256(path.read_bytes()).hexdigest()


def dynamic_admitted(gates,task):
    primary=any(gates[h]['stable_positive_signal'] for h in ['24h','3d'])
    return primary and gates[task]['stable_positive_signal']


def validate_plan(config,development):
    if development['status']!='development_complete' or not development['no_new_target_forecast_errors_accessed']:
        raise ValueError('A completed source-only development record is required.')
    for key,value in development['experiment'].items():
        if key=='conditional_dynamic':continue
        if config.get(key)!=value:raise ValueError('Source development plan changed: '+key)
    if config['context_days']!=14 or config['alpha_candidates']!=[0.,.1,.25,.5]:raise ValueError('Unregistered history or alpha grid.')
    all_ids=sum([sum(r.values(),[]) for r in config['domain_roles'].values()],[])
    if len(set(all_ids))!=len(all_ids) or set(all_ids)&set(config['retired_m100_v1_test']):raise ValueError('Domain isolation failed.')


def build_group(panel,roles,config,gates,folder):
    folder.mkdir(parents=True)
    references=fit_reference_gain(panel,roles,config,folder/'gain-reference')
    original=load_config('configs/gain_v2_models.yaml');built={};records=[]
    for task,h in horizon_points(config,panel.points_per_day).items():
        selection=gates[task];path=folder/task;path.mkdir(exist_ok=True)
        bank,training=fit_experts(panel,roles['source'],config['final_source_cutoff'],h,config['ridge_alpha'])
        (path/'expert-bank').mkdir();bank.save(path/'expert-bank'/'model')
        dev=source_batch(panel,roles['source'],44,h)
        validation=domain_windows(panel,roles['validation'],expanded_origins(config['validation_days'],task),14*panel.points_per_day,h)
        sapf=create_model('sapf',original['sapf'],config['seed']).fit(dev).calibrate(validation)
        write_json(path/'original-sapf-selection.json',sapf.calibration);sapf.refit(training)
        local_kind=selection['local_kind'];common=dict(bank=bank,local_kind=local_kind,bound_sigma=config['correction_bound_sigma'])
        models={name:create_model(name,dict(period=panel.points_per_day) if name=='seasonal_naive' else
            dict(alpha=config['ridge_alpha']) if name=='ridge' else {},config['seed']).fit(training)
            for name in ['seasonal_naive','three_day_mean','ridge']}
        models['external_mean']=RobustFixedFusion(**common,alpha=1.,name='robust_external').fit(training)
        for bounded in [False,True]:
            for alpha in config['alpha_candidates']:
                name=('bounded_' if bounded else '')+f'fixed_{alpha:g}'
                models[name]=RobustFixedFusion(**common,alpha=alpha,bounded=bounded).fit(training)
        for bounded,key in [(False,'fixed'),(True,'bounded')]:
            models['selected_'+key]=RobustFixedFusion(**common,alpha=selection['selected_alphas'][key],bounded=bounded).fit(training)
        models['sapf']=sapf;models['gain_v2_reference']=references[task]
        if dynamic_admitted(gates,task):
            analog=train_analog(panel,roles['source'],task,h,local_kind,config,path)
            models['dynamic']=RobustDynamicFusion(**common,analog=analog,name='robust_dynamic').fit(training)
        paths={}
        for name,model in models.items():
            dest=path/name;dest.mkdir(exist_ok=True);paths[name]=model.save(dest/'model')
        built[task]=(models,bank,paths)
        records.append(dict(horizon=task,source_domains=roles['source'],development_domains=roles['calibration'],validation_domains=roles['validation'],
            cutoff_day=config['final_source_cutoff'],last_training_label=str(max(pd.Timestamp(t) for t in training.inputs.target_times[:,-1])),
            training_windows=len(training),selected_alphas=selection['selected_alphas'],local_kind=local_kind,dynamic_executed='dynamic' in models,
            dynamic_gate=selection['stable_positive_signal'],model_names=list(models)))
        print(f'[{folder.name}/{task}] final source fit complete; dynamic={"dynamic" in models}; no new target scores',flush=True)
    write_json(folder/'training-records.json',records)
    return built,records


def routing_table(batch,task,name,prediction,info,local,external,config):
    local_loss=ape(batch.targets,local);loss=ape(batch.targets,prediction);_,sigma=history_scale(batch.inputs)
    correction=np.clip(external-local,-config['correction_bound_sigma']*sigma[:,None],config['correction_bound_sigma']*sigma[:,None])
    gain=np.stack([(local_loss-ape(batch.targets,local+a*correction))/np.maximum(local_loss,.1) for a in config['alpha_candidates']],axis=1)
    opportunity=gain.max(1)>=.02
    alpha=info.get('alpha',np.full(len(batch),np.nan))
    if 'transfer_gate' in info:alpha=info['transfer_gate']
    if 'local_weight' in info:alpha=1-info['local_weight']
    return pd.DataFrame(dict(target_domain=batch.inputs.domain_ids,origin=[str(o) for o in batch.inputs.origins],horizon=task,model=name,
        alpha=alpha,local_MAPE=local_loss,final_MAPE=loss,positive_opportunity=opportunity,
        any_positive_opportunity=gain.max(1)>1e-10,successful=(local_loss-loss)/np.maximum(local_loss,.1)>=.02,
        negative=loss>local_loss+1e-10,zero_transfer=np.isfinite(alpha)&(alpha==0),local_kind=info.get('local_kind','source_selected'),
        opportunity_scope='posthoc_same_zscore_pool_relative_to_source_selected_local_not_legacy_gate_training'))


def evaluate_study(folder,study,panels,groups,config):
    source=study['source_group']+'_'+study['frequency'].replace('min','');target=study['test_group']+'_'+study['frequency'].replace('min','')
    panel=panels[target];domains=config['domain_roles'][study['test_group']]['test'];child=folder/'studies'/study['name'];child.mkdir(parents=True)
    write_json(child/'test-start.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),global_freeze_sha256=digest(folder/'frozen-before-test.json')))
    rows=[];origins=[];routes=[];restores=[];checks={};model_paths={}
    for task,h in horizon_points(config,panel.points_per_day).items():
        batch=domain_windows(panel,domains,test_origins(config,study['test_group'],task),14*panel.points_per_day,h)
        if batch.skipped_windows:raise ValueError('A prespecified new test window is missing.')
        models,bank,paths=groups[source][task];model_paths[task]={k:v.relative_to(folder).as_posix() for k,v in paths.items()}
        raw,checks[task]=numerical_diagnostics(bank,batch,config,child/task,'test')
        local=local_prediction(batch.inputs,models['selected_fixed'].local_kind).astype(float);external=np.maximum(raw,0.).mean(1)
        for name,model in models.items():
            if isinstance(model,RobustFixedFusion):
                # Model-only inference and serialized replay; identical source pool for every intervention.
                prediction,info=model.predict_with_diagnostics(batch.inputs)
            elif hasattr(model,'predict_with_diagnostics'):prediction,info=model.predict_with_diagnostics(batch.inputs)
            else:prediction,info=model.predict(batch.inputs),{}
            prediction=np.maximum(prediction,0.)
            restored=np.maximum(joblib.load(paths[name]).predict(batch.inputs),0.)
            difference=float(np.max(np.abs(restored-prediction)))
            restores.append(dict(horizon=task,model=name,windows=len(batch),maximum_absolute_difference=difference,passed=difference<1e-7))
            _save_predictions(child,panel,batch,task,name,'test',prediction,rows,origins)
            routes.append(routing_table(batch,task,name,prediction,info,local,external,config))
        print(f'[{study["name"]}/{task}] single held-out evaluation and all saved-model restores complete',flush=True)
    metrics=pd.DataFrame(rows);origin=pd.concat(origins,ignore_index=True);route=pd.concat(routes,ignore_index=True)
    metrics.to_csv(child/'metrics.csv',index=False);origin.to_parquet(child/'per-origin.parquet',index=False);route.to_parquet(child/'routing.parquet',index=False)
    write_json(child/'saved-model-audit.json',restores);write_json(child/'numerical-audit.json',checks)
    write_json(child/'manifest.json',dict(status='complete',protocol='sapf_robust_study',research_method=study['name'],data='Measured node/server power',target_unit='kW',is_demo=False,
        experiment=study,requested_horizons=list(config['horizon_days']),source_group=source,target_group=target,model_paths=model_paths,
        domain_roles={**{k:v for k,v in config['domain_roles'][study['source_group']].items() if k!='test'},'test':domains},
        source_clock_audit={d:panel.audit[d] for d in domains},domain_metadata={d:panel.metadata[d].to_dict() for d in domains}))
    if not all(r['passed'] for r in restores):raise ValueError('Saved forecasts do not replay.')
    return metrics.assign(study=study['name']),origin.assign(study=study['name']),route.assign(study=study['name'])


def run_final(args,config):
    if not getattr(args,'development_run',None):raise ValueError('--development-run is required.')
    if any(getattr(args,k,None) for k in ['demo','models','horizons']):raise ValueError('Only the complete frozen protocol is allowed.')
    dev=resolve_path(args.development_run);development=json.loads((dev/'manifest.json').read_text(encoding='utf-8'));validate_plan(config,development)
    prereg=resolve_path('results/analysis/robust_direction/preregistration.json')
    registration=json.loads(prereg.read_text(encoding='utf-8'))
    for p,d in registration['computational_source_sha256'].items():
        if digest(PROJECT_ROOT/p)!=d:raise ValueError('Preregistered source changed: '+p)
    if registration['development_manifest_sha256']!=digest(dev/'manifest.json'):raise ValueError('Source admission changed.')
    folder=resolve_path(config['results_dir'])/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_sapf_robust_'+uuid.uuid4().hex[:6]);folder.mkdir()
    manifest=dict(status='running',protocol=config['protocol'],experiment=config,studies=config['studies'],is_demo=False,target_unit='kW',
        data='New-date M100/UCC measured node/server power; known centers; offline transfer',development_run=str(dev),source_sha256=registration['computational_source_sha256'])
    write_json(folder/'manifest.json',manifest);write_json(folder/'preregistration.json',registration)
    try:
        panels,provenance=load_robust_panels(config);manifest['data_provenance']=provenance;write_json(folder/'manifest.json',manifest)
        adapted={};groups={};records={}
        for key,panel in panels.items():
            for domain,frame in panel.series.items():
                path=folder/'adapted-data'/key/(domain+'.parquet');path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path)
                adapted[path.relative_to(folder).as_posix()]=digest(path)
        gates=development['source_h1_gates'];write_json(folder/'source-h1-gates.json',gates)
        with threadpool_limits(1,user_api='blas'):
            for key,panel in panels.items():groups[key],records[key]=build_group(panel,config['domain_roles'][key.split('_')[0]],config,gates[key],folder/'groups'/key)
            freeze_for_test(folder,global_domain_roles=config['domain_roles'],group_training_records=records,source_h1_gates=gates,
                adapted_data_sha256=adapted,preregistration_sha256=digest(folder/'preregistration.json'),
                training_artifact_sha256={p.relative_to(folder).as_posix():digest(p) for p in (folder/'groups').rglob('*') if p.is_file()})
            print('[Robust] All models and decisions frozen together. Starting one new-date target evaluation.',flush=True)
            outputs=[evaluate_study(folder,study,panels,groups,config) for study in config['studies']]
        metric,origin,route=[pd.concat([o[i] for o in outputs],ignore_index=True) for i in range(3)]
        metric.to_csv(folder/'metrics.csv',index=False);origin.to_parquet(folder/'per-origin.parquet',index=False);route.to_parquet(folder/'routing.parquet',index=False)
        from src.evaluation.robust_report import produce_analysis,print_robust_results
        produce_analysis(folder,metric,origin,route,config)
        from src.evaluation.robust_audit import audit_robust
        audit=audit_robust(folder);write_json(folder/'audit.json',audit)
        if not audit['passed']:raise ValueError('Robust frozen evaluation audit failed.')
        manifest.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat());write_json(folder/'manifest.json',manifest)
        print_robust_results(folder);return folder
    except Exception as exc:
        manifest.update(status='failed',error=type(exc).__name__+': '+str(exc));write_json(folder/'manifest.json',manifest);raise
