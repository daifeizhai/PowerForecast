"""Unified four-study Gain-v2 lifecycle: source fit, calibration, selection, freeze, test."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import shutil
import uuid

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from src.data.domains import assert_data_usage
from src.data.gain_v2_panel import load_gain_v2_panel, raw_power_frames
from src.data.google_panel import domain_windows
from src.data.splitting import WindowBatch
from src.evaluation.artifacts import save_result
from src.evaluation.backtest import score_predictions
from src.evaluation.cross_domain import summarize_transfer
from src.evaluation.metrics import forecast_metrics
from src.evaluation.gain_v2_diagnostics import value_summary, calibration_curves
from src.evaluation.reporting import print_saved_results
from src.evaluation.statistics import paired_statistics, risk_statistics
from src.models import create_model
from src.models.ml.benefit import pair_features
from src.models.ml.gain_v2 import GAIN_METHODS, OBJECTIVES, FixedExpertBlend, GainV2Policy, TransferValueLearner
from src.models.ml.level_shape import REPRESENTATIONS
from src.training.benefit_learning import reduce_priors
from src.training.gain_v2_learning import ape, fit_bank, horizons, make_value_table, source_batch, train_value_tables
from src.training.lifecycle import freeze_for_test
from src.utils import PROJECT_ROOT, load_config, resolve_path, write_json


def validate_config(config):
    if config['horizon_days'] != {'4h':1/6,'24h':1,'3d':3} or config['context_days'] != 14:
        raise ValueError('This frozen study requires 4h/24h/3d with 14-day histories.')
    ids=[]
    for roles in config['domain_roles'].values():
        if min(map(len,roles.values()))<2 or len(roles['source'])<3:
            raise ValueError('Separate source/calibration/selection/test identity groups are required.')
        ids.extend(sum(roles.values(),[]))
    if len(set(ids)) != len(ids) or set(ids)&set(config['retired_m100_v1_test']):
        raise ValueError('Global identities overlap or include retired M100 tests.')
    if max(config['prior_origins'])+3 > min(config['meta_cutoffs']) or max(config['meta_cutoffs'])+config['meta_block_days']>config['development_cutoff_day']:
        raise ValueError('Historical priors or source teacher labels cross the development boundary.')
    if min(config['calibration_origins'])<config['development_cutoff_day'] or max(config['calibration_origins'])+3>min(config['validation_origins']) or max(config['validation_origins'])+3>config['final_cutoff_day']:
        raise ValueError('Calibration, policy selection and final fitting cutoffs overlap future labels.')
    if min(min(v) for v in config['test_origins'].values())<config['final_cutoff_day']:
        raise ValueError('All new protocol test origins must follow the final source fitting coordinate.')


def _group_key(group,frequency):
    return group+'_'+frequency.replace('min','')


def _calibration_predictions(learner,table,objective):
    raw_gain,raw_risk=learner.table_raw(table)
    gain,risk=learner.apply_calibration(raw_gain,raw_risk)
    return table[['target_domain','horizon','stage','origin_day','gain_marginal']].assign(
        objective=objective,raw_gain=raw_gain,raw_risk=raw_risk,predicted_gain=gain,calibrated_risk=risk)


def _select_rule(model,validation,grid):
    candidates=[]
    for margin,risk in product(grid['margin_pp'],grid['max_risk']):
        model.rule=dict(margin_pp=margin,max_risk=risk)
        prediction=model.predict(validation.inputs)
        scores=[forecast_metrics(validation.targets[np.asarray(validation.inputs.domain_ids)==d],prediction[np.asarray(validation.inputs.domain_ids)==d])['MAPE'] for d in sorted(set(validation.inputs.domain_ids))]
        candidates.append(dict(**model.rule,mean_MAPE=float(np.mean(scores)),worst_MAPE=float(max(scores)),objective=float(np.mean(scores)+grid['worst_domain_penalty']*max(scores))))
    best=min(candidates,key=lambda r:(r['objective'],-r['margin_pp'],r['max_risk']))
    model.rule={k:best[k] for k in ['margin_pp','max_risk']}
    return dict(selected=best,candidates=candidates,validation_domains=sorted(set(validation.inputs.domain_ids)))


def _models_for_stage(panel,roles,config,models,task,learners,priors,cutoff,initial=None):
    h=horizons(config,panel.points_per_day)[task]
    banks={}
    for representation in REPRESENTATIONS:
        banks[representation],training=fit_bank(panel,roles['source'],cutoff,h,representation,models['ridge_alpha'])
    bank=banks['level_shape'];built={};choices={}
    validation=domain_windows(panel,roles['validation'],config['validation_origins'],14*panel.points_per_day,h) if initial is None else None
    if validation is not None and validation.skipped_windows:
        raise ValueError('No silently omitted policy-validation windows.')
    for name in models['default_models']:
        if name in GAIN_METHODS:
            objective='marginal' if name in {'gain_v2_forced','gain_v2_uncalibrated'} else name[8:]
            model=create_model(name,dict(bank=bank,learner=learners[objective],priors=reduce_priors(priors,bank.source_domains)),config['seed']).fit(training)
            if initial is not None:
                model.rule=deepcopy(initial[name].rule)
            elif name in {'gain_v2_forced','gain_v2_uncalibrated'}:
                model.rule=deepcopy(built['gain_v2_marginal'].rule)
                choices[name]=dict(inherited_from='gain_v2_marginal',rule=model.rule,only_intervention='force rejected actions' if name.endswith('forced') else 'remove learned calibration')
            else:
                choices[name]=_select_rule(model,validation,models['policy_grid'])
        elif name=='uniform_transfer':
            model=FixedExpertBlend(bank,1.,name).fit(training)
        elif name.startswith('fixed_blend_'):
            model=create_model(name,dict(bank=bank),config['seed']).fit(training)
        elif name=='sapf':
            if initial is None:
                model=create_model(name,models['sapf'],config['seed']).fit(training).calibrate(validation)
                choices[name]=model.calibration
            else:
                model=deepcopy(initial[name]).refit(training)
        else:
            model=create_model(name,dict(period=panel.points_per_day) if name=='seasonal_naive' else
                dict(alpha=models['ridge_alpha']) if name.startswith('ridge_') else {},config['seed']).fit(training)
        built[name]=model
    return built,banks,training,choices


def _save_predictions(folder,panel,batch,task,name,split,prediction,rows,origin_tables,oracle=False):
    path=folder/'predictions'/task/(name+'.parquet');path.parent.mkdir(parents=True,exist_ok=True)
    writer=None
    def sink(table,origin):
        nonlocal writer
        arrow=pa.Table.from_pandas(table,preserve_index=False)
        if writer is None:writer=pq.ParquetWriter(path,arrow.schema,compression='zstd')
        writer.write_table(arrow)
        origin_tables.append(origin)
    try:
        for domain in dict.fromkeys(batch.inputs.domain_ids):
            indices=np.flatnonzero(np.asarray(batch.inputs.domain_ids)==domain)
            part=WindowBatch(batch.inputs.take(indices),batch.targets[indices]);values=prediction[indices]
            scored=score_predictions(part,values,anchor=part.inputs.origins[0])
            extra=dict(target_domain=domain,target_unit='kW',source_strategy='target_history' if name in {'seasonal_naive','three_day_mean'} else 'oracle_analysis_only' if oracle else 'source_only_transfer',
                oracle_only=oracle,source_clock_start=panel.audit[domain]['source_clock_start'],protocol_anchor=str(panel.anchor),
                source_clock_timezone_verified=panel.metadata[domain].timezone_verified)
            save_result(folder,task,name,split,part,scored,rows,0.,'kW',False,row_metadata=extra,quiet=True,compact_sink=sink)
            if task=='3d':
                keep=np.asarray(((part.inputs.origins-part.inputs.origins[0])/pd.Timedelta(days=1))%3==0)
                rows.append(dict(horizon=task,model=name,split=split,evaluation='curve_nonoverlap',train_seconds=0.,inference_seconds=0.,**extra,**forecast_metrics(part.targets[keep],values[keep])))
    finally:
        if writer is not None:writer.close()


def _pair_diagnostics(batch,prediction,info,model,task,split):
    frames=[];n,s=info['experts'].shape[:2]
    target=np.asarray(batch.inputs.domain_ids);sources=np.asarray(model.source_domains)
    local_loss=ape(batch.targets,info['local']);final_loss=ape(batch.targets,prediction)
    for step in info['steps']:
        i,j=np.nonzero(step['available'])
        base_loss=ape(batch.targets,step['current'])
        mixed=.5*step['current'][:,None]+.5*info['experts']
        gain=base_loss[:,None]-ape(batch.targets[:,None],mixed)
        frames.append(pd.DataFrame(dict(target_domain=target[i],source_domain=sources[j],origin=batch.inputs.origins[i],
            horizon=task,model=model.name,split=split,stage=step['stage'],actual_gain_pp=gain[i,j],
            predicted_gain_pp=step['gain'][i,j],negative_risk=step['risk'][i,j],raw_gain_pp=step['raw_gain'][i,j],raw_risk=step['raw_risk'][i,j],
            selected=step['chosen'][i]==j,
            eligible=step['eligible'][i,j],final_source_weight=info['weights'][i,j],transfer_gate=1-info['local_weight'][i],
            local_MAPE=local_loss[i],final_MAPE=final_loss[i],source_marginal_negative_prior=model.learner.marginal_negative_prior)))
    return pd.concat(frames,ignore_index=True)


def _oracle(batch,bank,stored,alpha_grid):
    local=stored['three_day_mean'];experts=bank.expert_predictions(batch.inputs)
    best=local.copy();loss=ape(batch.targets,best);choice=np.full(len(batch),'local',dtype=object)
    def consider(values,label):
        nonlocal loss,best,choice
        score=ape(batch.targets,values);keep=score<loss-1e-12
        best[keep]=values[keep];loss[keep]=score[keep];choice[keep]=label
    for name,values in stored.items():consider(values,'frozen_'+name)
    for s,source in enumerate(bank.source_domains):
        for alpha in alpha_grid:consider((1-alpha)*local+alpha*experts[:,s],f'{source}:alpha={alpha}')
        for j,other in enumerate(bank.source_domains):
            if s!=j:consider(.25*local+.25*experts[:,s]+.5*experts[:,j],f'{source}:then:{other}')
    return best,choice


def _evaluate(folder,panel,domains,origins,task,models,banks,config,split,root=None,paths=None):
    h=horizons(config,panel.points_per_day)[task]
    batch=domain_windows(panel,domains,origins,14*panel.points_per_day,h)
    if batch.skipped_windows:
        raise ValueError('Fixed evaluation grids cannot silently omit windows.')
    rows,origin_tables,pairs,restores,stored=[],[],[],[],{}
    for name,model in models.items():
        if name in GAIN_METHODS:
            prediction,info=model.predict_with_diagnostics(batch.inputs)
            pairs.append(_pair_diagnostics(batch,prediction,info,model,task,split))
        else:prediction=model.predict(batch.inputs)
        prediction=np.maximum(prediction,0.);stored[name]=prediction
        if root is not None:
            restored=joblib.load(root/paths[name]).predict(batch.inputs)
            difference=float(np.max(np.abs(np.maximum(restored,0.)-prediction)))
            restores.append(dict(horizon=task,model=name,windows=len(batch),maximum_absolute_difference=difference,passed=difference<1e-7))
        _save_predictions(folder,panel,batch,task,name,split,prediction,rows,origin_tables)
    representation=[]
    for representation_name,bank in banks.items():
        predictions=bank.expert_predictions(batch.inputs)
        losses=ape(batch.targets[:,None],predictions)
        for i,domain in enumerate(batch.inputs.domain_ids):
            for j,source in enumerate(bank.source_domains):
                representation.append(dict(horizon=task,split=split,representation=representation_name,target_domain=domain,source_domain=source,
                    origin=str(batch.inputs.origins[i]),MAPE=float(losses[i,j])))
    oracle_rows=None
    if split=='test':
        oracle,choice=_oracle(batch,banks['level_shape'],stored,load_config(config['model_config'])['oracle']['alpha_grid'])
        _save_predictions(folder,panel,batch,task,'oracle_posthoc',split,oracle,rows,origin_tables,True)
        oracle_rows=pd.DataFrame(dict(target_domain=batch.inputs.domain_ids,origin=batch.inputs.origins,horizon=task,choice=choice,
            MAPE=ape(batch.targets,oracle),oracle_only=True))
    return rows,pd.concat(origin_tables,ignore_index=True),pd.concat(pairs,ignore_index=True),pd.DataFrame(representation),restores,oracle_rows


def _summary(metrics):
    rows=[]
    for (task,evaluation),part in metrics[metrics.split=='test'].groupby(['horizon','evaluation']):
        reference=part[part.model=='three_day_mean'].set_index('target_domain').MAPE
        for model,group in part.groupby('model',sort=False):
            differences=group.MAPE.to_numpy()-reference.reindex(group.target_domain).to_numpy()
            rows.append(dict(horizon=task,evaluation=evaluation,model=model,targets=len(group),mean_MAPE=group.MAPE.mean(),worst_MAPE=group.MAPE.max(),
                mean_WAPE=group.WAPE.mean(),mean_RMSE=group.RMSE.mean(),mean_MAE=group.MAE.mean(),total_points=group.n.sum(),
                negative_vs_local_domains=int((differences>1e-10).sum()),negative_vs_local_pct=100*float((differences>1e-10).mean()),
                oracle_only=bool(group.oracle_only.iloc[0])))
    return pd.DataFrame(rows)


def run_gain_v2(args,config):
    if any(getattr(args,key,None) for key in ['demo','models','horizons']):
        raise ValueError('Run the complete jointly frozen Gain-v2 suite; fixture tests cover synthetic data.')
    validate_config(config)
    model_config=load_config(config['model_config'])
    prereg_path=resolve_path(config.get('preregistration','results/analysis/gain_v2/preregistration.json'))
    if not prereg_path.exists() and not getattr(args,'test_fixture',False):
        raise ValueError('Create the pre-test source/config/quality preregistration before real forecasting.')
    if prereg_path.exists():
        prereg=json.loads(prereg_path.read_text(encoding='utf-8'))
        for path,digest in prereg['computational_source_sha256'].items():
            if sha256((PROJECT_ROOT/path).read_bytes()).hexdigest()!=digest:
                raise ValueError('Computational source differs from preregistration: '+path)
    identifier=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_sapf_gain_v2_'+uuid.uuid4().hex[:6]
    folder=resolve_path(config['results_dir'])/identifier;folder.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',protocol=config['protocol'],data='M100/UCC measured node power',target_unit='kW',is_demo=False,experiment=config,
        model_config=model_config,studies=config['studies'],requested_horizons=list(config['horizon_days']),
        source_sha256={p.relative_to(PROJECT_ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for d in ['src','scripts','configs'] for p in (PROJECT_ROOT/d).rglob('*') if p.suffix in {'.py','.yaml'}})
    write_json(folder/'manifest.json',manifest)
    write_json(folder/'predeclared-plan.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),config=config,models=model_config))
    if prereg_path.exists():write_json(folder/'preregistration.json',prereg)
    try:
        panels={};group_models={};group_banks={};group_paths={};group_records={}
        for group in ['m100','ucc']:
            raw,provenance=raw_power_frames(config,group,sum(config['domain_roles'][group].values(),[]))
            for frequency in (['30min','15min'] if group=='m100' else ['30min']):
                panels[_group_key(group,frequency)]=load_gain_v2_panel(config,group,frequency,raw,provenance)
        adapted_hashes={}
        manifest['data_provenance']={}
        manifest['panel_metadata']={}
        for key,panel in panels.items():
            manifest['data_provenance'].update(panel.sources)
            manifest['panel_metadata'][key]={d:panel.metadata[d].to_dict() for d in panel.series}
            for domain,frame in panel.series.items():
                path=folder/'adapted-data'/key/(domain+'.parquet');path.parent.mkdir(parents=True,exist_ok=True)
                frame.to_parquet(path);adapted_hashes[path.relative_to(folder).as_posix()]=sha256(path.read_bytes()).hexdigest()
        write_json(folder/'manifest.json',manifest)
        with threadpool_limits(config['blas_threads'],user_api='blas'):
            for key,panel in panels.items():
                group=key.split('_')[0];roles=config['domain_roles'][group];location=folder/'groups'/key;location.mkdir(parents=True)
                print(f'[{key}] source={len(roles["source"])} calibration={len(roles["calibration"])} selection={len(roles["validation"])}; source-internal value learning',flush=True)
                table,priors=train_value_tables(panel,roles['source'],config,model_config,location)
                dev_banks={};calibration_parts=[]
                for task,h in horizons(config,panel.points_per_day).items():
                    bank,_=fit_bank(panel,roles['source'],config['development_cutoff_day'],h,'level_shape',model_config['ridge_alpha']);dev_banks[task]=bank
                    calibration_parts.append(make_value_table(panel,roles['calibration'],config['calibration_origins'],bank,priors[task],task,h,config['development_cutoff_day'],False))
                calibration_table=pd.concat(calibration_parts,ignore_index=True);calibration_table.to_parquet(location/'calibration-data.parquet',index=False)
                learners={};calibration_predictions=[]
                for objective in OBJECTIVES:
                    learner=TransferValueLearner(objective,model_config['benefit'],config['seed']).fit(table,roles['source']).calibrate(calibration_table,roles['calibration'])
                    path=location/'learners'/objective;path.mkdir(parents=True);joblib.dump(learner,path/'model.joblib')
                    write_json(path/'calibration.json',learner.calibration);learners[objective]=learner
                    calibration_predictions.append(_calibration_predictions(learner,calibration_table,objective))
                calibration_prediction=pd.concat(calibration_predictions,ignore_index=True)
                calibration_prediction.to_parquet(location/'calibration-predictions.parquet',index=False)
                curve_input=calibration_prediction.rename(columns={'gain_marginal':'actual_gain_pp','raw_gain':'raw_gain_pp','predicted_gain':'predicted_gain_pp','calibrated_risk':'negative_risk'})
                curve_input['model']='gain_v2_'+curve_input.objective
                calibration_curves(curve_input).assign(split='calibration_in_sample').to_csv(location/'calibration-curves.csv',index=False)
                validation_rows=[];validation_origins=[];records=[]
                for task,h in horizons(config,panel.points_per_day).items():
                    initial,initial_banks,_,selection=_models_for_stage(panel,roles,config,model_config,task,learners,priors[task],config['development_cutoff_day'])
                    write_json(location/task/'validation-selection.json',selection)
                    val=_evaluate(location,panel,roles['validation'],config['validation_origins'],task,initial,initial_banks,config,'validation')
                    validation_rows.extend(val[0]);validation_origins.append(val[1])
                    final,banks,training,_=_models_for_stage(panel,roles,config,model_config,task,learners,priors[task],config['final_cutoff_day'],initial)
                    paths={}
                    for representation,bank in banks.items():
                        path=location/task/'expert-banks'/representation;path.mkdir(parents=True)
                        bank.save(path/'model')
                    for name,model in final.items():
                        path=location/task/name;path.mkdir(parents=True)
                        paths[name]=model.save(path/'model').relative_to(folder).as_posix()
                        records.append(dict(model=name,horizon=task,model_path=paths[name],source_domains=[] if name in {'seasonal_naive','three_day_mean'} else roles['source'],
                            calibration_domains=roles['calibration'] if name in GAIN_METHODS else [],selection_domains=roles['validation'] if name in GAIN_METHODS or name=='sapf' else [],
                            final_cutoff_day=config['final_cutoff_day'],last_training_label=str(max(pd.Timestamp(t) for t in training.inputs.target_times[:,-1])),
                            windows=len(training),rule=getattr(model,'rule',None)))
                    group_models[key,task],group_banks[key,task],group_paths[key,task]=final,banks,paths
                    print(f'[{key}] {task} final fit and independent selection complete; no test error accessed',flush=True)
                pd.DataFrame(validation_rows).to_csv(location/'validation-metrics.csv',index=False)
                pd.concat(validation_origins,ignore_index=True).to_parquet(location/'validation-per-origin.parquet',index=False)
                write_json(location/'training-records.json',records);group_records[key]=records
            # All directions/cadences/horizons freeze together, before any target errors.
            freeze_for_test(folder,global_domain_roles=config['domain_roles'],group_training_records=group_records,studies=config['studies'],
                adapted_data_sha256=adapted_hashes,
                training_artifact_sha256={p.relative_to(folder).as_posix():sha256(p.read_bytes()).hexdigest() for p in (folder/'groups').rglob('*') if p.suffix in {'.json','.csv','.parquet'}})
            print('[Gain-v2] All source groups frozen. Starting all new held-out tests.',flush=True)
            all_origins,all_pairs,all_metrics=[],[],[]
            for study in config['studies']:
                key=_group_key(study['source_group'],study['frequency']);target_key=_group_key(study['test_group'],study['frequency'])
                panel=panels[target_key];test=config['domain_roles'][study['test_group']]['test'];source_roles=config['domain_roles'][study['source_group']]
                child=folder/'studies'/study['name'];child.mkdir(parents=True)
                child_manifest=dict(status='running',protocol='sapf_gain_v2',research_method=study['name'],data=study['source_group'].upper()+' source -> '+study['test_group'].upper()+' new-period nodes',
                    target='node_power_kw',target_unit='kW',is_demo=False,experiment=dict(frequency=study['frequency']),requested_horizons=list(config['horizon_days']),
                    console_methods=model_config['default_models']+['oracle_posthoc'],domain_roles={**{k:v for k,v in source_roles.items() if k!='test'},'test':test},source_group=key,
                    target_dataset_labels_used_for_selection=study['source_group']==study['test_group'],domain_metadata={d:panel.metadata[d].to_dict() for d in test},
                    source_clock_audit={d:panel.audit[d] for d in test},model_paths={h:group_paths[key,h] for h in config['horizon_days']})
                write_json(child/'manifest.json',child_manifest)
                write_json(child/'test-start.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),global_freeze_sha256=sha256((folder/'frozen-before-test.json').read_bytes()).hexdigest()))
                for domain in test:
                    path=child/'adapted-data'/(domain+'.parquet');path.parent.mkdir(exist_ok=True)
                    shutil.copyfile(folder/'adapted-data'/target_key/(domain+'.parquet'),path)
                rows,origins,pairs,representation,restores,oracles=[],[],[],[],[],[]
                for task in config['horizon_days']:
                    result=_evaluate(child,panel,test,config['test_origins'][task],task,group_models[key,task],group_banks[key,task],config,'test',folder,group_paths[key,task])
                    rows.extend(result[0]);origins.append(result[1]);pairs.append(result[2]);representation.append(result[3]);restores.extend(result[4]);oracles.append(result[5])
                    print(f'[{study["name"]}] {task} test and saved-model replay complete',flush=True)
                metrics=pd.DataFrame(rows);origin=pd.concat(origins,ignore_index=True);pair=pd.concat(pairs,ignore_index=True)
                metrics.to_csv(child/'metrics.csv',index=False);_summary(metrics).to_csv(child/'domain-summary.csv',index=False)
                origin.to_parquet(child/'per-origin.parquet',index=False);pair.to_parquet(child/'gain-diagnostics.parquet',index=False)
                value_summary(pair).to_csv(child/'risk-summary.csv',index=False)
                calibration_curves(pair).to_csv(child/'calibration-curves.csv',index=False)
                pd.concat(representation,ignore_index=True).to_parquet(child/'representation-pairs.parquet',index=False)
                pd.concat(oracles,ignore_index=True).to_parquet(child/'oracle-choices.parquet',index=False)
                write_json(child/'saved-model-audit.json',restores)
                if not all(r['passed'] for r in restores):raise ValueError('A saved-model replay changed its forecast.')
                child_manifest.update(status='complete');write_json(child/'manifest.json',child_manifest)
                all_metrics.append(metrics.assign(study=study['name']));all_origins.append(origin.assign(study=study['name']));all_pairs.append(pair.assign(study=study['name']))
            metric=pd.concat(all_metrics,ignore_index=True);origin=pd.concat(all_origins,ignore_index=True);pair=pd.concat(all_pairs,ignore_index=True)
            metric.to_csv(folder/'metrics.csv',index=False)
            print('[Gain-v2] Frozen tests complete; target-cluster/block uncertainty and calibration diagnostics.',flush=True)
            paired_statistics(origin,config).to_csv(folder/'statistics.csv',index=False)
            risk_statistics(pair,config).to_csv(folder/'risk-statistics.csv',index=False)
        from src.evaluation.gain_v2_audit import audit_gain_v2
        audit=audit_gain_v2(folder);write_json(folder/'audit.json',audit)
        if not audit['passed']:raise ValueError('Gain-v2 freeze/score/source-boundary audit failed.')
        manifest.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat());write_json(folder/'manifest.json',manifest)
        print_saved_results(folder)
        return folder
    except Exception as exc:
        manifest.update(status='failed',error=type(exc).__name__+': '+str(exc));write_json(folder/'manifest.json',manifest)
        raise
