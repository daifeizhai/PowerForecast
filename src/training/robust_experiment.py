"""Source-only expert audit and fixed-fusion admission before any new target score."""
from datetime import datetime,timezone
from hashlib import sha256
import json,uuid
from pathlib import Path

import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits

from src.data.domains import assert_data_usage
from src.data.robust_panel import load_robust_panels
from src.data.google_panel import domain_windows
from src.evaluation.metrics import forecast_metrics
from src.models.ml.sapf import SelectiveTransferForecaster
from src.models.ml.robust_transfer import RobustFixedFusion,history_scale,local_prediction
from src.training.gain_v2_learning import source_batch
from src.utils import PROJECT_ROOT,resolve_path,write_json


def horizon_points(config,period):return {key:int(round(value*period)) for key,value in config['horizon_days'].items()}


def expanded_origins(days,task):
    return [day+part/6 for day in days for part in range(6)] if task=='4h' else list(days)


def test_origins(config,group,task):
    spec=config['dataset_clocks'][group];stop=spec['test_stop_day']-(2 if task=='3d' else 0)
    return expanded_origins(range(spec['test_start_day'],stop),task)


def fit_experts(panel,sources,cutoff,horizon,alpha):
    training=source_batch(panel,sources,cutoff,horizon)
    bank=SelectiveTransferForecaster(base_params={'alpha':alpha},top_k=[1,3],temperatures=[1.],feature_groups=['shape'],
        transfer_strengths=[0.,.1,.25,.5],reliability_powers=[0.,1.]).fit(training)
    return bank,training


def numerical_diagnostics(bank,batch,config,folder,split):
    raw=bank.expert_predictions(batch.inputs);mu,sigma=history_scale(batch.inputs)
    z=(raw-mu[:,None,None])/sigma[:,None,None]
    power=batch.inputs.history[:,:,0].astype(float);records=[];restoration=0.
    for j,expert in enumerate(bank.experts):
        transformed,location,scale=expert._transform(batch.inputs)
        direct=expert.model.predict(transformed)*sigma[:,None]+mu[:,None]
        restoration=max(restoration,float(np.max(np.abs(direct-raw[:,j]))))
    for i,(domain,origin) in enumerate(zip(batch.inputs.domain_ids,batch.inputs.origins)):
        cv=sigma[i]/max(abs(mu[i]),1e-8)
        for j,source in enumerate(bank.source_domains):
            values=raw[i,j];finite=np.isfinite(values).all()
            score=forecast_metrics(batch.targets[i],values) if finite else dict(MAPE=np.nan,WAPE=np.nan,RMSE=np.nan,MAE=np.nan,n=len(values),mape_n=0,mape_excluded=0)
            records.append(dict(target_domain=domain,source_domain=source,origin=str(origin),split=split,
                hardware_group='_'.join(domain.split('_')[:2]),history_mu_kw=mu[i],history_sigma_kw=sigma[i],history_cv=cv,
                history_state='low_variation' if cv<.05 else 'medium_variation' if cv<.2 else 'high_variation',
                history_min_kw=float(power[i].min()),history_max_kw=float(power[i].max()),prediction_min_kw=float(values.min()),prediction_max_kw=float(values.max()),
                prediction_mean_kw=float(values.mean()),negative_fraction=float((values<0).mean()),nonfinite_fraction=float((~np.isfinite(values)).mean()),
                max_abs_z=float(np.max(np.abs(z[i,j]))),mean_level_ratio=float(np.abs(values.mean())/max(abs(mu[i]),1e-8)),
                beyond_history_envelope_fraction=float(((values<power[i].min()-3*sigma[i])|(values>power[i].max()+3*sigma[i])).mean()),**score))
    table=pd.DataFrame(records);folder.mkdir(parents=True,exist_ok=True)
    table.to_parquet(folder/(split+'-expert-diagnostics.parquet'),index=False)
    np.savez_compressed(folder/(split+'-expert-predictions.npz'),raw_kw=raw,standardized=z,actual_kw=batch.targets,
        history_mu_kw=mu,history_sigma_kw=sigma,target_domain=np.asarray(batch.inputs.domain_ids),origin=np.asarray([str(o) for o in batch.inputs.origins]),sources=np.asarray(bank.source_domains))
    rule=config['numerical_gate'];negative=float((raw<0).mean());nonfinite=float((~np.isfinite(raw)).mean())
    severe=float((np.abs(z)>rule['severe_z_abs']).mean());extreme=float((np.abs(raw.mean(2))/np.maximum(np.abs(mu[:,None]),1e-8)>rule['extreme_level_ratio']).mean())
    passed=nonfinite<=rule['nonfinite_fraction_max'] and negative<=rule['negative_fraction_max'] and severe<=rule['severe_z_fraction_max'] and extreme<=rule['extreme_level_fraction_max'] and restoration<1e-8
    return raw,dict(passed=bool(passed),negative_fraction=negative,nonfinite_fraction=nonfinite,severe_z_fraction=severe,extreme_level_fraction=extreme,
        max_abs_z=float(np.max(np.abs(z))),restoration_max_difference_kw=restoration,raw_predictions_audited_before_nonnegative_clip=True)


def domain_scores(batch,prediction,task,primary=True):
    rows=[]
    for domain in dict.fromkeys(batch.inputs.domain_ids):
        idx=np.flatnonzero(np.asarray(batch.inputs.domain_ids)==domain)
        if task=='3d' and primary:idx=idx[::3]
        rows.append(dict(target_domain=domain,**forecast_metrics(batch.targets[idx],prediction[idx])))
    return pd.DataFrame(rows)


def _signal(batch,prediction,local_kind,task,rule):
    scores=domain_scores(batch,prediction,task).set_index('target_domain').MAPE
    baselines={kind:domain_scores(batch,local_prediction(batch.inputs,kind),task).set_index('target_domain').MAPE for kind in ['three_day_mean','seasonal_naive']}
    strongest=min(baselines,key=lambda k:baselines[k].mean());baseline=baselines[strongest]
    improvement=1-scores.mean()/baseline.mean();domain_delta=(baseline-scores).to_numpy()
    jackknife=min(float(np.delete(domain_delta,i).mean()) for i in range(len(domain_delta))) if len(domain_delta)>1 else float(domain_delta[0])
    tail=scores.max()/baseline.max()-1;fraction=float((scores<baseline-1e-10).mean())
    passed=improvement>=rule['relative_MAPE_gain_min'] and tail<=rule['worst_relative_increase_max'] and fraction>=rule['improved_domain_fraction_min'] and jackknife>=rule['leave_one_domain_out_gain_min']
    return dict(passed=bool(passed),relative_MAPE_gain=float(improvement),mean_MAPE=float(scores.mean()),strong_baseline=strongest,strong_baseline_MAPE=float(baseline.mean()),
        improved_domain_fraction=fraction,worst_relative_increase=float(tail),leave_one_domain_out_min_delta_pp=jackknife,local_rule_selected_on_development=local_kind)


def run_development(args,config):
    if getattr(args,'demo',False) or getattr(args,'models',None) or getattr(args,'horizons',None):raise ValueError('Run the full prespecified source-development protocol.')
    folder=resolve_path(config['results_dir'])/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_sapf_robust_dev_'+uuid.uuid4().hex[:6]);folder.mkdir(parents=True)
    manifest=dict(status='running',protocol='sapf_robust_development',evaluation_role='source_development_only',experiment=config,is_demo=False,
        data='Source-side validation only; no new target forecast errors',target_unit='kW')
    write_json(folder/'manifest.json',manifest)
    try:
        panels,provenance=load_robust_panels(config);write_json(folder/'data-fingerprints.json',provenance)
        all_rows=[];gates={};bank_paths={}
        with threadpool_limits(1,user_api='blas'):
            for key,panel in panels.items():
                group=key.split('_')[0];roles=config['domain_roles'][group];gates[key]={};bank_paths[key]={}
                for task,horizon in horizon_points(config,panel.points_per_day).items():
                    bank,training=fit_experts(panel,roles['source'],config['source_development_cutoff'],horizon,config['ridge_alpha'])
                    path=folder/'groups'/key/task;path.mkdir(parents=True);bank.save(path/'development-bank')
                    bank_paths[key][task]=(path/'development-bank.joblib').relative_to(folder).as_posix()
                    predictions={};batches={};numeric={}
                    for split,domains,days in [('development',roles['calibration'],config['development_days']),('validation',roles['validation'],config['validation_days'])]:
                        for domain in domains:assert_data_usage(panel.metadata[domain],'validation')
                        batch=domain_windows(panel,domains,expanded_origins(days,task),14*panel.points_per_day,horizon)
                        if batch.skipped_windows:raise ValueError('A declared source validation window is missing.')
                        raw,check=numerical_diagnostics(bank,batch,config,path,split);numeric[split]=check;batches[split]=batch
                        predictions[split]={'three_day_mean':local_prediction(batch.inputs,'three_day_mean').astype(float),'seasonal_naive':local_prediction(batch.inputs,'seasonal_naive').astype(float),'external_mean':np.maximum(raw,0.).mean(1)}
                    dev=batches['development'];values=predictions['development']
                    local_kind=min(['three_day_mean','seasonal_naive'],key=lambda k:domain_scores(dev,values[k],task).MAPE.mean())
                    selected={};candidates=[]
                    for bounded in [False,True]:
                        records=[]
                        for alpha in config['alpha_candidates']:
                            for split,batch in batches.items():
                                external=predictions[split]['external_mean'];local=predictions[split][local_kind];_,sigma=history_scale(batch.inputs);delta=external-local
                                if bounded:delta=np.clip(delta,-config['correction_bound_sigma']*sigma[:,None],config['correction_bound_sigma']*sigma[:,None])
                                prediction=local+alpha*delta;method=('bounded_' if bounded else '')+'fixed_'+str(alpha)
                                scored=domain_scores(batch,prediction,task).assign(source_group=key,horizon=task,split=split,model=method,alpha=alpha,bounded=bounded)
                                all_rows.append(scored)
                                if split=='development':records.append(dict(alpha=alpha,mean_MAPE=float(scored.MAPE.mean())))
                        chosen=min(records,key=lambda r:(r['mean_MAPE'],r['alpha']))['alpha'];selected['bounded' if bounded else 'fixed']=chosen;candidates.extend([dict(bounded=bounded,**r) for r in records])
                    validation_results={}
                    for split,batch in batches.items():
                        local=predictions[split][local_kind];external=predictions[split]['external_mean'];p=local+selected['fixed']*(external-local)
                        validation_results[split]=_signal(batch,p,local_kind,task,config['source_signal_gate'])
                        for name in ['three_day_mean','seasonal_naive','external_mean']:
                            all_rows.append(domain_scores(batch,predictions[split][name],task).assign(source_group=key,horizon=task,split=split,model=name))
                    admitted=all(c['passed'] for c in numeric.values());signal=admitted and selected['fixed']>0 and all(v['passed'] for v in validation_results.values())
                    gates[key][task]=dict(numerical= numeric,numerically_admitted=admitted,local_kind=local_kind,selected_alphas=selected,candidates=candidates,
                        source_signal=validation_results,stable_positive_signal=bool(signal),dynamic_allowed=bool(signal and task in ['24h','3d']))
                    write_json(path/'selection.json',gates[key][task]);print(f'[{key}/{task}] numerical={admitted} alpha={selected["fixed"]} stable-source-signal={signal}',flush=True)
        pd.concat(all_rows,ignore_index=True).to_csv(folder/'source-metrics.csv',index=False)
        write_json(folder/'source-h1-gates.json',gates)
        manifest.update(status='development_complete',data_provenance=provenance,bank_paths=bank_paths,completed_at_utc=datetime.now(timezone.utc).isoformat(),
            no_new_target_forecast_errors_accessed=True,source_h1_gates=gates)
        write_json(folder/'manifest.json',manifest)
        print(f'Source development saved: {folder}',flush=True)
        return folder
    except Exception as exc:
        manifest.update(status='failed',error=type(exc).__name__+': '+str(exc));write_json(folder/'manifest.json',manifest);raise


def train_analog(panel,sources,task,horizon,local_kind,config,folder):
    from src.models.ml.robust_transfer import AnalogFusionMap,analog_features
    dynamic=config['conditional_dynamic'];records=[];period=panel.points_per_day
    for cutoff in dynamic['teacher_cutoffs']:
        bank,_=fit_experts(panel,sources,cutoff,horizon,config['ridge_alpha'])
        stop=min(cutoff+dynamic['teacher_block_days'],dynamic['last_label_day'])
        hdays=horizon/period
        origins=expanded_origins(range(cutoff,stop),task) if task=='4h' else list(range(cutoff,stop-int(np.ceil(hdays))+1))
        origins=[d for d in origins if d+hdays<=stop+1e-8]
        for target in sources:
            batch=domain_windows(panel,[target],origins,14*period,horizon)
            raw=bank.expert_predictions(batch.inputs);keep=[d!=target for d in bank.source_domains]
            experts=[d for d in bank.source_domains if d!=target];external=np.maximum(raw[:,keep],0.).mean(1)
            local=local_prediction(batch.inputs,local_kind).astype(float);_,sigma=history_scale(batch.inputs)
            features=analog_features(batch.inputs,bank,external,local,exclude_source=target)
            correction=np.clip(external-local,-config['correction_bound_sigma']*sigma[:,None],config['correction_bound_sigma']*sigma[:,None])
            for i,origin in enumerate(batch.inputs.origins):
                day=float((origin-panel.anchor)/pd.Timedelta(days=1));local_loss=forecast_metrics(batch.targets[i],local[i])['MAPE']
                gains={f'gain_{alpha:g}':(local_loss-forecast_metrics(batch.targets[i],local[i]+alpha*correction[i])['MAPE'])/max(local_loss,.1) for alpha in config['alpha_candidates']}
                records.append(dict(target_domain=target,expert_domains=experts,teacher_cutoff_day=cutoff,origin_day=day,label_stop_day=day+hdays,
                    shape_distance=features[i,0],forecast_disagreement=features[i,1],local_MAPE=local_loss,**gains))
    table=pd.DataFrame(records);table.to_parquet(folder/'analog-training.parquet',index=False)
    model=AnalogFusionMap().fit(table,sources,config['alpha_candidates'],dynamic['neighbors_per_source'],dynamic['minimum_expected_relative_gain'])
    joblib.dump(model,folder/'analog.joblib');return model


def fit_reference_gain(panel,roles,config,folder):
    """Unchanged Gain-v2 structure as a required reference, not new gate research."""
    from src.training.gain_v2_learning import train_value_tables,fit_bank,make_value_table
    from src.models.ml.gain_v2 import TransferValueLearner,GainV2Policy
    from src.training.benefit_learning import reduce_priors
    from src.training.gain_v2_experiment import _select_rule
    from src.utils import load_config
    original=load_config('configs/gain_v2_models.yaml');plan=dict(config,prior_cutoff_day=20,prior_origins=[20,21],meta_cutoffs=[24,30,36],meta_block_days=6)
    table,priors=train_value_tables(panel,roles['source'],plan,original,folder)
    parts=[];banks={}
    for task,h in horizon_points(config,panel.points_per_day).items():
        bank,_=fit_bank(panel,roles['source'],44,h,'level_shape',config['ridge_alpha']);banks[task]=bank
        parts.append(make_value_table(panel,roles['calibration'],expanded_origins(config['development_days'],task),bank,priors[task],task,h,44,False))
    calibration=pd.concat(parts,ignore_index=True);calibration.to_parquet(folder/'calibration-data.parquet',index=False)
    learner=TransferValueLearner('marginal',original['benefit'],config['seed']).fit(table,roles['source']).calibrate(calibration,roles['calibration'])
    joblib.dump(learner,folder/'learner.joblib');write_json(folder/'calibration.json',learner.calibration)
    final={}
    for task,h in horizon_points(config,panel.points_per_day).items():
        training=source_batch(panel,roles['source'],44,h)
        policy=GainV2Policy(bank=banks[task],learner=learner,priors=reduce_priors(priors[task],banks[task].source_domains)).fit(training)
        validation=domain_windows(panel,roles['validation'],expanded_origins(config['validation_days'],task),14*panel.points_per_day,h)
        choices=_select_rule(policy,validation,original['policy_grid']);write_json(folder/task/'validation-selection.json',choices)
        bank,training=fit_bank(panel,roles['source'],config['final_source_cutoff'],h,'level_shape',config['ridge_alpha'])
        final[task]=GainV2Policy(name='gain_v2_marginal',bank=bank,learner=learner,priors=policy.priors,rule=policy.rule).fit(training)
    return final
