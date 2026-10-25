"""Source-only chronological teacher labels for complementary transfer value."""
from math import ceil

import numpy as np
import pandas as pd

from src.data.domains import assert_data_usage
from src.data.google_panel import domain_windows
from src.models.ml.benefit import PRIOR_FEATURES
from src.models.ml.gain_v2 import VALUE_FEATURES, value_features
from src.models.ml.level_shape import RepresentationBank
from src.models.statistical.baselines import ThreeDayMeanForecaster
from src.training.benefit_learning import reduce_priors


def ape(actual, prediction):
    actual,prediction = np.asarray(actual,float),np.asarray(prediction,float)
    valid = np.abs(actual)>1e-8
    if np.any(valid.sum(-1)==0):
        raise ValueError('No finite MAPE denominator in this origin.')
    return 100*np.divide(np.abs(actual-prediction),np.abs(actual),out=np.zeros_like(prediction,dtype=float),where=valid).sum(-1)/valid.sum(-1)


def horizons(config, period):
    return {key:int(round(value*period)) for key,value in config['horizon_days'].items()}


def source_batch(panel, sources, cutoff, horizon):
    for domain in sources:
        assert_data_usage(panel.metadata[domain],'training')
    # Daily origins: even a 4h label must finish before the fitting cutoff.
    origins = list(range(14,cutoff-ceil(horizon/panel.points_per_day)+1))
    batch = domain_windows(panel,sources,origins,14*panel.points_per_day,horizon)
    if batch.skipped_windows:
        raise ValueError('Assigned source training windows must be complete; no silent deletion.')
    return batch


def fit_bank(panel,sources,cutoff,horizon,representation,alpha):
    training = source_batch(panel,sources,cutoff,horizon)
    return RepresentationBank(representation,alpha).fit(training),training


def make_value_table(panel,domains,origins,bank,prior_table,task,horizon,cutoff,exclude_pseudo_target):
    rows=[]
    for domain in domains:
        batch = domain_windows(panel,[domain],origins,14*panel.points_per_day,horizon)
        if batch.skipped_windows:
            raise ValueError('Predeclared meta/calibration windows must be complete.')
        experts = bank.expert_predictions(batch.inputs)
        local = ThreeDayMeanForecaster().predict(batch.inputs)
        priors = reduce_priors(prior_table,bank.source_domains,domain if exclude_pseudo_target else None)
        permissible = [i for i,d in enumerate(bank.source_domains) if d != domain]
        context_indices = np.asarray([permissible[(cutoff+int((origin-panel.anchor)/pd.Timedelta(days=1)))%len(permissible)] for origin in batch.inputs.origins])
        for stage in (0,1):
            current = local if stage == 0 else .5*local+.5*experts[np.arange(len(batch)),context_indices]
            features = value_features(batch.inputs,bank,priors,experts,local,current,stage)
            local_loss,base_loss = ape(batch.targets,local),ape(batch.targets,current)
            expert_loss = np.stack([ape(batch.targets,experts[:,s]) for s in range(experts.shape[1])],axis=1)
            fixed_loss = np.stack([ape(batch.targets,.5*local+.5*experts[:,s]) for s in range(experts.shape[1])],axis=1)
            marginal_loss = np.stack([ape(batch.targets,.5*current+.5*experts[:,s]) for s in range(experts.shape[1])],axis=1)
            for i,origin in enumerate(batch.inputs.origins):
                day = int((origin-panel.anchor)/pd.Timedelta(days=1))
                for s,source in enumerate(bank.source_domains):
                    if source == domain or (stage == 1 and s == context_indices[i]):
                        continue
                    rows.append(dict(horizon=task,target_domain=domain,expert_domain=source,stage=stage,
                        context_domain='' if stage == 0 else bank.source_domains[context_indices[i]],
                        expert_fit_cutoff_day=cutoff,origin_day=day,label_end_day=day+ceil(horizon/panel.points_per_day),
                        prior_label_end_day=int(prior_table.label_end_day.max()),local_MAPE=local_loss[i],current_MAPE=base_loss[i],
                        expert_MAPE=expert_loss[i,s],fixed_MAPE=fixed_loss[i,s],marginal_MAPE=marginal_loss[i,s],
                        gain_single=local_loss[i]-expert_loss[i,s],gain_fixed=local_loss[i]-fixed_loss[i,s],
                        gain_marginal=base_loss[i]-marginal_loss[i,s],
                        **{'feature_'+key:float(value) for key,value in zip(VALUE_FEATURES,features[i,s])}))
    return pd.DataFrame(rows)


def train_value_tables(panel,sources,config,models,folder):
    tables,priors={},{}
    for task,horizon in horizons(config,panel.points_per_day).items():
        bank,_ = fit_bank(panel,sources,config['prior_cutoff_day'],horizon,'level_shape',models['ridge_alpha'])
        location = folder/task/'prior';location.mkdir(parents=True)
        bank.save(location/'model')
        past = domain_windows(panel,sources,config['prior_origins'],14*panel.points_per_day,horizon)
        values = bank.expert_predictions(past.inputs);local=ThreeDayMeanForecaster().predict(past.inputs)
        history=[]
        for i,(domain,origin) in enumerate(zip(past.inputs.domain_ids,past.inputs.origins)):
            for s,source in enumerate(bank.source_domains):
                if source != domain:
                    history.append(dict(expert_domain=source,target_domain=domain,horizon=task,origin=str(origin),
                        expert_MAPE=float(ape(past.targets[i],values[i,s])),local_MAPE=float(ape(past.targets[i],local[i])),
                        label_end_day=int((origin-panel.anchor)/pd.Timedelta(days=1))+ceil(horizon/panel.points_per_day)))
        prior=pd.DataFrame(history);priors[task]=prior
        prior.to_csv(folder/task/'past-performance.csv',index=False)
        parts=[]
        for cutoff in config['meta_cutoffs']:
            bank,_ = fit_bank(panel,sources,cutoff,horizon,'level_shape',models['ridge_alpha'])
            location=folder/task/f'cutoff{cutoff}';location.mkdir(parents=True)
            bank.save(location/'model')
            origins=list(range(cutoff,cutoff+config['meta_block_days']-ceil(horizon/panel.points_per_day)+1))
            parts.append(make_value_table(panel,sources,origins,bank,prior,task,horizon,cutoff,True))
        tables[task]=pd.concat(parts,ignore_index=True)
    table=pd.concat(tables.values(),ignore_index=True)
    table.to_parquet(folder/'gain-training.parquet',index=False)
    if (table[table.stage==0].gain_fixed-table[table.stage==0].gain_marginal).abs().max()>1e-10:
        raise AssertionError('At local-only stage, fixed 50% and marginal 50% gain must coincide.')
    return table,priors
