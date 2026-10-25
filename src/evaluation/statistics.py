"""Paired domain/block uncertainty and auxiliary HAC Diebold-Mariano tests."""
from math import ceil

import numpy as np
import pandas as pd
from scipy.stats import norm


def holm(pvalues):
    values=np.asarray(pvalues,float)
    if np.any((values<0)|(values>1)|~np.isfinite(values)):
        raise ValueError('Holm needs finite p values in [0,1].')
    order=np.argsort(values,kind='stable');adjusted=np.empty(len(values))
    adjusted[order]=np.minimum(1,np.maximum.accumulate((len(values)-np.arange(len(values)))*values[order]))
    return adjusted


def dm_hac(differences, horizon_days, minimum=8):
    d=np.asarray(differences,float);n=len(d)
    if n<minimum:
        return dict(dm_statistic=None,p_value=None,hac_lags=None,reason='Too few forecast origins; auxiliary DM not reported.')
    lag=min(n-1,max(2,ceil(horizon_days)-1))
    centered=d-d.mean();variance=float(centered@centered/n)
    for k in range(1,lag+1):
        variance+=2*(1-k/(lag+1))*float(centered[k:]@centered[:-k]/n)
    if variance<=1e-20:
        return dict(dm_statistic=0. if np.allclose(d,0) else None,p_value=1. if np.allclose(d,0) else None,hac_lags=lag,reason='Zero HAC variance.')
    statistic=float(d.mean()/np.sqrt(variance/n))
    return dict(dm_statistic=statistic,p_value=float(2*norm.sf(abs(statistic))),hac_lags=lag,
                reason='Auxiliary asymptotic normal HAC-DM on per-origin mean APE; short series, not primary evidence.')


def paired_statistics(per_origin,config,*,include_nonoverlap=True,origin_stride_days=1):
    settings=config['statistics'];rng=np.random.default_rng(settings['seed']);reps=settings['replicates']
    rows=[]
    for (study,task),part in per_origin.groupby(['study','horizon'],sort=False):
        pivot=part.pivot(index=['target_domain','origin'],columns='model',values='MAPE')
        for baseline in ['three_day_mean','seasonal_naive','fixed_blend_50']:
            difference=(pivot.gain_v2_marginal-pivot[baseline]).unstack('origin').to_numpy()
            main_matrix=pivot.gain_v2_marginal.unstack('origin').to_numpy()
            baseline_matrix=pivot[baseline].unstack('origin').to_numpy()
            if not np.isfinite(difference).all():
                raise ValueError('Paired statistics need the same complete target/origin grid.')
            domains=sorted(part.target_domain.unique());n,t=difference.shape
            block=min(ceil(settings['block_origins']/origin_stride_days),t)
            domain_boot=np.empty(reps);hierarchical=np.empty(reps);domain_tail=np.empty(reps);hierarchical_tail=np.empty(reps)
            for b in range(reps):
                ids=rng.integers(n,size=n)
                domain_boot[b]=difference[ids].mean()
                domain_tail[b]=main_matrix[ids].mean(1).max()-baseline_matrix[ids].mean(1).max()
                starts=rng.integers(t,size=(n,ceil(t/block)))
                indices=((starts[:,:,None]+np.arange(block))%t).reshape(n,-1)[:,:t]
                hierarchical[b]=difference[ids[:,None],indices].mean()
                hierarchical_tail[b]=main_matrix[ids[:,None],indices].mean(1).max()-baseline_matrix[ids[:,None],indices].mean(1).max()
            estimate=float(difference.mean())
            for kind,boot,tail in [('domain_bootstrap',domain_boot,domain_tail),('domain_and_origin_block_bootstrap',hierarchical,hierarchical_tail)]:
                low,high=np.quantile(boot,[.025,.975])
                p=float((1+np.sum(np.abs(boot-estimate)>=abs(estimate)))/(reps+1))
                rows.append(dict(study=study,horizon=task,model='gain_v2_marginal',baseline=baseline,kind=kind,
                    target_domain='',domains=n,origins_per_domain=t,delta_MAPE_pp=estimate,median_domain_delta_pp=float(np.median(difference.mean(1))),
                    improved_domain_fraction=float((difference.mean(1)<-1e-10).mean()),ci_low=low,ci_high=high,p_value=p,block_origins=block,
                    worst_delta_MAPE_pp=float(main_matrix.mean(1).max()-baseline_matrix.mean(1).max()),
                    worst_ci_low=float(np.quantile(tail,.025)),worst_ci_high=float(np.quantile(tail,.975)),
                    interpretation='Negative delta favors Gain-v2. Node clusters within one site do not imply independent centers.'))
            for i,domain in enumerate(domains):
                values=dm_hac(difference[i],config['horizon_days'][task],settings['dm_min_origins'])
                rows.append(dict(study=study,horizon=task,model='gain_v2_marginal',baseline=baseline,kind='dm_hac',target_domain=domain,
                    domains=1,origins_per_domain=t,delta_MAPE_pp=float(difference[i].mean()),**values))
    frame=pd.DataFrame(rows)
    frame['evaluation_grid']='daily_rolling'
    if include_nonoverlap:
        part=per_origin[per_origin.horizon=='3d'].copy()
        if not part.empty:
            clock=pd.to_datetime(part.origin,utc=True)
            starts=clock.groupby(part.study).transform('min')
            part=part[((clock-starts)/pd.Timedelta(days=1))%3==0]
            extra=paired_statistics(part,config,include_nonoverlap=False,origin_stride_days=3)
            extra['evaluation_grid']='nonoverlap_3d'
            frame=pd.concat([frame,extra],ignore_index=True)
    for kind,indices in frame.groupby('kind').groups.items():
        # Missing/degenerate tests remain in the declared family as p=1.
        frame.loc[indices,'p_holm']=holm(frame.loc[indices,'p_value'].fillna(1).to_numpy())
        frame.loc[indices,'holm_family_size']=len(indices)
    return frame


def weighted_auc(y,score,weights):
    order=np.argsort(score,kind='stable');y=np.asarray(y)[order];q=np.asarray(score)[order];w=np.asarray(weights)[order]
    starts=np.r_[0,np.flatnonzero(q[1:]!=q[:-1])+1]
    pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(~y),starts)
    if pos.sum()==0 or neg.sum()==0:
        return np.nan
    return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))


def risk_statistics(pair_table,config):
    rows=[];rng=np.random.default_rng(config['statistics']['seed'])
    selected_models=['gain_v2_single','gain_v2_fixed','gain_v2_marginal','gain_v2_rank']
    for (study,task,model),part in pair_table[(pair_table.stage==0)&pair_table.model.isin(selected_models)].groupby(['study','horizon','model'],sort=False):
        part=part.reset_index(drop=True);domains=sorted(part.target_domain.unique());codes=pd.Categorical(part.target_domain,categories=domains).codes
        y=part.actual_gain_pp.to_numpy() < -1e-10;p=part.negative_risk.to_numpy();g=part.predicted_gain_pp.to_numpy();positive=part.actual_gain_pp.to_numpy()>1e-10
        prior=float(part.source_marginal_negative_prior.iloc[0]);n=len(domains)
        order=np.argsort(p,kind='stable');ordered_y=y[order];ordered_p=p[order]
        starts=np.r_[0,np.flatnonzero(ordered_p[1:]!=ordered_p[:-1])+1]
        selected=part.selected.to_numpy(bool);gain_error=np.abs(g-part.actual_gain_pp.to_numpy())
        def values(weights):
            divide=lambda a,b:float(a/b) if b else np.nan
            total=weights.sum();pred=g>0
            pos=np.add.reduceat(weights[order]*ordered_y,starts);neg=np.add.reduceat(weights[order]*(~ordered_y),starts)
            auc=divide(np.sum(pos*(np.cumsum(neg)-.5*neg)),pos.sum()*neg.sum())
            return dict(AUROC=auc,Brier=float(np.sum(weights*(p-y)**2)/total),
                Brier_delta_source_constant=float(np.sum(weights*((p-y)**2-(prior-y)**2))/total),
                positive_precision=divide(np.sum(weights*pred*positive),np.sum(weights*pred)),
                positive_recall=divide(np.sum(weights*pred*positive),np.sum(weights*positive)),
                negative_precision=divide(np.sum(weights*(p>=.5)*y),np.sum(weights*(p>=.5))),
                negative_recall=divide(np.sum(weights*(p>=.5)*y),np.sum(weights*y)),
                selected_positive_precision=divide(np.sum(weights*selected*positive),np.sum(weights*selected)),
                selected_positive_recall=divide(np.sum(weights*selected*positive),np.sum(weights*positive)),
                gain_MAE_pp=float(np.sum(weights*gain_error)/total))
        point=values(np.ones(len(part)));samples={key:[] for key in point}
        for _ in range(config['statistics']['replicates']):
            counts=np.bincount(rng.integers(n,size=n),minlength=n)
            for key,value in values(counts[codes]).items():samples[key].append(value)
        for key,estimate in point.items():
            finite=np.asarray(samples[key]);finite=finite[np.isfinite(finite)]
            low,high=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
            rows.append(dict(study=study,horizon=task,model=model,metric=key,estimate=estimate,ci_low=low,ci_high=high,
                target_clusters=n,expert_pairs=len(part),valid_bootstrap_replicates=len(finite),
                interpretation='Target-domain cluster bootstrap; common stage-0 marginal fusion labels, fixed 50% action.'))
    return pd.DataFrame(rows)
