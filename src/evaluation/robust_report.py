"""Predeclared paired inference and complete descriptive transfer reporting."""
import json
from math import ceil
import numpy as np,pandas as pd
from src.evaluation.statistics import holm
from src.utils import write_json


def primary_grid(table,task):
    if task!='3d':return table
    clock=pd.to_datetime(table.origin,utc=True);start=clock.min()
    return table[np.asarray((clock-start)/pd.Timedelta(days=1))%3==0]


def paired_interval(part,model,baseline,task,config,rng):
    pivot=primary_grid(part,task).pivot(index=['target_domain','origin'],columns='model',values='MAPE')
    a=pivot[model].unstack('origin').to_numpy();b=pivot[baseline].unstack('origin').to_numpy()
    counts=primary_grid(part,task).pivot(index=['target_domain','origin'],columns='model',values='mape_n')
    weights=counts[baseline].unstack('origin').to_numpy()
    if not np.array_equal(weights,counts[model].unstack('origin').to_numpy()):raise ValueError('Different MAPE denominators across paired methods.')
    if not np.isfinite(a+b).all():raise ValueError('A comparison changed its paired node/origin grid.')
    n,t=a.shape;domain_a=(a*weights).sum(1)/weights.sum(1);domain_b=(b*weights).sum(1)/weights.sum(1)
    block=min(t,18 if task=='4h' else 3 if task=='24h' else 1)
    reps=config['statistics']['replicates'];samples=np.empty(reps);tails=np.empty(reps)
    for i in range(reps):
        ids=rng.integers(n,size=n)
        # Shared time draws preserve contemporaneous shocks across different nodes.
        starts=rng.integers(t,size=ceil(t/block));ts=((starts[:,None]+np.arange(block))%t).ravel()[:t]
        aa=a[ids[:,None],ts];bb=b[ids[:,None],ts];ww=weights[ids[:,None],ts]
        aa=(aa*ww).sum(1)/ww.sum(1);bb=(bb*ww).sum(1)/ww.sum(1)
        samples[i]=(bb-aa).mean();tails[i]=aa.max()-bb.max()
    domain_delta=domain_b-domain_a;estimate=float(domain_delta.mean());low,high=np.quantile(samples,[.025,.975])
    p=(1+np.sum(np.abs(samples-estimate)>=abs(estimate)))/(reps+1)
    return dict(executed=True,domains=n,origins_per_domain=t,block_origins=block,improvement_MAPE_pp=estimate,
        relative_MAPE_gain=1-float(domain_a.mean()/domain_b.mean()),ci_low=float(low),ci_high=float(high),p_value=float(p),
        improved_domain_fraction=float((domain_delta>1e-10).mean()),worst_relative_increase=float(domain_a.max()/domain_b.max()-1),
        worst_delta_MAPE_pp=float(domain_a.max()-domain_b.max()),worst_ci_low=float(np.quantile(tails,.025)),worst_ci_high=float(np.quantile(tails,.975)),
        jackknife_min_improvement_pp=float(min(np.delete(domain_delta,i).mean() for i in range(n))) if n>1 else estimate)


def produce_analysis(folder,metrics,origins,routes,config):
    summary=[];risk=[]
    for (study,task,evaluation),part in metrics.groupby(['study','horizon','evaluation'],sort=False):
        for name,g in part.groupby('model',sort=False):
            refs={k:part[part.model==k].set_index('target_domain').MAPE.reindex(g.target_domain).to_numpy() for k in ['seasonal_naive','three_day_mean']}
            row=dict(study=study,horizon=task,evaluation=evaluation,model=name,domains=len(g),mean_MAPE=g.MAPE.mean(),worst_MAPE=g.MAPE.max(),
                mean_WAPE=g.WAPE.mean(),mean_MAE=g.MAE.mean(),mean_RMSE=g.RMSE.mean(),points=int(g.n.sum()),mape_excluded=int(g.mape_excluded.sum()))
            for k,ref in refs.items():
                row['negative_pct_vs_'+k]=float(100*(g.MAPE.to_numpy()>ref+1e-10).mean()) if evaluation!='daily_mean' or task!='4h' else np.nan
            summary.append(row)
    pd.DataFrame(summary).to_csv(folder/'domain-summary.csv',index=False)
    for (study,task,name),g in routes.groupby(['study','horizon','model'],sort=False):
        g=primary_grid(g,task);op=g.positive_opportunity;success=g.successful
        risk.append(dict(study=study,horizon=task,model=name,windows=len(g),zero_transfer_pct=100*g.zero_transfer.mean() if g.alpha.notna().any() else np.nan,
            positive_opportunity_pct=100*op.mean(),positive_opportunity_utilization_pct=100*(success&op).sum()/op.sum() if op.any() else np.nan,
            missed_opportunity_pct=100*(op&~success).sum()/op.sum() if op.any() else np.nan,
            zero_missed_opportunity_pct=100*(op&g.zero_transfer).sum()/op.sum() if op.any() else np.nan,
            successful_pct=100*success.mean(),negative_window_pct=100*g.negative.mean()))
    pd.DataFrame(risk).to_csv(folder/'routing-summary.csv',index=False)
    rows=[];rng=np.random.default_rng(config['statistics']['seed'])
    for study in [s['name'] for s in config['studies']]:
        for task in ['4h','24h','3d']:
            part=origins[(origins.study==study)&(origins.horizon==task)]
            for family,model,refs in [('H1','selected_fixed',['seasonal_naive','three_day_mean']),
                ('H2','dynamic',['selected_bounded','selected_fixed','seasonal_naive','three_day_mean'])]:
                for ref in refs:
                    row=dict(study=study,horizon=task,family=family,model=model,baseline=ref,main_comparison=study!='m100_internal_15' and task!='4h',
                        evaluation='curve_nonoverlap' if task=='3d' else 'curve')
                    row.update(paired_interval(part,model,ref,task,config,rng) if model in part.model.unique() else dict(executed=False,p_value=1.,reason='Not admitted by the prespecified source-side gate.'))
                    rows.append(row)
    stats=pd.DataFrame(rows);stats['p_holm']=np.nan
    for family in ['H1','H2']:
        mask=(stats.family==family)&stats.main_comparison;stats.loc[mask,'p_holm']=holm(stats.loc[mask,'p_value'].to_numpy())
        stats.loc[mask,'holm_family_size']=mask.sum()
    r=config['decision'];stats['stable_positive']=stats.executed&(stats.relative_MAPE_gain>=r['relative_MAPE_gain_min'])&(stats.ci_low>0)&(stats.p_holm<=.05)&(stats.worst_relative_increase<=r['worst_relative_increase_max'])&(stats.improved_domain_fraction>=r['improved_domain_fraction_min'])&(stats.jackknife_min_improvement_pp>=0)
    numerical={(s['name'],h):v['passed'] for s in config['studies'] for h,v in json.loads((folder/'studies'/s['name']/'numerical-audit.json').read_text(encoding='utf-8')).items()}
    stats['expert_numerical_admission']=[numerical[(s,h)] for s,h in zip(stats.study,stats.horizon)]
    stats['stable_positive'] &= stats.expert_numerical_admission
    stats.to_csv(folder/'statistics.csv',index=False)
    # Hardware/cohort summaries are descriptive; no post-test model or threshold selection.
    cohorts=metrics.copy();cohorts['hardware_group']=cohorts.target_domain.map(lambda d:'_'.join(d.split('_')[:2]))
    cohorts.groupby(['study','horizon','evaluation','model','hardware_group'],sort=False).agg(mean_MAPE=('MAPE','mean'),worst_MAPE=('MAPE','max'),nodes=('MAPE','size')).reset_index().to_csv(folder/'hardware-summary.csv',index=False)
    routes.sort_values('final_MAPE',ascending=False).groupby(['study','horizon','model'],sort=False).head(10).to_csv(folder/'extreme-cases.csv',index=False)
    main=stats[stats.main_comparison];h1=main[main.family=='H1'];h2=main[main.family=='H2']
    write_json(folder/'decision.json',dict(H1_passed_comparisons=int(h1.stable_positive.sum()),H1_family_size=len(h1),
        H2_passed_comparisons=int(h2.stable_positive.sum()),H2_executed_comparisons=int(h2.executed.sum()),H2_family_size=len(h2),
        automatic_success=bool(h1.stable_positive.all() and h2.stable_positive.all()),
        no_post_test_parameter_changes=True,independence_scope='New unscored dates at known centers; offline cross-dataset benchmark; not new-center prospective evidence.'))


def print_robust_results(folder):
    summary=pd.read_csv(folder/'domain-summary.csv');risk=pd.read_csv(folder/'routing-summary.csv')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for study in manifest['studies']:
        name=study['name'];print(f'\n配置：{name} | 14天历史 | {study["frequency"]} | 节点/服务器实测kW')
        for task in ['4h','24h','3d']:
            p=summary[(summary.study==name)&(summary.horizon==task)&(summary.evaluation==('curve_nonoverlap' if task=='3d' else 'curve'))]
            print(f'{task} | 方法 | 平均MAPE% | 最差MAPE% | WAPE% | MAE kW | RMSE kW | 负迁移节点%（前三天均值） | 点数')
            for row in p.itertuples():
                print(f'{task} | {row.model} | {row.mean_MAPE:.4f} | {row.worst_MAPE:.4f} | {row.mean_WAPE:.4f} | {row.mean_MAE:.6f} | {row.mean_RMSE:.6f} | {row.negative_pct_vs_three_day_mean:.1f} | {row.points}')
            if 'dynamic' not in p.model.to_list():print(f'{task} | dynamic | N/A：未通过源侧准入，未执行')
        print('4h日均：N/A；3d表为非重叠主检验，每日滚动与日均详见metrics.csv。')
        print(risk[(risk.study==name)&risk.model.isin(['selected_fixed','selected_bounded','dynamic'])][['horizon','model','zero_transfer_pct','positive_opportunity_utilization_pct','missed_opportunity_pct']].to_string(index=False))
    print(f'\n结果保存：{folder}',flush=True)
