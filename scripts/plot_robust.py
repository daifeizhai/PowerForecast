"""Render saved robust evidence only; never select or retrain forecasting methods."""
from pathlib import Path
import argparse,sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import resolve_path


def plot(folder):
    output=folder/'figures';output.mkdir(exist_ok=True)
    summary=pd.read_csv(folder/'domain-summary.csv');stats=pd.read_csv(folder/'statistics.csv')
    studies=summary.study.drop_duplicates().to_list();tasks=['4h','24h','3d']
    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for i,study in enumerate(studies):
        for j,task in enumerate(tasks):
            ax=axes[i,j];p=summary[(summary.study==study)&(summary.horizon==task)&(summary.evaluation==('curve_nonoverlap' if task=='3d' else 'curve'))]
            for prefix,label in [('fixed_','Unbounded'),('bounded_fixed_','Bounded')]:
                q=p[p.model.str.startswith(prefix)].copy();q['alpha']=q.model.str.replace(prefix,'',regex=False).astype(float);q=q.sort_values('alpha')
                ax.plot(q.alpha,q.mean_MAPE,'o-',label=label)
            for method,color in [('seasonal_naive','gray'),('three_day_mean','black')]:
                ax.axhline(p[p.model==method].mean_MAPE.iloc[0],ls='--',color=color,label=method)
            if 'dynamic' in p.model.to_list():ax.axhline(p[p.model=='dynamic'].mean_MAPE.iloc[0],color='tab:red',label='Dynamic')
            ax.set_title(study+' / '+task);ax.set_xlabel('Transfer fraction (source-selected for primary claim)');ax.set_ylabel('Equal-node MAPE (%)');ax.grid(alpha=.2)
            if i==0 and j==0:ax.legend(fontsize=8)
    fig.suptitle('Held-out weight sensitivity; 3d uses nonoverlapping windows',fontsize=15);fig.savefig(output/'weight-sensitivity.png',dpi=160);plt.close(fig)
    main=stats[(stats.main_comparison)&stats.executed].copy();labels=[]
    fig,ax=plt.subplots(figsize=(12,max(5,len(main)*.44)),constrained_layout=True)
    for i,r in enumerate(main.itertuples()):
        ax.errorbar(r.improvement_MAPE_pp,i,xerr=[[r.improvement_MAPE_pp-r.ci_low],[r.ci_high-r.improvement_MAPE_pp]],fmt='o',color='tab:blue' if r.family=='H1' else 'tab:orange')
        labels.append(f'{r.family} {r.study} {r.horizon} vs {r.baseline}')
    ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel('MAPE improvement (percentage points), paired node + synchronized-block 95% CI')
    ax.set_title('Positive favors frozen fusion; statistical significance also requires Holm correction');fig.savefig(output/'paired-intervals.png',dpi=160);plt.close(fig)
    for study in studies:
        child=folder/'studies'/study
        fig,axes=plt.subplots(1,3,figsize=(16,6),constrained_layout=True)
        for ax,task in zip(axes,tasks):
            d=pd.read_parquet(child/task/'test-expert-diagnostics.parquet');clock=pd.to_datetime(d.origin,utc=True)
            if task=='3d':d=d[np.asarray((clock-clock.min())/pd.Timedelta(days=1))%3==0]
            table=d.pivot_table(index='target_domain',columns='source_domain',values='MAPE',aggfunc='mean')
            im=ax.imshow(table.to_numpy(),aspect='auto',cmap='magma',vmin=0,vmax=np.quantile(table.to_numpy(),.95))
            ax.set_title(task+' single-source raw MAPE');ax.set_xlabel('Source expert');ax.set_ylabel('Target node/server')
            ax.set_xticks(range(len(table.columns)),[x.rsplit('_',1)[-1] for x in table.columns],rotation=90,fontsize=7)
            if len(table)<15:ax.set_yticks(range(len(table)),[x.rsplit('_',1)[-1] for x in table.index],fontsize=7)
            fig.colorbar(im,ax=ax,label='MAPE % (color capped at p95)')
        fig.suptitle(study+' : raw experts before shared nonnegative clipping');fig.savefig(output/(study+'-expert-matrix.png'),dpi=160);plt.close(fig)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run');args=parser.parse_args();print(plot(resolve_path(args.run)))
