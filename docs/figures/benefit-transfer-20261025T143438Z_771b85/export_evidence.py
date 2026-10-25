"""Package frozen outcomes and descriptive analyses; never fit or select a model."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import hashlib
import json
import shutil
import tarfile
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.ml.benefit import FEATURE_NAMES

run = ROOT/'results/runs/20261025T143438Z_sapf_transfer_771b85'
analysis = ROOT/'results/analysis/benefit_transfer_v1'
out = ROOT/'docs/figures/benefit-transfer-20261025T143438Z_771b85'
out.mkdir(parents=True, exist_ok=False)
manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
assert manifest['status'] == 'complete'
for name in ['independent-verification.json', 'native-verification.json']:
    assert json.loads((analysis/name).read_text(encoding='utf-8'))['passed']

log_sources = {}

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == '.log':
        raw = src.read_bytes()
        log_sources[src.name] = hashlib.sha256(raw).hexdigest()
        text = raw.decode('utf-16') if raw[:2] in [b'\xff\xfe', b'\xfe\xff'] else raw.decode('utf-8')
        dst.write_text(text.replace('\r\r\n', '\n').replace('\r\n', '\n'), encoding='utf-8')
    else:
        shutil.copyfile(src, dst)

for name in ['manifest.json', 'predeclared-plan.json', 'frozen-before-test.json', 'audit.json']:
    copy(run/name, out/name)
for name in ['preregistration.json', 'quality-eligibility.csv', 'pytest-final.log', 'real-run.log', 'results-latest.log',
             'independent-verification.json', 'native-verification.json', 'verify_independently.py', 'verify_native_power.py']:
    copy(analysis/name, out/name)
copy(Path(__file__), out/'export_evidence.py')
copy(ROOT/'docs/BENEFIT_TRANSFER_PROTOCOL.md', out/'protocol-at-preregistration.md')
prereg = json.loads((analysis/'preregistration.json').read_text(encoding='utf-8'))
assert hashlib.sha256((ROOT/'docs/BENEFIT_TRANSFER_PROTOCOL.md').read_bytes()).hexdigest() == prereg['protocol_document_sha256']
with tarfile.open(out/'frozen-computational-source.tar.gz', 'w:gz') as archive:
    for path, digest in prereg['computational_source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
        archive.add(ROOT/path, arcname=path)

summaries, risks, metrics, cohorts, gain_details, candidates, importances, training = [], [], [], [], [], [], [], []
for group in ['m100', 'ucc']:
    for path in (run/'groups'/group/'benefit').glob('*.csv'):
        copy(path, out/'groups'/group/path.name)
    learner = joblib.load(run/'groups'/group/'joint_predictor/model.joblib')
    training.append(dict(source_group=group, rows=learner.training_rows, gain_mean_pp=learner.training_gain_mean,
                         negative_prevalence=learner.negative_prevalence, sources=len(learner.source_domains)))
    for feature, reg, cls in zip(FEATURE_NAMES, learner.regressor.feature_importances_, learner.classifier.feature_importances_):
        importances.append(dict(source_group=group, feature=feature, gain_importance=reg, risk_importance=cls))
    for task in ['24h', '3d']:
        records = json.loads((run/'groups'/group/task/'training-records.json').read_text(encoding='utf-8'))
        copy(run/'groups'/group/task/'training-records.json', out/'groups'/group/task/'training-records.json')
        record = next(r for r in records if r['model'] == 'sapf_gain')
        for candidate in record['selected']['candidates']:
            candidates.append(dict(source_group=group, horizon=task, **candidate,
                                   selected=candidate == record['selected']['selected']))

for study in manifest['studies']:
    name = study['name']; child = run/'studies'/name
    for filename in ['manifest.json', 'predeclared-plan.json', 'frozen-before-test.json', 'test-start.json', 'audit.json',
                     'metrics.csv', 'domain-summary.csv', 'negative-transfer.csv', 'risk-summary.csv',
                     'gain-diagnostics.csv', 'saved-model-audit.json']:
        copy(child/filename, out/'studies'/name/filename)
    summary = pd.read_csv(child/'domain-summary.csv').assign(study=name); summaries.append(summary)
    metric = pd.read_csv(child/'metrics.csv').assign(study=name); metrics.append(metric)
    risk = pd.read_csv(child/'risk-summary.csv').assign(study=name)
    prior = next(t['negative_prevalence'] for t in training if t['source_group'] == study['source_group'])
    risk['Brier_constant_0_5'] = .25
    risk['Brier_constant_source_prevalence'] = risk.negative_prevalence*(1-prior)**2+(1-risk.negative_prevalence)*prior**2
    risks.append(risk)
    curve = metric[(metric.split == 'test') & (metric.evaluation == 'curve')].copy()
    curve['cohort'] = (curve.target_domain.str.extract(r'(ucc_d[12]|m100_rack[01])')[0])
    base = curve[curve.model == 'three_day_mean'].set_index(['target_domain', 'horizon']).MAPE
    for (task, model, cohort), rows in curve.groupby(['horizon', 'model', 'cohort']):
        losses = rows.MAPE.to_numpy() > base.reindex(pd.MultiIndex.from_frame(rows[['target_domain', 'horizon']])).to_numpy()+1e-10
        cohorts.append(dict(study=name, horizon=task, model=model, cohort=cohort, domains=len(rows),
                            mean_MAPE=rows.MAPE.mean(), worst_MAPE=rows.MAPE.max(), mean_WAPE=rows.WAPE.mean(),
                            mean_RMSE_kw=rows.RMSE.mean(), mean_MAE_kw=rows.MAE.mean(), negative_vs_local=int(losses.sum())))
    diag = pd.read_csv(child/'gain-diagnostics.csv')
    diag = diag[(diag.model == 'sapf_gain') & (diag.split == 'test')]
    for task, rows in diag.groupby('horizon'):
        gain_details.append(dict(study=name, horizon=task, mean_predicted_gain_pp=rows.predicted_gain_pp.mean(),
            mean_actual_gain_pp=rows.actual_gain_pp.mean(), signed_prediction_bias_pp=(rows.predicted_gain_pp-rows.actual_gain_pp).mean(),
            min_predicted_gain_pp=rows.predicted_gain_pp.min(), max_predicted_gain_pp=rows.predicted_gain_pp.max(),
            actually_beneficial_pairs=int((rows.actual_gain_pp > 1e-10).sum()), predicted_positive_pairs=int((rows.predicted_gain_pp > 0).sum()),
            selected_pairs=int((rows.weight > 0).sum())))

summary = pd.concat(summaries, ignore_index=True); risk = pd.concat(risks, ignore_index=True)
metric = pd.concat(metrics, ignore_index=True)
summary.to_csv(out/'all-domain-summary.csv', index=False)
risk.to_csv(out/'all-risk-summary.csv', index=False)
metric.to_csv(out/'all-metrics.csv', index=False)
pd.DataFrame(cohorts).to_csv(out/'cohort-summary.csv', index=False)
pd.DataFrame(gain_details).to_csv(out/'gain-calibration-diagnostics.csv', index=False)
pd.DataFrame(candidates).to_csv(out/'source-validation-candidates.csv', index=False)
pd.DataFrame(importances).to_csv(out/'feature-importances.csv', index=False)
pd.DataFrame(training).to_csv(out/'gain-training-summary.csv', index=False)
aggregate = metric.groupby(['study', 'horizon', 'split', 'evaluation', 'model']).agg(
    domains=('target_domain', 'nunique'), mean_MAPE=('MAPE', 'mean'), worst_MAPE=('MAPE', 'max'),
    mean_WAPE=('WAPE', 'mean'), mean_RMSE_kw=('RMSE', 'mean'), mean_MAE_kw=('MAE', 'mean'),
    total_points=('n', 'sum'), mape_excluded=('mape_excluded', 'sum')).reset_index()
aggregate.to_csv(out/'all-metric-summary.csv', index=False)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
studies = ['m100_gain', 'm100_to_ucc', 'ucc_to_m100']
titles = ['M100 new-node holdout', 'M100 -> UCC', 'UCC -> M100']
methods = ['three_day_mean', 'fixed_blend', 'sapf', 'sapf_gain', 'sapf_gain_no_veto']
labels = ['Local history', 'Fixed 50% blend', 'Prior SAPF', 'Learned gain + veto', 'Gain, forced transfer']
colors = ['#6b7280', '#3b82f6', '#eab308', '#059669', '#be123c']
for field, filename, label in [('mean_MAPE', 'mean-mape.png', 'Equal-node mean MAPE (%)'), ('worst_MAPE', 'worst-mape.png', 'Worst-node MAPE (%)')]:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for j, (name, title) in enumerate(zip(studies, titles)):
        for i, task in enumerate(['24h', '3d']):
            ax = axes[i,j]
            rows = summary[(summary.study == name)&(summary.horizon == task)].set_index('model')
            values = rows.loc[methods, field].to_numpy()
            ax.barh(np.arange(len(methods)), values, color=colors, height=.66)
            ax.invert_yaxis(); ax.set_yticks(range(len(methods)), labels if j == 0 else ['']*len(labels))
            ax.set_xlim(0, max(values)*1.2)
            for k, v in enumerate(values): ax.text(v+max(values)*.015, k, f'{v:.3f}', va='center', fontsize=9)
            ax.set_title(title+' | '+task); ax.set_xlabel(label); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    fig.suptitle('Frozen 30-minute measured-power protocol | 14-day history | Offline comparison', fontsize=14)
    fig.savefig(out/filename, dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout='constrained')
x = np.arange(6); tags = [title+'\n'+task for title in titles for task in ['24h', '3d']]
ordered = risk.set_index(['study', 'horizon']).loc[[(s,h) for s in studies for h in ['24h', '3d']]]
axes[0].bar(x, ordered.risk_AUROC, color='#2563eb'); axes[0].axhline(.5, color='#6b7280', linestyle='--')
axes[0].set_ylim(0, 1); axes[0].set_ylabel('Negative-transfer risk AUROC'); axes[0].set_title('Risk ranking on all held-out expert pairs')
axes[1].bar(x, 100*ordered.zero_transfer_fraction, color='#059669'); axes[1].set_ylim(0, 110)
axes[1].set_ylabel('Origins with exact local fallback (%)'); axes[1].set_title('A veto can be conservative despite useful ranking')
for ax in axes:
    ax.set_xticks(x, tags, rotation=25, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
for i, row in enumerate(ordered.itertuples()):
    axes[0].text(i, row.risk_AUROC+.02, f'{row.risk_AUROC:.3f}', ha='center', fontsize=9)
    axes[1].text(i, 100*row.zero_transfer_fraction+2, f'{100*row.zero_transfer_fraction:.1f}', ha='center', fontsize=9)
fig.savefig(out/'risk-and-fallback.png', dpi=170); plt.close(fig)

evidence = dict(run=str(run.relative_to(ROOT)), passed=True, real_suite_runs=1, studies=studies,
    preregistration_utc=prereg['created_at_utc'], freeze_utc=json.loads((run/'frozen-before-test.json').read_text(encoding='utf-8'))['frozen_at_utc'],
    completed_utc=manifest['completed_at_utc'], model_states_frozen=118, metric_rows=len(metric),
    independent_verification=json.loads((analysis/'independent-verification.json').read_text(encoding='utf-8')) | {'metric_records': 'see independent-verification.json', 'mixture_records': 'see independent-verification.json'},
    native_verification={k:v for k,v in json.loads((analysis/'native-verification.json').read_text(encoding='utf-8')).items() if k != 'records'},
    scientific_source_hashes_match_preregistration=True, raw_models_predictions_left_in_original_run=True, original_log_byte_sha256=log_sources,
    note='Post-test descriptive tables and figures only; no fit, threshold update, policy selection, or Eagle access.',
    files_sha256={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})
(out/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
print(out)
print(pd.DataFrame(training).to_string(index=False))
print(risk[['study','horizon','risk_AUROC','risk_Brier','Brier_constant_source_prevalence']].to_string(index=False))
print(pd.DataFrame(cohorts).query("study == 'm100_to_ucc' and model in ['three_day_mean', 'fixed_blend', 'sapf_gain']").to_string(index=False))
