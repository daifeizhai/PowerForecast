"""Gain-v2 panels with role-specific, original-clock holdout periods."""
from io import BytesIO
from pathlib import Path
import tarfile

import pandas as pd

from src.data.contracts import PublishedPowerAdapter, attach_panel_contract
from src.data.domains import DomainMetadata, validate_compatible
from src.data.google_panel import DomainPanel
from src.data.transfer30 import file_hash
from src.utils import load_config, resolve_path


def raw_power_frames(config, group, domains):
    """Open assigned records only; never Eagle or retired M100 observations."""
    result, provenance = {}, {}
    if group == 'm100':
        spec = load_config('configs/datasets/m100_nodes.yaml')
        for archive_spec in spec['archives']:
            path = resolve_path(spec['raw_data'])/archive_spec['filename']
            digest = file_hash(path)
            if digest != archive_spec['sha256']:
                raise ValueError('M100 original bytes changed.')
            provenance['M100/'+path.name] = dict(path=str(path), sha256=digest)
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    domain = f"m100_rack{archive_spec['rack_id']}_node{Path(member.name).stem}"
                    if domain in domains:
                        result[domain] = pd.read_parquet(BytesIO(archive.extractfile(member).read()), columns=['timestamp', 'total_power_avg'])
    elif group == 'ucc':
        for spec in load_config('configs/datasets/transfer30.yaml')['ucc_files']:
            path = resolve_path('data/raw/UCC2023')/spec['filename']
            digest = file_hash(path)
            if digest != spec['sha256']:
                raise ValueError('UCC original bytes changed.')
            provenance['UCC/'+path.name] = dict(path=str(path), sha256=digest)
            raw = pd.read_csv(path, usecols=['Series', 'Time', 'Power'])
            raw.Series = raw.Series.str.strip()
            for host, part in raw.groupby('Series', sort=True):
                domain = 'ucc_'+spec['cohort']+'_'+host.replace('-', '_')
                if domain in domains:
                    result[domain] = part
    else:
        raise ValueError('Only audited M100 and UCC measured-power groups are allowed.')
    if set(result) != set(domains):
        raise ValueError('An assigned native power domain is missing.')
    return result, provenance


def load_gain_v2_panel(config, group, frequency, raw=None, provenance=None):
    roles = config['domain_roles'][group]
    domains = sum(roles.values(), [])
    if set(domains) & set(config['retired_m100_v1_test']):
        raise ValueError('Retired original M100 tests cannot enter the new study.')
    if raw is None:
        raw, provenance = raw_power_frames(config, group, domains)
    interval = pd.Timedelta(frequency)
    ppd = int(pd.Timedelta(days=1)/interval)
    anchor = pd.Timestamp(config['protocol_anchor']).as_unit('ns')
    grid = pd.date_range(anchor, periods=config['days']*ppd, freq=frequency, name='timestamp')
    series, metadata, audit, coverage = {}, {}, {}, {}
    native = 15 if group == 'm100' else 10
    adapter = PublishedPowerAdapter('timestamp' if group == 'm100' else 'Time', 'total_power_avg' if group == 'm100' else 'Power', native)
    for domain in domains:
        test = domain in roles['test']
        start = config['source_clock_periods'][group]['test_start' if test else 'development_start']
        frame = adapter.adapt(raw[domain], start=start, days=config['days'], frequency=frequency)
        original = frame.index[0].isoformat()
        frame.index = grid
        series[domain] = frame
        coverage[domain] = (frame.quality_mask == 0).astype(int)*(int(interval.total_seconds()/60)//native)
        metadata[domain] = DomainMetadata(dataset_id=group+'_gain_v2', site_id='marconi100' if group == 'm100' else 'ucc_research_cloud',
            domain_id=domain, node_id=domain.split('_')[-1], target_field='node_power_kw', target_kind='active_power', target_scope='node',
            target_unit='kW', label_origin='measured', timezone='UTC' if group == 'm100' else 'unverified_source_local_clock',
            timezone_verified=group == 'm100', raw_frequency=f'{native}min', evaluation_frequency=frequency, sampling_semantics='interval_mean',
            interval_definition=f'Arithmetic mean of complete unique native published slots in {frequency} source-clock buckets.',
            coverage='Explicit missing/duplicate/nonfinite/negative quality bitmask; no interpolation.', source_version='Official local file SHA256 pinned.',
            usage_role='final_holdout' if test else 'development', feature_availability={'target':'history_only','trace_calendar':'future_known'},
            assumptions=('Offline role-specific collection dates; synthetic protocol coordinates are not physical UTC.', 'Available by bucket label plus interval; actual aggregation edges and extra reporting delay unverified.'))
        audit[domain] = dict(source_clock_start=original, source_clock_timezone_verified=group == 'm100',
            history_availability_upper_bound_minutes=interval.total_seconds()/60, missing_bins=int((frame.quality_mask != 0).sum()))
    validate_compatible(list(metadata.values()))
    return attach_panel_contract(DomainPanel(series, metadata, coverage, provenance, anchor, audit))


def audit_gain_v2_contracts(config):
    """Quality/availability only: no forecast, error, model fit, or Eagle read."""
    from src.data.google_panel import domain_windows, load_google_domains
    from src.training.gain_v2_learning import horizons, source_batch
    records=[];provenance={}
    for group in ['m100','ucc']:
        roles=config['domain_roles'][group]
        raw,files=raw_power_frames(config,group,sum(roles.values(),[]));provenance.update(files)
        for frequency in (['15min','30min'] if group=='m100' else ['30min']):
            panel=load_gain_v2_panel(config,group,frequency,raw,files)
            for task,horizon in horizons(config,panel.points_per_day).items():
                training=source_batch(panel,roles['source'],config['final_cutoff_day'],horizon)
                for role in ['calibration','validation','test']:
                    origins=config['test_origins'][task] if role=='test' else config[role+'_origins']
                    batch=domain_windows(panel,roles[role],origins,14*panel.points_per_day,horizon)
                    if batch.skipped_windows or len(batch)!=len(roles[role])*len(origins):
                        raise ValueError('An assigned required power window is incomplete.')
                    records.append(dict(dataset=group,frequency=frequency,horizon=task,role=role,domains=len(roles[role]),windows=len(batch),
                        history_points=14*panel.points_per_day,forecast_points=horizon,measured_scope='compute_node' if group=='m100' else 'physical_server',
                        target_unit='kW',source_timezone_verified=group=='m100',mask_available=True,available_at_available=True,passed=True))
    google=attach_panel_contract(load_google_domains(load_config('configs/datasets/google_cross_domain.yaml')))
    for domain,frame in google.series.items():
        records.append(dict(dataset='google',domain=domain,frequency='15min',target_unit=google.metadata[domain].target_unit,
            measured_scope=google.metadata[domain].target_scope,rows=len(frame),invalid_bins=int((frame.quality_mask!=0).sum()),passed=True))
    return dict(passed=True,source_files=provenance,contracts=records,google_domains=len(google.series),
        no_forecasts_or_error_metrics_computed=True,eagle_observations_opened=False,
        quality_bits={'missing':1,'duplicate':2,'nonfinite':4,'negative':8},
        availability_assumption='Source bucket label plus one published aggregate interval, no independently verified extra reporting delay.')
