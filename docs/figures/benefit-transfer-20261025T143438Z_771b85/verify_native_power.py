"""Read-only native-file aggregation check, independent of the adapter implementation."""
from pathlib import Path
from io import BytesIO
import argparse
import hashlib
import json
import tarfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = ROOT / args.output
if out.exists():
    raise FileExistsError('Use a new output path.')
run = ROOT / 'results/runs/20261025T143438Z_sapf_transfer_771b85'
manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
assert manifest['status'] == 'complete'
roles = manifest['experiment']['domain_roles']
retired = set(roles['retired_m100_previous_test'])
caches = {}
for study in manifest['studies']:
    for path in (run/'studies'/study['name']/'adapted-data').glob('*.parquet'):
        caches.setdefault(path.stem, path)
assert not (set(caches) & retired)
records, hashes = [], {}

def check(raw, time, power, domain, native):
    saved = pd.read_parquet(caches[domain])
    source_clock = pd.DatetimeIndex(pd.to_datetime(saved.source_clock_timestamp)).as_unit('ns')
    clock = pd.DatetimeIndex(pd.to_datetime(raw[time])).as_unit('ns')
    keep = (clock >= source_clock[0]) & (clock < source_clock[-1]+pd.Timedelta(minutes=30))
    clock = clock[keep]
    values = pd.to_numeric(raw.loc[keep, power]).to_numpy(dtype=float)
    assert not clock.duplicated().any()
    assert np.isfinite(values).all() and (values >= 0).all()
    assert np.array_equal(clock.sort_values().asi8, pd.date_range(source_clock[0], periods=len(source_clock)*30//native, freq=f'{native}min').asi8)
    # Explicit source-clock group sums/counts; no call to the production adapter.
    table = pd.DataFrame({'power_w': values, 'bucket': clock.floor('30min')})
    groups = table.groupby('bucket').power_w.agg(['sum', 'count']).reindex(source_clock)
    assert (groups['count'] == 30//native).all()
    independent = groups['sum'].to_numpy()/groups['count'].to_numpy()/1000.
    difference = float(np.max(np.abs(independent-saved.node_power_kw.to_numpy())))
    assert difference < 1e-12, (domain, difference)
    records.append(dict(domain=domain, native_minutes=native, native_rows=len(values), labels=len(saved), maximum_absolute_kw_difference=difference, passed=True))

for rack in (0, 1):
    path = ROOT/'data/raw/M100'/f'{rack}.tar'
    hashes['M100/'+path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            domain = f'm100_rack{rack}_node{Path(member.name).stem}'
            if domain not in caches:
                continue
            raw = pd.read_parquet(BytesIO(archive.extractfile(member).read()), columns=['timestamp', 'total_power_avg'])
            check(raw, 'timestamp', 'total_power_avg', domain, 15)
    print('native M100 rack', rack, 'checked', flush=True)
for cohort in ('d1', 'd2'):
    path = ROOT/'data/raw/UCC2023'/f'dataset{cohort[-1]}_combined.csv'
    hashes['UCC/'+path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = pd.read_csv(path, usecols=['Series', 'Time', 'Power'])
    raw.Series = raw.Series.str.strip()
    for host, part in raw.groupby('Series'):
        domain = f'ucc_{cohort}_'+host.replace('-', '_')
        if domain in caches:
            check(part, 'Time', 'Power', domain, 10)
    print('native UCC', cohort, 'checked', flush=True)
assert len(records) == len(caches) == 132
for key, value in hashes.items():
    assert value == manifest['data_provenance'][key]['sha256']
report = dict(passed=True, domains=len(records), aggregate_labels=sum(r['labels'] for r in records), native_rows=sum(r['native_rows'] for r in records),
              maximum_absolute_kw_difference=max(r['maximum_absolute_kw_difference'] for r in records), original_sha256=hashes,
              retired_m100_records_not_opened=True, records=records,
              scope='Independent group sums/counts from assigned-domain raw power only; no forecast, fit, interpolation, or implicit timezone assignment.')
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print({k:v for k,v in report.items() if k != 'records'}, flush=True)
