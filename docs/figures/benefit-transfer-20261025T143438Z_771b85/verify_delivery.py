"""Read-only final source/evidence/archive/link verification."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import re
import tarfile

ROOT = Path(__file__).resolve().parents[3]
package = ROOT/'docs/figures/benefit-transfer-20261025T143438Z_771b85'
prereg = json.loads((package/'preregistration.json').read_text(encoding='utf-8'))
hash_bytes = lambda value: hashlib.sha256(value).hexdigest()
for name, digest in prereg['computational_source_sha256'].items():
    assert hash_bytes((ROOT/name).read_bytes()) == digest, name
with tarfile.open(package/'frozen-computational-source.tar.gz') as archive:
    contents = {m.name:hash_bytes(archive.extractfile(m).read()) for m in archive.getmembers() if m.isfile()}
assert contents == prereg['computational_source_sha256']
snapshot = ROOT/'archive/pre_gain_transfer_20261025'
old = json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
assert hash_bytes((snapshot/'source.tar.gz').read_bytes()) == old['archive_sha256']
with tarfile.open(snapshot/'source.tar.gz') as archive:
    contents = {m.name:hash_bytes(archive.extractfile(m).read()) for m in archive.getmembers() if m.isfile()}
assert contents == old['files']
sealed = {k:v for k,v in old['files'].items() if k.startswith('docs/figures/')}
for name, digest in sealed.items():
    assert hash_bytes((ROOT/name).read_bytes()) == digest, name
old_states = 0
for run_id in ['20261025T082346Z_sapf_5644ac', '20261025T121710Z_sapf_4355ef']:
    run = ROOT/'results/runs'/run_id
    freeze = json.loads((run/'frozen-before-test.json').read_text(encoding='utf-8'))
    for name, digest in freeze['model_sha256'].items():
        assert hash_bytes((run/name).read_bytes()) == digest, name
        old_states += 1
docs = ['README.md', 'docs/BENEFIT_TRANSFER_PROTOCOL.md', 'docs/BENEFIT_TRANSFER_RESULTS.md',
        'docs/figures/benefit-transfer-20261025T143438Z_771b85/README.md', 'archive/pre_gain_transfer_20261025/README.md']
links = 0
for name in docs:
    path = ROOT/name
    text = path.read_text(encoding='utf-8')
    assert '\ufffd' not in text and '??' not in text and '<!-- FULL_METHOD_TABLES -->' not in text, name
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
        if target.startswith(('https:', 'http:', '#')):
            continue
        target = target.split('#')[0]
        assert (path.parent/target).resolve().exists(), (name, target)
        links += 1
report = dict(passed=True, computational_files_matched=len(prereg['computational_source_sha256']),
              frozen_source_archive_exact=True, pre_extension_snapshot_exact=True, snapshot_files=len(old['files']),
              old_evidence_files_unchanged=len(sealed), old_frozen_model_states_unchanged=old_states,
              documentation_links_checked=links, source_code_changed_after_preregistration=False,
              scope='Hashes and document links only; no fit, forecast, or Eagle observations accessed.')
out = ROOT/'results/analysis/benefit_transfer_v1/delivery-verification.json'
if out.exists():
    raise FileExistsError('Do not overwrite the final verification record.')
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(report)
