"""Final no-fit source, original-outcome, archive, and documentation checks."""
from pathlib import Path
from io import BytesIO
from datetime import datetime,timezone
import hashlib,json,re,subprocess,tarfile
ROOT=Path(__file__).resolve().parents[3]
out=ROOT/'docs/figures/gain-v2-20261025T175359Z_42a876'
run=ROOT/'results/runs/20261025T175359Z_sapf_gain_v2_42a876'
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
original=read(out/'preregistration.json');amend=read(out/'audit-amendment.json');changed=[]
assert read(run/'manifest.json')['status']=='complete'
assert read(out/'independent-verification.json')['passed']
for name,value in original['computational_source_sha256'].items():
    if digest(ROOT/name)!=value:
        assert name=='src/evaluation/gain_v2_audit.py'
        assert amend['allowed_changes'][name]==dict(before_sha256=value,after_sha256=digest(ROOT/name))
        changed.append(name)
assert changed==['src/evaluation/gain_v2_audit.py']
assert all(digest(run/name)==value for name,value in amend['preserved_outcome_sha256'].items())
frozen=read(run/'frozen-before-test.json')
assert all(digest(run/name)==value for name,value in frozen['model_sha256'].items())
assert read(run/'manifest.json')['experiment']==original['config']
current=read(out/'replication-preregistration.json')
with tarfile.open(out/'frozen-current-source.tar.gz') as archive:
    hashes={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers() if m.isfile()}
assert hashes==current['computational_source_sha256']
for name,value in hashes.items():assert digest(ROOT/name)==value
baseline_assets=0
payload=subprocess.check_output(['git','archive','f89dc7d','--','docs/figures'],cwd=ROOT)
with tarfile.open(fileobj=BytesIO(payload)) as archive:
    for member in archive.getmembers():
        if member.isfile():
            assert digest(ROOT/member.name)==hashlib.sha256(archive.extractfile(member).read()).hexdigest(),member.name
            baseline_assets+=1
old_states=0
for name in ['20261025T082346Z_sapf_5644ac','20261025T121710Z_sapf_4355ef','20261025T143438Z_sapf_transfer_771b85']:
    parent=ROOT/'results/runs'/name
    for path,value in read(parent/'frozen-before-test.json')['model_sha256'].items():
        assert digest(parent/path)==value;old_states+=1
docs=['README.md','docs/SAPF_GAIN_V2_RESULTS.md','docs/SAPF_GAIN_V2_PROTOCOL.md','docs/SAPF_GAIN_V2_AUDIT_AMENDMENT.md',
      'docs/figures/gain-v2-20261025T175359Z_42a876/README.md']
links=0
for name in docs:
    path=ROOT/name;text=path.read_text(encoding='utf-8')
    assert '\ufffd' not in text and '??' not in text and '<!-- LABEL_TABLE -->' not in text,name
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        if target.startswith(('http:','https:','#')):continue
        assert (path.parent/target.split('#')[0]).resolve().exists(),(name,target)
        links+=1
report=dict(passed=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),scientific_and_test_files_unchanged=84,
    original_computational_files=85,explicit_audit_only_amendment=changed,original_outcomes_unchanged=len(amend['preserved_outcome_sha256']),
    new_frozen_models_unchanged=len(frozen['model_sha256']),baseline_evidence_files_unchanged=baseline_assets,
    old_frozen_model_states_unchanged=old_states,manifest_config_matches_preregistered_config=True,
    current_frozen_source_archive_exact=True,documentation_links_checked=links,
    scope='Read-only hashes and links. No refit, model/threshold update, forecast rewrite, or Eagle observation access.')
path=ROOT/'results/analysis/gain_v2/delivery-verification.json'
if path.exists():raise FileExistsError('Use a new verification record.')
path.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
