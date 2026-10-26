"""Write an append-only pre-test registration from completed source development."""
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import json,platform,sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import PROJECT_ROOT,write_json


def register():
    location=PROJECT_ROOT/'results/analysis/robust_direction';path=location/'preregistration.json'
    if path.exists():raise FileExistsError('Do not overwrite a pre-test registration.')
    development=PROJECT_ROOT/'results/runs/20261026T080606Z_sapf_robust_dev_8b8652'
    state=json.loads((development/'manifest.json').read_text(encoding='utf-8'))
    if state['status']!='development_complete' or not state['no_new_target_forecast_errors_accessed']:raise ValueError('Source-only admission is incomplete.')
    if any((p/'studies').exists() for p in (PROJECT_ROOT/'results/runs').glob('*_sapf_robust_*') if '_dev_' not in p.name):raise ValueError('A robust target evaluation already exists; do not register after test access.')
    digest=lambda p:sha256(p.read_bytes()).hexdigest()
    code={p.relative_to(PROJECT_ROOT).as_posix():digest(p) for directory in ['src','scripts','configs','tests'] for p in (PROJECT_ROOT/directory).rglob('*') if p.suffix in {'.py','.yaml'}}
    evidence={name:digest(location/name) for name in ['new-period-availability.csv','prior-score-inventory.csv','previous-model-preservation.json','robust-tests.log']}
    record=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),target_scores_accessed=False,computational_source_sha256=code,
        development_manifest_sha256=digest(development/'manifest.json'),development_artifact_sha256={p.relative_to(development).as_posix():digest(p) for p in development.rglob('*') if p.is_file()},
        evidence_sha256=evidence,protocol_sha256=digest(PROJECT_ROOT/'docs/SAPF_ROBUST_PROTOCOL.md'),python=platform.python_version(),
        decision='Source-only weights and conditional dynamic admission fixed before any new target forecast score. H1 family 8; H2 family 16 including unexecuted cells.')
    write_json(path,record)
    print(path)


if __name__=='__main__':register()
