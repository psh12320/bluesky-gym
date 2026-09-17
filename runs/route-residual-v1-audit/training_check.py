import hashlib, json, subprocess, sys, zipfile
from pathlib import Path
from atc.residual import configuration
from atc.route_input import configuration as route_configuration

rows = []
for name in ('smoke-route-residual-v1-train-400', 'smoke-route-residual-v1-resume-200', 'smoke-route-residual-v1-sa-train-200'):
    p = Path('runs')/name
    config = json.loads((p/'config.json').read_text())
    with zipfile.ZipFile(p/'model.zip') as archive:
        data = json.loads(archive.read('data'))
    assert config['guard_static'] and config['guard_traffic']
    assert all(config.get(key) == value for key, value in {**configuration(), **route_configuration()}.items())
    assert data['navigation_warmup_std'] == .15
    rows.append(dict(run=name, timesteps=data['num_timesteps'], optimizer_updates=data['_n_updates'],
        observation_shape=config['observation_shape'], metadata_matches=True))
assert [row['timesteps'] for row in rows] == [400, 600, 200]
assert [row['optimizer_updates'] for row in rows] == [120, 200, 100]
assert [row['observation_shape'] for row in rows] == [[124], [124], [131]]
result = dict(training_checks=rows)
# A stale residual definition must be rejected before creating a run directory.
source = Path('runs/smoke-route-residual-v1-train-400')
stale = Path('runs/route-residual-v1-audit/stale-checkpoint')
stale.mkdir()
config = json.loads((source/'config.json').read_text())
config['route_residual_heading_limit_deg'] = 30
(stale/'config.json').write_text(json.dumps(config, indent=2))
(stale/'model.zip').write_bytes((source/'model.zip').read_bytes())
output = Path('runs/route-residual-v1-rejected-resume')
assert not output.exists()
command = [sys.executable, '-m', 'atc.train', '--env', 'ma', '--algorithm', 'sac',
    '--recipe', 'public_route_residual', '--steps', '100', '--seed', '2813',
    '--resume', str(stale/'model.zip'), '--resume-replay', str(source/'replay.pkl'), '--run-dir', str(output)]
completed = subprocess.run(command, text=True, capture_output=True)
assert completed.returncode != 0 and 'Route residual mapping changed; use fresh replay' in completed.stderr
assert not output.exists()
result['stale_replay_mapping_rejected_before_run_creation'] = True
Path('runs/route-residual-v1-audit/training-verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
