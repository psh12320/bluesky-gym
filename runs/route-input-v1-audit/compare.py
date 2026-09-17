import json
from pathlib import Path
from atc.compare import load_evaluation, paired_comparison

cases = [
    ('runs/sac-ma-public-600k-seed1400/validation-joint-v1-individual-20', 'runs/public-route-input-v1-joint-391k/validation-20', 'compare-ma-joint-route-input-v1'),
    ('runs/sac-ma-public-600k-seed1400/validation-final-individual-20', 'runs/public-route-input-v1-joint-391k/validation-20', 'compare-ma-public-route-input-v1'),
    ('runs/goal-route-input-v1-joint-ma-20', 'runs/public-route-input-v1-joint-391k/validation-20', 'compare-ma-classical-learned-route-input-v1'),
    ('runs/goal-route-input-v1-joint-sa-20', 'runs/public-route-input-v1-joint-sa-transfer-391k/validation-20', 'compare-sa-classical-learned-route-input-v1'),
    ('runs/sac-ma-public-600k-seed1400/validation-final-individual-20', 'runs/sac-ma-public-learner1m-seed2500/validation-final-individual-20', 'compare-ma-public-larger-347k'),
    ('runs/neutral-ma-20', 'runs/sac-ma-public-learner1m-seed2500/validation-final-individual-20', 'compare-ma-neutral-larger-347k'),
]
# Read all required artifacts before writing any comparison.
loaded = []
for a, b, name in cases:
    am, ar = load_evaluation(Path(a))
    bm, br = load_evaluation(Path(b))
    assert all(am[k] == bm[k] for k in ('env', 'seed', 'episodes'))
    assert not Path('runs', name + '.json').exists()
    loaded.append((a, b, name, am, ar, br))
for a, b, name, am, ar, br in loaded:
    result = dict(reference=a, candidate=b, env=am['env'], seed=am['seed'], episodes=am['episodes'],
        interval='95% paired scenario bootstrap, pointwise; does not include training-seed uncertainty',
        metrics=paired_comparison(ar, br, am['episodes']))
    Path('runs', name + '.json').write_text(json.dumps(result, indent=2))
    print(name, json.dumps({k: v for k, v in result['metrics'].items() if k in
        ('waypoint_reached', 'flight_time', 'intrusion_time', 'time_in_restricted_area', 'clean_completion_rate')}))
