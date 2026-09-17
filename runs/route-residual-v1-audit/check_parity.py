import json
from pathlib import Path
from atc.compare import load_evaluation
from atc.metrics import METRICS

pairs = dict(ma=('runs/goal-route-input-v1-joint-ma-20', 'runs/route-residual-v1-zero-ma-20'),
             sa=('runs/goal-route-input-v1-joint-sa-20', 'runs/route-residual-v1-zero-sa-20'))

def check(kind):
    reference, candidate = pairs[kind]
    old_meta, old = load_evaluation(Path(reference))
    new_meta, new = load_evaluation(Path(candidate))
    assert all(old_meta[key] == new_meta[key] for key in ('env', 'seed', 'episodes'))
    a = {(row['episode'], row['agent']): row for row in old}
    b = {(row['episode'], row['agent']): row for row in new}
    assert a.keys() == b.keys()
    differences = {metric: max(abs(a[key][metric] - b[key][metric]) for key in a) for metric in METRICS}
    assert not any(differences.values()), differences
    stats = new_meta['residual_statistics']
    assert stats['actions'] > 0
    assert not any(value for key, value in stats.items() if key != 'actions')
    result = dict(env=kind, episodes=20, agent_episodes=len(a), exact_zero_adjustment_parity=True,
                  max_absolute_metric_differences=differences, residual_statistics=stats,
                  model_sha256=new_meta['model_sha256'], reference=reference, candidate=candidate)
    target = Path('runs/route-residual-v1-audit') / (kind + '-zero-parity.json')
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    import sys
    check(sys.argv[1])
