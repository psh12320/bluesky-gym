"""Verify the expanded evaluation prefix and describe its new scenario suffix."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atc.compare import load_evaluation, paired_comparison
from atc.metrics import METRICS, summarize

old_prefix = ROOT / 'runs/sac-ma-public-600k-seed1400/validation-final-individual-20'
new_prefix = ROOT / 'runs/sac-ma-public-600k-seed1400/validation-individual-200'
_, previous = load_evaluation(old_prefix)
metadata, current = load_evaluation(new_prefix)
neutral_metadata, neutral = load_evaluation(ROOT / 'runs/neutral-ma-200')
expected = {(r['episode'], r['agent']): r for r in previous}
observed = {(r['episode'], r['agent']): r for r in current if r['episode'] < 20}
assert expected.keys() == observed.keys()
assert all(expected[k][m] == observed[k][m] for k in expected for m in METRICS)
assert metadata['model_sha256'] == 'b4c7a671dbd0fae8146522fa8278f19f4119c859caf2e1b07fe0bb56eae45093'
assert metadata['inference_mode'] == 'per_aircraft'
assert metadata['seed'] == neutral_metadata['seed'] == 2026
assert metadata['episodes'] == neutral_metadata['episodes'] == 200

def suffix(rows):
    return [dict(r, episode=r['episode'] - 20) for r in rows if r['episode'] >= 20]

def tails(rows):
    return {metric: dict(zip(('p90', 'p95', 'p99', 'maximum'), map(float, np.quantile([r[metric] for r in rows], [.9, .95, .99, 1]))))
            for metric in ('intrusion_time', 'time_in_restricted_area', 'time_outside_sector', 'flight_time')}

comparison = paired_comparison(suffix(neutral), suffix(current), 180)
result = dict(exact_20_scenario_prefix_match=True, checked_aircraft=200, checked_metrics=list(METRICS),
    csv_sha256=hashlib.sha256(new_prefix.with_suffix('.csv').read_bytes()).hexdigest(),
    full_summary=summarize(current, 200, 10), neutral_tail_metrics=tails(neutral), learned_tail_metrics=tails(current),
    failed_arrivals=[r for r in current if not r['waypoint_reached']],
    suffix=dict(original_episode_range=[20, 199], episodes=180,
                interval='95% paired scenario bootstrap, pointwise; excludes training-seed uncertainty',
                metrics=comparison))
(Path(__file__).parent / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(dict(prefix_exact=True, failed_arrivals=len(result['failed_arrivals']),
    learned_tail_metrics=result['learned_tail_metrics'], suffix_metrics={k: comparison[k] for k in (
    'waypoint_reached', 'flight_time', 'intrusion_time', 'time_in_restricted_area', 'time_outside_sector', 'clean_completion_rate')}), indent=2))
