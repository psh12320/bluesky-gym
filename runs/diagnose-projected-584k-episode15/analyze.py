"""Measure the longest sampled conflict in the reproduced development scenario."""
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atc.compare import load_evaluation
from atc.metrics import METRICS
from core.tools import kwikqdrdist

HERE = Path(__file__).parent
summary = json.loads((HERE / 'summary.json').read_text())
scenario = json.loads((HERE / 'scenario.json').read_text())
_, reference = load_evaluation(ROOT / 'runs/sac-ma-projected-finetune-seed2200/validation-final-individual-20')
expected = {r['agent']: r for r in reference if r['episode'] == 15}
actual = {r['agent']: r for r in summary['final_metrics']}
assert actual.keys() == expected.keys()
assert all(actual[a][m] == expected[a][m] for a in actual for m in METRICS)
goals = {spec['ac_id']: spec['goal'] for spec in scenario['agents']}
with (HERE / 'trajectory.csv').open(newline='') as stream:
    raw = list(csv.DictReader(stream))
rows = {}
for row in raw:
    agent = row['agent']
    values = {k: (v == 'True' if v in ('True', 'False') else float(v)) for k, v in row.items() if k != 'agent'}
    bearing, _ = kwikqdrdist(values['lat'], values['lon'], *goals[agent])
    values['heading_deg'] = float((bearing + values['heading_error_deg']) % 360)
    rows.setdefault(agent, {})[values['time']] = values

pairs = []
for a, b in itertools.combinations(sorted(rows), 2):
    times = sorted(rows[a].keys() & rows[b].keys())
    samples, intervals, interval = [], [], []
    for time in times:
        x, y = rows[a][time], rows[b][time]
        _, distance = kwikqdrdist(x['lat'], x['lon'], y['lat'], y['lon'])
        heading_difference = abs((x['heading_deg'] - y['heading_deg'] + 180) % 360 - 180)
        sample = dict(time=time, distance_nm=float(distance), heading_difference_deg=heading_difference,
            speed_a_mps=x['tas_mps'], speed_b_mps=y['tas_mps'],
            action_a=x['heading_action'], action_b=y['heading_action'],
            filter_a=x['static_projection_intervened'], filter_b=y['static_projection_intervened'],
            goal_distance_a_km=x['goal_distance_km'], goal_distance_b_km=y['goal_distance_km'])
        samples.append(sample)
        if distance < 5:
            if interval and time - interval[-1]['time'] != 10:
                intervals.append(interval)
                interval = []
            interval.append(sample)
        elif interval:
            intervals.append(interval)
            interval = []
    if interval:
        intervals.append(interval)
    longest = max(intervals, key=len, default=[])
    if longest:
        pairs.append(dict(agents=[a, b], sampled_intrusion_count=sum(len(i) for i in intervals),
            longest_sampled_interval_start=longest[0]['time'], longest_sampled_interval_end=longest[-1]['time'],
            longest_sampled_interval_count=len(longest),
            minimum_separation_nm=min(v['distance_nm'] for v in longest),
            median_separation_nm=float(np.median([v['distance_nm'] for v in longest])),
            median_heading_difference_deg=float(np.median([v['heading_difference_deg'] for v in longest])),
            fraction_heading_difference_under_15_deg=float(np.mean([v['heading_difference_deg'] < 15 for v in longest])),
            median_absolute_tas_difference_mps=float(np.median([abs(v['speed_a_mps'] - v['speed_b_mps']) for v in longest])),
            filter_interventions_in_interval=[sum(v[k] for v in longest) for k in ('filter_a', 'filter_b')],
            samples=samples))
pairs.sort(key=lambda p: p['longest_sampled_interval_count'], reverse=True)
worst = pairs[0]
with (HERE / 'longest_pair.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(worst['samples'][0]))
    writer.writeheader()
    writer.writerows(worst['samples'])
report = dict(episode=15, seed=2026, exact_reference_metrics_match=True,
    sampling_seconds=10, note='Sample intervals are diagnostic. Original one-second scorer totals remain authoritative.',
    pairs=[{k: v for k, v in p.items() if k != 'samples'} for p in pairs])
(HERE / 'conflict_analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
samples = worst['samples']
t = [v['time'] for v in samples]
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
axes[0].plot(t, [v['distance_nm'] for v in samples], color='#2563eb')
axes[0].axhline(5, color='#dc2626', linestyle='--', label='5 NM separation')
axes[0].set(ylabel='Separation (NM)', title='Development scenario 15: ' + ' / '.join(worst['agents']))
axes[0].legend()
axes[1].plot(t, [v['heading_difference_deg'] for v in samples], color='#7c3aed')
axes[1].set(ylabel='Heading difference (degrees)')
for key, label, color in zip(('speed_a_mps', 'speed_b_mps'), worst['agents'], ('#2563eb', '#ea580c')):
    axes[2].plot(t, [v[key] for v in samples], label=label, color=color)
axes[2].set(xlabel='Simulation time (seconds)', ylabel='True airspeed (m/s)')
axes[2].legend()
for ax in axes:
    ax.axvspan(worst['longest_sampled_interval_start'], worst['longest_sampled_interval_end'], alpha=.08, color='#dc2626')
    ax.grid(alpha=.15)
fig.savefig(HERE / 'longest_pair.png', dpi=150)
plt.close(fig)
print(json.dumps({k: v for k, v in worst.items() if k != 'samples'}, indent=2))
