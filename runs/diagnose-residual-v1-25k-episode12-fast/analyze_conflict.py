"""Inspect the matched episode-12 conflict without changing the controller."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from core.tools import kwikqdrdist

root = Path(__file__).resolve().parent
summary = json.loads((root / 'summary.json').read_text())
assert summary['reference_comparison']['matches_reference']
assert summary['scenario_rng_unchanged_during_rollout']
scenario = json.loads((root / 'scenario.json').read_text())
with (root / 'trajectory.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
identities = ('KL003', 'KL005')
groups = {agent: {float(row['time']): row for row in rows if row['agent'] == agent} for agent in identities}
times = sorted(set(groups[identities[0]]) & set(groups[identities[1]]))
separations = []
for time in times:
    a, b = (groups[agent][time] for agent in identities)
    _, separation = kwikqdrdist(float(a['lat']), float(a['lon']), float(b['lat']), float(b['lon']))
    separations.append(float(separation))
minimum = int(np.argmin(separations))
window = []
for time in times:
    if 490 <= time <= 600:
        for agent in identities:
            row = groups[agent][time]
            window.append(dict(time=time, agent=agent,
                nearest_nm=float(row['nearest_nm']),
                chosen_turn_deg=float(row['commanded_heading_turn_deg']),
                chosen_speed_action=float(row['commanded_speed_action']),
                nominal_traffic_conflict=row['nominal_traffic_conflict'] == 'True',
                no_jointly_feasible_candidate=row['no_jointly_feasible_candidate'] == 'True',
                predicted_minimum_separation_km=float(row['predicted_minimum_separation_km'])))
report = dict(episode=12, seed=2026, matches_reference=True, sampling_seconds=10,
    minimum_sampled_pair_separation_nm=separations[minimum], time_of_sampled_minimum=times[minimum],
    official_intrusion_seconds_per_aircraft=22,
    interpretation='The conflict was predicted before the violation. Candidate commands repeatedly failed joint feasibility; later replanning also changed a temporarily feasible forecast. This trace does not prove physical unavoidability or a general prediction guarantee.',
    comparison_scope='Current selected scenario only; all nine final aircraft metrics match the saved full-prefix evaluation exactly.',
    selected_commands=window)
(root / 'conflict-analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

center = np.array(scenario['center'])
def xy(points):
    p = np.array(points, dtype=float) - center
    return np.column_stack((p[:, 1] * 60 * 1.852 * np.cos(np.deg2rad(center[0])), p[:, 0] * 60 * 1.852))
fig, (map_ax, sep_ax) = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
map_ax.add_patch(Polygon(xy(scenario['sector']), closed=True, fill=False, edgecolor='#64748b'))
for obstacle in scenario['obstacles']:
    map_ax.add_patch(Polygon(xy(obstacle['vertices']), closed=True, facecolor='#fee2e2', edgecolor='#ef4444'))
colors = ('#15803d', '#7c3aed')
for agent, color in zip(identities, colors):
    selected = [row for time, row in groups[agent].items() if 400 <= time <= 650]
    points = xy([(float(row['lat']), float(row['lon'])) for row in selected])
    map_ax.plot(*points.T, color=color, label=agent, linewidth=2)
    for time in (490, 540, 590, 640):
        row = groups[agent][time]
        point = xy([(float(row['lat']), float(row['lon']))])[0]
        map_ax.scatter(*point, color=color, s=20)
        map_ax.annotate(f'{time}s', point, xytext=(5, 3), textcoords='offset points', color=color, fontsize=8)
    selected = [groups[agent][t] for t in times if 450 <= t <= 650]
    command_times = [float(r['time']) for r in selected]
    sep_ax.plot(command_times, [float(r['predicted_minimum_separation_km']) / 1.852 for r in selected],
                color=color, linestyle='--', alpha=.8, label=f'{agent} chosen forecast minimum')
map_ax.set(xlim=(-55, 40), ylim=(-155, -95), xlabel='East (km)', ylabel='North (km)', title='Opposing flights around a restricted area')
map_ax.set_aspect('equal')
map_ax.legend(loc='upper right')
sep_ax.plot(times, separations, color='#0f172a', linewidth=2, label='Actual pair distance, sampled every 10 s')
sep_ax.axhline(5, color='#dc2626', linewidth=1.2, label='Official 5 NM separation')
sep_ax.axhline(10.26 / 1.852, color='#64748b', linestyle=':', label='Filter margin')
sep_ax.set(xlim=(450, 650), ylim=(4, 20), xlabel='Scenario time (s)', ylabel='Separation (NM)', title='The filter anticipated the conflict')
sep_ax.legend(loc='upper right', fontsize=7)
for ax in (map_ax, sep_ax):
    ax.grid(alpha=.15)
fig.savefig(root / 'conflict-analysis.png', dpi=150)
plt.close(fig)
print(json.dumps({key: value for key, value in report.items() if key != 'selected_commands'}, indent=2))
