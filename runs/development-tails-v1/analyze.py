"""Describe aircraft-level tails on completed, matched development evaluations."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atc.compare import load_evaluation
from atc.metrics import SAFETY, summarize

OUTPUT = Path(__file__).resolve().parent
SOURCES = [
    ('ma_classical', 'ma', 'Classical', 'runs/goal-route-choice-v1-fast-joint-ma-200'),
    ('ma_learned', 'ma', 'Learned, original reward', 'runs/sac-ma-fast-reference-v1-25k-seed2900/validation-200'),
    ('sa_classical', 'sa', 'Classical', 'runs/goal-route-choice-v1-fast-joint-sa-200'),
    ('sa_learned', 'sa', 'Learned, original reward', 'runs/fast-reference-v1-ma25k-sa/validation-200'),
    ('sa_reach250', 'sa', 'Learned, arrival reward 250', 'runs/fast-reference-reach250-v1-ma25k-sa/validation-200'),
]
TIMES = ('flight_time', 'intrusion_time', 'time_in_restricted_area', 'time_outside_sector')
results, raw, table = {}, {}, []
for identity, track, label, prefix in SOURCES:
    path = ROOT/prefix
    metadata, records = load_evaluation(path)
    assert metadata['env'] == track and metadata['seed'] == 2026 and metadata['episodes'] == 200
    expected = summarize(records, 200, 10 if track == 'ma' else 1)
    assert all(metadata[key] == value for key, value in expected.items())
    raw[identity] = records
    result = {'track':track, 'label':label, 'reference':prefix, 'aircraft_records':len(records),
              'csv_sha256':hashlib.sha256(path.with_suffix('.csv').read_bytes()).hexdigest(),
              'metadata_sha256':hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest(),
              'model_sha256':metadata['model_sha256'], 'times':{}}
    for metric in TIMES:
        values = np.array([r[metric] for r in records])
        stats = {'mean':float(values.mean()), 'median':float(np.median(values)),
                 'p95':float(np.quantile(values,.95,method='linear')),
                 'p99':float(np.quantile(values,.99,method='linear')),
                 'maximum':float(values.max()), 'positive_aircraft':int(np.count_nonzero(values)),
                 'positive_rate':float(np.mean(values>0)),
                 'mean_if_positive':float(values[values>0].mean()) if np.any(values>0) else None}
        result['times'][metric] = stats
        result.setdefault('largest_exposures',{})[metric] = sorted(records,key=lambda r:r[metric],reverse=True)[:5]
        table.append({'controller':identity,'metric':metric,**stats})
    arrivals = [r for r in records if r['waypoint_reached']]
    result['arrival_rate'] = len(arrivals)/len(records)
    result['arrival_only_mean_flight'] = float(np.mean([r['flight_time'] for r in arrivals]))
    result['missed_arrivals'] = [r for r in records if not r['waypoint_reached']]
    result['clean_completion_rate'] = expected['clean_completion_rate']
    result['slots'] = {}
    for agent in sorted({r['agent'] for r in records}):
        group = [r for r in records if r['agent']==agent]
        result['slots'][agent] = {'count':len(group), 'arrival_rate':float(np.mean([r['waypoint_reached'] for r in group])),
            'mean_flight':float(np.mean([r['flight_time'] for r in group])),
            'mean_intrusion':float(np.mean([r['intrusion_time'] for r in group])),
            'clean_completion_rate':float(np.mean([bool(r['waypoint_reached'] and all(r[k]==0 for k in SAFETY)) for r in group]))}
    results[identity] = result

manifest = {'scope':'Completed seed-2026 development evaluations; no held-out outcomes read',
            'quantile_method':'NumPy linear interpolation over all aircraft records, including timeouts',
            'caveats':['Descriptive quantiles; no uncertainty intervals or training-seed generalization claim',
                       'MA aircraft are dependent within each scenario; per-aircraft safety counts double count pair conflicts',
                       'KL identifiers are simulation slots, not demographic or airline groups; slot differences do not establish unfairness or causality',
                       'Flight duration is not route excess or delay; start-goal distances and geometry differ'],
            'evaluations':results}
(OUTPUT/'tails.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with (OUTPUT/'quantiles.csv').open('w',newline='',encoding='utf-8') as out:
    writer = csv.DictWriter(out,fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors = {'Classical':'#64748b','Learned, original reward':'#2563eb','Learned, arrival reward 250':'#c2410c'}
for row, track in enumerate(('ma','sa')):
    for identity, kind, label, prefix in SOURCES:
        if kind != track:
            continue
        records = raw[identity]
        flight = np.sort([r['flight_time'] for r in records])
        axes[row,0].step(np.r_[0,flight],np.r_[0,100*np.arange(1,len(flight)+1)/len(flight)],where='post',color=colors[label],label=label,lw=1.8)
        intrusion = np.array([r['intrusion_time'] for r in records])
        x = np.unique(np.r_[0,intrusion])
        y = np.array([100*np.mean(intrusion>v) for v in x])
        axes[row,1].step(x,y,where='post',color=colors[label],label=label,lw=1.8)
    count = 2000 if track=='ma' else 200
    axes[row,0].set(title=f'{track.upper()}: flight duration ({count:,} aircraft)',xlabel='Flight time (seconds)',ylabel='Aircraft at or below duration (%)',xlim=(0,3000),ylim=(0,101))
    axes[row,1].set(title=f'{track.upper()}: conflict exposure ({count:,} aircraft)',xlabel='Intrusion time (seconds)',ylabel='Aircraft exceeding exposure (%)',xlim=(0,None),ylim=(0,None))
    for axis in axes[row]:
        axis.grid(alpha=.18)
    axes[row,0].legend(loc='lower right',fontsize=8,frameon=False)
fig.suptitle('High arrival averages still contain long flights and concentrated safety failures\n200 shared development scenarios per track; ten-second control',fontsize=14)
fig.savefig(OUTPUT/'development-tails.png',dpi=160)
fig.savefig(OUTPUT/'development-tails.svg')
plt.close(fig)
for identity,result in results.items():
    print(identity,json.dumps({metric:result['times'][metric] for metric in ('flight_time','intrusion_time')}))
