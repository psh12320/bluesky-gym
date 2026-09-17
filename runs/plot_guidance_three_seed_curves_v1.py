from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();parent=root/'runs/ppo-guidance-replication-v1'
out=parent/'three-seed-curves';out.mkdir(exist_ok=False)
paths=[root/'runs/ppo-guidance-pilot-v1/curve-dev20/curve.json',
       parent/'seed-49920/curve-three-points-dev20/curve.json',
       parent/'seed-49940/curve-three-points-dev20/curve.json']
curves=[json.loads(path.read_text(encoding='utf-8')) for path in paths]
cohort=json.loads((parent/'three-seed-results.json').read_text())
assert [curve['training_seed'] for curve in curves]==cohort['training_seeds']
for curve,row in zip(curves,cohort['per_seed']):
    points=curve['points']
    assert len(points)==3 and points[0]['live_transitions']==0
    assert 50000<=points[1]['live_transitions']<55120
    assert points[-1]['live_transitions']==row['training_audit']['live_transitions']
    assert curve['worlds']==20 and curve['classical_means']==curves[0]['classical_means']
    for metric,expected in row['final_means'].items():
        assert np.isclose(points[-1]['metrics'][metric]['mean'],expected,atol=1e-12,rtol=0)
panels=[('waypoint_reached','Arrival (%)',100),('clean_completion','Clean completion (%)',100),
        ('flight_time','Flight time (seconds)',1),('intrusion_time','Intrusion time (seconds)',1),
        ('time_in_restricted_area','Restricted-area time (seconds)',1),('time_outside_sector','Outside-sector time (seconds)',1)]
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
colors=['#1764a0','#df7a18','#218d70']
for ax,(metric,label,scale) in zip(axes.flat,panels):
    for curve,color in zip(curves,colors):
        x=[point['live_transitions']/1000 for point in curve['points']]
        y=[point['metrics'][metric]['mean']*scale for point in curve['points']]
        ax.plot(x,y,'o-',color=color,lw=1.8,ms=5,label=f"Seed {curve['training_seed']}")
    ax.axhline(curves[0]['classical_means'][metric]*scale,color='#333333',ls='--',lw=1.5,label='Fixed classical benchmark')
    ax.set_title(label,fontsize=11);ax.set_xlabel('Live aircraft transitions (thousands)')
    ax.set_xlim(-3,106);ax.grid(alpha=.2)
    if metric=='waypoint_reached':ax.set_ylim(94,101)
    elif metric=='clean_completion':ax.set_ylim(0,100)
    elif metric!='flight_time':ax.set_ylim(bottom=0)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
fig.suptitle('Guidance-only PPO: three training seeds\n20 paired development worlds; conflict filtering off',fontsize=15)
fig.savefig(out/'learning-curves.png',dpi=180)
plt.close(fig)
record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'training_seeds':cohort['training_seeds'],
        'curves':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        'metrics_and_actual_counts':[{'seed':c['training_seed'],'points':c['points']} for c in curves],
        'interpretation':'Each line is one independent training seed on the same twenty development worlds. No unseen evaluation. The classical benchmark has different support; own initial/final changes measure learning.',
        'uncertainty':'Lines show seed-specific means. World-bootstrap intervals remain in each input curve.json; no training-seed confidence interval is implied.'}
(out/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({'plot':str(out/'learning-curves.png'),'seeds':[{'seed':c['training_seed'],'live':[p['live_transitions'] for p in c['points']],
                  'clean':[p['metrics']['clean_completion']['mean'] for p in c['points']]} for c in curves]}))
