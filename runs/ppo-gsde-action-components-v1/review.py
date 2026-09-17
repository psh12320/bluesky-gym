"""Inspect completed action interventions and export their paired effects."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PARENT=Path(__file__).resolve().parent
ROOT=PARENT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=read(PARENT/'protocol.json')
    result=read(PARENT/'results.json')
    if result['diagnostic_protocol_sha256']!=sha(PARENT/'protocol.json'):
        raise ValueError('Protocol changed after the completed diagnostic')
    if result['promoted_to_candidate'] or not result['all_arms_use_identical_scenarios']:
        raise ValueError('Unexpected diagnostic status')
    for name,digest in plan['protected_inputs'].items():
        if sha(Path(name))!=digest:raise ValueError('Protected input changed: '+name)
    source=Path(plan['source'])
    manifest=read(source/'cluster-manifest.json')
    for name,digest in manifest['files'].items():
        if sha(source/name)!=digest:raise ValueError('Frozen source changed: '+name)
    sys.path.insert(0,str(source))
    from atc.metrics import METRICS,SAFETY,summarize
    arms=['zero_both','zero_speed','zero_heading','full_policy']
    labels=['Zero learned actions','Heading only','Speed only','Heading + speed']
    world_values={}; provenance={}; raw_rows={}
    for arm in arms:
        directory=PARENT/arm;summary=read(directory/'summary.json')
        if summary!=result['results'][arm]:raise ValueError('Arm summary differs')
        if sha(directory/'aircraft.csv')!=summary['csv_sha256']:raise ValueError('CSV changed')
        with (directory/'aircraft.csv').open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            row['episode']=int(row['episode'])
            for key in METRICS:row[key]=float(row[key])
        keyed={(r['episode'],r['agent']):r for r in rows}
        if len(rows)!=200 or len(keyed)!=200:raise ValueError('Incomplete or duplicate rows')
        recomputed=summarize(rows,20,10)
        for key,value in recomputed.items():
            if summary[key]!=value:raise ValueError('Summary mismatch: '+key)
        clean=lambda row:float(bool(row['waypoint_reached']) and all(row[key]==0 for key in SAFETY))
        worlds={i:[r for r in rows if r['episode']==i] for i in range(20)}
        for i,records in worlds.items():
            if len(records)!=10 or any(r['scenario_sha256']!=summary['scenarios'][i] for r in records):
                raise ValueError('Scenario mismatch')
        values={metric:np.array([np.mean([r[metric] for r in worlds[i]]) for i in range(20)]) for metric in METRICS}
        values['clean_completion']=np.array([np.mean([clean(r) for r in worlds[i]]) for i in range(20)])
        values['all_aircraft_clean_completion']=np.array([all(clean(r) for r in worlds[i]) for i in range(20)],dtype=float)
        for metric,array in values.items():
            if float(array.mean())!=result['means'][arm][metric]:raise ValueError('Mean mismatch')
        world_values[arm]=values;raw_rows[arm]=keyed
        provenance[arm]={'csv_sha256':sha(directory/'aircraft.csv'),'summary_sha256':sha(directory/'summary.json')}
    canonical=raw_rows['full_policy']
    for rows in raw_rows.values():
        if rows.keys()!=canonical.keys():raise ValueError('Aircraft sets differ')
        for key in rows:
            if rows[key]['scenario_sha256']!=canonical[key]['scenario_sha256']:raise ValueError('Scenarios differ')
    controls=0
    for arm,stage in [('full_policy','final'),('zero_both','initial')]:
        ref=Path(result['replay_checks'][arm]['reference'])
        with (ref/'aircraft.csv').open(newline='',encoding='utf-8') as stream:
            original={(int(r['episode']),r['agent']):r for r in csv.DictReader(stream)}
        if original.keys()!=raw_rows[arm].keys():raise ValueError('Replay row set differs')
        for key,row in raw_rows[arm].items():
            if row['scenario_sha256']!=original[key]['scenario_sha256']:raise ValueError('Replay scenario differs')
            for metric in METRICS:
                if row[metric]!=float(original[key][metric]):raise ValueError('Replay metric differs')
                controls+=1
    indices=np.random.default_rng(701).integers(0,20,size=(10000,20))
    pairs={'full_minus_zero':('full_policy','zero_both'),'heading_only_minus_zero':('zero_speed','zero_both'),
           'speed_only_minus_zero':('zero_heading','zero_both'),'heading_only_minus_full':('zero_speed','full_policy'),
           'speed_only_minus_full':('zero_heading','full_policy')}
    effect_rows=[]
    for comparison,(first,second) in pairs.items():
        for metric in world_values[first]:
            delta=world_values[first][metric]-world_values[second][metric]
            band=np.quantile(delta[indices].mean(axis=1),[.025,.975])
            recorded=result['paired_effects'][comparison][metric]
            if float(delta.mean())!=recorded['mean_difference'] or band.tolist()!=recorded['paired_world_bootstrap_95_interval']:
                raise ValueError('Paired effect differs')
            effect_rows.append({'comparison':comparison,'metric':metric,'mean_difference':float(delta.mean()),
                                'world_bootstrap_lower':float(band[0]),'world_bootstrap_upper':float(band[1])})
    destination=PARENT/'review';destination.mkdir(exist_ok=False)
    with (destination/'paired-effects.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(effect_rows[0]));writer.writeheader();writer.writerows(effect_rows)
    panels=[('clean_completion','Clean completion (%)',100),('flight_time','Flight time (seconds)',1),
            ('intrusion_time','Intrusion time (seconds)',1),('time_in_restricted_area','Restricted-area time (seconds)',1)]
    fig,axes=plt.subplots(2,2,figsize=(12.8,7.8),layout='constrained')
    colors=['#7e8a94','#356ea4','#b76b2b','#438167']
    for ax,(metric,title,scale) in zip(axes.flat,panels):
        for i,(arm,color) in enumerate(zip(arms,colors)):
            values=world_values[arm][metric]*scale
            ax.scatter(np.full(20,i)+np.linspace(-.07,.07,20),values,s=13,color=color,alpha=.28,zorder=2)
            mean=float(values.mean());band=np.quantile(values[indices].mean(axis=1),[.025,.975])
            ax.plot([i,i],band,color=color,linewidth=2.5,zorder=3)
            ax.plot(i,mean,'o',color=color,markersize=7,zorder=4)
            ax.annotate(f'{mean:.1f}' if scale==100 or metric!='time_in_restricted_area' else f'{mean:.3f}',
                        (i,mean),xytext=(7,6),textcoords='offset points',fontsize=9)
        ax.set_title(title);ax.set_xticks(range(4),labels,rotation=12);ax.grid(axis='y',alpha=.2)
        ax.set_xlim(-.45,3.55);ax.set_ylim(bottom=0)
        if scale==100:ax.set_ylim(0,105)
    fig.suptitle('PPO action components: one selected checkpoint, 20 reused development worlds\n'
                 'Dots: world means; solid markers: mean with world-bootstrap 95% intervals',fontsize=12)
    fig.text(.5,-.025,'All modes retain route guidance and static-area filtering. Zero heading can still turn toward the route. '
             'Evaluation intervention only; no retraining.',ha='center',fontsize=9)
    fig.savefig(destination/'action-components.png',dpi=170,bbox_inches='tight')
    fig.savefig(destination/'action-components.pdf',bbox_inches='tight');plt.close(fig)
    review={'results_sha256':sha(PARENT/'results.json'),'protocol_sha256':sha(PARENT/'protocol.json'),
            'csv_provenance':provenance,'exact_control_metric_checks':controls,'all_four_conditions_audited':True,
            'all_effects_exported':len(effect_rows),'means':result['means'],'limitations':plan['limits'],
            'promoted_to_candidate':False}
    with (destination/'audit.json').open('x',encoding='utf-8') as f:json.dump(review,f,indent=2)
    print(json.dumps({'figure':str(destination/'action-components.png'),'means':result['means'],'exact_replay_checks':controls}))

if __name__=='__main__':main()
