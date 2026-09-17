"""Compare complete, paired development evaluations without selecting a checkpoint."""
from pathlib import Path
import argparse
import csv
import hashlib
import json


def load_evaluation(directory):
    import numpy as np
    from atc.metrics import METRICS, SAFETY, summarize
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    path=directory/'aircraft.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=summary['csv_sha256']:
        raise ValueError('Evaluation CSV integrity mismatch: '+str(directory))
    with path.open(newline='',encoding='utf-8') as stream:
        records=list(csv.DictReader(stream))
    for row in records:
        row['episode']=int(row['episode'])
        for metric in METRICS:row[metric]=float(row[metric])
    recomputed=summarize(records,summary['episodes'],10)
    if recomputed['metrics']!=summary['metrics']:
        raise ValueError('Recorded summary differs from recomputed metrics')
    worlds={}
    for row in records:worlds.setdefault(row['episode'],[]).append(row)
    fingerprints={episode:{r['scenario_sha256'] for r in rows} for episode,rows in worlds.items()}
    if any(len(s)!=1 for s in fingerprints.values()):raise ValueError('Mixed scenario identity')
    values={metric:np.array([np.mean([r[metric] for r in worlds[i]]) for i in sorted(worlds)]) for metric in METRICS}
    clean=lambda r:bool(r['waypoint_reached'] and all(r[k]==0 for k in SAFETY))
    values['clean_completion']=np.array([np.mean([clean(r) for r in worlds[i]]) for i in sorted(worlds)])
    values['all_aircraft_clean_completion']=np.array([all(clean(r) for r in worlds[i]) for i in sorted(worlds)],dtype=float)
    protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    return summary,protocol,fingerprints,values


def main():
    import numpy as np
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--trained',type=Path,required=True)
    parser.add_argument('--classical',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    loaded={name:load_evaluation(getattr(args,name)) for name in ('initial','trained','classical')}
    first=loaded['initial'];trained=loaded['trained'];classical=loaded['classical']
    from atc_rl.exploration import require_matching
    require_matching(first[1],trained[1])
    from atc_rl.traffic_scaling import require_matching as require_matching_positions
    require_matching_positions(first[1],trained[1])
    if not first[2]==trained[2]==classical[2]:raise ValueError('Scenarios are not paired')
    for key in ('seed','guidance','filter','algorithm','source_sha256'):
        if first[1][key]!=trained[1][key]:raise ValueError('Initial/trained comparison changed '+key)
    if first[1].get('static_filter',False)!=trained[1].get('static_filter',False):
        raise ValueError('Initial/trained comparison changed static_filter')
    if first[1].get('mask_conflict_features',False)!=trained[1].get('mask_conflict_features',False):
        raise ValueError('Initial/trained comparison changed mask_conflict_features')
    if first[1].get('conflict_features',False)!=trained[1].get('conflict_features',False):
        raise ValueError('Initial/trained comparison changed conflict_features')
    if first[1].get('action_reference','direct')!=trained[1].get('action_reference','direct'):
        raise ValueError('Initial/trained comparison changed action reference')
    for item in (first,trained):
        if item[1].get('evaluation_progress_scale',0.0)!=0 or item[1].get('evaluation_reward_scale',1.0)!=1:
            raise ValueError('Learning comparisons require native evaluation rewards')
    if first[1]['checkpoint']['live_transitions']!=0 or first[1]['checkpoint']['counted_transitions']!=0:
        raise ValueError('Initial checkpoint was already trained')
    if trained[1]['checkpoint']['live_transitions']<=0:raise ValueError('Trained checkpoint has no live training')
    from atc_rl.checkpoint_identity import policy_fingerprint,verified_checkpoint
    initial_path=verified_checkpoint(Path(first[1]['model_path']).parent,first[1]['checkpoint'])
    trained_parent=Path(trained[1]['model_path']).parent
    trained_config_path=trained_parent/'config.json'
    trained_config=json.loads(trained_config_path.read_text(encoding='utf-8-sig'))
    if trained_config.get('action_reference','direct')!=trained[1].get('action_reference','direct'):
        raise ValueError('Evaluator did not use the trained action reference')
    if trained_config.get('static_filter',False)!=trained[1].get('static_filter',False):
        raise ValueError('Evaluator did not use the trained static_filter setting')
    if trained_config.get('mask_conflict_features',False)!=trained[1].get('mask_conflict_features',False):
        raise ValueError('Evaluator did not use the trained mask_conflict_features setting')
    if trained_config.get('conflict_features',False)!=trained[1].get('conflict_features',False):
        raise ValueError('Evaluator did not use the trained conflict_features setting')
    require_matching(trained_config,trained[1])
    require_matching_positions(trained_config,trained[1])
    trained_manifest=json.loads((trained_parent/'checkpoints.json').read_text(encoding='utf-8-sig'))
    initial_records=[r for r in trained_manifest if r['file']=='initial-model.zip' and r['live_transitions']==0]
    if len(initial_records)!=1:raise ValueError('Trained run has no unique untrained reference')
    reference_path=verified_checkpoint(trained_parent,initial_records[0])
    if policy_fingerprint(initial_path)!=policy_fingerprint(reference_path):
        raise ValueError('Evaluated initial policy differs from the trained run initialization')
    if args.out.exists():raise ValueError('Choose a fresh comparison output directory')
    args.out.mkdir(parents=True)
    rng=np.random.default_rng(701)
    count=len(first[2]);indices=rng.integers(0,count,size=(10000,count))
    comparisons={}
    for label,reference in (('learning_trained_minus_initial',first),('system_trained_minus_classical',classical)):
        metrics={}
        for metric,current in trained[3].items():
            difference=current-reference[3][metric]
            bootstrap=difference[indices].mean(axis=1)
            metrics[metric]={'mean_difference':float(difference.mean()),
                'paired_world_bootstrap_95_percent_interval':np.quantile(bootstrap,[.025,.975]).tolist()}
        comparisons[label]=metrics
    result={'worlds':count,'aircraft_per_policy':10*count,'training_seeds':1,
            'checkpoint':trained[1]['checkpoint'],
            'training_configuration':trained_config,
            'training_configuration_sha256':hashlib.sha256(trained_config_path.read_bytes()).hexdigest(),
            'initial_reference_reused':initial_path.parent.resolve()!=trained_parent.resolve(),
            'initial_reference_validation':'Exact full policy tensor identity with the trained run initial checkpoint; identical evaluation source, settings and scenarios.',
            'support':{name:{k:item[1].get(k,False) for k in ('guidance','filter','static_filter')} for name,item in loaded.items()},
            'means':{name:{key:float(value.mean()) for key,value in item[3].items()} for name,item in loaded.items()},
            'comparisons':comparisons,
            'limitations':['Development scenarios, one training seed; not final generalization evidence.',
                'Intervals resample whole worlds; they do not estimate variation across training seeds.',
                'Classical system has guidance/filter support; only the initial/trained pair isolates learning.',
                'No official aggregate scalar score is invented; all physical metrics are retained.']}
    (args.out/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    metrics=[('waypoint_reached','Arrival (%)',100),('clean_completion','Clean completion (%)',100),
             ('flight_time','Flight time (seconds)',1),('intrusion_time','Intrusion time (seconds)',1),
             ('time_in_restricted_area','Restricted-area time (seconds)',1),('time_outside_sector','Outside-sector time (seconds)',1)]
    names=list(loaded)
    for ax,(metric,label,scale) in zip(axes.flat,metrics):
        ax.bar(names,[result['means'][name][metric]*scale for name in names],color=['#a2adba','#2466a8','#2a8358'])
        ax.set_title(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if scale==100:ax.set_ylim(0,105)
    fig.suptitle(trained[1]['algorithm'].upper()+f' pilot: {count} paired development worlds; one training seed',fontsize=13)
    fig.savefig(args.out/'performance.png',dpi=160);plt.close(fig)
    print(json.dumps({'means':result['means'],'learning':comparisons['learning_trained_minus_initial']}))


if __name__=='__main__':main()
