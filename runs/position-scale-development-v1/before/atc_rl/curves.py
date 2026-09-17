"""Plot physical performance versus verified experience along one training run."""
import argparse
import json
from pathlib import Path
from atc_rl.checkpoint_identity import policy_fingerprint,verified_checkpoint
from atc_rl.compare import load_evaluation


def validate_configuration(source,canonical,protocol,live):
    from atc_rl.exploration import require_matching
    require_matching(source,canonical)
    require_matching(protocol,canonical)
    # At zero experience, reward transformations have never updated the policy.
    # Full tensor identity is checked separately against this run's own initial model.
    fields=[('algorithm',None),('seed',None),('workers',None),('guidance',False),('filter',False),('static_filter',False),
            ('initial_action_std',.6065306597126334),('neutral_action_mean',False),('action_reference','direct'),('conflict_features',False),('mask_conflict_features',False)]
    if live>0:fields += [('progress_scale',0.0),('reward_scale',1.0)]
    for key,default in fields:
        if source.get(key,default)!=canonical.get(key,default):
            raise ValueError('Curve mixes training configurations: '+key)
    for key,default in [('algorithm',None),('guidance',False),('filter',False),('static_filter',False),('action_reference','direct'),('conflict_features',False),('mask_conflict_features',False)]:
        if protocol.get(key,default)!=canonical.get(key,default):
            raise ValueError('Curve evaluation changed '+key)
    if protocol.get('evaluation_reward_scale',1.)!=1. or protocol.get('evaluation_progress_scale',0.)!=0.:
        raise ValueError('Curve requires native evaluation rewards')


def main():
    import numpy as np
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,action='append',required=True)
    parser.add_argument('--classical',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Choose a fresh curve directory')
    if len(args.evaluation)<2:parser.error('At least two evaluated checkpoints are required')
    config=json.loads((args.training_run/'config.json').read_text(encoding='utf-8-sig'))
    manifest=json.loads((args.training_run/'checkpoints.json').read_text(encoding='utf-8-sig'))
    canonical={}
    for record in manifest:
        canonical.setdefault(record['live_transitions'],[]).append(record)
    results=[]
    fingerprints={}
    for directory in args.evaluation:
        loaded=load_evaluation(directory)
        summary,protocol,scenarios,values=loaded
        if protocol['checkpoint'] is None:raise ValueError('A curve point must evaluate a checkpoint')
        checkpoint=verified_checkpoint(Path(protocol['model_path']).parent,protocol['checkpoint'])
        live=protocol['checkpoint']['live_transitions']
        fingerprint=policy_fingerprint(checkpoint)
        matches=[]
        for record in canonical.get(live,[]):
            path=verified_checkpoint(args.training_run,record)
            if path not in fingerprints:fingerprints[path]=policy_fingerprint(path)
            if fingerprints[path]==fingerprint:matches.append(record)
        if not matches:raise ValueError('Evaluation is not a matching checkpoint from the declared training lineage')
        source_config=json.loads((checkpoint.parent/'config.json').read_text(encoding='utf-8-sig'))
        validate_configuration(source_config,config,protocol,live)
        if results:
            first=results[0]['loaded']
            if scenarios!=first[2]:raise ValueError('Curve scenarios are not paired')
            if protocol['source_sha256']!=first[1]['source_sha256']:
                raise ValueError('Curve mixes evaluation implementations')
        results.append({'directory':str(directory.resolve()),'loaded':loaded,'live':live,'policy_fingerprint':fingerprint})
    results.sort(key=lambda x:x['live'])
    if len({r['live'] for r in results})!=len(results):raise ValueError('Duplicate experience point')
    if results[0]['live']!=0:raise ValueError('Include the untrained initial policy')
    classical=load_evaluation(args.classical)
    if classical[1]['algorithm']!='classical':raise ValueError('Expected the fixed classical benchmark')
    if classical[2]!=results[0]['loaded'][2]:raise ValueError('Classical scenarios differ')
    count=len(classical[2]);rng=np.random.default_rng(701)
    indices=rng.integers(0,count,size=(10000,count))
    points=[]
    first_values=results[0]['loaded'][3]
    for item in results:
        metrics={}
        for metric,values in item['loaded'][3].items():
            bootstrap=values[indices].mean(axis=1)
            difference=values.astype(float)-first_values[metric]
            paired=difference[indices].mean(axis=1)
            metrics[metric]={'mean':float(values.mean()),'world_bootstrap_95_interval':np.quantile(bootstrap,[.025,.975]).tolist(),
                'difference_from_initial':float(difference.mean()),'paired_difference_95_interval':np.quantile(paired,[.025,.975]).tolist()}
        points.append({'live_transitions':item['live'],'evaluation':item['directory'],
                       'policy_fingerprint':item['policy_fingerprint'],'metrics':metrics})
    record={'training_run':str(args.training_run.resolve()),'algorithm':config['algorithm'],'training_seed':config['seed'],
        'worlds':count,'points':points,'classical_means':{k:float(v.mean()) for k,v in classical[3].items()},
        'limitations':'One training seed; paired development worlds. Bootstrap bands describe scenario variation, not training-seed uncertainty.',
        'checkpoint_reuse':'Reuse requires exact full policy tensors, live experience, action/support settings and evaluation source. Untrained references may differ only in unused training reward transformations; learned points must match those too.'}
    args.out.mkdir(parents=True)
    (args.out/'curve.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[('waypoint_reached','Arrival (%)',100),('clean_completion','Clean completion (%)',100),
            ('flight_time','Flight time (seconds)',1),('intrusion_time','Intrusion time (seconds)',1),
            ('time_in_restricted_area','Restricted-area time (seconds)',1),('time_outside_sector','Outside-sector time (seconds)',1)]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    x=np.array([p['live_transitions'] for p in points])/1e6
    for ax,(metric,title,scale) in zip(axes.flat,panels):
        y=np.array([p['metrics'][metric]['mean'] for p in points])*scale
        intervals=np.array([p['metrics'][metric]['world_bootstrap_95_interval'] for p in points])*scale
        ax.plot(x,y,'o-',label='Policy checkpoint')
        ax.fill_between(x,intervals[:,0],intervals[:,1],alpha=.18,label='Scenario bootstrap 95% interval')
        ax.axhline(record['classical_means'][metric]*scale,color='#287f58',linestyle='--',label='Fixed classical system')
        ax.set_title(title);ax.set_xlabel('Active aircraft transitions (millions)');ax.grid(alpha=.2)
        if scale==100:ax.set_ylim(-2,102)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle(config['algorithm'].upper()+f': {count} paired development worlds; one training seed')
    fig.savefig(args.out/'performance-curve.png',dpi=150)
    plt.close(fig)
    print(json.dumps({'points':[p['live_transitions'] for p in points],'arrival_means':[p['metrics']['waypoint_reached']['mean'] for p in points]}))


if __name__=='__main__':main()
