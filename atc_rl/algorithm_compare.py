"""Compare matched PPO and MAPPO learning on paired development worlds."""
import argparse
import io
import json
from pathlib import Path
import zipfile
from atc_rl.compare import load_evaluation
from atc_rl.checkpoint_identity import verified_checkpoint,policy_fingerprint


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def validate_recipes(ppo,mappo,ppo_record,mappo_record):
    if ppo['algorithm']!='ppo' or mappo['algorithm']!='mappo':raise ValueError('Expected PPO and MAPPO training runs')
    from atc_rl.exploration import require_matching
    require_matching(ppo,mappo)
    from atc_rl.traffic_scaling import require_matching as require_matching_positions
    require_matching_positions(ppo,mappo)
    keys=('seed','world_seeds','device','reward_recipe','workers','rollout_steps','batch_size','epochs','guidance','filter','action_reference',
          'reward_scale','progress_scale','initial_action_std','neutral_action_mean','gamma','gae_lambda',
          'learning_rate','clip_range','entropy_coefficient','value_coefficient','max_grad_norm','target_kl',
          'actor_widths','critic_widths','decision_interval_seconds')
    for key in keys:
        if ppo[key]!=mappo[key]:raise ValueError('Algorithm comparison changed '+key)
    if ppo.get('static_filter',False)!=mappo.get('static_filter',False):
        raise ValueError('Algorithm comparison changed static_filter')
    if ppo.get('mask_conflict_features',False)!=mappo.get('mask_conflict_features',False):
        raise ValueError('Algorithm comparison changed mask_conflict_features')
    if ppo.get('conflict_features',False)!=mappo.get('conflict_features',False):
        raise ValueError('Algorithm comparison changed conflict_features')
    counts=[r['live_transitions'] for r in (ppo_record,mappo_record)]
    tolerance=ppo['workers']*10*ppo['rollout_steps']
    if min(counts)<=0 or abs(counts[0]-counts[1])>tolerance:
        raise ValueError('Checkpoint experience differs by more than one full rollout')
    return tolerance


def checkpoint(protocol):
    model=Path(protocol['model_path']).resolve()
    path=verified_checkpoint(model.parent,protocol['checkpoint'])
    if path!=model:raise ValueError('Evaluation checkpoint path differs from manifest')
    config=read(model.parent/'config.json')
    if config['algorithm']!=protocol['algorithm']:raise ValueError('Evaluation algorithm differs from checkpoint')
    return path,config


def initial_for(training,initial_protocol):
    manifest=read(training.parent/'checkpoints.json')
    rows=[r for r in manifest if r['file']=='initial-model.zip']
    if len(rows)!=1 or any(rows[0][k]!=0 for k in ('live_transitions','counted_transitions','optimizer_steps')):
        raise ValueError('Training run has no verified zero-experience initial reference')
    own=verified_checkpoint(training.parent,rows[0])
    evaluated=verified_checkpoint(Path(initial_protocol['model_path']).parent,initial_protocol['checkpoint'])
    if any(initial_protocol['checkpoint'][k]!=0 for k in ('live_transitions','counted_transitions','optimizer_steps')):
        raise ValueError('Evaluated initial policy already contains training')
    if policy_fingerprint(own)!=policy_fingerprint(evaluated):raise ValueError('Initial evaluation does not match the trained run')
    return own


def actor_match(ppo,mappo):
    import torch
    from atc_rl.actor_reference import is_actor_parameter
    def state(path):
        with zipfile.ZipFile(path) as z:return torch.load(io.BytesIO(z.read('policy.pth')),map_location='cpu',weights_only=True)
    left,right=state(ppo),state(mappo)
    actors={k for k in left if is_actor_parameter(k)}
    if not actors or actors!={k for k in right if is_actor_parameter(k)} or not all(torch.equal(left[k],right[k]) for k in actors):
        raise ValueError('PPO and MAPPO initial actors are not identical')
    return {'initial_actor_tensors_identical':True,'actor_parameters':sum(left[k].numel() for k in actors),
        'ppo_critic_parameters':sum(v.numel() for k,v in left.items() if k not in actors),
        'mappo_critic_parameters':sum(v.numel() for k,v in right.items() if k not in actors)}


def compare(ppo_initial,ppo,mappo_initial,mappo):
    import numpy as np
    directories={'ppo_initial':ppo_initial,'ppo':ppo,'mappo_initial':mappo_initial,'mappo':mappo}
    loaded={name:load_evaluation(path) for name,path in directories.items()}
    reference=loaded['ppo_initial'];first_protocol=reference[1]
    for name,data in loaded.items():
        protocol=data[1]
        if data[2]!=reference[2]:raise ValueError('Algorithm comparison scenarios are not paired')
        for key in ('seed','episodes','source_sha256','guidance','filter','action_reference'):
            if protocol[key]!=first_protocol[key]:raise ValueError('Algorithm evaluation changed '+key)
        if protocol.get('static_filter',False)!=first_protocol.get('static_filter',False):
            raise ValueError('Algorithm evaluation changed static_filter')
        if protocol.get('mask_conflict_features',False)!=first_protocol.get('mask_conflict_features',False):
            raise ValueError('Algorithm evaluation changed mask_conflict_features')
        if protocol.get('conflict_features',False)!=first_protocol.get('conflict_features',False):
            raise ValueError('Algorithm evaluation changed conflict_features')
        if protocol.get('evaluation_reward_scale',1.)!=1. or protocol.get('evaluation_progress_scale',0.)!=0.:
            raise ValueError('Algorithm comparison requires native scoring')
        expected='mappo' if name.startswith('mappo') else 'ppo'
        if protocol['algorithm']!=expected:raise ValueError('Wrong algorithm evaluation')
    ppo_path,ppo_config=checkpoint(loaded['ppo'][1]);mappo_path,mappo_config=checkpoint(loaded['mappo'][1])
    tolerance=validate_recipes(ppo_config,mappo_config,loaded['ppo'][1]['checkpoint'],loaded['mappo'][1]['checkpoint'])
    from atc_rl.exploration import require_matching
    for config,name in ((ppo_config,'ppo'),(mappo_config,'mappo')):
        require_matching(config,loaded[name][1])
        from atc_rl.traffic_scaling import require_matching as require_matching_positions
        require_matching_positions(config,loaded[name][1])
        for key in ('guidance','filter','action_reference'):
            if config[key]!=loaded[name][1][key]:raise ValueError('Evaluator ignored training '+key)
        if config.get('static_filter',False)!=loaded[name][1].get('static_filter',False):
            raise ValueError('Evaluator ignored training static_filter')
        if config.get('mask_conflict_features',False)!=loaded[name][1].get('mask_conflict_features',False):
            raise ValueError('Evaluator ignored training mask_conflict_features')
        if config.get('conflict_features',False)!=loaded[name][1].get('conflict_features',False):
            raise ValueError('Evaluator ignored training conflict_features')
    identity=actor_match(initial_for(ppo_path,loaded['ppo_initial'][1]),initial_for(mappo_path,loaded['mappo_initial'][1]))
    for metric in reference[3]:
        if not np.array_equal(reference[3][metric],loaded['mappo_initial'][3][metric]):
            raise ValueError('Matched starting policies have different evaluated behavior')
    count=len(reference[2]);rng=np.random.default_rng(701);indices=rng.integers(0,count,size=(10000,count))
    effects={}
    for metric in reference[3]:
        ppo_gain=loaded['ppo'][3][metric].astype(float)-reference[3][metric]
        mappo_gain=loaded['mappo'][3][metric].astype(float)-loaded['mappo_initial'][3][metric]
        differences={'ppo_learning':ppo_gain,'mappo_learning':mappo_gain,
            'mappo_minus_ppo_final':loaded['mappo'][3][metric].astype(float)-loaded['ppo'][3][metric],
            'mappo_minus_ppo_learning':mappo_gain-ppo_gain}
        effects[metric]={name:{'mean_difference':float(value.mean()),
            'paired_world_bootstrap_95_interval':np.quantile(value[indices].mean(axis=1),[.025,.975]).tolist()}
            for name,value in differences.items()}
    sources={name:read(path.parent/'provenance.json')['source_sha256'] for name,path in (('ppo',ppo_path),('mappo',mappo_path))}
    differences=sorted(k for k in set(sources['ppo'])|set(sources['mappo']) if sources['ppo'].get(k)!=sources['mappo'].get(k))
    return {'training_seeds':1,'worlds':count,'initial_identity':identity,'live_count_tolerance':tolerance,
        'training_source_differences':differences,'training_source_identical':not differences,
        'source_caveat':'Changed training sources require separate verification before attributing differences solely to the critic.',
        'checkpoints':{k:loaded[k][1]['checkpoint'] for k in ('ppo','mappo')},
        'training_configurations':{'ppo':ppo_config,'mappo':mappo_config},
        'evaluations':{k:{'directory':str(v.resolve()),'csv_sha256':loaded[k][0]['csv_sha256']} for k,v in directories.items()},
        'means':{k:{metric:float(value.mean()) for metric,value in data[3].items()} for k,data in loaded.items()},
        'effects':effects,
        'limitations':['One training seed on paired development scenarios; no general algorithm ranking.',
            'Joint critic input changes parameter count even with the same hidden widths; this is not a capacity-matched information-only ablation.',
            'Intervals resample worlds, not independent training seeds. Actual experience counts and partial runs remain explicit.',
            'The initial navigation prior is controlled by separate initial-policy evaluations.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ppo-initial','ppo','mappo-initial','mappo','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Choose a fresh algorithm comparison file')
    result=compare(args.ppo_initial,args.ppo,args.mappo_initial,args.mappo)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'worlds':result['worlds'],'checkpoints':result['checkpoints'],'initial_identity':result['initial_identity']}))


if __name__=='__main__':main()
