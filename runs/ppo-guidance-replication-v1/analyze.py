"""Analyze all precommitted guided-PPO seeds, retaining every outcome."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
import numpy as np
root=Path(__file__).resolve().parents[2];parent=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from atc_rl.audit import audit
from atc_rl.compare import load_evaluation
from atc_rl.algorithm_compare import initial_for
from atc_rl.checkpoint_identity import verified_checkpoint
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(parent/'protocol.json')
seeds=[protocol['first_seed'],*protocol['additional_seeds']]
assert len(seeds)==len(set(seeds))==3
first=Path(protocol['first_run']).parent
pairs=[(seeds[0],first,root/'runs/initial-support-ablation-v1/eval-g1-f0-dev20',first/'eval-final-dev20')]
pairs += [(seed,parent/f'seed-{seed}',parent/f'seed-{seed}'/'eval-initial-dev20',parent/f'seed-{seed}'/'eval-final-dev20') for seed in protocol['additional_seeds']]
classical=root/'runs/ppo-direct-pilot-v1/eval-classical-dev20'
reference=load_evaluation(classical)
fields=('algorithm','workers','live_steps','rollout_steps','batch_size','epochs','device','guidance','filter','action_reference',
        'initial_action_std','neutral_action_mean','reward_scale','progress_scale','gamma','gae_lambda','learning_rate',
        'clip_range','entropy_coefficient','value_coefficient','max_grad_norm','target_kl','actor_widths','critic_widths',
        'decision_interval_seconds','reward_recipe')
canonical=read(first/'train/config.json');source=read(first/'train/provenance.json')
rows=[];used_world_seeds=set();evaluation_source=None
# Validate the complete precommitted cohort before producing any aggregate.
for seed,directory,initial_dir,final_dir in pairs:
    training=directory/'train';checked=audit(training);config=read(training/'config.json')
    assert checked['status']=='complete' and config['seed']==seed
    assert all(config[key]==canonical[key] for key in fields)
    assert config['world_seeds']==protocol['world_seeds'][str(seed)]
    assert not used_world_seeds.intersection(config['world_seeds'])
    used_world_seeds.update(config['world_seeds'])
    assert canonical['live_steps']<=checked['live_transitions']<canonical['live_steps']+canonical['workers']*10*canonical['rollout_steps']
    provenance=read(training/'provenance.json')
    assert provenance['source_sha256']==source['source_sha256'] and provenance['packages']==source['packages'] and provenance['python']==source['python']
    initial=load_evaluation(initial_dir);final=load_evaluation(final_dir)
    assert initial[2]==final[2]==reference[2]
    for data in (initial,final):
        execution=data[1]
        assert execution['seed']==protocol['evaluation_seed'] and execution['episodes']==protocol['evaluation_worlds']
        assert execution.get('evaluation_reward_scale',1)==1 and execution.get('evaluation_progress_scale',0)==0
        assert all(execution[key]==config[key] for key in ('algorithm','guidance','filter','action_reference'))
        if evaluation_source is None:evaluation_source=execution['source_sha256']
        assert execution['source_sha256']==evaluation_source
    assert initial[0]['runtime']==final[0]['runtime']
    checkpoint=verified_checkpoint(training,final[1]['checkpoint'])
    assert checkpoint==Path(final[1]['model_path']).resolve()
    assert sha(checkpoint)==checked['model_sha256'] and final[1]['checkpoint']['live_transitions']==checked['live_transitions']
    initial_for(checkpoint,initial[1])
    rows.append({'seed':seed,'directory':directory,'initial_dir':initial_dir,'final_dir':final_dir,
                 'initial':initial,'final':final,'audit':checked})
results=[]
for row in rows:
    output=row['directory']/'comparison-final-dev20'
    if row['seed']==protocol['first_seed']:
        comparison=read(output/'comparison.json')
        assert comparison['checkpoint']['sha256']==row['audit']['model_sha256']
    else:
        subprocess.run([sys.executable,'-m','atc_rl.compare','--initial',str(row['initial_dir']),
             '--trained',str(row['final_dir']),'--classical',str(classical),'--out',str(output)],cwd=root,check=True)
        comparison=read(output/'comparison.json')
    means={stage:{metric:float(value.mean()) for metric,value in row[stage][3].items()} for stage in ('initial','final')}
    assert comparison['means']['initial']==means['initial'] and comparison['means']['trained']==means['final']
    gate=protocol['advance_screen'];effects=comparison['comparisons']['learning_trained_minus_initial']
    checks={'arrival_at_least':means['final']['waypoint_reached']>=gate['arrival_at_least'],
            'arrival_change_at_least':effects['waypoint_reached']['mean_difference']>=gate['arrival_change_at_least'],
            'clean_completion_change_at_least':effects['clean_completion']['mean_difference']>=gate['clean_completion_change_at_least'],
            **{metric:effects[metric]['mean_difference']<=gate['safety_time_changes_at_most'] for metric in gate['metrics']}}
    results.append({'seed':row['seed'],'training_audit':row['audit'],'initial_means':means['initial'],'final_means':means['final'],
        'learning_effects':effects,'registered_screen_checks':checks,'screen_passed':all(checks.values()),
        'initial_csv_sha256':row['initial'][0]['csv_sha256'],'final_csv_sha256':row['final'][0]['csv_sha256'],
        'comparison_sha256':sha(output/'comparison.json')})
metrics={}
for metric in results[0]['initial_means']:
    before=np.array([row['initial_means'][metric] for row in results]);after=np.array([row['final_means'][metric] for row in results]);delta=after-before
    metrics[metric]={'initial_mean':float(before.mean()),'final_mean':float(after.mean()),'mean_learning_change':float(delta.mean()),
        'learning_change_sample_standard_deviation':float(delta.std(ddof=1)),
        'per_seed_learning_changes':{str(row['seed']):float(d) for row,d in zip(results,delta)},
        'minimum_learning_change':float(delta.min()),'maximum_learning_change':float(delta.max())}
record={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'registered_protocol_sha256':sha(parent/'protocol.json'),
        'training_seeds':seeds,'evaluation_worlds':protocol['evaluation_worlds'],'all_registered_seeds_included':True,
        'source_and_recipes_identical_except_seeds':True,'disjoint_training_world_seeds':sorted(used_world_seeds),
        'per_seed':results,'aggregate':metrics,'classical_means':{m:float(v.mean()) for m,v in reference[3].items()},
        'uncertainty':'Report every training-seed effect and their sample standard deviation. Within-seed intervals resample whole worlds. Three seeds are too few to establish a precise training-seed distribution.',
        'unseen_scenarios_used':False,'limitations':protocol['limitations']}
with (parent/'three-seed-results.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps({'seeds':seeds,'aggregate':metrics,'per_seed_screen_passed':{row['seed']:row['screen_passed'] for row in results}}))
