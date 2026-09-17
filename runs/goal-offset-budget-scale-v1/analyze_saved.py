"""Analyze all registered surviving checkpoints and label interrupted endpoints."""
from pathlib import Path
import json,subprocess,sys,hashlib
import numpy as np
from atc_rl.compare import load_evaluation

root=Path(__file__).resolve().parents[2];parent=Path(__file__).resolve().parent
initial=root/'runs/goal-offset-v1/eval-initial-dev20';classical=root/'runs/ppo-direct-pilot-v1/eval-classical-dev20'
paths={'native':{'initial':initial,'100k':root/'runs/goal-offset-v1/eval-trained-dev20','300k':parent/'eval-native-300k-dev20','last':parent/'eval-native-last-dev20'},
       'scaled':{'initial':initial,'100k':parent/'eval-scaled-100k-dev20','300k':parent/'eval-scaled-300k-dev20','last':parent/'eval-scaled-last-dev20'}}
loaded={kind:{stage:load_evaluation(directory) for stage,directory in stages.items()} for kind,stages in paths.items()}
reference=loaded['native']['initial'];classical_data=load_evaluation(classical)
for stages in loaded.values():
    for data in stages.values():
        assert data[0]['episodes']==20 and data[0]['agent_episodes']==200
        assert data[2]==reference[2]==classical_data[2]
        assert data[1]['source_sha256']==reference[1]['source_sha256']
        assert data[1]['evaluation_reward_scale']==1 and data[1]['evaluation_progress_scale']==0
        assert data[1]['action_reference']=='goal_offset' and not data[1]['guidance'] and not data[1]['filter']
for kind in paths:
    args=[sys.executable,'-m','atc_rl.compare','--initial',str(initial),'--trained',str(paths[kind]['last']),
          '--classical',str(classical),'--out',str(parent/f'comparison-{kind}-last-dev20')]
    subprocess.run(args,cwd=root,check=True,capture_output=True,text=True)
    args=[sys.executable,'-m','atc_rl.curves','--training-run',str(parent/kind),'--classical',str(classical),'--out',str(parent/f'curve-{kind}-dev20')]
    for directory in paths[kind].values():args+=['--evaluation',str(directory)]
    subprocess.run(args,cwd=root,check=True,capture_output=True,text=True)

rng=np.random.default_rng(701);indices=rng.integers(0,20,size=(10000,20));scale_effects={};screens=[]
protocol=json.loads((parent/'protocol.json').read_text())
for stage in ('100k','300k','last'):
    native=loaded['native'][stage];scaled=loaded['scaled'][stage];metrics={}
    for metric in native[3]:
        delta=scaled[3][metric].astype(float)-native[3][metric]
        metrics[metric]={'scaled_minus_native':float(delta.mean()),'paired_world_bootstrap_95_interval':np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist()}
    scale_effects[stage]={'native_checkpoint':native[1]['checkpoint'],'scaled_checkpoint':scaled[1]['checkpoint'],'metrics':metrics}
for kind,stages in loaded.items():
    for stage,data in stages.items():
        if stage=='initial':continue
        means={k:float(v.mean()) for k,v in data[3].items()};changes={k:float((v.astype(float)-reference[3][k]).mean()) for k,v in data[3].items()}
        gate=protocol['advancement_screen'];checks={
            'arrival_at_least':means['waypoint_reached']>=gate['arrival_at_least'],
            'arrival_change_at_least':changes['waypoint_reached']>=gate['arrival_change_at_least'],
            'clean_completion_change_at_least':changes['clean_completion']>=gate['clean_completion_change_at_least'],
            **{k:changes[k]<=gate['safety_time_changes_at_most'] for k in gate['metrics']}}
        screens.append({'kind':kind,'stage':stage,'actual_live_transitions':data[1]['checkpoint']['live_transitions'],
                       'means':means,'learning_changes':changes,'checks':checks,'screen_passed':all(checks.values()),
                       'csv_sha256':data[0]['csv_sha256']})
record={'status':'interrupted_partial_training_evaluated','requested_live_transitions_per_run':1000000,
        'million_step_results_present':False,'training_seeds':1,'development_worlds':20,
        'scale_effects':scale_effects,'learning_screens':screens,
        'initial_means':{k:float(v.mean()) for k,v in reference[3].items()},
        'classical_means':{k:float(v.mean()) for k,v in classical_data[3].items()},
        'limitations':['Supplemental endpoints are the last durable policies, not completed million-step models.',
            'All results use one training seed and development worlds. Scenario intervals do not establish training-seed robustness.',
            'Reward-scale comparisons have small rollout-boundary differences in live experience; actual counts are retained.',
            'Arrival from the initial navigation prior is not evidence of learning.']}
with (parent/'saved-checkpoint-results-v1.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps({'screens':[{'kind':r['kind'],'stage':r['stage'],'screen_passed':r['screen_passed'],'clean_completion':r['means']['clean_completion']} for r in screens]}))
