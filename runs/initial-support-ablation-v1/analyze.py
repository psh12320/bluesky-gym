"""Measure initial guidance/filter effects with paired whole-world uncertainty."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from atc_rl.compare import load_evaluation
from atc_rl.checkpoint_identity import policy_fingerprint,verified_checkpoint
parent=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(parent/'protocol.json')
loaded={}
for cell in protocol['controls']:
    data=load_evaluation(Path(cell['evaluation']))
    summary,execution,scenarios,values=data
    assert execution['algorithm']=='ppo'
    assert execution['seed']==protocol['evaluation_seed'] and execution['episodes']==protocol['evaluation_worlds']
    assert execution['guidance']==cell['guidance'] and execution['filter']==cell['filter']
    assert execution['action_reference']=='goal_offset'
    assert execution.get('evaluation_reward_scale',1)==1 and execution.get('evaluation_progress_scale',0)==0
    assert all(execution['checkpoint'][name]==0 for name in ('live_transitions','counted_transitions','padded_transitions','optimizer_steps','policy_epochs','rollouts'))
    model=verified_checkpoint(Path(execution['model_path']).parent,execution['checkpoint'])
    assert policy_fingerprint(model)==protocol['full_policy_fingerprint']
    assert sha(Path(cell['checkpoint']))==protocol['source_initial_model_sha256']
    if loaded:
        first=next(iter(loaded.values()))
        assert scenarios==first[2]
        assert execution['source_sha256']==first[1]['source_sha256']
    loaded[cell['name']]=data
classical=load_evaluation(root/'runs/ppo-direct-pilot-v1/eval-classical-dev20')
assert classical[1]['algorithm']=='classical' and classical[2]==loaded['g0-f0'][2]
count=len(classical[2]);indices=np.random.default_rng(701).integers(0,count,size=(10000,count))
contrasts={
    'guidance_without_filter':{'g1-f0':1,'g0-f0':-1},
    'guidance_with_filter':{'g1-f1':1,'g0-f1':-1},
    'filter_without_guidance':{'g0-f1':1,'g0-f0':-1},
    'filter_with_guidance':{'g1-f1':1,'g1-f0':-1},
    'guidance_filter_interaction':{'g1-f1':1,'g1-f0':-1,'g0-f1':-1,'g0-f0':1}}
effects={}
for label,terms in contrasts.items():
    metrics={}
    for metric in loaded['g0-f0'][3]:
        difference=sum(weight*loaded[cell][3][metric].astype(float) for cell,weight in terms.items())
        metrics[metric]={'mean_difference':float(difference.mean()),
            'paired_world_bootstrap_95_interval':np.quantile(difference[indices].mean(axis=1),[.025,.975]).tolist()}
    effects[label]=metrics
record={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'worlds':count,'aircraft_per_cell':10*count,
    'policy_experience':0,'policy_fingerprint':protocol['full_policy_fingerprint'],
    'protocol_sha256':sha(parent/'protocol.json'),
    'evaluations':{cell['name']:{'directory':cell['evaluation'],'csv_sha256':loaded[cell['name']][0]['csv_sha256']} for cell in protocol['controls']},
    'means':{name:{metric:float(values.mean()) for metric,values in item[3].items()} for name,item in loaded.items()},
    'classical_means':{metric:float(values.mean()) for metric,values in classical[3].items()},
    'contrasts':contrasts,'initial_support_effects':effects,
    'limitations':protocol['limitations']+['World bootstrap intervals describe scenario variation, not training-seed variation.','These are the registered development worlds, not unseen generalization scenarios.']}
with (parent/'support-effects.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps({'means':record['means'],'effects':effects}))
