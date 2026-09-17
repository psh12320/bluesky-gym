"""Validate the 100k paired comparison without modifying training artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from atc_rl.algorithm_compare import validate_recipes

root=Path(__file__).resolve().parents[2]
parent=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
comparison_path=parent/'ppo-mappo-100k-dev20.json'
comparison=read(comparison_path)
ppo=root/'runs/goal-offset-scaled-million-retry-v1/train'
mappo=parent/'train'
proof_path=ppo.parent/'100k-prefix-reproduction.json'
proof=read(proof_path)
assert proof['all_policy_tensors_identical'] and proof['optimizer_state_identical']
assert proof['all_nontiming_learning_fields_identical']
old=root/'runs/goal-offset-budget-scale-v1/scaled/policy-live-100699.zip'
retry=ppo/'policy-live-100699.zip'
assert sha(old)==proof['original_checkpoint_sha256']==comparison['checkpoints']['ppo']['sha256']
assert sha(retry)==proof['retry_checkpoint_sha256']
provenance={name:read(path/'provenance.json') for name,path in [('ppo',ppo),('mappo',mappo)]}
assert provenance['ppo']==provenance['mappo']
frozen=root/'runs/onpolicy-durable-source-v1-verify'
for name,digest in provenance['ppo']['source_sha256'].items():
    assert sha(frozen/name)==digest,name
configs={name:read(path/'config.json') for name,path in [('ppo',ppo),('mappo',mappo)]}
validate_recipes(configs['ppo'],configs['mappo'],comparison['checkpoints']['ppo'],comparison['checkpoints']['mappo'])
protocol=read(parent/'protocol.json')
gate=protocol['advancement_screen']
means=comparison['means']['mappo']
changes={key:value['mappo_learning']['mean_difference'] for key,value in comparison['effects'].items()}
checks={
    'arrival_at_least':means['waypoint_reached']>=gate['arrival_at_least'],
    'arrival_change_at_least':changes['waypoint_reached']>=gate['arrival_change_at_least'],
    'clean_completion_change_at_least':changes['clean_completion']>=gate['clean_completion_change_at_least'],
    **{key:changes[key]<=gate['safety_time_changes_at_most'] for key in gate['metrics']}}
record={
    'checked_at_utc':datetime.now(timezone.utc).isoformat(),
    'comparison_sha256':sha(comparison_path),'registered_protocol_sha256':sha(parent/'protocol.json'),
    'training_seed_count':1,'paired_development_worlds':20,
    'checkpoint':comparison['checkpoints']['mappo'],
    'source_equivalence':{
        'current_ppo_and_mappo_provenance_identical':True,
        'frozen_source_files_verified':len(provenance['ppo']['source_sha256']),
        'original_ppo_checkpoint_reproduced_exactly':True,
        'proof_sha256':sha(proof_path),'proof_file':str(proof_path),
        'original_ppo_sha256':sha(old),'current_ppo_sha256':sha(retry),
        'training_recipe_controls_rechecked':True,
        'interpretation':'The reused PPO evaluation represents an exactly reproduced policy from the same source used for current MAPPO. Original evaluation provenance remains unchanged. Critic capacity and one-seed limitations remain.'},
    'means':means,'learning_changes':changes,'advancement_checks':checks,'advancement_screen_passed':all(checks.values()),
    'next_registered_target_live_transitions':300000,
    'full_budget_training_continues':True,'unseen_evaluation_used':False,
    'optimizer_diagnostic':{
        'file_sha256':sha(parent/'optimizer-diagnostics-100k.json'),
        'observation':'Both actors have nearly identical median saved Adam epsilon attenuation (~0.915). This does not support a substantially stronger epsilon bottleneck for MAPPO at this checkpoint.',
        'limitations':'Moments are post-clipping and collected on different learned trajectories. They cannot identify clipping dominance or establish causality.',
        'reference_implementation':'https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py',
        'verified_reference_difference':'The reference implementation clips actor and critic gradients separately; our registered runs use SB3 joint clipping. No running optimization settings have been changed.'}}
with (parent/'validated-100k-results.json').open('x',encoding='utf-8') as stream:
    json.dump(record,stream,indent=2)
print(json.dumps({'source_equivalence_verified':True,'checks':checks,'screen_passed':all(checks.values()),'learning_changes':changes}))