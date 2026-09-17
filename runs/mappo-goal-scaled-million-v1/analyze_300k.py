from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
from atc_rl.algorithm_compare import validate_recipes
root=Path.cwd();parent=root/'runs/mappo-goal-scaled-million-v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
comparison=read(parent/'ppo-mappo-300k-dev20.json');protocol=read(parent/'protocol.json')
ppo=root/'runs/goal-offset-scaled-million-retry-v1/train';mappo=parent/'train'
proof_path=ppo.parent/'300k-prefix-reproduction.json';proof=read(proof_path)
assert all(proof[key] for key in ('all_policy_tensors_identical','optimizer_state_identical','nontiming_learning_fields_identical'))
old=root/'runs/goal-offset-budget-scale-v1/scaled/policy-live-301272.zip'
retry=ppo/'policy-live-301272.zip'
assert sha(old)==proof['original_checkpoint_sha256']==comparison['checkpoints']['ppo']['sha256']
assert sha(retry)==proof['retry_checkpoint_sha256']
provenance=read(ppo/'provenance.json');assert provenance==read(mappo/'provenance.json')
frozen=root/'runs/onpolicy-durable-source-v1-verify'
assert all(sha(frozen/name)==digest for name,digest in provenance['source_sha256'].items())
validate_recipes(read(ppo/'config.json'),read(mappo/'config.json'),comparison['checkpoints']['ppo'],comparison['checkpoints']['mappo'])
means=comparison['means']['mappo'];gate=protocol['advancement_screen']
changes={key:value['mappo_learning']['mean_difference'] for key,value in comparison['effects'].items()}
checks={'arrival_at_least':means['waypoint_reached']>=gate['arrival_at_least'],
        'arrival_change_at_least':changes['waypoint_reached']>=gate['arrival_change_at_least'],
        'clean_completion_change_at_least':changes['clean_completion']>=gate['clean_completion_change_at_least'],
        **{key:changes[key]<=gate['safety_time_changes_at_most'] for key in gate['metrics']}}
record=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),checkpoint=comparison['checkpoints']['mappo'],
    comparison_sha256=sha(parent/'ppo-mappo-300k-dev20.json'),registered_protocol_sha256=sha(parent/'protocol.json'),
    source_equivalence=dict(current_ppo_mappo_provenance_identical=True,frozen_source_files_verified=len(provenance['source_sha256']),
                           reproduced_ppo_checkpoint_sha256=sha(retry),original_ppo_checkpoint_sha256=sha(old),proof_sha256=sha(proof_path)),
    means=means,learning_changes=changes,advancement_checks=checks,advancement_screen_passed=all(checks.values()),
    training_seed_count=1,paired_development_worlds=20,unseen_evaluation_used=False,
    next_registered_target_live_transitions=1000000,full_budget_training_continues=True,
    limitations=comparison['limitations'])
with (parent/'validated-300k-results.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps({'checks':checks,'screen_passed':all(checks.values())}))