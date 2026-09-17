"""Create evaluation-only support controls from one unchanged untrained policy."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,shutil,zipfile
import torch
from atc_rl.checkpoint_identity import policy_fingerprint,verified_checkpoint
root=Path(__file__).resolve().parents[2];parent=Path(__file__).resolve().parent
torch.set_num_threads(1)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=root/'runs/goal-offset-scaled-million-retry-v1/train'
manifest=read(source/'checkpoints.json');records=[r for r in manifest if r['file']=='initial-model.zip']
assert len(records)==1;record=records[0];model=verified_checkpoint(source,record)
assert all(record[key]==0 for key in ('live_transitions','counted_transitions','padded_transitions','optimizer_steps','policy_epochs','rollouts'))
with zipfile.ZipFile(model) as z:
    data=json.loads(z.read('data'))
    optimizer=torch.load(io.BytesIO(z.read('policy.optimizer.pth')),map_location='cpu',weights_only=True)
    state=torch.load(io.BytesIO(z.read('policy.pth')),map_location='cpu',weights_only=True)
assert data['num_timesteps']==data['_n_updates']==0 and not optimizer['state']
assert torch.count_nonzero(state['action_net.weight'])==0 and torch.count_nonzero(state['action_net.bias'])==0
config=read(source/'config.json')
assert config['algorithm']=='ppo' and config['action_reference']=='goal_offset'
assert config['neutral_action_mean'] and not config['guidance'] and not config['filter']
fingerprint=policy_fingerprint(model)
existing=root/'runs/goal-offset-v1/eval-initial-dev20'
existing_protocol=read(existing/'protocol.json')
assert existing_protocol['checkpoint']['live_transitions']==0
assert not existing_protocol['guidance'] and not existing_protocol['filter']
assert policy_fingerprint(Path(existing_protocol['model_path']))==fingerprint
controls=[]
for guidance,filtering in ((False,False),(True,False),(False,True),(True,True)):
    name=f'g{int(guidance)}-f{int(filtering)}';directory=parent/name;directory.mkdir()
    shutil.copyfile(model,directory/'initial-model.zip')
    assert sha(directory/'initial-model.zip')==record['sha256']
    changed=dict(config,guidance=guidance,filter=filtering,run_dir=str(directory),live_steps=0,
        checkpoint_role='evaluation-only zero-experience support intervention',original_initial_checkpoint=str(model))
    (directory/'config.json').write_text(json.dumps(changed,indent=2),encoding='utf-8')
    (directory/'checkpoints.json').write_text(json.dumps([record],indent=2),encoding='utf-8')
    origin=dict(model_sha256=sha(model),full_policy_fingerprint=fingerprint,source_checkpoint=str(model),
        source_configuration_sha256=sha(source/'config.json'),source_provenance_sha256=sha(source/'provenance.json'),
        changed_execution_flags={'guidance':guidance,'filter':filtering},additional_training_transitions=0,
        interpretation='Exact untrained policy archive; only evaluator support flags differ. No training or classical-controller refinement occurred.')
    (directory/'origin.json').write_text(json.dumps(origin,indent=2),encoding='utf-8')
    controls.append(dict(name=name,guidance=guidance,filter=filtering,checkpoint=str(directory/'initial-model.zip'),
        evaluation=str(existing) if not guidance and not filtering else str(parent/('eval-'+name+'-dev20')),
        reused_evaluation=not guidance and not filtering))
protocol=dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),purpose='Separate route-guidance and conflict-filter contributions before learning.',
    controls=controls,source_initial_model_sha256=sha(model),full_policy_fingerprint=fingerprint,
    source_serialized_training_counters={'num_timesteps':data['num_timesteps'],'optimizer_updates':data['_n_updates']},
    evaluation_seed=20260,evaluation_worlds=20,evaluation_source='runs/onpolicy-goal-offset-source-v1-verify',
    inference='deterministic per aircraft; identical actor/critic tensors in all cells',
    reuse_note='The g0-f0 reference has exact full initial-policy identity; unused learning reward scale does not affect its native-scoring evaluation.',
    physical_metrics='All nine native metrics plus aircraft and whole-world clean completion',
    analysis='Paired-world guidance effects at each filter setting, filter effects at each guidance setting, and their interaction.',
    scheduling='Run one evaluation simulator at a time. Final evaluations of the existing million-step learners take priority.',
    limitations=['These are zero-experience support effects, not learned improvements.',
        'They do not replace training with each support combination, three training seeds, or unseen scenarios.',
        'The fixed classical benchmark remains separate; its speed reference is +1 whereas this initial neural actor outputs zero speed adjustment.'],
    trained_runs_created=0,unseen_scenarios_used=False)
with (parent/'protocol.json').open('x',encoding='utf-8') as stream:json.dump(protocol,stream,indent=2)
assert sha(model)==record['sha256']
print(json.dumps({'untrained_policy_sha256':sha(model),'policy_fingerprint':fingerprint,'control_cells':len(controls),'training_steps':0}))