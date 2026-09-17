"""Audit completed on-policy training artifacts without rerunning the simulator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory):
    import torch
    from stable_baselines3 import PPO
    from atc_rl.actor_reference import is_actor_parameter
    directory=Path(directory).resolve()
    read=lambda name:json.loads((directory/name).read_text(encoding='utf-8-sig'))
    summary=read('training_summary.json')
    config=read('config.json')
    provenance=read('provenance.json')
    manifest=read('checkpoints.json')
    if summary['status'] not in ('complete','wall_limit_before_budget'):
        raise ValueError('Run has not finished')
    with zipfile.ZipFile(directory/'source.zip') as archive:
        if set(archive.namelist())!=set(provenance['source_sha256']):
            raise ValueError('Source archive and provenance have different files')
        for name,digest in provenance['source_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:
                raise ValueError('Archived source integrity mismatch: '+name)
    previous=-1
    for record in manifest:
        checkpoint=(directory/record['file']).resolve()
        if checkpoint.parent!=directory or sha256(checkpoint)!=record['sha256']:
            raise ValueError('Checkpoint integrity mismatch')
        if record['counted_transitions']!=record['live_transitions']+record['padded_transitions']:
            raise ValueError('Checkpoint transition accounting mismatch')
        if record['live_transitions']<previous:raise ValueError('Non-monotone checkpoint experience')
        previous=record['live_transitions']
    initial_record=[r for r in manifest if r['file']=='initial-model.zip']
    final_record=[r for r in manifest if r['file']=='model.zip']
    if len(initial_record)!=1 or len(final_record)!=1 or initial_record[0]['counted_transitions']!=0:
        raise ValueError('Missing initial/final checkpoint or nonzero initial experience')
    final_record=final_record[0]
    for name in ('live_transitions','counted_transitions','padded_transitions','optimizer_steps','policy_epochs','rollouts'):
        if summary[name]!=final_record[name]:raise ValueError('Summary/checkpoint mismatch: '+name)
    if final_record['sha256']!=summary['model_sha256']:raise ValueError('Final model digest differs')
    with (directory/'learning.csv').open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=summary['rollouts']:raise ValueError('Missing learning-curve rows')
    for name in ('live_transitions','counted_transitions','padded_transitions','optimizer_steps','policy_epochs'):
        if int(rows[-1][name])!=summary[name]:raise ValueError('Learning-curve mismatch: '+name)
    for row in rows:
        if int(row['counted_transitions'])!=int(row['live_transitions'])+int(row['padded_transitions']):
            raise ValueError('Learning-curve transition accounting mismatch')
    current=Path(__file__).with_name('policy.py')
    if sha256(current)!=provenance['source_sha256']['atc_rl/policy.py']:
        raise ValueError('Audit requires the recorded policy implementation')
    torch.set_num_threads(1)
    initial=PPO.load(directory/'initial-model.zip',device='cpu')
    model=PPO.load(directory/'model.zip',device='cpu')
    from atc_rl.exploration import validate_model
    for restored in (initial,model):
        if restored.gamma!=config['gamma'] or restored.gae_lambda!=config['gae_lambda']:
            raise ValueError('Serialized discount or GAE lambda differs from training configuration')
        if restored.rollout_buffer.gae_lambda!=config['gae_lambda']:
            raise ValueError('Restored rollout buffer changed GAE lambda')
    validate_model(initial,config)
    exploration=validate_model(model,config)
    if model.num_timesteps!=summary['counted_transitions'] or model._n_updates!=summary['policy_epochs']:
        raise ValueError('Serialized training counters differ')
    if model.policy.centralized!=(config['algorithm']=='mappo'):
        raise ValueError('Serialized critic architecture differs')
    before=dict(initial.policy.named_parameters())
    after=dict(model.policy.named_parameters())
    if set(before)!=set(after) or not all(torch.isfinite(p).all() for p in after.values()):
        raise ValueError('Policy has invalid parameter layout or non-finite weights')
    changed=[name for name,p in after.items() if not torch.equal(p,before[name])]
    actor_changed=any(is_actor_parameter(name) for name in changed)
    critic_changed=any(not is_actor_parameter(name) for name in changed)
    if not actor_changed or not critic_changed:raise ValueError('Actor or critic did not change')
    steps={int(state['step'].item()) for state in model.policy.optimizer.state.values() if 'step' in state}
    if steps!={summary['optimizer_steps']}:raise ValueError('Serialized optimizer step counters differ')
    observation={key:torch.full((3,*space.shape),.2,requires_grad=(key=='critic'))
                 for key,space in model.observation_space.spaces.items()}
    mean=model.policy.get_distribution(observation).distribution.mean
    actor_gradient=torch.autograd.grad(mean.sum(),observation['critic'],allow_unused=True)[0]
    if actor_gradient is not None and torch.count_nonzero(actor_gradient):
        raise ValueError('Actor depends on centralized critic information')
    values=model.policy.predict_values(observation)
    critic_gradient=torch.autograd.grad(values.sum(),observation['critic'],allow_unused=True)[0]
    critic_uses_context=bool(critic_gradient is not None and torch.count_nonzero(critic_gradient))
    if critic_uses_context!=model.policy.centralized:raise ValueError('Critic context dependency differs')
    with (directory/'training-aircraft.csv').open(newline='',encoding='utf-8') as stream:
        aircraft=list(csv.DictReader(stream))
    if len(aircraft)!=summary['aircraft_completed']:raise ValueError('Missing completed training aircraft')
    return {'run_directory':str(directory),'algorithm':config['algorithm'],**exploration,
        'gamma':model.gamma,'gae_lambda':model.gae_lambda,
        'status':summary['status'],'live_transitions':summary['live_transitions'],
        'padded_transitions':summary['padded_transitions'],'optimizer_steps':summary['optimizer_steps'],
        'verified_checkpoints':len(manifest),'verified_source_files':len(provenance['source_sha256']),
        'learning_curve_rows':len(rows),'aircraft_completed':len(aircraft),
        'actor_parameter_count':sum(p.numel() for n,p in after.items() if is_actor_parameter(n)),
        'critic_parameter_count':sum(p.numel() for n,p in after.items() if not is_actor_parameter(n)),
        'actor_and_critic_changed':True,'actor_gradient_to_joint_context_zero':True,
        'critic_uses_joint_context':critic_uses_context,
        'structural_probe':'Synthetic bounded inputs verify computation dependencies; these are not performance scenarios.',
        'model_sha256':summary['model_sha256'],'policy_quality_assessed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Choose a fresh audit output path')
    result=audit(args.run)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
