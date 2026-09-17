"""Match MAPPO's actor initialization to an untrained PPO checkpoint."""
from pathlib import Path
import hashlib
import inspect
import json
import shutil

ACTOR_PREFIXES = ('mlp_extractor.policy_net.', 'action_net.')


def is_actor_parameter(name):
    return name == 'log_std' or name.startswith(ACTOR_PREFIXES)


def match_initial_actor(model, checkpoint, output_directory):
    """Copy only the initial actor and reconstruct PPO's post-construction RNG.

    PPO.load constructs a fresh local-critic PPO with the recorded seed before
    loading weights. That reproduces its learner RNG initialization. Simulator
    workers have separate RNGs and retain their explicitly assigned seeds.
    The source must contain no learned experience; trained warm starts are rejected.
    """
    checkpoint=Path(checkpoint).resolve()
    output_directory=Path(output_directory)
    config_path=checkpoint.parent/'config.json'
    provenance_path=checkpoint.parent/'provenance.json'
    config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    manifest=json.loads((checkpoint.parent/'checkpoints.json').read_text(encoding='utf-8-sig'))
    provenance=json.loads(provenance_path.read_text(encoding='utf-8-sig'))
    entries=[entry for entry in manifest if entry['file']==checkpoint.name]
    if len(entries)!=1 or entries[0]['live_transitions']!=0 or entries[0]['counted_transitions']!=0:
        raise ValueError('Reference must be a recorded zero-experience PPO checkpoint')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if entries[0]['sha256']!=digest:raise ValueError('Reference checkpoint integrity mismatch')
    if config['algorithm']!='ppo' or config['seed']!=model.seed:
        raise ValueError('Reference must use PPO with the same learner seed')
    if config['workers']*10!=model.n_envs:
        raise ValueError('Reference must use the same simulator worker count')
    current_source=Path(inspect.getfile(type(model.policy)))
    if hashlib.sha256(current_source.read_bytes()).hexdigest()!=provenance['source_sha256']['atc_rl/policy.py']:
        raise ValueError('Reference policy implementation differs')
    if output_directory.exists():raise ValueError('Choose a fresh actor-reference directory')
    import torch
    from stable_baselines3 import PPO
    reference=PPO.load(checkpoint,device=model.device)
    from atc_rl.exploration import validate_model
    validate_model(reference,config)
    validate_model(model,config)
    if reference.num_timesteps!=0 or reference._n_updates!=0 or reference.policy.centralized:
        raise ValueError('Reference is trained or has a centralized critic')
    if reference.observation_space!=model.observation_space or reference.action_space!=model.action_space:
        raise ValueError('Reference spaces differ')
    source_parameters=dict(reference.policy.named_parameters())
    target_parameters=dict(model.policy.named_parameters())
    actor_names={name for name in target_parameters if is_actor_parameter(name)}
    if actor_names!={name for name in source_parameters if is_actor_parameter(name)}:
        raise ValueError('Actor parameter layouts differ')
    if any(target_parameters[name].shape!=source_parameters[name].shape for name in actor_names):
        raise ValueError('Actor parameter dimensions differ')
    critic_before={name:p.detach().clone() for name,p in target_parameters.items() if name not in actor_names}
    with torch.no_grad():
        for name in actor_names:target_parameters[name].copy_(source_parameters[name])
    assert all(torch.equal(target_parameters[name],source_parameters[name]) for name in actor_names)
    assert all(torch.equal(target_parameters[name],value) for name,value in critic_before.items())
    output_directory.mkdir(parents=True)
    shutil.copyfile(checkpoint,output_directory/checkpoint.name)
    shutil.copyfile(config_path,output_directory/'config.json')
    shutil.copyfile(provenance_path,output_directory/'provenance.json')
    (output_directory/'checkpoints.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    record={'reference_checkpoint':str(checkpoint),'reference_sha256':digest,
            'reference_live_transitions':0,'actor_parameters_identical':True,
            'critic_parameters_unchanged':True,'actor_parameter_names':sorted(actor_names),
            'actor_parameter_count':sum(target_parameters[name].numel() for name in actor_names),
            'critic_parameter_count':sum(p.numel() for name,p in target_parameters.items() if name not in actor_names),
            'learner_rng':'Reconstructed through seeded local-critic PPO loading before the first rollout.',
            'simulation_rng':'Original per-world seeds retained; no simulator steps executed by this helper.'}
    (output_directory/'matching.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record
