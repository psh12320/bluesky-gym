import hashlib
import inspect
import json
import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from atc_rl.actor_reference import is_actor_parameter, match_initial_actor
from atc_rl.policy import AircraftPolicy


class ToyAircraft(Env):
    observation_space=spaces.Dict({'actor':spaces.Box(-1,1,(3,),dtype=np.float32),
                                   'critic':spaces.Box(-1,1,(8,),dtype=np.float32)})
    action_space=spaces.Box(-1,1,(2,),dtype=np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        return {'actor':np.zeros(3,dtype=np.float32),'critic':np.zeros(8,dtype=np.float32)},{}
    def step(self,action):
        obs,_=self.reset()
        return obs,0.,False,False,{}


def make_model(centralized):
    return PPO(AircraftPolicy,DummyVecEnv([ToyAircraft]*10),seed=64,n_steps=2,batch_size=10,n_epochs=1,device='cpu',
               policy_kwargs={'centralized':centralized,'actor_width':8,'critic_width':8,'log_std_init':-.5})


def save_reference(model,directory):
    from pathlib import Path
    directory.mkdir()
    checkpoint=directory/'initial-model.zip';model.save(checkpoint)
    (directory/'config.json').write_text(json.dumps({'algorithm':'ppo','seed':64,'workers':1}))
    (directory/'checkpoints.json').write_text(json.dumps([{'file':checkpoint.name,'live_transitions':0,
        'counted_transitions':0,'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}]))
    policy_file=Path(inspect.getfile(AircraftPolicy))
    (directory/'provenance.json').write_text(json.dumps({'source_sha256':{
        'atc_rl/policy.py':hashlib.sha256(policy_file.read_bytes()).hexdigest()}}))
    return checkpoint


def test_matching_preserves_critic_and_reproduces_initial_actor_and_sampling(tmp_path):
    torch.set_num_threads(1)
    reference=make_model(False)
    checkpoint=save_reference(reference,tmp_path/'reference')
    expected_rng=torch.get_rng_state().clone()
    observation={'actor':torch.ones((10,3))*.2,'critic':torch.ones((10,8))*.4}
    expected_actions=reference.policy(observation)[0].detach()
    target=make_model(True)
    before={name:p.detach().clone() for name,p in target.policy.named_parameters() if not is_actor_parameter(name)}
    record=match_initial_actor(target,checkpoint,tmp_path/'copied')
    assert torch.equal(torch.get_rng_state(),expected_rng)
    assert torch.equal(target.policy(observation)[0],expected_actions)
    assert all(torch.equal(dict(target.policy.named_parameters())[name],p) for name,p in before.items())
    assert record['actor_parameters_identical'] and record['critic_parameters_unchanged']
    context=torch.ones((1,8),requires_grad=True)
    value=target.policy.predict_values({'actor':torch.ones((1,3)),'critic':context})
    assert torch.count_nonzero(torch.autograd.grad(value.sum(),context)[0])>0
    reference.env.close();target.env.close()


def test_trained_reference_is_rejected_before_copying_actor(tmp_path):
    reference=make_model(False)
    checkpoint=save_reference(reference,tmp_path/'reference')
    path=checkpoint.parent/'checkpoints.json'
    entries=json.loads(path.read_text());entries[0]['live_transitions']=10;path.write_text(json.dumps(entries))
    target=make_model(True)
    before={name:p.detach().clone() for name,p in target.policy.named_parameters()}
    with pytest.raises(ValueError,match='zero-experience'):
        match_initial_actor(target,checkpoint,tmp_path/'copied')
    assert all(torch.equal(dict(target.policy.named_parameters())[name],p) for name,p in before.items())
    assert not (tmp_path/'copied').exists()
    reference.env.close();target.env.close()
