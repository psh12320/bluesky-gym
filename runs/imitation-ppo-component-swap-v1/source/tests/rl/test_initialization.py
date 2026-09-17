import numpy as np
import pytest
import torch
from gymnasium import Env,spaces
from stable_baselines3 import PPO
from atc_rl.initialization import neutral_action_mean
from atc_rl.policy import AircraftPolicy


class AircraftExample(Env):
    observation_space=spaces.Dict({'actor':spaces.Box(-1,1,(3,),dtype=np.float32),
                                   'critic':spaces.Box(-1,1,(8,),dtype=np.float32)})
    action_space=spaces.Box(-1,1,(2,),dtype=np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        return {'actor':np.full(3,.2,dtype=np.float32),'critic':np.full(8,.2,dtype=np.float32)},{}
    def step(self,action):return self.reset()[0],1.,False,False,{}


def make_model(std):
    torch.set_num_threads(1)
    return PPO(AircraftPolicy,AircraftExample(),seed=64,n_steps=4,batch_size=4,n_epochs=1,device='cpu',
               policy_kwargs={'actor_width':8,'critic_width':8,'log_std_init':float(np.log(std))})


def test_neutral_initialization_changes_only_action_head_and_preserves_rng():
    model=make_model(.05)
    before={name:p.detach().clone() for name,p in model.policy.named_parameters()}
    rng=torch.get_rng_state().clone()
    neutral_action_mean(model)
    assert torch.equal(rng,torch.get_rng_state())
    for name,p in model.policy.named_parameters():
        if not name.startswith('action_net.'):
            assert torch.equal(p,before[name])
    observation={'actor':torch.randn(16,3),'critic':torch.randn(16,8)}
    distribution=model.policy.get_distribution(observation).distribution
    assert torch.count_nonzero(distribution.mean)==0
    torch.testing.assert_close(distribution.stddev,torch.full((16,2),.05))
    before_update=model.policy.action_net.weight.detach().clone()
    model.learn(4)
    assert not torch.equal(before_update,model.policy.action_net.weight)
    model.env.close()


def test_neutral_initialization_rejects_trained_model_without_mutating_weights():
    model=make_model(.05);model.learn(4)
    before={name:p.detach().clone() for name,p in model.policy.named_parameters()}
    with pytest.raises(ValueError,match='untrained'):
        neutral_action_mean(model)
    assert all(torch.equal(p,before[name]) for name,p in model.policy.named_parameters())
    model.env.close()


def test_smaller_exploration_preserves_all_initial_mean_and_value_weights():
    baseline=make_model(.6065306597126334)
    reduced=make_model(.05)
    left=dict(baseline.policy.named_parameters());right=dict(reduced.policy.named_parameters())
    assert not torch.equal(left['log_std'],right['log_std'])
    assert all(torch.equal(p,right[name]) for name,p in left.items() if name!='log_std')
    observations={'actor':torch.full((4,3),.2),'critic':torch.full((4,8),.4)}
    a=baseline.policy.get_distribution(observations).distribution
    b=reduced.policy.get_distribution(observations).distribution
    assert torch.equal(a.mean,b.mean)
    baseline.env.close();reduced.env.close()
