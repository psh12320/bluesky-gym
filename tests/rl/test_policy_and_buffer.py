import numpy as np
import pytest
import torch
from gymnasium import spaces
from atc_rl.policy import AircraftPolicy
from atc_rl.buffer import ActiveRolloutBuffer

def observation_space():
    return spaces.Dict({'actor':spaces.Box(-1,1,shape=(3,),dtype=np.float32),
                        'critic':spaces.Box(-1,1,shape=(8,),dtype=np.float32)})

def policy(centralized):
    torch.manual_seed(12)
    return AircraftPolicy(observation_space(),spaces.Box(-1,1,shape=(2,),dtype=np.float32),
        lambda _:3e-4,centralized=centralized,actor_width=8,critic_width=8)

@pytest.mark.parametrize('centralized',[False,True])
def test_actor_cannot_use_centralized_inputs(centralized):
    model=policy(centralized)
    actor=torch.tensor([[.1,.2,.3]])
    context=torch.zeros((1,8),requires_grad=True)
    obs={'actor':actor,'critic':context}
    mean=model.get_distribution(obs).distribution.mean
    gradient=torch.autograd.grad(mean.sum(),context)[0]
    assert torch.count_nonzero(gradient)==0
    other=model.get_distribution({'actor':actor,'critic':torch.ones((1,8))}).distribution.mean
    assert torch.equal(mean,other)

@pytest.mark.parametrize('centralized',[False,True])
def test_only_centralized_critic_uses_joint_information(centralized):
    model=policy(centralized)
    actor=torch.tensor([[.1,.2,.3]])
    context=torch.zeros((1,8),requires_grad=True)
    value=model.predict_values({'actor':actor,'critic':context})
    gradient=torch.autograd.grad(value.sum(),context)[0]
    assert bool(torch.count_nonzero(gradient))==centralized

@pytest.mark.parametrize('centralized',[False,True])
def test_policy_serialization_retains_information_boundary(tmp_path,centralized):
    model=policy(centralized);path=tmp_path/'policy.pt';model.save(path)
    restored=AircraftPolicy.load(path, device=model.device)
    obs={'actor':torch.ones((2,3)),'critic':torch.ones((2,8))}
    assert restored.centralized==centralized
    assert torch.equal(model.predict_values(obs),restored.predict_values(obs))
    assert torch.equal(model.get_distribution(obs).distribution.mean,restored.get_distribution(obs).distribution.mean)

def filled_buffer():
    b=ActiveRolloutBuffer(3,observation_space(),spaces.Box(-1,1,shape=(2,),dtype=np.float32),n_envs=2,gamma=.9,gae_lambda=1)
    for step in range(3):
        b.pending_valid=np.array([True,step==0])
        b.add({'actor':np.array([[step,0,0],[100+step,0,0]],dtype=np.float32),'critic':np.zeros((2,8),dtype=np.float32)},
              np.zeros((2,2)),np.array([1,2 if step==0 else 0]),np.array([step==0,True]),torch.zeros(2),torch.zeros(2))
    b.compute_returns_and_advantage(torch.zeros(2),np.array([True,True]))
    return b

def test_padding_excluded_across_all_epochs():
    b=filled_buffer()
    for _ in range(3):
        samples=list(b.get(2))
        identifiers=torch.cat([s.observations['actor'][:,0] for s in samples]).tolist()
        assert sorted(identifiers)==[0,1,2,100]
        assert sum(len(s.advantages) for s in samples)==4

def test_terminal_does_not_bootstrap_through_padding():
    b=filled_buffer()
    assert b.returns[0,1]==pytest.approx(2)
    assert b.returns[:,0].tolist()==pytest.approx([2.71,1.9,1])

def test_missing_or_reused_mask_fails():
    b=ActiveRolloutBuffer(2,observation_space(),spaces.Box(-1,1,shape=(2,),dtype=np.float32),n_envs=1)
    args=({'actor':np.zeros((1,3)),'critic':np.zeros((1,8))},np.zeros((1,2)),np.zeros(1),np.ones(1),torch.zeros(1),torch.zeros(1))
    with pytest.raises(RuntimeError,match='fresh'):b.add(*args)
    b.pending_valid=np.ones(1,dtype=bool);b.add(*args)
    with pytest.raises(RuntimeError,match='fresh'):b.add(*args)
