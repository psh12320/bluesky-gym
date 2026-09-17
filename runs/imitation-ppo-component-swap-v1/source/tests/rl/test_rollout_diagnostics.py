import numpy as np
import pytest
import torch
from gymnasium import spaces
from atc_rl.buffer import ActiveRolloutBuffer
from atc_rl.rollout_diagnostics import rollout_diagnostics


def buffer():
    obs_space=spaces.Dict({'actor':spaces.Box(-1,1,(1,),dtype=np.float32),
                           'critic':spaces.Box(-1,1,(1,),dtype=np.float32)})
    b=ActiveRolloutBuffer(3,obs_space,spaces.Box(-1,1,(2,),dtype=np.float32),n_envs=2,gamma=0.)
    for step in range(3):
        b.pending_valid=np.array([True,step==0])
        b.add({'actor':np.zeros((2,1)),'critic':np.zeros((2,1))},
              np.array([[.2,1.2 if step==0 else 0.],[0.,0.] if step==0 else [10.,10.]]),
              np.array([1.,3. if step==0 else 0.]),np.ones(2),
              torch.tensor([0.,0. if step==0 else 500.]),torch.zeros(2))
    b.compute_returns_and_advantage(torch.zeros(2),np.ones(2))
    return b


def flattened():
    b=buffer();list(b.get(2));return b


def test_live_fit_ignores_absorbing_slots_and_raw_action_clipping():
    b=flattened();result=rollout_diagnostics(b)
    assert result=={'rollout_live_samples':4,'rollout_return_variance_live':.75,
        'rollout_value_mse_live':3.,'rollout_value_explained_variance_live':0.,
        'rollout_action_box_clip_fraction_live':.25}
    inactive=np.setdiff1d(np.arange(6),b.valid_indices)
    b.values[inactive]=float('nan');b.returns[inactive]=999.;b.actions[inactive]=float('nan')
    assert rollout_diagnostics(b)==result


def test_diagnostics_preserve_buffer_and_random_state():
    b=flattened();before={key:getattr(b,key).copy() for key in ('values','returns','actions','valid','valid_indices')}
    state=np.random.get_state();torch_state=torch.random.get_rng_state().clone()
    rollout_diagnostics(b)
    after_state=np.random.get_state()
    assert state[0]==after_state[0] and np.array_equal(state[1],after_state[1]) and state[2:]==after_state[2:]
    assert torch.equal(torch_state,torch.random.get_rng_state())
    for key,value in before.items():np.testing.assert_array_equal(value,getattr(b,key))


def test_zero_variance_is_undefined_instead_of_a_false_perfect_fit():
    b=flattened();b.returns[b.valid_indices]=2.
    result=rollout_diagnostics(b)
    assert result['rollout_value_explained_variance_live'] is None
    assert result['rollout_value_mse_live']==4.


def test_diagnostics_reject_unflattened_or_mismatched_masks():
    b=buffer()
    with pytest.raises(ValueError,match='flattened'):rollout_diagnostics(b)
    list(b.get(2));b.valid_indices=np.array([0,1,2])
    with pytest.raises(ValueError,match='mask'):rollout_diagnostics(b)


def test_nonfinite_live_samples_are_not_silently_removed():
    b=flattened();b.values[0]=float('nan')
    with pytest.raises(ValueError,match='Non-finite'):rollout_diagnostics(b)