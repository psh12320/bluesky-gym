import numpy as np
import pytest
from gymnasium import spaces
from pettingzoo import ParallelEnv
from atc.baselines import goal_tracker
from atc_rl.goal_offset import GoalOffsetActions,heading_commands


@pytest.mark.parametrize('drift,offset,expected',[(0,0,0),(90,0,-1),(-90,0,1),(0,.5,1),(-170,.5,-1),(170,-.5,1)])
def test_goal_offset_turns_and_wraparound(drift,offset,expected):
    angle=np.deg2rad(drift)
    command=heading_commands([[offset,-.25]],[[np.cos(angle),np.sin(angle)]],0,1)
    np.testing.assert_allclose(command,[[expected,-.25]],atol=1e-6)


def test_zero_offset_exactly_matches_observed_bearing_tracker():
    dictionary=spaces.Dict({'cos_drift':spaces.Box(-1,1,(1,),dtype=np.float32),
                            'sin_drift':spaces.Box(-1,1,(1,),dtype=np.float32)})
    angles=np.linspace(-np.pi,np.pi,10001,dtype=np.float32)
    observations=np.stack((np.cos(angles),np.sin(angles)),axis=-1)
    actions=np.zeros((len(angles),2),dtype=np.float32);actions[:,1]=1
    np.testing.assert_array_equal(heading_commands(actions,observations,0,1),goal_tracker(dictionary,1)(observations))


class ExampleWorld(ParallelEnv):
    possible_agents=['a'];d_heading=45
    def __init__(self):self.agents=['a'];self.angle=0.;self.received=None
    def observation_space(self,agent):
        return spaces.Dict({'cos_drift':spaces.Box(-1,1,(1,)),'sin_drift':spaces.Box(-1,1,(1,))})
    def reset(self,seed=None,options=None):
        self.angle=0.;self.agents=['a']
        return {'a':np.array([1.,0.])},{'a':{'reset_marker':1}}
    def step(self,actions):
        self.received=actions;self.angle=np.pi/2
        self.returned=({'a':np.array([0.,1.])},{'a':-3.},{'a':False},{'a':False},{'a':{'intrusion_time':5.}})
        return self.returned


def test_wrapper_preserves_outputs_and_uses_updated_observation():
    world=ExampleWorld();wrapper=GoalOffsetActions(world)
    observations,infos=wrapper.reset()
    assert infos=={'a':{'reset_marker':1}}
    result=wrapper.step({'a':np.array([0.,.4])})
    assert result is world.returned
    np.testing.assert_allclose(world.received['a'],[0,.4])
    wrapper.step({'a':np.array([0.,.4])})
    np.testing.assert_allclose(world.received['a'],[-1,.4])
    assert world.returned[1:]==({'a':-3.},{'a':False},{'a':False},{'a':{'intrusion_time':5.}})


@pytest.mark.parametrize('action',[[float('nan'),0],[2,0]])
def test_invalid_latent_actions_are_rejected(action):
    with pytest.raises(ValueError,match='finite and bounded'):
        heading_commands([action],[[1,0]],0,1)
