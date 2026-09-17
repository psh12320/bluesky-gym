import numpy as np
import pytest
from gymnasium import spaces
from pettingzoo import ParallelEnv
from atc.conflicts import FEATURE_NAMES
from atc.envs import FlattenObservations
from atc_rl.feature_mask import MaskConflictFeatures
from atc_rl.world_pool import WorldPool

class ExampleWorld(ParallelEnv):
    possible_agents=["a"]
    def __init__(self):
        self.agents=["a"]
        self.dictionary=spaces.Dict({"cos_drift":spaces.Box(-1,1,(1,),dtype=np.float64),
                                     "x_r":spaces.Box(-100,100,(9,),dtype=np.float64),
                                     **{name:spaces.Box(0,4,(9,),dtype=np.float64) for name in FEATURE_NAMES}})
    def observation_space(self,agent):return self.dictionary
    def action_space(self,agent):return spaces.Box(-1,1,(2,),dtype=np.float32)
    def reset(self,seed=None,options=None):
        self.observation={name:np.full(space.shape,.5) for name,space in self.dictionary.spaces.items()}
        self.info={"a":{"marker":17}}
        return {"a":self.observation},self.info
    def step(self,actions):
        self.received=actions
        obs,_=self.reset()
        self.scoring=({"a":-3.},{"a":False},{"a":False},{"a":{"metric":7.}})
        return obs,*self.scoring

def test_mask_retains_spaces_and_only_zeros_the_added_channels():
    original=ExampleWorld();flat=FlattenObservations(original);masked=MaskConflictFeatures(flat)
    assert masked.observation_space("a")==flat.observation_space("a")
    observation,info=masked.reset()
    restored=spaces.unflatten(original.dictionary,observation["a"])
    for name in FEATURE_NAMES:assert np.count_nonzero(restored[name])==0
    np.testing.assert_array_equal(restored["x_r"],original.observation["x_r"])
    np.testing.assert_array_equal(restored["cos_drift"],original.observation["cos_drift"])
    assert info is original.info and np.all(original.observation["traffic_dcpa"]==.5)
    assert masked.observation_space("a").contains(observation["a"])

def test_mask_does_not_modify_actions_rewards_terminals_or_scoring():
    original=ExampleWorld();masked=MaskConflictFeatures(FlattenObservations(original))
    masked.reset();actions={"a":np.array([.2,-.1],dtype=np.float32)}
    result=masked.step(actions)
    assert original.received is actions
    assert all(result[i+1] is item for i,item in enumerate(original.scoring))

def test_mask_rejects_wrong_wrapper_order_and_missing_feature_flag(tmp_path):
    with pytest.raises(ValueError,match="flattening"):MaskConflictFeatures(ExampleWorld())
    with pytest.raises(ValueError,match="requires conflict_features"):
        WorldPool(1,tmp_path,mask_conflict_features=True)
