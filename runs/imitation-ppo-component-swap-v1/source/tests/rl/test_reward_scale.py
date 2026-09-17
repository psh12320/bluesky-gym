import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from atc_rl.reward_scale import ScaledLearningRewards


class ScriptedWorld(VecEnv):
    def __init__(self):
        self.actions=None
        self.position=0
        self.native_metrics={'waypoint_reached':1.0,'total_reward':30.0}
        super().__init__(2,spaces.Box(-1,1,(1,),dtype=np.float32),spaces.Box(-1,1,(1,),dtype=np.float32))

    def reset(self):
        self.position=0
        return np.zeros((2,1),dtype=np.float32)

    def step_async(self,actions):self.actions=actions

    def step_wait(self):
        self.position+=1
        if self.position==1:
            rewards=np.array([10.0,4.0],dtype=np.float32)
            dones=np.array([False,True])
            infos=[{'inactive':False,'native_reward':10.0},
                {'inactive':False,'native_reward':4.0,'learning_episode_return':4.0,'episode':{'r':4.0,'l':1}}]
        else:
            rewards=np.array([25.0,0.0],dtype=np.float32)
            dones=np.array([True,True])
            infos=[{'inactive':False,'native_reward':20.0,'shaping_reward':5.0,
                    'learning_episode_return':35.0,'episode':{'r':35.0,'l':2},'metrics':self.native_metrics},
                   {'inactive':True,'aircraft_done':False}]
        self.last_packet=(np.full((2,1),self.position,dtype=np.float32),rewards,dones,infos)
        return self.last_packet

    def close(self):pass
    def get_attr(self,attr_name,indices=None):return [None,None]
    def set_attr(self,attr_name,value,indices=None):raise NotImplementedError
    def env_method(self,method_name,*args,indices=None,**kwargs):raise NotImplementedError
    def env_is_wrapped(self,wrapper_class,indices=None):return [False,False]


@pytest.mark.parametrize('scale',[1.0,.01,3.0])
def test_scaling_preserves_actions_scoring_padding_and_episode_boundaries(scale):
    original=ScriptedWorld();wrapped=ScaledLearningRewards(original,scale)
    assert np.array_equal(wrapped.reset(),np.zeros((2,1)))
    actions=np.array([[.2],[-.3]],dtype=np.float32)
    obs,reward,done,info=wrapped.step(actions)
    assert original.actions is actions
    np.testing.assert_allclose(reward,np.array([10.,4.])*scale)
    assert info[1]['learning_episode_return']==4*scale
    obs,reward,done,info=wrapped.step(actions)
    packet=original.last_packet
    assert obs is packet[0] and done is packet[2]
    np.testing.assert_allclose(reward,np.array([25.,0.])*scale)
    assert info[1]=={'inactive':True,'aircraft_done':False}
    assert info[0]['metrics']=={'waypoint_reached':1.0,'total_reward':30.0}
    assert info[0]['native_reward']==20 and info[0]['shaping_reward']==5
    assert info[0]['learning_episode_return']==35*scale
    assert info[0]['unscaled_learning_episode_return']==35
    assert info[0]['episode']=={'r':35*scale,'l':2}
    assert packet[3][0]['learning_episode_return']==35
    assert packet[3][0]['episode']['r']==35
    wrapped.close()


@pytest.mark.parametrize('scale',[0,-1,float('nan'),float('inf')])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError,match='finite and positive'):
        ScaledLearningRewards(ScriptedWorld(),scale)
