"""Goal-relative action representation for learned heading and speed control."""
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper


def heading_commands(actions,observations,cosine_index,sine_index):
    """Convert learned +/-90 degree goal offsets to bounded +/-45 degree turns.

    Only the current actor-visible bearing is used. With route guidance enabled,
    that observation carries its reference bearing; otherwise it is the original
    destination bearing. Speed commands are passed through unchanged.
    """
    actions=np.asarray(actions,dtype=np.float32)
    observations=np.asarray(observations,dtype=np.float32)
    if actions.ndim!=2 or actions.shape[1]!=2 or observations.ndim!=2 or len(actions)!=len(observations):
        raise ValueError('Expected one heading/speed action and observation per aircraft')
    if not np.isfinite(actions).all() or np.any(np.abs(actions)>1):
        raise ValueError('Latent actions must be finite and bounded')
    cosine=observations[:,cosine_index];sine=observations[:,sine_index]
    if not np.isfinite(cosine).all() or not np.isfinite(sine).all() or np.any(cosine*cosine+sine*sine<.5):
        raise ValueError('Missing or invalid destination bearing')
    desired=-np.arctan2(sine,cosine)+actions[:,0]*np.float32(np.pi/2)
    desired=np.where(desired>np.float32(np.pi),desired-np.float32(2*np.pi),desired)
    desired=np.where(desired<-np.float32(np.pi),desired+np.float32(2*np.pi),desired)
    result=actions.copy()
    result[:,0]=np.clip(desired/(np.pi/4),-1,1)
    return result


class GoalOffsetActions(BaseParallelWrapper):
    """Apply the action representation before existing optional conflict filtering.

    Observations, rewards, terminal flags and scoring records are returned from
    the underlying environment unchanged. A zero policy follows the bearing;
    its performance must therefore be measured separately from learned gains.
    """
    def __init__(self,environment):
        super().__init__(environment)
        world=environment.unwrapped
        if world.d_heading!=45:raise ValueError('Goal-offset experiments use 45-degree command bounds')
        dictionary=world.observation_space(environment.possible_agents[0])
        offsets={};cursor=0
        for name,space in dictionary.spaces.items():
            offsets[name]=cursor;cursor+=spaces.flatdim(space)
        self.cosine_index=offsets['cos_drift'];self.sine_index=offsets['sin_drift']
        self._observations=None
        self.last_commands={}

    def reset(self,seed=None,options=None):
        observations,infos=self.env.reset(seed=seed,options=options)
        self._observations={agent:np.asarray(value,dtype=np.float32).copy() for agent,value in observations.items()}
        self.last_commands={}
        return observations,infos

    def step(self,actions):
        if self._observations is None:raise RuntimeError('Reset before applying goal-offset actions')
        agents=list(actions)
        if agents:
            commands=heading_commands([actions[a] for a in agents],[self._observations[a] for a in agents],
                                      self.cosine_index,self.sine_index)
            mapped={agent:commands[i] for i,agent in enumerate(agents)}
        else:mapped={}
        self.last_commands={agent:value.copy() for agent,value in mapped.items()}
        result=self.env.step(mapped)
        self._observations={agent:np.asarray(value,dtype=np.float32).copy() for agent,value in result[0].items()}
        return result
