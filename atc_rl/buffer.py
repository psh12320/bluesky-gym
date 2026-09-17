"""Exclude absorbing aircraft slots from every PPO loss and normalization."""
import numpy as np
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

class ActiveRolloutBuffer(DictRolloutBuffer):
    def reset(self):
        super().reset()
        self.valid=np.zeros((self.buffer_size,self.n_envs),dtype=bool)
        self.pending_valid=None
        self.valid_indices=None

    def add(self,*args,**kwargs):
        if self.pending_valid is None:
            raise RuntimeError('Every rollout transition requires a fresh active-aircraft mask')
        mask=np.asarray(self.pending_valid,dtype=bool)
        if mask.shape!=(self.n_envs,):raise ValueError('Wrong active-aircraft mask shape')
        self.valid[self.pos]=mask
        self.pending_valid=None
        super().add(*args,**kwargs)

    def get(self,batch_size=None):
        if not self.full:raise RuntimeError('Cannot sample an incomplete on-policy rollout')
        if not self.generator_ready:
            self.valid_indices=np.flatnonzero(self.swap_and_flatten(self.valid).reshape(-1))
            if not len(self.valid_indices):raise RuntimeError('No live transitions in rollout')
            for key,obs in self.observations.items():self.observations[key]=self.swap_and_flatten(obs)
            for name in ('actions','values','log_probs','advantages','returns'):
                self.__dict__[name]=self.swap_and_flatten(self.__dict__[name])
            self.generator_ready=True
        indices=np.random.permutation(self.valid_indices)
        size=len(indices) if batch_size is None else batch_size
        if size<1:raise ValueError('Batch size must be positive')
        for start in range(0,len(indices),size):yield self._get_samples(indices[start:start+size])

class RolloutAccounting(BaseCallback):
    """Give the buffer fresh masks and count live experience separately from padding."""
    def __init__(self):
        super().__init__()
        self.live_transitions=0
        self.padded_transitions=0
        self.completed_aircraft=[]
        self.completed_learning_returns=[]
        self.rollouts=0
        self.completed_worlds=0

    def _on_step(self):
        infos=self.locals['infos']
        if any('inactive' not in info for info in infos):
            raise RuntimeError('World adapter did not report aircraft activity')
        mask=np.array([not info['inactive'] for info in infos],dtype=bool)
        self.completed_worlds+=sum(bool(info.get('world_completed',False)) for info in infos)
        self.model.rollout_buffer.pending_valid=mask
        self.live_transitions+=int(mask.sum())
        self.padded_transitions+=int((~mask).sum())
        for info,valid in zip(infos,mask):
            if valid and info.get('aircraft_done'):
                self.completed_aircraft.append(info['metrics'])
                self.completed_learning_returns.append(float(info['learning_episode_return']))
        return True

    def _on_rollout_end(self):
        self.rollouts+=1
